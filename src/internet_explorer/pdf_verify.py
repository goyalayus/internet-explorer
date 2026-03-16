from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from internet_explorer.canonicalize import canonicalize_url, homepage_url_for_domain, registrable_domain
from internet_explorer.fetcher import AsyncWebFetcher
from internet_explorer.llm import LLMClient
from internet_explorer.models import PdfVerificationResult, SourceEvidenceItem
from internet_explorer.telemetry import Telemetry


PDF_VERIFY_SYSTEM_PROMPT = """
You verify whether a PDF is relevant evidence for a datasource-discovery intent.
Return only strict JSON.
"""


class _PdfDecisionEnvelope(BaseModel):
    relevant: bool
    title: str = ""
    summary: str = ""
    reasoning: str
    extracted_signals: list[str] = Field(default_factory=list)


class PdfVerifierService:
    def __init__(self, fetcher: AsyncWebFetcher, llm: LLMClient, telemetry: Telemetry) -> None:
        self.fetcher = fetcher
        self.llm = llm
        self.telemetry = telemetry

    async def verify(self, *, url_id: str, intent: str, pdf_url: str) -> PdfVerificationResult:
        started = self.telemetry.timed()
        canonical_pdf_url = canonicalize_url(pdf_url)
        try:
            fetched = await self.fetcher.fetch_binary(canonical_pdf_url)
            if fetched.status_code and fetched.status_code >= 400:
                result = PdfVerificationResult(
                    url=canonical_pdf_url,
                    final_url=fetched.final_url,
                    status_code=fetched.status_code,
                    content_type=fetched.content_type,
                    relevant=False,
                    reasoning=f"PDF fetch returned HTTP {fetched.status_code}.",
                    summary="PDF could not be downloaded successfully.",
                    fallback_urls=_fallback_urls(fetched.final_url),
                    error=f"http_status:{fetched.status_code}",
                )
                self._emit(url_id=url_id, intent=intent, pdf_url=canonical_pdf_url, result=result, decision="pdf_fetch_failed", started=started)
                return result

            decision = await self.llm.complete_pdf_json(
                system_prompt=PDF_VERIFY_SYSTEM_PROMPT,
                user_prompt=(
                    "Determine whether this PDF is good evidence for the intent and summarize what it proves.\n\n"
                    f"Intent:\n{intent}\n\n"
                    f"PDF URL:\n{canonical_pdf_url}\n\n"
                    "Return JSON with EXACT keys:\n"
                    "- relevant (boolean)\n"
                    "- title (string)\n"
                    "- summary (string)\n"
                    "- reasoning (string that explains both why this PDF matches the intent and what rough recurring access path this suggests for the site/domain)\n"
                    "- extracted_signals (list of short strings)\n"
                ),
                pdf_bytes=fetched.content_bytes,
                schema=_PdfDecisionEnvelope,
                temperature=0.0,
                max_completion_tokens=1400,
            )
            source_evidence = SourceEvidenceItem(
                kind="pdf",
                url=canonical_pdf_url,
                title=decision.title.strip(),
                summary=(decision.summary.strip() or decision.reasoning.strip())[:1200],
            )
            result = PdfVerificationResult(
                url=canonical_pdf_url,
                final_url=fetched.final_url,
                title=decision.title.strip(),
                status_code=fetched.status_code,
                content_type=fetched.content_type,
                relevant=decision.relevant,
                reasoning=decision.reasoning.strip(),
                summary=decision.summary.strip(),
                extracted_signals=[item.strip() for item in decision.extracted_signals if item.strip()][:16],
                fallback_urls=_fallback_urls(fetched.final_url),
                source_evidence=source_evidence,
            )
            self._emit(url_id=url_id, intent=intent, pdf_url=canonical_pdf_url, result=result, decision="pdf_verified", started=started)
            return result
        except ValueError as exc:
            message = str(exc)
            if message.startswith("pdf_too_large_for_inline_gemini:"):
                fallback_url = fetched.final_url if "fetched" in locals() else canonical_pdf_url
                extracted_signals = _keyword_signals_from_url(fallback_url)
                strong_signal = any(item in {"rfp", "tender", "procurement", "solicitation", "bid"} for item in extracted_signals)
                reasoning = (
                    "PDF is too large for inline verification in this run. "
                    "Classified using URL/path signals and parent fallback pages."
                )
                result = PdfVerificationResult(
                    url=canonical_pdf_url,
                    final_url=fallback_url,
                    status_code=fetched.status_code if "fetched" in locals() else None,
                    content_type=fetched.content_type if "fetched" in locals() else "application/pdf",
                    relevant=strong_signal,
                    reasoning=reasoning,
                    summary="Large PDF fallback used; confirm relevance from nearby listing pages.",
                    extracted_signals=extracted_signals[:12],
                    fallback_urls=_fallback_urls(fallback_url),
                    source_evidence=(
                        SourceEvidenceItem(
                            kind="pdf",
                            url=canonical_pdf_url,
                            title="",
                            summary=(
                                "Large PDF encountered; URL-based signals suggest procurement relevance."
                                if strong_signal
                                else "Large PDF encountered; no strong procurement signal in URL path."
                            ),
                        )
                        if strong_signal
                        else None
                    ),
                    error=message,
                )
                self._emit(
                    url_id=url_id,
                    intent=intent,
                    pdf_url=canonical_pdf_url,
                    result=result,
                    decision="pdf_verify_large_fallback",
                    started=started,
                    error_code="PdfTooLarge",
                )
                return result
            raise
        except Exception as exc:
            error_text = f"{type(exc).__name__}:{exc}"
            if "The document has no pages." in error_text:
                fallback_url = fetched.final_url if "fetched" in locals() else canonical_pdf_url
                result = PdfVerificationResult(
                    url=canonical_pdf_url,
                    final_url=fallback_url,
                    status_code=fetched.status_code if "fetched" in locals() else None,
                    content_type=fetched.content_type if "fetched" in locals() else "application/pdf",
                    relevant=False,
                    reasoning="PDF appears unreadable or empty for model-based parsing. Use parent listing pages for source assessment.",
                    summary="Unreadable/empty PDF fallback used.",
                    extracted_signals=_keyword_signals_from_url(fallback_url)[:12],
                    fallback_urls=_fallback_urls(fallback_url),
                    error=error_text,
                )
                self._emit(
                    url_id=url_id,
                    intent=intent,
                    pdf_url=canonical_pdf_url,
                    result=result,
                    decision="pdf_verify_unreadable_fallback",
                    started=started,
                    error_code="PdfUnreadable",
                )
                return result
            result = PdfVerificationResult(
                url=canonical_pdf_url,
                relevant=False,
                reasoning="PDF verification failed before the document could be judged.",
                summary="Inspect telemetry and retry this document later.",
                fallback_urls=_fallback_urls(canonical_pdf_url),
                error=error_text,
            )
            self._emit(url_id=url_id, intent=intent, pdf_url=canonical_pdf_url, result=result, decision="pdf_verify_error", started=started, error_code=type(exc).__name__)
            return result

    def _emit(
        self,
        *,
        url_id: str,
        intent: str,
        pdf_url: str,
        result: PdfVerificationResult,
        decision: str,
        started: float,
        error_code: str | None = None,
    ) -> None:
        self.telemetry.emit(
            phase="pdf_verify",
            actor="normal_agent",
            url_id=url_id,
            input_payload={"intent": intent, "pdf_url": pdf_url},
            output_summary=result.model_dump(mode="json"),
            decision=decision,
            latency_ms=self.telemetry.elapsed_ms(started),
            error_code=error_code,
        )


def pdf_verification_tool_payload(result: PdfVerificationResult) -> dict[str, Any]:
    return {
        "relevant": result.relevant,
        "title": result.title,
        "summary": result.summary,
        "reasoning": result.reasoning,
        "extracted_signals": result.extracted_signals,
        "fallback_urls": result.fallback_urls,
        "error": result.error,
    }


def _fallback_urls(url: str) -> list[str]:
    canonical = canonicalize_url(url)
    parsed = urlparse(canonical)
    domain = registrable_domain(canonical)
    urls: list[str] = []
    homepage = homepage_url_for_domain(domain, scheme=parsed.scheme or "https")
    if homepage:
        urls.append(homepage)
    path = PurePosixPath(parsed.path or "/")
    parents = list(path.parents)[:3]
    for parent in parents:
        if str(parent) in {"", "."}:
            continue
        parent_url = f"{parsed.scheme or 'https'}://{parsed.netloc}{str(parent)}"
        if not parent_url.endswith("/"):
            parent_url += "/"
        canonical_parent = canonicalize_url(parent_url)
        if canonical_parent not in urls:
            urls.append(canonical_parent)
    return urls[:4]


def _keyword_signals_from_url(url: str) -> list[str]:
    lowered = canonicalize_url(url).lower()
    mapping = (
        ("data annotation", ("data-annotation", "data_annotation", "annotation", "labeling", "labelling")),
        ("rfp", ("rfp", "request-for-proposal", "request_for_proposal")),
        ("tender", ("tender", "tenders")),
        ("procurement", ("procurement", "purchase", "purchasing")),
        ("solicitation", ("solicitation", "solicitations")),
        ("bid", ("bid", "bids")),
        ("ai", ("ai", "artificial-intelligence", "machine-learning", "machine_learning", "ml")),
    )
    found: list[str] = []
    for label, patterns in mapping:
        if any(pattern in lowered for pattern in patterns):
            found.append(label)
    return found
