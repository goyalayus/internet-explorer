from __future__ import annotations

from pydantic import BaseModel, Field

from internet_explorer.browser_delegate import BrowserDelegationManager
from internet_explorer.api_probe import ApiProbeService
from internet_explorer.canonicalize import canonical_domain, canonicalize_url
from internet_explorer.config import AppConfig
from internet_explorer.fetcher import AsyncWebFetcher, analyze_page, detect_render_profile, extract_relevant_links
from internet_explorer.llm import LLMClient
from internet_explorer.models import EvaluationDecision, PageEvidence, UrlCandidate, UrlEvaluation
from internet_explorer.telemetry import Telemetry


class _DecisionEnvelope(BaseModel):
    useful: bool
    relevance_score: float
    outcome: str
    why_useful: str
    how_to_use: str
    api_stage: str
    notes: list[str] = Field(default_factory=list)


DECISION_SYSTEM_PROMPT = """
You decide whether a website is a useful datasource for a given intent.
Be strict and evidence-based. Return only JSON.
"""

VALID_OUTCOMES = {
    "data_on_site",
    "api_available",
    "contact_sales_only",
    "paywall",
    "irrelevant",
    "unknown",
}
VALID_API_STAGES = {"none", "api_detected", "api_accessible", "api_relevant", "api_viable"}


class UrlEvaluator:
    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        fetcher: AsyncWebFetcher,
        telemetry: Telemetry,
        browser_manager: BrowserDelegationManager,
    ) -> None:
        self.config = config
        self.llm = llm
        self.fetcher = fetcher
        self.telemetry = telemetry
        self.browser_manager = browser_manager
        self.api_probe = ApiProbeService(telemetry, llm, timeout_seconds=max(config.request_timeout_seconds, 30))

    async def evaluate(self, *, intent: str, candidate: UrlCandidate) -> UrlEvaluation:
        started = self.telemetry.timed()
        try:
            page_evidence: list[PageEvidence] = []
            browser_result = None

            homepage = await self.fetcher.fetch(candidate.canonical_url)
            self.telemetry.emit(
                phase="page_fetch",
                actor="system",
                strategy_id=candidate.strategy_id,
                query_id=candidate.query_id,
                url_id=candidate.url_id,
                input_payload={"url": candidate.canonical_url},
                output_summary={"status_code": homepage.status_code, "final_url": homepage.final_url},
                decision="homepage_fetched",
            )
            render_profile = detect_render_profile(homepage.html)
            self.telemetry.emit(
                phase="render_detect",
                actor="normal_agent",
                url_id=candidate.url_id,
                input_payload={"url": homepage.final_url},
                output_summary={"render_profile": render_profile},
                decision=render_profile,
            )
            homepage_evidence = analyze_page(homepage)
            page_evidence.append(homepage_evidence)

            if render_profile != "csr_shell":
                links = extract_relevant_links(homepage.final_url, homepage.html, self.config.max_internal_links)
                for depth, link in enumerate(links[: self.config.max_internal_links], start=1):
                    if depth > self.config.max_internal_links or depth > self.config.max_link_depth * self.config.max_internal_links:
                        break
                    if canonical_domain(link) != candidate.domain:
                        continue
                    fetched = await self.fetcher.fetch(link)
                    detail_evidence = analyze_page(fetched)
                    page_evidence.append(detail_evidence)
                    self.telemetry.emit(
                        phase="link_follow",
                        actor="normal_agent",
                        url_id=candidate.url_id,
                        input_payload={"url": link},
                        output_summary={"status_code": fetched.status_code, "final_url": fetched.final_url},
                        decision="followed",
                    )
                    if detail_evidence.captcha_present:
                        break
                    if detail_evidence.paywall_present or detail_evidence.api_signal.detected or detail_evidence.contact_sales_present:
                        break
            else:
                browser_result = await self.browser_manager.delegate(url=candidate.canonical_url, intent=intent, url_id=candidate.url_id)

            if browser_result and browser_result.classification == "paywall":
                decision = EvaluationDecision(
                    useful=True,
                    relevance_score=max(browser_result.confidence, 0.5),
                    outcome="paywall",
                    why_useful=browser_result.why_useful or "Relevant source discovered, but access appears gated behind payment.",
                    how_to_use=browser_result.how_to_use or "Store for human review and possible paid access evaluation.",
                    api_stage="none",
                    notes=["browser_detected_paywall"],
                )
            else:
                decision = await self._decide(intent=intent, candidate=candidate, render_profile=render_profile, evidence=page_evidence, browser_result=browser_result)

            api_probe = None
            merged_signal = homepage_evidence.api_signal
            for evidence in page_evidence[1:]:
                merged_signal.detected = merged_signal.detected or evidence.api_signal.detected
                merged_signal.doc_links.extend(link for link in evidence.api_signal.doc_links if link not in merged_signal.doc_links)
                merged_signal.openapi_links.extend(link for link in evidence.api_signal.openapi_links if link not in merged_signal.openapi_links)
                merged_signal.graphql_hints.extend(token for token in evidence.api_signal.graphql_hints if token not in merged_signal.graphql_hints)
                merged_signal.auth_required = merged_signal.auth_required or evidence.api_signal.auth_required

            if merged_signal.detected:
                api_probe = await self.api_probe.probe(
                    url_id=candidate.url_id,
                    intent=intent,
                    evidence=page_evidence,
                )
                if api_probe and api_probe.error:
                    decision.notes.append(f"api_probe_error:{api_probe.error[:200]}")
                if api_probe and api_probe.url:
                    decision.notes.append(f"api_probe_url:{api_probe.url}")
                if api_probe and api_probe.accessible and decision.api_stage == "api_detected":
                    decision.api_stage = "api_accessible"
                if api_probe and api_probe.relevant_guess and decision.api_stage in {"api_detected", "api_accessible"}:
                    decision.api_stage = "api_relevant"
                if api_probe and api_probe.viable_guess:
                    decision.api_stage = "api_viable"

            evaluation = UrlEvaluation(
                url_id=candidate.url_id,
                canonical_url=candidate.canonical_url,
                domain=candidate.domain,
                novelty=candidate.novelty,
                render_profile=render_profile,
                outcome=decision.outcome,
                useful=decision.useful,
                relevance_score=decision.relevance_score,
                why_useful=decision.why_useful,
                how_to_use=decision.how_to_use,
                api_stage=decision.api_stage,
                browser_delegated=browser_result is not None,
                data_on_site=decision.outcome == "data_on_site",
                api_signal=merged_signal,
                api_probe=api_probe,
                contact_sales_only=decision.outcome == "contact_sales_only",
                paywall_present=decision.outcome == "paywall" or any(e.paywall_present for e in page_evidence),
                auth_required=any(e.auth_required for e in page_evidence) or bool(browser_result and browser_result.auth_required),
                captcha_present=any(e.captcha_present for e in page_evidence) or bool(browser_result and browser_result.captcha_present),
                evidence=page_evidence,
                browser_result=browser_result,
                notes=decision.notes,
            )
            self.telemetry.emit(
                phase="final_decision",
                actor="normal_agent",
                strategy_id=candidate.strategy_id,
                query_id=candidate.query_id,
                url_id=candidate.url_id,
                input_payload={"candidate": candidate.model_dump()},
                output_summary=evaluation.model_dump(mode="json"),
                decision=evaluation.outcome,
                latency_ms=self.telemetry.elapsed_ms(started),
            )
            return evaluation
        except Exception as exc:
            evaluation = UrlEvaluation(
                url_id=candidate.url_id,
                canonical_url=candidate.canonical_url,
                domain=candidate.domain,
                novelty=candidate.novelty,
                render_profile="hybrid",
                outcome="unknown",
                useful=False,
                why_useful="Evaluation failed before the source could be fully assessed.",
                how_to_use="Inspect the logged events and retry this URL.",
                notes=[f"evaluation_error:{type(exc).__name__}", str(exc)],
            )
            self.telemetry.emit(
                phase="final_decision",
                actor="system",
                strategy_id=candidate.strategy_id,
                query_id=candidate.query_id,
                url_id=candidate.url_id,
                input_payload={"candidate": candidate.model_dump()},
                output_summary=evaluation.model_dump(mode="json"),
                decision="error_fallback",
                latency_ms=self.telemetry.elapsed_ms(started),
                error_code=type(exc).__name__,
            )
            return evaluation

    async def _decide(
        self,
        *,
        intent: str,
        candidate: UrlCandidate,
        render_profile: str,
        evidence: list[PageEvidence],
        browser_result,
    ) -> EvaluationDecision:
        prompt = {
            "intent": intent,
            "candidate": candidate.model_dump(),
            "render_profile": render_profile,
            "browser_result": browser_result.model_dump(mode="json") if browser_result else None,
            "evidence": [item.model_dump(mode="json") for item in evidence],
        }
        response = await self.llm.complete_json(
            system_prompt=DECISION_SYSTEM_PROMPT,
            user_prompt=(
                "Decide whether this website is a useful datasource for the intent.\n\n"
                f"{prompt}\n\n"
                "Rules:\n"
                "- `data_on_site` when valuable data appears directly on the site or portal.\n"
                "- `api_available` when API/docs/endpoints are present and seem workable.\n"
                "- `contact_sales_only` when only sales/demo access exists.\n"
                "- `paywall` when payment/upgrade is required for access.\n"
                "- `irrelevant` when it does not materially help the intent.\n"
                "- Novelty matters, but a non-novel source can still be useful.\n"
                "- Be strict. Return only JSON.\n"
            ),
            schema=_DecisionEnvelope,
            temperature=0.0,
            max_completion_tokens=2048,
        )
        outcome = response.outcome if response.outcome in VALID_OUTCOMES else "unknown"
        api_stage = response.api_stage if response.api_stage in VALID_API_STAGES else "none"
        return EvaluationDecision(
            useful=response.useful,
            relevance_score=float(response.relevance_score),
            outcome=outcome,
            why_useful=response.why_useful,
            how_to_use=response.how_to_use,
            api_stage=api_stage,
            notes=response.notes,
        )
