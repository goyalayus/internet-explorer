from __future__ import annotations

from pydantic import BaseModel, Field

from internet_explorer.api_probe import ApiProbeService
from internet_explorer.browser_delegate import BrowserDelegationManager
from internet_explorer.config import AppConfig
from internet_explorer.fetcher import AsyncWebFetcher, analyze_page, detect_render_profile
from internet_explorer.llm import LLMClient
from internet_explorer.models import ApiSignal, EvaluationDecision, UrlCandidate, UrlEvaluation
from internet_explorer.site_graph import SiteGraph
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
            page_evidence = []
            browser_result = None
            render_profile = "hybrid"

            site_graph = SiteGraph(
                config=self.config,
                telemetry=self.telemetry,
                url_id=candidate.url_id,
                intent=intent,
                seed_url=candidate.canonical_url,
                domain=candidate.domain,
            )
            await site_graph.bootstrap(self.fetcher)

            for visit_index in range(self.config.max_site_graph_visits):
                frontier = site_graph.next_frontier(limit=1)
                if not frontier:
                    break
                target = frontier[0]
                try:
                    fetched = await self.fetcher.fetch(target.canonical_url)
                except Exception as exc:
                    site_graph.update_node_from_tool(
                        url=target.canonical_url,
                        title=target.title,
                        page_type_guess=target.page_type_guess,
                        summary=f"Fetch failed: {type(exc).__name__}: {exc}",
                        signals=["fetch_failed"],
                        status="failed",
                        relevant_links=[],
                    )
                    self.telemetry.emit(
                        phase="page_fetch",
                        actor="system",
                        strategy_id=candidate.strategy_id,
                        query_id=candidate.query_id,
                        url_id=candidate.url_id,
                        input_payload={"url": target.canonical_url, "visit_index": visit_index + 1},
                        output_summary={"error": str(exc)},
                        decision="frontier_page_failed",
                        error_code=type(exc).__name__,
                    )
                    continue
                self.telemetry.emit(
                    phase="page_fetch",
                    actor="system",
                    strategy_id=candidate.strategy_id,
                    query_id=candidate.query_id,
                    url_id=candidate.url_id,
                    input_payload={"url": target.canonical_url, "visit_index": visit_index + 1},
                    output_summary={"status_code": fetched.status_code, "final_url": fetched.final_url},
                    decision="frontier_page_fetched",
                )

                current_render_profile = detect_render_profile(fetched.html)
                if visit_index == 0:
                    render_profile = current_render_profile
                    self.telemetry.emit(
                        phase="render_detect",
                        actor="normal_agent",
                        url_id=candidate.url_id,
                        input_payload={"url": fetched.final_url},
                        output_summary={"render_profile": render_profile},
                        decision=render_profile,
                    )

                evidence = analyze_page(fetched)
                page_evidence.append(evidence)
                site_graph.record_analysis(
                    url=fetched.final_url,
                    render_profile=current_render_profile,
                    evidence=evidence,
                )
                site_graph.add_links(fetched.final_url, evidence.relevant_links, discovered_via="html_link")

                if target.canonical_url != candidate.canonical_url:
                    self.telemetry.emit(
                        phase="link_follow",
                        actor="normal_agent",
                        url_id=candidate.url_id,
                        input_payload={"url": target.canonical_url, "visit_index": visit_index + 1},
                        output_summary={"status_code": fetched.status_code, "final_url": fetched.final_url},
                        decision="frontier_followed",
                    )

                if browser_result is None and current_render_profile == "csr_shell":
                    browser_result = await self.browser_manager.delegate(
                        url=fetched.final_url,
                        intent=intent,
                        url_id=candidate.url_id,
                        site_graph=site_graph,
                    )
                    site_graph.record_browser_result(url=fetched.final_url, result=browser_result)

                if evidence.captcha_present or evidence.paywall_present:
                    break
                if browser_result and (
                    browser_result.classification != "unknown"
                    or browser_result.captcha_present
                    or browser_result.paywall_present
                ):
                    break

            merged_signal = self._merge_api_signal(page_evidence, browser_result)
            api_probe = None
            if merged_signal.detected:
                api_probe = await self.api_probe.probe(
                    url_id=candidate.url_id,
                    intent=intent,
                    evidence=page_evidence,
                )

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
                decision = await self._decide(
                    intent=intent,
                    candidate=candidate,
                    render_profile=render_profile,
                    site_graph=site_graph,
                    browser_result=browser_result,
                    api_probe=api_probe,
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
                paywall_present=decision.outcome == "paywall" or any(item.paywall_present for item in page_evidence),
                auth_required=any(item.auth_required for item in page_evidence) or bool(browser_result and browser_result.auth_required),
                captcha_present=any(item.captcha_present for item in page_evidence) or bool(browser_result and browser_result.captcha_present),
                evidence=page_evidence,
                site_graph=site_graph.snapshot(),
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
        site_graph: SiteGraph,
        browser_result,
        api_probe,
    ) -> EvaluationDecision:
        prompt = {
            "intent": intent,
            "candidate": candidate.model_dump(),
            "render_profile": render_profile,
            "site_graph": site_graph.prompt_context(max_nodes=max(self.config.max_site_graph_visits + 4, 8)),
            "browser_result": browser_result.model_dump(mode="json") if browser_result else None,
            "api_probe": api_probe.model_dump(mode="json") if api_probe else None,
        }
        response = await self.llm.complete_json(
            system_prompt=DECISION_SYSTEM_PROMPT,
            user_prompt=(
                "Decide whether this website is a useful datasource for the intent.\n\n"
                f"{prompt}\n\n"
                "Rules:\n"
                "- Use only the visited page summaries from `site_graph.visited_pages` as page-history evidence.\n"
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

    def _merge_api_signal(self, page_evidence: list, browser_result) -> ApiSignal:
        merged = ApiSignal()
        for evidence in page_evidence:
            merged.detected = merged.detected or evidence.api_signal.detected
            merged.auth_required = merged.auth_required or evidence.api_signal.auth_required
            merged.doc_links.extend(link for link in evidence.api_signal.doc_links if link not in merged.doc_links)
            merged.openapi_links.extend(link for link in evidence.api_signal.openapi_links if link not in merged.openapi_links)
            merged.graphql_hints.extend(token for token in evidence.api_signal.graphql_hints if token not in merged.graphql_hints)

        if browser_result:
            merged.detected = merged.detected or browser_result.api_detected
            merged.auth_required = merged.auth_required or browser_result.auth_required
            for link in browser_result.relevant_links:
                lowered = link.lower()
                if any(token in lowered for token in ("api", "docs", "developer", "reference", "graphql")) and link not in merged.doc_links:
                    merged.doc_links.append(link)
                if any(token in lowered for token in ("openapi", "swagger", ".json")) and link not in merged.openapi_links:
                    merged.openapi_links.append(link)
                if "graphql" in lowered and link not in merged.graphql_hints:
                    merged.graphql_hints.append(link)
        return merged
