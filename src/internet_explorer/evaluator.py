from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from internet_explorer.api_probe import ApiProbeService
from internet_explorer.browser_delegate import BrowserDelegationManager
from internet_explorer.canonicalize import canonicalize_url, registrable_domain
from internet_explorer.config import AppConfig
from internet_explorer.fetcher import AsyncWebFetcher, analyze_page, detect_render_profile
from internet_explorer.llm import LLMClient
from internet_explorer.models import (
    ApiSignal,
    EvaluationDecision,
    NavigationMemoryEntry,
    ToolDuplicateSignal,
    UrlCandidate,
    UrlEvaluation,
)
from internet_explorer.site_graph import SiteGraph
from internet_explorer.telemetry import Telemetry
from internet_explorer.tool_inventory import ToolInventory


class _DecisionEnvelope(BaseModel):
    useful: bool
    relevance_score: float
    outcome: str
    why_useful: str
    how_to_use: str
    api_stage: str
    notes: list[str] = Field(default_factory=list)


class _DecisionRawEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    useful: bool | str | None = None
    relevance_score: float | int | str | None = None
    outcome: str | None = None
    why_useful: str | None = None
    how_to_use: str | None = None
    api_stage: str | None = None
    notes: list[str] | str | None = None

    # Common alternate keys observed in model outputs.
    reason: str | None = None
    category: str | None = None
    usefulness: bool | str | None = None
    useful_datasource: bool | str | None = None
    verdict: str | None = None
    evidence: str | None = None
    confidence: float | int | str | None = None
    score: float | int | str | None = None
    recommendation: str | None = None


class _NavigationPlanEnvelope(BaseModel):
    reasoning: str
    action: str
    target_url: str = ""
    action_notes: list[str] = Field(default_factory=list)


class _ToolTermEnvelope(BaseModel):
    terms: list[str] = Field(default_factory=list)
    reason: str = ""


DECISION_SYSTEM_PROMPT = """
You decide whether a website is a useful datasource for a given intent.
Be strict and evidence-based. Return only JSON.
"""

NAVIGATION_SYSTEM_PROMPT = """
You are the normal agent planning one navigation step at a time.
Use bounded exploration and pick only one action.
Return only JSON.
"""

TOOL_TERM_SYSTEM_PROMPT = """
Extract concise provider/tool identity terms to check if a candidate duplicates known tooling.
Return only JSON.
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
VALID_NAV_ACTIONS = {"fetch_url", "read_node", "delegate_browser", "stop"}


class UrlEvaluator:
    def __init__(
        self,
        config: AppConfig,
        llm: LLMClient,
        fetcher: AsyncWebFetcher,
        telemetry: Telemetry,
        browser_manager: BrowserDelegationManager,
        tool_inventory: ToolInventory | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        self.fetcher = fetcher
        self.telemetry = telemetry
        self.browser_manager = browser_manager
        self.tool_inventory = tool_inventory
        self.api_probe = ApiProbeService(telemetry, llm, timeout_seconds=max(config.request_timeout_seconds, 30))
        self._domain_bootstrap_cache: dict[str, list[str]] = {}

    async def evaluate(self, *, intent: str, candidate: UrlCandidate) -> UrlEvaluation:
        started = self.telemetry.timed()
        try:
            graph, initial_links = await self._bootstrap_graph(candidate=candidate, intent=intent)
            page_evidence = []
            browser_result = None
            visited_memory: list[NavigationMemoryEntry] = []
            visited_urls: set[str] = set()
            previous_reasoning = ""
            node_context_for_next_turn: dict[str, Any] = {}
            render_profile = "hybrid"
            fetch_failure_count = 0

            step_limit = max(1, self.config.max_site_graph_visits)
            for step_no in range(1, step_limit + 1):
                node_context = node_context_for_next_turn
                node_context_for_next_turn = {}

                plan = await self._plan_navigation_step(
                    intent=intent,
                    candidate=candidate,
                    graph=graph,
                    initial_links=initial_links,
                    page_evidence=page_evidence,
                    visited_memory=visited_memory,
                    previous_reasoning=previous_reasoning,
                    node_context=node_context,
                    step_no=step_no,
                )
                previous_reasoning = plan.reasoning[:1200]
                self.telemetry.emit(
                    phase="triage",
                    actor="normal_agent",
                    strategy_id=candidate.strategy_id,
                    query_id=candidate.query_id,
                    url_id=candidate.url_id,
                    input_payload={"step_no": step_no},
                    output_summary=plan.model_dump(mode="json"),
                    decision=f"navigation_{plan.action}",
                )

                if plan.action == "stop":
                    break

                if plan.action == "read_node":
                    read_url = self._resolve_target_url(
                        target_url=plan.target_url,
                        candidate=candidate,
                        initial_links=initial_links,
                        visited_urls=visited_urls,
                        graph=graph,
                        allow_visited=True,
                    )
                    node = graph.get_node(read_url) if read_url else None
                    node_context_for_next_turn = (
                        {
                            "url": node.canonical_url,
                            "title": node.title,
                            "page_type_guess": node.page_type_guess,
                            "status": node.status,
                            "summary": node.summary,
                            "signals": node.signals,
                        }
                        if node
                        else {}
                    )
                    self.telemetry.emit(
                        phase="site_graph_tool",
                        actor="normal_agent",
                        strategy_id=candidate.strategy_id,
                        query_id=candidate.query_id,
                        url_id=candidate.url_id,
                        input_payload={"url": read_url},
                        output_summary={"found": bool(node)},
                        decision="normal_agent_read_node",
                    )
                    continue

                if plan.action == "delegate_browser":
                    delegate_url = self._resolve_target_url(
                        target_url=plan.target_url,
                        candidate=candidate,
                        initial_links=initial_links,
                        visited_urls=visited_urls,
                        graph=graph,
                        allow_visited=True,
                    )
                    if not delegate_url:
                        continue

                    browser_result = await self.browser_manager.delegate(
                        url=delegate_url,
                        intent=intent,
                        url_id=candidate.url_id,
                        initial_links=initial_links,
                    )
                    graph.record_browser_result(url=delegate_url, result=browser_result)
                    visited_urls.add(delegate_url)
                    visited_memory.append(
                        NavigationMemoryEntry(
                            step_no=step_no,
                            url=delegate_url,
                            summary=_one_line_browser_summary(browser_result),
                        )
                    )
                    if browser_result.captcha_present or browser_result.paywall_present:
                        break
                    continue

                target_url = self._resolve_target_url(
                    target_url=plan.target_url,
                    candidate=candidate,
                    initial_links=initial_links,
                    visited_urls=visited_urls,
                    graph=graph,
                    allow_visited=False,
                )
                if not target_url:
                    break

                fetch_started = self.telemetry.timed()
                try:
                    fetched = await self.fetcher.fetch(target_url)
                except Exception as exc:
                    fetch_failure_count += 1
                    graph.update_node_from_tool(
                        url=target_url,
                        title="",
                        page_type_guess="",
                        summary=f"Fetch failed: {type(exc).__name__}",
                        signals=["fetch_failed"],
                        status="failed",
                        relevant_links=[],
                    )
                    visited_urls.add(target_url)
                    visited_memory.append(
                        NavigationMemoryEntry(
                            step_no=step_no,
                            url=target_url,
                            summary=f"fetch failed ({type(exc).__name__})",
                        )
                    )
                    self.telemetry.emit(
                        phase="page_fetch",
                        actor="system",
                        strategy_id=candidate.strategy_id,
                        query_id=candidate.query_id,
                        url_id=candidate.url_id,
                        input_payload={"url": target_url, "step_no": step_no},
                        output_summary={"error": _describe_exception(exc)},
                        decision="fetch_failed",
                        error_code=type(exc).__name__,
                        latency_ms=self.telemetry.elapsed_ms(fetch_started),
                    )
                    continue

                self.telemetry.emit(
                    phase="page_fetch",
                    actor="normal_agent",
                    strategy_id=candidate.strategy_id,
                    query_id=candidate.query_id,
                    url_id=candidate.url_id,
                    input_payload={"url": target_url, "step_no": step_no},
                    output_summary={"status_code": fetched.status_code, "final_url": fetched.final_url},
                    decision="page_fetched",
                    latency_ms=self.telemetry.elapsed_ms(fetch_started),
                )

                render_profile = detect_render_profile(fetched.html)
                self.telemetry.emit(
                    phase="render_detect",
                    actor="normal_agent",
                    strategy_id=candidate.strategy_id,
                    query_id=candidate.query_id,
                    url_id=candidate.url_id,
                    input_payload={"url": fetched.final_url, "step_no": step_no},
                    output_summary={"render_profile": render_profile},
                    decision=render_profile,
                )

                evidence = analyze_page(fetched)
                page_evidence.append(evidence)
                graph.record_analysis(
                    url=fetched.final_url,
                    render_profile=render_profile,
                    evidence=evidence,
                )
                graph.add_links(fetched.final_url, evidence.relevant_links, discovered_via="html_link")
                visited_urls.add(canonicalize_url(fetched.final_url))
                visited_memory.append(
                    NavigationMemoryEntry(
                        step_no=step_no,
                        url=canonicalize_url(fetched.final_url),
                        summary=_one_line_page_summary(evidence),
                    )
                )

                if evidence.captcha_present or evidence.paywall_present:
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
            elif not page_evidence and browser_result is None and fetch_failure_count > 0:
                decision = EvaluationDecision(
                    useful=False,
                    relevance_score=0.0,
                    outcome="unknown",
                    why_useful="No page evidence was collected because all fetch attempts failed.",
                    how_to_use="Retry this source later or run with lower concurrency.",
                    api_stage="none",
                    notes=["unknown_fetch_failure", f"fetch_failures:{fetch_failure_count}"],
                )
            else:
                decision = await self._decide(
                    intent=intent,
                    candidate=candidate,
                    render_profile=render_profile,
                    initial_links=initial_links,
                    page_evidence=page_evidence,
                    browser_result=browser_result,
                    api_probe=api_probe,
                    visited_memory=visited_memory,
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

            if decision.outcome == "unknown" and "unknown_fetch_failure" in decision.notes:
                tool_duplicate_signal = ToolDuplicateSignal(checked=False, reason="skipped_unknown_fetch_failure")
            else:
                tool_duplicate_signal = await self._assess_tool_duplicate(
                    intent=intent,
                    candidate=candidate,
                    page_evidence=page_evidence,
                    decision=decision,
                )
            if tool_duplicate_signal.duplicate_detected:
                decision.useful = False
                decision.outcome = "irrelevant"
                decision.relevance_score = min(decision.relevance_score, 0.35)
                decision.notes.append(
                    "duplicate_tool_inventory:" + ",".join(tool_duplicate_signal.matched_tools[:10])
                )
                decision.why_useful = (
                    "This source appears to duplicate an existing tool/provider already present in the tool-flow baseline."
                )
                decision.how_to_use = "Skip for novelty goals unless a human explicitly wants redundancy."

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
                site_graph=graph.snapshot(max_nodes=min(self.config.max_site_graph_nodes, 60)),
                visited_memory=visited_memory,
                tool_duplicate_signal=tool_duplicate_signal,
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
                notes=[f"evaluation_error:{type(exc).__name__}", _describe_exception(exc)],
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

    async def _bootstrap_graph(self, *, candidate: UrlCandidate, intent: str) -> tuple[SiteGraph, list[str]]:
        graph = SiteGraph(
            config=self.config,
            telemetry=self.telemetry,
            url_id=candidate.url_id,
            intent=intent,
            seed_url=candidate.canonical_url,
            domain=candidate.domain,
        )
        cached = self._domain_bootstrap_cache.get(candidate.domain)
        if cached is not None:
            for link in cached:
                graph.add_url(link, discovered_via="cache", parent_url=graph.root_url)
            self.telemetry.emit(
                phase="site_graph",
                actor="system",
                strategy_id=candidate.strategy_id,
                query_id=candidate.query_id,
                url_id=candidate.url_id,
                output_summary={"domain": candidate.domain, "cached_urls": len(cached)},
                decision="bootstrap_cache_hit",
            )
        else:
            await graph.bootstrap(self.fetcher)
            cached_frontier = graph.next_frontier(limit=self.config.max_site_graph_nodes)
            self._domain_bootstrap_cache[candidate.domain] = [
                node.canonical_url
                for node in cached_frontier
                if node.canonical_url and node.canonical_url != candidate.canonical_url
            ]
            self.telemetry.emit(
                phase="site_graph",
                actor="system",
                strategy_id=candidate.strategy_id,
                query_id=candidate.query_id,
                url_id=candidate.url_id,
                output_summary={"domain": candidate.domain, "cached_urls": len(self._domain_bootstrap_cache[candidate.domain])},
                decision="bootstrap_cache_store",
            )

        frontier = graph.next_frontier(limit=max(self.config.max_site_graph_frontier, self.config.max_internal_links))
        links = [node.canonical_url for node in frontier if node.canonical_url and node.canonical_url != candidate.canonical_url]
        return graph, _unique_links(links)

    async def _plan_navigation_step(
        self,
        *,
        intent: str,
        candidate: UrlCandidate,
        graph: SiteGraph,
        initial_links: list[str],
        page_evidence: list,
        visited_memory: list[NavigationMemoryEntry],
        previous_reasoning: str,
        node_context: dict[str, Any],
        step_no: int,
    ) -> _NavigationPlanEnvelope:
        frontier = graph.next_frontier(limit=min(self.config.max_site_graph_frontier, 8))
        prompt = {
            "intent": intent,
            "candidate": {
                "url": candidate.canonical_url,
                "domain": candidate.domain,
                "title": candidate.source_title,
                "snippet": candidate.source_snippet,
            },
            "step_no": step_no,
            "max_steps": max(1, self.config.max_site_graph_visits),
            "previous_reasoning": previous_reasoning,
            "visited_memory": [entry.model_dump(mode="json") for entry in visited_memory[-12:]],
            "frontier": [
                {
                    "url": node.canonical_url,
                    "page_type_guess": node.page_type_guess,
                    "signals": node.signals,
                    "priority_score": node.priority_score,
                }
                for node in frontier
            ],
            "last_page_signals": [
                {
                    "url": evidence.url,
                    "title": evidence.title,
                    "api_detected": evidence.api_signal.detected,
                    "paywall": evidence.paywall_present,
                    "contact_sales": evidence.contact_sales_present,
                    "captcha": evidence.captcha_present,
                }
                for evidence in page_evidence[-4:]
            ],
            "temporary_node_context": node_context,
            "initial_links": initial_links[:40],
        }
        try:
            response = await self.llm.complete_json(
                system_prompt=NAVIGATION_SYSTEM_PROMPT,
                user_prompt=(
                    "Plan one next action for datasource evaluation.\n\n"
                    f"{prompt}\n\n"
                    "Rules:\n"
                    "- Allowed actions: fetch_url, read_node, delegate_browser, stop.\n"
                    "- Use read_node when you need a long summary of one page node.\n"
                    "- Keep target_url in the same registrable domain as candidate when possible.\n"
                    "- Prefer fetch_url before delegate_browser unless JS interaction is likely necessary.\n"
                    "- If captcha is already present in signals, prefer stop or classify via final decision.\n"
                    "- Return strict JSON only.\n"
                ),
                schema=_NavigationPlanEnvelope,
                temperature=0.0,
                max_completion_tokens=700,
            )
            action = response.action if response.action in VALID_NAV_ACTIONS else "fetch_url"
            return _NavigationPlanEnvelope(
                reasoning=response.reasoning.strip(),
                action=action,
                target_url=response.target_url.strip(),
                action_notes=response.action_notes,
            )
        except Exception as exc:
            return self._fallback_navigation_step(
                candidate=candidate,
                frontier=frontier,
                visited_memory=visited_memory,
                error=type(exc).__name__,
            )

    def _fallback_navigation_step(self, *, candidate: UrlCandidate, frontier: list, visited_memory: list[NavigationMemoryEntry], error: str) -> _NavigationPlanEnvelope:
        visited_urls = {entry.url for entry in visited_memory}
        if candidate.canonical_url not in visited_urls:
            return _NavigationPlanEnvelope(
                reasoning=f"LLM plan fallback ({error}); fetch seed first.",
                action="fetch_url",
                target_url=candidate.canonical_url,
            )
        for node in frontier:
            if node.canonical_url not in visited_urls:
                return _NavigationPlanEnvelope(
                    reasoning=f"LLM plan fallback ({error}); fetch best frontier page.",
                    action="fetch_url",
                    target_url=node.canonical_url,
                )
        return _NavigationPlanEnvelope(
            reasoning=f"LLM plan fallback ({error}); no strong next page.",
            action="stop",
            target_url="",
        )

    def _resolve_target_url(
        self,
        *,
        target_url: str,
        candidate: UrlCandidate,
        initial_links: list[str],
        visited_urls: set[str],
        graph: SiteGraph,
        allow_visited: bool,
    ) -> str:
        candidate_url = candidate.canonical_url
        if not visited_urls:
            return candidate_url

        candidates: list[str] = []
        if target_url:
            candidates.append(target_url)
        candidates.extend(initial_links[: self.config.max_internal_links])
        candidates.extend(node.canonical_url for node in graph.next_frontier(limit=self.config.max_site_graph_frontier))

        for raw_url in candidates:
            try:
                canonical = canonicalize_url(raw_url)
            except Exception:
                continue
            if not canonical:
                continue
            if registrable_domain(canonical) != candidate.domain:
                continue
            if not allow_visited and canonical in visited_urls:
                continue
            return canonical
        return ""

    async def _decide(
        self,
        *,
        intent: str,
        candidate: UrlCandidate,
        render_profile: str,
        initial_links: list[str],
        page_evidence,
        browser_result,
        api_probe,
        visited_memory: list[NavigationMemoryEntry],
    ) -> EvaluationDecision:
        prompt = {
            "intent": intent,
            "candidate": candidate.model_dump(),
            "render_profile": render_profile,
            "initial_links": initial_links[:80],
            "visited_memory": [entry.model_dump(mode="json") for entry in visited_memory],
            "visited_pages": [
                {
                    "url": evidence.url,
                    "title": evidence.title,
                    "text_excerpt": evidence.text_excerpt[:500],
                    "api_signal": evidence.api_signal.model_dump(mode="json"),
                    "paywall_present": evidence.paywall_present,
                    "contact_sales_present": evidence.contact_sales_present,
                    "auth_required": evidence.auth_required,
                    "captcha_present": evidence.captcha_present,
                    "data_signals": evidence.data_signals,
                }
                for evidence in page_evidence
            ],
            "browser_result": browser_result.model_dump(mode="json") if browser_result else None,
            "api_probe": api_probe.model_dump(mode="json") if api_probe else None,
        }
        response = await self.llm.complete_json(
            system_prompt=DECISION_SYSTEM_PROMPT,
            user_prompt=(
                "Decide whether this website is a useful datasource for the intent.\n\n"
                f"{prompt}\n\n"
                "Rules:\n"
                "- Use only visited_pages, browser_result, and api_probe as evidence.\n"
                "- `data_on_site` when valuable data appears directly on the site or portal.\n"
                "- `api_available` when API/docs/endpoints are present and seem workable.\n"
                "- `contact_sales_only` when only sales/demo access exists.\n"
                "- `paywall` when payment/upgrade is required for access.\n"
                "- `irrelevant` when it does not materially help the intent.\n"
                "- Novelty matters, but a non-novel source can still be useful.\n"
                "- Return JSON with EXACT keys:\n"
                "  useful (boolean), relevance_score (0..1 float), outcome,\n"
                "  why_useful, how_to_use, api_stage, notes (list).\n"
                "- Be strict. Return only JSON.\n"
            ),
            schema=_DecisionRawEnvelope,
            temperature=0.0,
            max_completion_tokens=2048,
        )
        return _normalize_decision_response(response)

    async def _assess_tool_duplicate(
        self,
        *,
        intent: str,
        candidate: UrlCandidate,
        page_evidence: list,
        decision: EvaluationDecision,
    ) -> ToolDuplicateSignal:
        if self.tool_inventory is None or not self.tool_inventory.tool_names:
            return ToolDuplicateSignal(checked=False, reason="tool_inventory_unavailable")

        prompt = {
            "intent": intent,
            "candidate": {
                "domain": candidate.domain,
                "title": candidate.source_title,
                "snippet": candidate.source_snippet,
                "canonical_url": candidate.canonical_url,
            },
            "evidence": [
                {
                    "url": item.url,
                    "title": item.title,
                    "data_signals": item.data_signals,
                }
                for item in page_evidence[:4]
            ],
            "decision": {
                "outcome": decision.outcome,
                "why_useful": decision.why_useful,
                "how_to_use": decision.how_to_use,
            },
        }
        try:
            response = await self.llm.complete_json(
                system_prompt=TOOL_TERM_SYSTEM_PROMPT,
                user_prompt=(
                    "Extract up to 8 concise tool/provider identity terms from this candidate source.\n"
                    "Focus on brand/platform/provider words that can be matched against an existing tool inventory.\n"
                    "Avoid generic words like api, data, docs unless part of a name.\n\n"
                    f"{prompt}\n\n"
                    "Return strict JSON with `terms` and optional `reason`."
                ),
                schema=_ToolTermEnvelope,
                temperature=0.0,
                max_completion_tokens=300,
            )
            terms = response.terms[:8]
            reason = response.reason.strip()
        except Exception as exc:
            # Keep fallback deterministic when the LLM extraction call fails.
            terms = [candidate.domain, candidate.source_title]
            reason = f"fallback:{type(exc).__name__}"

        match = self.tool_inventory.match_terms(terms)
        signal = ToolDuplicateSignal(
            checked=True,
            search_terms=match.terms,
            matched_tools=match.matched_tools,
            duplicate_detected=match.duplicate_detected,
            reason=reason or match.reason,
        )
        self.telemetry.emit(
            phase="triage",
            actor="normal_agent",
            strategy_id=candidate.strategy_id,
            query_id=candidate.query_id,
            url_id=candidate.url_id,
            input_payload={"terms": terms, "candidate_domain": candidate.domain},
            output_summary=signal.model_dump(mode="json"),
            decision="tool_duplicate_checked",
        )
        return signal

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


def _one_line_page_summary(evidence) -> str:
    title = evidence.title.strip() if evidence.title else ""
    bits = []
    if evidence.api_signal.detected:
        bits.append("api")
    if evidence.contact_sales_present:
        bits.append("contact-sales")
    if evidence.paywall_present:
        bits.append("paywall")
    if evidence.captcha_present:
        bits.append("captcha")
    if evidence.data_signals:
        bits.append("data-signals")
    signal = f" [{', '.join(bits)}]" if bits else ""
    base = title or evidence.url
    return f"{base}{signal}"[:220]


def _one_line_browser_summary(result) -> str:
    reason = (result.why_useful or result.how_to_use or "browser delegation completed").strip()
    return f"{result.classification}: {reason}"[:220]


def _unique_links(links: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for link in links:
        if not link or link in seen:
            continue
        seen.add(link)
        unique.append(link)
    return unique


def _normalize_decision_response(response: _DecisionRawEnvelope) -> EvaluationDecision:
    notes = _normalize_notes(response.notes)
    standard_fields_present = all(
        [
            response.useful is not None,
            response.relevance_score is not None,
            bool((response.outcome or "").strip()),
            bool((response.why_useful or "").strip()),
            bool((response.how_to_use or "").strip()),
            bool((response.api_stage or "").strip()),
        ]
    )
    if not standard_fields_present:
        notes.append("decision_shape_normalized")

    outcome = _normalize_outcome(response)
    useful = _normalize_useful(response, outcome=outcome)
    relevance_score = _normalize_score(response, useful=useful, outcome=outcome)
    why_useful = (
        (response.why_useful or "").strip()
        or (response.evidence or "").strip()
        or (response.reason or "").strip()
        or ("Relevant signals were found for this intent." if useful else "Insufficient evidence of intent relevance.")
    )
    how_to_use = (
        (response.how_to_use or "").strip()
        or (response.recommendation or "").strip()
        or _default_how_to_use(outcome=outcome, useful=useful)
    )
    api_stage = _normalize_api_stage(response, outcome=outcome)

    return EvaluationDecision(
        useful=useful,
        relevance_score=relevance_score,
        outcome=outcome,
        why_useful=why_useful,
        how_to_use=how_to_use,
        api_stage=api_stage,
        notes=notes,
    )


def _normalize_notes(raw_notes: list[str] | str | None) -> list[str]:
    if raw_notes is None:
        return []
    if isinstance(raw_notes, list):
        return [str(item).strip() for item in raw_notes if str(item).strip()]
    value = str(raw_notes).strip()
    return [value] if value else []


def _normalize_outcome(response: _DecisionRawEnvelope) -> str:
    raw_candidates = [
        response.outcome,
        response.category,
        response.verdict,
        str(response.usefulness) if response.usefulness is not None else "",
    ]
    for candidate in raw_candidates:
        normalized = _map_outcome(candidate)
        if normalized != "unknown":
            return normalized
    return "unknown"


def _map_outcome(value: str | None) -> str:
    lowered = (value or "").strip().lower()
    if not lowered:
        return "unknown"
    if lowered in VALID_OUTCOMES:
        return lowered
    if lowered in {"data", "site_data", "onsite_data", "on_site_data", "portal_data", "dataset", "rfp_data"}:
        return "data_on_site"
    if lowered in {"api", "has_api", "api_viable", "developer_api", "api_docs"}:
        return "api_available"
    if lowered in {"contact_sales", "sales_only", "contact_only", "demo_required"}:
        return "contact_sales_only"
    if lowered in {"paid", "payment_required", "subscription_required", "upgrade_required"}:
        return "paywall"
    if lowered in {"not_relevant", "not useful", "unrelated", "irrelevant_source"}:
        return "irrelevant"
    return "unknown"


def _normalize_useful(response: _DecisionRawEnvelope, *, outcome: str) -> bool:
    for raw in (response.useful, response.useful_datasource, response.usefulness):
        coerced = _coerce_bool(raw)
        if coerced is not None:
            return coerced
    if outcome in {"data_on_site", "api_available", "contact_sales_only", "paywall"}:
        return True
    if outcome == "irrelevant":
        return False
    return False


def _normalize_score(response: _DecisionRawEnvelope, *, useful: bool, outcome: str) -> float:
    for raw in (response.relevance_score, response.confidence, response.score):
        value = _coerce_float(raw)
        if value is not None:
            return max(0.0, min(value, 1.0))
    if useful:
        return 0.8 if outcome in {"data_on_site", "api_available"} else 0.6
    return 0.1


def _normalize_api_stage(response: _DecisionRawEnvelope, *, outcome: str) -> str:
    raw = (response.api_stage or "").strip().lower()
    if raw in VALID_API_STAGES:
        return raw
    if outcome == "api_available":
        return "api_detected"
    return "none"


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes", "y", "relevant", "useful"}:
        return True
    if lowered in {"false", "0", "no", "n", "irrelevant", "not useful"}:
        return False
    return None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _default_how_to_use(*, outcome: str, useful: bool) -> str:
    if not useful:
        return "Skip this source for now."
    return {
        "data_on_site": "Use page scraping or feed export where available.",
        "api_available": "Use documented API endpoints with auth/rate-limit handling.",
        "contact_sales_only": "Store for human follow-up and access evaluation.",
        "paywall": "Store for human review and paid-access decision.",
    }.get(outcome, "Store this source for manual follow-up.")


def _describe_exception(exc: Exception) -> str:
    text = str(exc).strip()
    if text:
        return text
    return type(exc).__name__
