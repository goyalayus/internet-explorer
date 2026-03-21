from __future__ import annotations

import traceback
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from internet_explorer.api_probe import ApiProbeService
from internet_explorer.browser_delegate import BrowserDelegationManager
from internet_explorer.canonicalize import canonicalize_url, registrable_domain
from internet_explorer.config import AppConfig
from internet_explorer.fetcher import AsyncWebFetcher, analyze_page, detect_render_profile, is_pdf_fetch
from internet_explorer.llm import LLMClient
from internet_explorer.models import (
    ApiSignal,
    EvaluationDecision,
    NavigationMemoryEntry,
    PageEvidence,
    SourceEvidenceItem,
    ToolDuplicateSignal,
    UrlCandidate,
    UrlEvaluation,
)
from internet_explorer.pdf_verify import PdfVerifierService
from internet_explorer.site_graph import SiteGraph
from internet_explorer.telemetry import Telemetry
from internet_explorer.tool_inventory import ToolInventory


class _DecisionEnvelope(BaseModel):
    useful: bool
    relevance_score: float
    outcome: str
    reasoning: str
    api_stage: str
    source_evidence: list[SourceEvidenceItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class _DecisionRawEnvelope(BaseModel):
    model_config = ConfigDict(extra="allow")

    useful: bool | str | None = None
    relevance_score: float | int | str | None = None
    outcome: str | None = None
    reasoning: str | None = None
    api_stage: str | None = None
    source_evidence: Any | None = None
    notes: list[str] | str | None = None

    reason: str | None = None
    category: str | None = None
    usefulness: bool | str | None = None
    useful_datasource: bool | str | None = None
    verdict: str | None = None
    evidence: str | None = None
    confidence: float | int | str | None = None
    score: float | int | str | None = None
    recommendation: str | None = None
    why_useful: str | None = None
    how_to_use: str | None = None


class _NavigationPlanEnvelope(BaseModel):
    reasoning: str
    action: str
    target_url: str = ""
    action_notes: list[str] = Field(default_factory=list)


class _ToolTermEnvelope(BaseModel):
    terms: list[str] = Field(default_factory=list)
    reason: str = ""


DECISION_SYSTEM_PROMPT = """
You decide whether a website or domain is a useful datasource for a given intent.
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
GENERIC_TOOL_IDENTITY_TERMS = {
    "api",
    "apis",
    "data",
    "docs",
    "documentation",
    "developer",
    "developers",
    "platform",
    "portal",
    "service",
    "services",
    "home",
    "homepage",
    "contact",
    "pricing",
}


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
        self.pdf_verifier = PdfVerifierService(fetcher, llm, telemetry)
        self._domain_bootstrap_cache: dict[str, list[str]] = {}

    async def evaluate(self, *, intent: str, candidate: UrlCandidate) -> UrlEvaluation:
        started = self.telemetry.timed()
        try:
            graph, initial_links = await self._bootstrap_graph(candidate=candidate, intent=intent)
            page_evidence: list[PageEvidence] = []
            source_evidence: list[SourceEvidenceItem] = []
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
                    source_evidence=source_evidence,
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
                    if browser_result.source_evidence:
                        source_evidence = _merge_source_evidence(source_evidence, browser_result.source_evidence)
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
                    output_summary={"status_code": fetched.status_code, "final_url": fetched.final_url, "content_type": fetched.content_type},
                    decision="page_fetched",
                    latency_ms=self.telemetry.elapsed_ms(fetch_started),
                )

                visited_urls.add(canonicalize_url(target_url))
                visited_urls.add(canonicalize_url(fetched.final_url))

                if is_pdf_fetch(fetched):
                    pdf_result = await self.pdf_verifier.verify(url_id=candidate.url_id, intent=intent, pdf_url=fetched.final_url)
                    pdf_evidence = PageEvidence(
                        url=pdf_result.final_url or fetched.final_url,
                        title=pdf_result.title,
                        content_type=pdf_result.content_type,
                        content_kind="pdf",
                        text_excerpt=pdf_result.summary[:1200],
                        relevant_links=pdf_result.fallback_urls,
                        data_signals=pdf_result.extracted_signals,
                    )
                    page_evidence.append(pdf_evidence)
                    graph.update_node_from_tool(
                        url=fetched.final_url,
                        title=pdf_result.title,
                        page_type_guess="rfp" if any("rfp" in signal.lower() for signal in pdf_result.extracted_signals) else "document",
                        summary=(pdf_result.summary or pdf_result.reasoning)[:1200],
                        signals=["pdf"] + [item.lower().replace(" ", "_") for item in pdf_result.extracted_signals[:8]],
                        status="analyzed",
                        relevant_links=pdf_result.fallback_urls,
                    )
                    if pdf_result.source_evidence is not None:
                        source_evidence = _merge_source_evidence(source_evidence, [pdf_result.source_evidence])
                    visited_memory.append(
                        NavigationMemoryEntry(
                            step_no=step_no,
                            url=canonicalize_url(fetched.final_url),
                            summary=_one_line_pdf_summary(pdf_result),
                        )
                    )
                    continue

                if fetched.is_binary and not fetched.html:
                    binary_evidence = PageEvidence(
                        url=fetched.final_url,
                        title="",
                        content_type=fetched.content_type,
                        content_kind="binary_file",
                        text_excerpt="Binary file encountered.",
                    )
                    page_evidence.append(binary_evidence)
                    graph.update_node_from_tool(
                        url=fetched.final_url,
                        title="",
                        page_type_guess="file",
                        summary=f"Binary file encountered ({fetched.content_type or 'unknown content type'}).",
                        signals=["binary_file"],
                        status="analyzed",
                        relevant_links=[],
                    )
                    visited_memory.append(
                        NavigationMemoryEntry(
                            step_no=step_no,
                            url=canonicalize_url(fetched.final_url),
                            summary=f"binary file ({fetched.content_type or 'unknown'})",
                        )
                    )
                    continue

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
                    reasoning=browser_result.reasoning or "Relevant source discovered, but access appears gated behind payment.",
                    api_stage="none",
                    source_evidence=browser_result.source_evidence,
                    notes=["browser_detected_paywall"],
                )
            elif not page_evidence and browser_result is None and fetch_failure_count > 0:
                decision = EvaluationDecision(
                    useful=False,
                    relevance_score=0.0,
                    outcome="unknown",
                    reasoning="No page evidence was collected because all fetch attempts failed. Retry later or lower concurrency.",
                    api_stage="none",
                    source_evidence=[],
                    notes=["unknown_fetch_failure", f"fetch_failures:{fetch_failure_count}"],
                )
            else:
                decision = await self._decide(
                    intent=intent,
                    candidate=candidate,
                    render_profile=render_profile,
                    initial_links=initial_links,
                    page_evidence=page_evidence,
                    source_evidence=source_evidence,
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

            if not decision.source_evidence:
                decision.source_evidence = _infer_source_evidence(page_evidence, browser_result, source_evidence, api_probe)
            else:
                decision.source_evidence = _merge_source_evidence(
                    _infer_source_evidence(page_evidence, browser_result, source_evidence, api_probe),
                    decision.source_evidence,
                )

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
                decision.notes.append("duplicate_tool_inventory:" + ",".join(tool_duplicate_signal.matched_tools[:10]))
                decision.reasoning = (
                    "This source appears to duplicate an existing tool/provider already present in the tool-flow baseline. "
                    "Skip it for novelty-focused discovery unless a human explicitly wants redundancy."
                )

            evaluation = UrlEvaluation(
                url_id=candidate.url_id,
                canonical_url=candidate.canonical_url,
                start_url=candidate.start_url,
                homepage_url=candidate.homepage_url,
                domain=candidate.domain,
                novelty=candidate.novelty,
                render_profile=render_profile,
                outcome=decision.outcome,
                useful=decision.useful,
                relevance_score=decision.relevance_score,
                reasoning=decision.reasoning,
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
                source_evidence=decision.source_evidence,
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
            traceback_text = traceback.format_exc(limit=8)
            evaluation = UrlEvaluation(
                url_id=candidate.url_id,
                canonical_url=candidate.canonical_url,
                start_url=candidate.start_url,
                homepage_url=candidate.homepage_url,
                domain=candidate.domain,
                novelty=candidate.novelty,
                render_profile="hybrid",
                outcome="unknown",
                useful=False,
                reasoning="Evaluation failed before the source could be fully assessed.",
                notes=[
                    f"evaluation_error:{type(exc).__name__}",
                    _describe_exception(exc),
                    f"traceback:{traceback_text[:1400]}",
                ],
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
            seed_url=candidate.start_url,
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
                if node.canonical_url and node.canonical_url != candidate.start_url
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
        links = [node.canonical_url for node in frontier if node.canonical_url and node.canonical_url != candidate.start_url]
        return graph, _unique_links(links)

    async def _plan_navigation_step(
        self,
        *,
        intent: str,
        candidate: UrlCandidate,
        graph: SiteGraph,
        initial_links: list[str],
        page_evidence: list[PageEvidence],
        source_evidence: list[SourceEvidenceItem],
        visited_memory: list[NavigationMemoryEntry],
        previous_reasoning: str,
        node_context: dict[str, Any],
        step_no: int,
    ) -> _NavigationPlanEnvelope:
        frontier = graph.next_frontier(limit=min(self.config.max_site_graph_frontier, 8))
        prompt = {
            "intent": intent,
            "candidate": {
                "entry_url": candidate.canonical_url,
                "start_url": candidate.start_url,
                "homepage_url": candidate.homepage_url,
                "domain": candidate.domain,
                "title": candidate.source_title,
                "snippet": candidate.source_snippet,
                "start_mode": candidate.start_mode,
                "content_kind_hint": candidate.content_kind_hint,
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
                    "content_kind": evidence.content_kind,
                    "api_detected": evidence.api_signal.detected,
                    "paywall": evidence.paywall_present,
                    "contact_sales": evidence.contact_sales_present,
                    "captcha": evidence.captcha_present,
                    "data_signals": evidence.data_signals,
                }
                for evidence in page_evidence[-4:]
            ],
            "source_evidence": [item.model_dump(mode="json") for item in source_evidence[-6:]],
            "temporary_node_context": node_context,
            "initial_links": initial_links[:40],
        }
        try:
            allowed_actions = set(VALID_NAV_ACTIONS)
            if not self.config.enable_browser_delegation:
                allowed_actions.discard("delegate_browser")
            browser_rule = (
                "- Prefer fetch_url before delegate_browser unless JS interaction is likely necessary.\n"
                if "delegate_browser" in allowed_actions
                else "- Browser delegation is disabled for this run, so stay in normal-agent navigation.\n"
            )
            response = await self.llm.complete_json(
                system_prompt=NAVIGATION_SYSTEM_PROMPT,
                user_prompt=(
                    "Plan one next action for datasource evaluation.\n\n"
                    f"{prompt}\n\n"
                    "Rules:\n"
                    f"- Allowed actions: {', '.join(sorted(allowed_actions))}.\n"
                    "- Use read_node when you need a longer summary from the site graph.\n"
                    "- Keep target_url in the same registrable domain as candidate when possible.\n"
                    f"{browser_rule}"
                    "- If captcha is already present in signals, prefer stop or classify via final decision.\n"
                    "- If a PDF already proved relevance, you can still fetch a portal/homepage page to discover a reusable recurring access surface.\n"
                    "- Return strict JSON only.\n"
                ),
                schema=_NavigationPlanEnvelope,
                temperature=0.0,
                max_completion_tokens=700,
            )
            action = response.action if response.action in allowed_actions else "fetch_url"
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

    def _fallback_navigation_step(
        self,
        *,
        candidate: UrlCandidate,
        frontier: list,
        visited_memory: list[NavigationMemoryEntry],
        error: str,
    ) -> _NavigationPlanEnvelope:
        visited_urls = {entry.url for entry in visited_memory}
        if candidate.start_url not in visited_urls:
            return _NavigationPlanEnvelope(
                reasoning=f"LLM plan fallback ({error}); fetch start URL first.",
                action="fetch_url",
                target_url=candidate.start_url,
            )
        for preferred in (candidate.homepage_url, candidate.canonical_url):
            if preferred and preferred not in visited_urls:
                return _NavigationPlanEnvelope(
                    reasoning=f"LLM plan fallback ({error}); fetch candidate surface.",
                    action="fetch_url",
                    target_url=preferred,
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
        if not visited_urls:
            return candidate.start_url

        candidates: list[str] = []
        if target_url:
            candidates.append(target_url)
        candidates.extend(url for url in (candidate.homepage_url, candidate.canonical_url) if url)
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
        page_evidence: list[PageEvidence],
        source_evidence: list[SourceEvidenceItem],
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
                    "content_type": evidence.content_type,
                    "content_kind": evidence.content_kind,
                    "text_excerpt": evidence.text_excerpt[:500],
                    "api_signal": evidence.api_signal.model_dump(mode="json"),
                    "paywall_present": evidence.paywall_present,
                    "contact_sales_present": evidence.contact_sales_present,
                    "auth_required": evidence.auth_required,
                    "captcha_present": evidence.captcha_present,
                    "data_signals": evidence.data_signals,
                    "relevant_links": evidence.relevant_links[:12],
                }
                for evidence in page_evidence
            ],
            "source_evidence": [item.model_dump(mode="json") for item in source_evidence],
            "browser_result": browser_result.model_dump(mode="json") if browser_result else None,
            "api_probe": api_probe.model_dump(mode="json") if api_probe else None,
        }
        inferred_source_evidence = _infer_source_evidence(page_evidence, browser_result, source_evidence, api_probe)
        try:
            response = await self.llm.complete_json(
                system_prompt=DECISION_SYSTEM_PROMPT,
                user_prompt=(
                    "Decide whether this domain/source is a useful datasource for the intent.\n\n"
                    f"{prompt}\n\n"
                    "Rules:\n"
                    "- Use only visited_pages, source_evidence, browser_result, and api_probe as evidence.\n"
                    "- `data_on_site` when valuable data appears directly on the site, files, portal, or listings.\n"
                    "- `api_available` when API/docs/endpoints are present and seem workable.\n"
                    "- `contact_sales_only` when only sales/demo access exists.\n"
                    "- `paywall` when payment/upgrade is required for access.\n"
                    "- `irrelevant` when it does not materially help the intent.\n"
                    "- Novelty matters, but a non-novel source can still be useful.\n"
                    "- `reasoning` must explain both why the source is correct and the rough recurring access path on this domain.\n"
                    "- Return JSON with EXACT keys:\n"
                    "  useful (boolean), relevance_score (0..1 float), outcome,\n"
                    "  reasoning, api_stage, source_evidence (list), notes (list).\n"
                    "- Be strict. Return only JSON.\n"
                ),
                schema=_DecisionRawEnvelope,
                temperature=0.0,
                max_completion_tokens=2200,
            )
            return _normalize_decision_response(
                response,
                inferred_source_evidence=inferred_source_evidence,
            )
        except Exception as exc:
            return _fallback_decision_from_evidence(
                page_evidence=page_evidence,
                browser_result=browser_result,
                api_probe=api_probe,
                source_evidence=inferred_source_evidence,
                error=type(exc).__name__,
            )

    async def _assess_tool_duplicate(
        self,
        *,
        intent: str,
        candidate: UrlCandidate,
        page_evidence: list[PageEvidence],
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
                "entry_url": candidate.canonical_url,
            },
            "evidence": [
                {
                    "url": item.url,
                    "title": item.title,
                    "content_kind": item.content_kind,
                    "data_signals": item.data_signals,
                }
                for item in page_evidence[:4]
            ],
            "decision": {
                "outcome": decision.outcome,
                "reasoning": decision.reasoning,
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
            terms = _normalize_tool_terms(response.terms)
            reason = response.reason.strip()
        except Exception as exc:
            terms = []
            reason = f"fallback:{type(exc).__name__}"

        seed_terms = _seed_tool_identity_terms(candidate=candidate, page_evidence=page_evidence)
        if not _has_meaningful_tool_terms(terms):
            terms = seed_terms
            reason = f"{reason}|seeded_terms" if reason else "fallback:empty_terms|seeded_terms"
        else:
            terms = _merge_tool_terms(terms, seed_terms)[:8]

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

    def _merge_api_signal(self, page_evidence: list[PageEvidence], browser_result) -> ApiSignal:
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


def _one_line_page_summary(evidence: PageEvidence) -> str:
    title = evidence.title.strip() if evidence.title else ""
    bits = [evidence.content_kind]
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


def _one_line_pdf_summary(result) -> str:
    verdict = "pdf-match" if result.relevant else "pdf-nonmatch"
    base = result.title.strip() or result.url
    return f"{base} [{verdict}]"[:220]


def _one_line_browser_summary(result) -> str:
    reason = (result.reasoning or "browser delegation completed").strip()
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


def _normalize_tool_terms(raw_terms: list[str]) -> list[str]:
    terms: list[str] = []
    for raw_term in raw_terms:
        term = str(raw_term or "").strip()
        if not term:
            continue
        if term.lower() in GENERIC_TOOL_IDENTITY_TERMS:
            continue
        terms.append(term)
    return _merge_tool_terms(terms, [])[:8]


def _seed_tool_identity_terms(*, candidate: UrlCandidate, page_evidence: list[PageEvidence]) -> list[str]:
    terms: list[str] = []
    terms.append(candidate.domain)
    terms.append(candidate.source_title)
    terms.append(candidate.source_snippet)

    for evidence in page_evidence[:3]:
        terms.append(evidence.title)
        terms.extend(evidence.data_signals[:3])

    return _merge_tool_terms(terms, [])[:8]


def _has_meaningful_tool_terms(terms: list[str]) -> bool:
    for term in terms:
        normalized = str(term or "").strip().lower()
        if not normalized:
            continue
        if normalized in GENERIC_TOOL_IDENTITY_TERMS:
            continue
        return True
    return False


def _merge_tool_terms(primary: list[str], secondary: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for raw_term in [*primary, *secondary]:
        term = str(raw_term or "").strip()
        if not term:
            continue
        lowered = term.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        merged.append(term)
    return merged


def _normalize_decision_response(
    response: _DecisionRawEnvelope,
    *,
    inferred_source_evidence: list[SourceEvidenceItem],
) -> EvaluationDecision:
    notes = _normalize_notes(response.notes)
    standard_fields_present = all(
        [
            response.useful is not None,
            response.relevance_score is not None,
            bool((response.outcome or "").strip()),
            bool((response.reasoning or "").strip()),
            bool((response.api_stage or "").strip()),
        ]
    )
    if not standard_fields_present:
        notes.append("decision_shape_normalized")

    outcome = _normalize_outcome(response)
    useful = _normalize_useful(response, outcome=outcome)
    reasoning = (
        (response.reasoning or "").strip()
        or (response.why_useful or "").strip()
        or (
            f"{(response.evidence or '').strip()} {(response.how_to_use or response.recommendation or '').strip()}".strip()
        )
        or (response.reason or "").strip()
        or ("Relevant signals were found for this intent." if useful else "Insufficient evidence of intent relevance.")
    )
    api_stage = _normalize_api_stage(response, outcome=outcome)
    source_evidence = _normalize_source_evidence(response.source_evidence) or inferred_source_evidence
    outcome, useful = _reconcile_unknown_useful_outcome(
        outcome=outcome,
        useful=useful,
        api_stage=api_stage,
        source_evidence=source_evidence,
        notes=notes,
    )
    relevance_score = _normalize_score(response, useful=useful, outcome=outcome)

    return EvaluationDecision(
        useful=useful,
        relevance_score=relevance_score,
        outcome=outcome,
        reasoning=reasoning,
        api_stage=api_stage,
        source_evidence=source_evidence,
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


def _reconcile_unknown_useful_outcome(
    *,
    outcome: str,
    useful: bool,
    api_stage: str,
    source_evidence: list[SourceEvidenceItem],
    notes: list[str],
) -> tuple[str, bool]:
    if outcome != "unknown" or not useful:
        return outcome, useful

    inferred_outcome = _infer_outcome_from_source_evidence(
        api_stage=api_stage,
        source_evidence=source_evidence,
    )
    if inferred_outcome != "unknown":
        notes.append("unknown_useful_outcome_inferred_from_evidence")
        return inferred_outcome, True

    notes.append("unknown_useful_outcome_demoted")
    return "unknown", False


def _infer_outcome_from_source_evidence(*, api_stage: str, source_evidence: list[SourceEvidenceItem]) -> str:
    if api_stage != "none":
        return "api_available"

    kinds = {item.kind for item in source_evidence}
    if "paywall" in kinds:
        return "paywall"
    if "contact_sales" in kinds:
        return "contact_sales_only"
    if "api" in kinds:
        return "api_available"
    if kinds.intersection({"page", "pdf", "dataset", "file", "browser_finding"}):
        return "data_on_site"
    return "unknown"


def _normalize_source_evidence(raw: Any) -> list[SourceEvidenceItem]:
    if not isinstance(raw, list):
        return []
    items: list[SourceEvidenceItem] = []
    for value in raw:
        item = _coerce_source_evidence_item(value)
        if item is not None:
            items.append(item)
    return _merge_source_evidence([], items)


def _coerce_source_evidence_item(value: Any) -> SourceEvidenceItem | None:
    if isinstance(value, SourceEvidenceItem):
        return value
    if isinstance(value, dict):
        payload = dict(value)
        raw_url = str(payload.get("url") or "").strip()
        if raw_url:
            payload["url"] = canonicalize_url(raw_url)
        if not payload.get("kind"):
            payload["kind"] = _guess_evidence_kind(raw_url)
        try:
            return SourceEvidenceItem.model_validate(payload)
        except Exception:
            return None
    if isinstance(value, str):
        raw_url = value.strip()
        if raw_url.startswith(("http://", "https://")):
            return SourceEvidenceItem(
                kind=_guess_evidence_kind(raw_url),
                url=canonicalize_url(raw_url),
                summary="",
            )
    return None


def _guess_evidence_kind(url: str) -> str:
    lowered = url.lower()
    if lowered.endswith(".pdf"):
        return "pdf"
    if any(token in lowered for token in ("openapi", "swagger", "graphql", "/api", "developer")):
        return "api"
    if any(token in lowered for token in ("dataset", "/data", "catalog")):
        return "dataset"
    return "page"


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


def _infer_source_evidence(
    page_evidence: list[PageEvidence],
    browser_result,
    source_evidence: list[SourceEvidenceItem],
    api_probe,
) -> list[SourceEvidenceItem]:
    inferred = list(source_evidence)
    for item in page_evidence:
        if item.content_kind == "pdf":
            inferred.append(SourceEvidenceItem(kind="pdf", url=item.url, title=item.title, summary=item.text_excerpt[:800]))
        elif item.api_signal.detected:
            inferred.append(SourceEvidenceItem(kind="api", url=item.url, title=item.title, summary=item.text_excerpt[:800]))
        elif item.contact_sales_present:
            inferred.append(SourceEvidenceItem(kind="contact_sales", url=item.url, title=item.title, summary=item.text_excerpt[:800]))
        elif item.paywall_present:
            inferred.append(SourceEvidenceItem(kind="paywall", url=item.url, title=item.title, summary=item.text_excerpt[:800]))
        elif item.data_signals:
            inferred.append(SourceEvidenceItem(kind="page", url=item.url, title=item.title, summary=item.text_excerpt[:800]))
    if api_probe and api_probe.url:
        inferred.append(
            SourceEvidenceItem(
                kind="api",
                url=api_probe.url,
                summary=(api_probe.response_excerpt or api_probe.planner_reason or "API probe attempted.")[:800],
            )
        )
    if browser_result and browser_result.source_evidence:
        inferred.extend(browser_result.source_evidence)
    return _merge_source_evidence([], inferred)


def _merge_source_evidence(existing: list[SourceEvidenceItem], incoming: list[SourceEvidenceItem]) -> list[SourceEvidenceItem]:
    merged: list[SourceEvidenceItem] = []
    seen: set[tuple[str, str, str]] = set()
    for item in [*existing, *incoming]:
        if not item.url and not item.summary:
            continue
        key = (item.kind, item.url.strip(), item.summary.strip()[:120])
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _fallback_decision_from_evidence(
    *,
    page_evidence: list[PageEvidence],
    browser_result,
    api_probe,
    source_evidence: list[SourceEvidenceItem],
    error: str,
) -> EvaluationDecision:
    has_paywall = any(item.paywall_present for item in page_evidence) or bool(browser_result and browser_result.paywall_present)
    has_contact_sales = any(item.contact_sales_present for item in page_evidence) or bool(browser_result and browser_result.contact_sales_only)
    has_data = bool(
        any(item.data_signals for item in page_evidence)
        or any(item.content_kind == "pdf" and item.text_excerpt.strip() for item in page_evidence)
        or bool(browser_result and browser_result.data_on_site)
    )
    has_api = bool(
        any(item.api_signal.detected for item in page_evidence)
        or (api_probe and (api_probe.url or api_probe.accessible or api_probe.relevant_guess))
        or bool(browser_result and browser_result.api_detected)
    )

    if has_paywall:
        return EvaluationDecision(
            useful=True,
            relevance_score=0.55,
            outcome="paywall",
            reasoning="Access appears relevant but gated behind a paywall/upgrade flow. Capture for human follow-up.",
            api_stage="none",
            source_evidence=source_evidence,
            notes=[f"decision_fallback:{error}", "decision_fallback_path:paywall"],
        )
    if has_data:
        return EvaluationDecision(
            useful=True,
            relevance_score=0.72,
            outcome="data_on_site",
            reasoning="On-site signals indicate recurring procurement/data listings that can be monitored from this domain.",
            api_stage="none",
            source_evidence=source_evidence,
            notes=[f"decision_fallback:{error}", "decision_fallback_path:data_on_site"],
        )
    if has_api:
        api_stage = "api_detected"
        if api_probe and api_probe.accessible:
            api_stage = "api_accessible"
        if api_probe and api_probe.relevant_guess:
            api_stage = "api_relevant"
        if api_probe and api_probe.viable_guess:
            api_stage = "api_viable"
        return EvaluationDecision(
            useful=True,
            relevance_score=0.68,
            outcome="api_available",
            reasoning="API-like signals were found and appear potentially usable for recurring extraction.",
            api_stage=api_stage,
            source_evidence=source_evidence,
            notes=[f"decision_fallback:{error}", "decision_fallback_path:api_available"],
        )
    if has_contact_sales:
        return EvaluationDecision(
            useful=True,
            relevance_score=0.45,
            outcome="contact_sales_only",
            reasoning="Relevant area appears present, but access is primarily sales/demo-gated.",
            api_stage="none",
            source_evidence=source_evidence,
            notes=[f"decision_fallback:{error}", "decision_fallback_path:contact_sales_only"],
        )
    return EvaluationDecision(
        useful=False,
        relevance_score=0.1,
        outcome="irrelevant",
        reasoning="Insufficient direct relevance evidence was found on explored pages.",
        api_stage="none",
        source_evidence=source_evidence,
        notes=[f"decision_fallback:{error}", "decision_fallback_path:irrelevant"],
    )


def _describe_exception(exc: Exception) -> str:
    text = str(exc).strip()
    if text:
        return text
    return type(exc).__name__
