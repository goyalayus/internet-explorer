from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from internet_explorer.browser_delegate import BrowserDelegationManager
from internet_explorer.canonicalize import canonicalize_url, homepage_url_for_domain, load_baseline_domains, registrable_domain
from internet_explorer.config import AppConfig
from internet_explorer.evaluator import UrlEvaluator
from internet_explorer.fetcher import AsyncWebFetcher
from internet_explorer.llm import LLMClient
from internet_explorer.models import QueryPlan, RunSummary, SearchResult, Strategy, UrlCandidate
from internet_explorer.planning_cache import DiscoveryPlanningCacheStore
from internet_explorer.pdf_verify import PdfVerifierService
from internet_explorer.persistence import MongoPersistence
from internet_explorer.search import GoogleSearchCollector
from internet_explorer.strategy import StrategyPlanner
from internet_explorer.telemetry import Telemetry
from internet_explorer.tool_inventory import ToolInventory
from internet_explorer.vpn import GenericVpnManager


_PROCUREMENT_MARKERS = (
    "rfp",
    "tender",
    "procurement",
    "solicitation",
    "contract opportunities",
    "contract opportunity",
    "vendor",
    "supplier",
    "bid",
    "bids",
    "eprocure",
    "opportunities",
)
_LOW_SIGNAL_MARKERS = (
    "faculty",
    "profile",
    "staff",
    "team",
    "directory",
    "alumni",
    "blog",
    "newsroom",
    "press release",
    "publication",
    "journal",
    "case study",
    "whitepaper",
    "github repository",
    "readthedocs",
)
_LOW_SIGNAL_URL_MARKERS = (
    "/faculty",
    "/profile",
    "/profiles",
    "/people/",
    "/person/",
    "/staff/",
    "/team/",
    "/news/",
    "/blog/",
    "/article/",
    "/articles/",
    "/press/",
    "/wiki/",
)
_PROCUREMENT_URL_MARKERS = (
    "/rfp",
    "/tender",
    "/tenders",
    "/procurement",
    "/solicitation",
    "/solicitations",
    "/contract",
    "/contracts",
    "/bids",
    "/bid",
    "/vendor",
    "/supplier",
)
_NOISE_HOSTS = {
    "facebook.com",
    "linkedin.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "youtube.com",
    "tiktok.com",
    "medium.com",
    "github.com",
    "readthedocs.io",
    "substack.com",
    "pinterest.com",
    "google.com",
    "bing.com",
    "duckduckgo.com",
    "yahoo.com",
}


@dataclass(slots=True)
class _DomainResult:
    canonical_url: str
    result: SearchResult


class IntentDiscoveryService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.persistence = MongoPersistence(config)
        self.llm = LLMClient(config)

    async def run(self, intent: str) -> RunSummary:
        run_id = f"run_{uuid.uuid4().hex}"
        summary = RunSummary(
            run_id=run_id,
            intent=intent,
            candidate_start_mode=self.config.candidate_start_mode,
            started_at=datetime.utcnow(),
        )
        telemetry = Telemetry(self.persistence, run_id=run_id, intent_id=f"intent_{uuid.uuid4().hex[:8]}")
        vpn_manager = GenericVpnManager(self.config)
        vpn_started_by_service = False
        self.persistence.create_run(
            summary,
            metadata={
                "env_file_path": str(self.config.env_file_path),
                "selected_vpn_start_script": str(self.config.vpn_start_script) if self.config.vpn_start_script else None,
                "vpn_ovpn_config": str(self.config.vpn_ovpn_config) if self.config.vpn_ovpn_config else None,
                "vpn_defaults_file": str(self.config.vpn_defaults_file) if self.config.vpn_defaults_file else None,
                "vpn_docdb_host": self.config.vpn_docdb_host,
                "vpn_docdb_port": self.config.vpn_docdb_port,
                "vpn_log_dir": str(self.config.vpn_log_dir),
                "eu_swarm_path": str(self.config.eu_swarm_path) if self.config.eu_swarm_path else None,
                "discovery_cache_mode": self.config.discovery_cache_mode,
                "discovery_cache_dir": str(self.config.discovery_cache_dir),
                "discovery_cache_key": self.config.discovery_cache_key or None,
            },
        )

        baseline_domains = load_baseline_domains(self.config.baseline_domains_file)
        tool_inventory = ToolInventory.from_file(self.config.known_tools_file)
        self.persistence.update_run(
            run_id,
            {
                "tool_inventory_count": len(tool_inventory.tool_names),
                "tool_inventory_preview": tool_inventory.tool_names[:40],
            },
        )
        planner = StrategyPlanner(self.config, self.llm, telemetry)
        searcher = GoogleSearchCollector(self.config, telemetry)
        fetcher = AsyncWebFetcher(self.config)
        pdf_verifier = PdfVerifierService(fetcher, self.llm, telemetry)
        browser_manager = BrowserDelegationManager(
            self.config,
            telemetry,
            lambda fields: self.persistence.update_run(run_id, fields),
            pdf_verifier=pdf_verifier,
        )
        evaluator = UrlEvaluator(
            self.config,
            self.llm,
            fetcher,
            telemetry,
            browser_manager,
            tool_inventory=tool_inventory,
        )

        try:
            if self.config.auto_start_vpn:
                telemetry.emit(
                    phase="vpn_start",
                    actor="system",
                    output_summary={"auto_start_vpn": True},
                    decision="requested",
                )
                vpn_status = await asyncio.to_thread(vpn_manager.ensure_started)
                vpn_started_by_service = vpn_status.started_by_this_call
                self.persistence.update_run(run_id, {"vpn_status": vpn_status.model_dump(mode="json")})
                telemetry.emit(
                    phase="vpn_start",
                    actor="system",
                    output_summary=vpn_status.model_dump(mode="json"),
                    decision=vpn_status.message,
                )

            strategies, queries, raw_results, cache_info = await self._resolve_discovery_inputs(
                intent=intent,
                planner=planner,
                searcher=searcher,
                telemetry=telemetry,
            )
            summary.strategy_count = len(strategies)
            summary.query_count = len(queries)
            self.persistence.update_run(
                run_id,
                {
                    "strategy_count": summary.strategy_count,
                    "query_count": summary.query_count,
                    "planning_cache": cache_info,
                },
            )

            summary.raw_result_count = len(raw_results)
            telemetry.emit(
                phase="url_extract",
                actor="system",
                input_payload={"result_count": len(raw_results)},
                output_summary={"raw_result_count": len(raw_results)},
                decision="raw_urls_ready",
            )

            deduped_candidates = self._dedupe_results(raw_results, baseline_domains, telemetry)
            summary.unique_url_count = len(deduped_candidates)
            summary.unique_source_count = len(deduped_candidates)
            self.persistence.update_run(
                run_id,
                {
                    "raw_result_count": summary.raw_result_count,
                    "unique_url_count": summary.unique_url_count,
                    "unique_source_count": summary.unique_source_count,
                },
            )

            evaluation_counts = {"evaluated": 0, "useful": 0}

            def _record_evaluation(evaluation) -> None:
                evaluation_counts["evaluated"] += 1
                if evaluation.useful:
                    evaluation_counts["useful"] += 1
                self.persistence.upsert_url_summary(run_id, evaluation)
                telemetry.emit(
                    phase="db_write",
                    actor="system",
                    url_id=evaluation.url_id,
                    output_summary={"url_id": evaluation.url_id, "useful": evaluation.useful},
                    decision="url_summary_upserted",
                )
                self.persistence.update_run(
                    run_id,
                    {
                        "evaluated_url_count": evaluation_counts["evaluated"],
                        "useful_url_count": evaluation_counts["useful"],
                        "evaluated_source_count": evaluation_counts["evaluated"],
                        "useful_source_count": evaluation_counts["useful"],
                        "browser_peak_active": browser_manager.peak,
                    },
                )

            await self._evaluate_candidates(
                intent,
                deduped_candidates,
                evaluator,
                telemetry,
                on_result=_record_evaluation,
            )
            summary.evaluated_url_count = evaluation_counts["evaluated"]
            summary.useful_url_count = evaluation_counts["useful"]
            summary.evaluated_source_count = summary.evaluated_url_count
            summary.useful_source_count = summary.useful_url_count
            summary.browser_peak_active = browser_manager.peak
            summary.finished_at = datetime.utcnow()
            summary.completed_at = summary.finished_at
            summary.status = "completed"

            self.persistence.update_run(run_id, summary.model_dump())
            return summary
        except asyncio.CancelledError as exc:
            summary.status = "failed"
            summary.error = "run_cancelled"
            summary.finished_at = datetime.utcnow()
            summary.completed_at = summary.finished_at
            self.persistence.update_run(
                run_id,
                {
                    "status": summary.status,
                    "error": summary.error,
                    "finished_at": summary.finished_at,
                    "completed_at": summary.completed_at,
                },
            )
            telemetry.emit(
                phase="run_cancelled",
                actor="system",
                output_summary={"error": str(exc)},
                decision="cancelled",
                error_code=type(exc).__name__,
            )
            raise
        except Exception as exc:
            summary.status = "failed"
            summary.error = str(exc)
            summary.finished_at = datetime.utcnow()
            summary.completed_at = summary.finished_at
            self.persistence.update_run(
                run_id,
                {
                    "status": summary.status,
                    "error": summary.error,
                    "finished_at": summary.finished_at,
                    "completed_at": summary.completed_at,
                },
            )
            telemetry.emit(
                phase="run_failed",
                actor="system",
                output_summary={"error": str(exc)},
                decision="failed",
                error_code=type(exc).__name__,
            )
            raise
        finally:
            await fetcher.close()
            await searcher.close()
            await self.llm.close()
            if self.config.auto_start_vpn and vpn_started_by_service:
                try:
                    vpn_status = await asyncio.to_thread(vpn_manager.stop)
                    self.persistence.update_run(run_id, {"vpn_status_after_stop": vpn_status.model_dump(mode="json")})
                    telemetry.emit(
                        phase="vpn_stop",
                        actor="system",
                        output_summary=vpn_status.model_dump(mode="json"),
                        decision=vpn_status.message,
                    )
                except Exception as exc:
                    telemetry.emit(
                        phase="vpn_stop",
                        actor="system",
                        output_summary={"error": str(exc)},
                        decision="stop_failed",
                        error_code=type(exc).__name__,
                    )

    async def _resolve_discovery_inputs(
        self,
        *,
        intent: str,
        planner: StrategyPlanner,
        searcher: GoogleSearchCollector,
        telemetry: Telemetry,
    ) -> tuple[list[Strategy], list[QueryPlan], list[SearchResult], dict]:
        mode = self.config.discovery_cache_mode
        if mode == "off":
            strategies, queries, raw_results = await self._build_discovery_inputs_live(intent, planner, searcher)
            return strategies, queries, raw_results, {"mode": mode, "status": "disabled"}

        cache_store = DiscoveryPlanningCacheStore(self.config)
        cache_path = cache_store.path_for_intent(intent)
        cache_key = cache_store.key_for_intent(intent)

        if mode in {"read_only", "read_write"}:
            cached_snapshot = None
            try:
                cached_snapshot = cache_store.load(intent)
            except Exception as exc:
                telemetry.emit(
                    phase="planning_cache",
                    actor="system",
                    input_payload={"mode": mode, "cache_key": cache_key, "cache_path": str(cache_path)},
                    output_summary={"error": str(exc)},
                    decision="cache_load_error",
                    error_code=type(exc).__name__,
                )
                if mode == "read_only":
                    raise RuntimeError(
                        f"DISCOVERY_CACHE_MODE=read_only could not load cache at {cache_path}: {exc}"
                    ) from exc
            if cached_snapshot is not None:
                telemetry.emit(
                    phase="planning_cache",
                    actor="system",
                    input_payload={"mode": mode, "cache_key": cache_key, "cache_path": str(cache_path)},
                    output_summary={
                        "strategies": len(cached_snapshot.strategies),
                        "queries": len(cached_snapshot.queries),
                        "raw_results": len(cached_snapshot.raw_results),
                    },
                    decision="cache_hit",
                )
                return (
                    cached_snapshot.strategies,
                    cached_snapshot.queries,
                    cached_snapshot.raw_results,
                    {
                        "mode": mode,
                        "status": "hit",
                        "cache_key": cache_key,
                        "cache_path": str(cache_path),
                        "created_at": cached_snapshot.created_at.isoformat(),
                    },
                )
            telemetry.emit(
                phase="planning_cache",
                actor="system",
                input_payload={"mode": mode, "cache_key": cache_key, "cache_path": str(cache_path)},
                output_summary={},
                decision="cache_miss",
            )
            if mode == "read_only":
                raise RuntimeError(f"DISCOVERY_CACHE_MODE=read_only cache file not found: {cache_path}")

        strategies, queries, raw_results = await self._build_discovery_inputs_live(intent, planner, searcher)
        if mode in {"read_write", "refresh"}:
            snapshot = cache_store.save(
                intent=intent,
                strategies=strategies,
                queries=queries,
                raw_results=raw_results,
            )
            telemetry.emit(
                phase="planning_cache",
                actor="system",
                input_payload={"mode": mode, "cache_key": cache_key, "cache_path": str(cache_path)},
                output_summary={
                    "strategies": len(strategies),
                    "queries": len(queries),
                    "raw_results": len(raw_results),
                },
                decision="cache_write",
            )
            return (
                strategies,
                queries,
                raw_results,
                {
                    "mode": mode,
                    "status": "written",
                    "cache_key": cache_key,
                    "cache_path": str(cache_path),
                    "created_at": snapshot.created_at.isoformat(),
                },
            )

        return strategies, queries, raw_results, {"mode": mode, "status": "generated_no_write"}

    async def _build_discovery_inputs_live(
        self,
        intent: str,
        planner: StrategyPlanner,
        searcher: GoogleSearchCollector,
    ) -> tuple[list[Strategy], list[QueryPlan], list[SearchResult]]:
        strategies = await planner.generate_strategies(intent)
        queries: list[QueryPlan] = []
        for strategy in strategies:
            queries.extend(await planner.generate_queries(intent, strategy))
        raw_results = await searcher.collect_many(queries)
        return strategies, queries, raw_results

    def _dedupe_results(self, raw_results, baseline_domains: set[str], telemetry: Telemetry) -> list[UrlCandidate]:
        grouped: dict[str, list[_DomainResult]] = {}
        domain_order: list[str] = []

        for result in raw_results:
            canonical_url = canonicalize_url(result.url)
            domain = registrable_domain(canonical_url)
            if not domain:
                continue

            if domain not in grouped:
                grouped[domain] = []
                domain_order.append(domain)
            grouped[domain].append(_DomainResult(canonical_url=canonical_url, result=result))

        deduped: dict[str, UrlCandidate] = {}
        skipped_domains: list[dict[str, str]] = []
        for domain in domain_order:
            domain_results = grouped.get(domain) or []
            if not domain_results:
                continue

            selected = max(
                domain_results,
                key=lambda item: (
                    _score_search_result(item.result, item.canonical_url, domain=domain),
                    -int(item.result.rank),
                    -int(item.result.serp_page),
                ),
            )
            selected_score = _score_search_result(selected.result, selected.canonical_url, domain=domain)
            skip_reason = _skip_reason_for_domain_candidate(
                domain=domain,
                result=selected.result,
                canonical_url=selected.canonical_url,
                score=selected_score,
            )
            if skip_reason:
                skipped_domains.append(
                    {
                        "domain": domain,
                        "reason": skip_reason,
                        "url": selected.canonical_url,
                        "title": selected.result.title,
                    }
                )
                continue

            homepage_url = homepage_url_for_domain(domain)
            start_url = homepage_url if self.config.candidate_start_mode == "domain_homepage" else selected.canonical_url
            deduped[domain] = UrlCandidate(
                url_id=f"url_{uuid.uuid4().hex[:10]}",
                strategy_id=selected.result.strategy_id,
                query_id=selected.result.query_id,
                raw_url=selected.result.url,
                canonical_url=selected.canonical_url,
                start_url=start_url,
                homepage_url=homepage_url,
                domain=domain,
                novelty=domain not in baseline_domains,
                start_mode=self.config.candidate_start_mode,  # type: ignore[arg-type]
                source_title=selected.result.title,
                source_snippet=selected.result.snippet,
                serp_rank=selected.result.rank,
                serp_page=selected.result.serp_page,
                content_kind_hint=_guess_content_kind(selected.canonical_url),
            )

        candidates = list(deduped.values())
        telemetry.emit(
            phase="dedup",
            actor="system",
            input_payload={"raw_results": len(raw_results)},
            output_summary=[candidate.model_dump() for candidate in candidates[:50]],
            decision=f"deduped_domains_to_{len(candidates)}",
            extra={
                "deduped_total": len(candidates),
                "dedupe_key": "registrable_domain",
                "skipped_domain_count": len(skipped_domains),
                "skipped_domains_preview": skipped_domains[:40],
            },
        )
        return candidates

    async def _evaluate_candidates(
        self,
        intent: str,
        candidates: list[UrlCandidate],
        evaluator: UrlEvaluator,
        telemetry: Telemetry,
        on_result: Callable[[object], None] | None = None,
    ):
        semaphore = asyncio.Semaphore(self.config.max_url_concurrency) if self.config.max_url_concurrency > 0 else None
        batch_size = max(1, self.config.url_batch_size)

        async def run_one(candidate: UrlCandidate):
            if semaphore is None:
                return await evaluator.evaluate(intent=intent, candidate=candidate)
            async with semaphore:
                return await evaluator.evaluate(intent=intent, candidate=candidate)

        all_results = []
        total = len(candidates)
        for batch_index, start in enumerate(range(0, total, batch_size), start=1):
            batch = candidates[start : start + batch_size]
            telemetry.emit(
                phase="triage",
                actor="system",
                input_payload={
                    "batch_index": batch_index,
                    "batch_size": len(batch),
                    "batch_start": start,
                    "batch_end": start + len(batch),
                    "total_candidates": total,
                },
                output_summary={"url_ids": [candidate.url_id for candidate in batch[:50]]},
                decision="batch_start",
            )
            tasks = [asyncio.create_task(run_one(candidate)) for candidate in batch]
            batch_results = []
            try:
                for finished in asyncio.as_completed(tasks):
                    result = await finished
                    batch_results.append(result)
                    all_results.append(result)
                    if on_result is not None:
                        on_result(result)
            except Exception:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            telemetry.emit(
                phase="triage",
                actor="system",
                input_payload={
                    "batch_index": batch_index,
                    "batch_size": len(batch),
                    "completed_total": len(all_results),
                    "total_candidates": total,
                },
                output_summary={"useful_in_batch": sum(1 for item in batch_results if item.useful)},
                decision="batch_done",
            )
        return all_results


def _guess_content_kind(url: str) -> str:
    lowered = url.lower()
    if lowered.endswith(".pdf") or ".pdf?" in lowered:
        return "pdf"
    return "html_page"


def _score_search_result(result: SearchResult, canonical_url: str, *, domain: str) -> float:
    text = f"{result.title} {result.snippet} {canonical_url}".lower()
    score = 0.0

    procurement_hits = sum(1 for marker in _PROCUREMENT_MARKERS if marker in text)
    score += min(procurement_hits, 5) * 1.1

    procurement_url_hits = sum(1 for marker in _PROCUREMENT_URL_MARKERS if marker in canonical_url.lower())
    score += min(procurement_url_hits, 4) * 1.0

    low_signal_hits = sum(1 for marker in _LOW_SIGNAL_MARKERS if marker in text)
    score -= min(low_signal_hits, 4) * 1.0

    low_signal_url_hits = sum(1 for marker in _LOW_SIGNAL_URL_MARKERS if marker in canonical_url.lower())
    score -= min(low_signal_url_hits, 3) * 1.2

    rank = max(1, int(result.rank))
    score += max(0.0, 1.2 - (rank - 1) * 0.08)

    page = max(1, int(result.serp_page))
    score += 0.2 if page == 1 else 0.0

    lowered_domain = domain.lower()
    if _is_government_or_public_sector_domain(lowered_domain):
        score += 0.8

    if canonical_url.lower().endswith(".pdf") or ".pdf?" in canonical_url.lower():
        score -= 1.0

    return score


def _skip_reason_for_domain_candidate(*, domain: str, result: SearchResult, canonical_url: str, score: float) -> str:
    lowered_domain = domain.lower()
    if any(lowered_domain == host or lowered_domain.endswith(f".{host}") for host in _NOISE_HOSTS):
        return "noise_host_filtered"

    if score <= -1.5 and not _is_government_or_public_sector_domain(lowered_domain):
        return "very_low_signal_candidate"

    text = f"{result.title} {result.snippet} {canonical_url}".lower()
    has_procurement_signal = any(marker in text for marker in _PROCUREMENT_MARKERS)
    if not has_procurement_signal and any(marker in canonical_url.lower() for marker in _LOW_SIGNAL_URL_MARKERS):
        return "low_signal_profile_page_only"

    return ""


def _is_government_or_public_sector_domain(domain: str) -> bool:
    lowered = (domain or "").lower()
    if not lowered:
        return False
    if lowered.endswith(".gov") or ".gov." in lowered:
        return True
    if lowered.endswith(".gob.mx") or lowered.endswith(".gov.in"):
        return True
    if lowered.endswith(".edu") or ".edu." in lowered:
        return True
    return False
