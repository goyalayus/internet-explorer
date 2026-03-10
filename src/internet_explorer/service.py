from __future__ import annotations

import asyncio
import uuid
from datetime import datetime

from internet_explorer.browser_delegate import BrowserDelegationManager
from internet_explorer.canonicalize import canonical_domain, canonicalize_url, load_baseline_domains, registrable_domain
from internet_explorer.config import AppConfig
from internet_explorer.evaluator import UrlEvaluator
from internet_explorer.fetcher import AsyncWebFetcher
from internet_explorer.llm import LLMClient
from internet_explorer.models import RunSummary, UrlCandidate
from internet_explorer.persistence import MongoPersistence
from internet_explorer.search import GoogleSearchCollector
from internet_explorer.strategy import StrategyPlanner
from internet_explorer.telemetry import Telemetry
from internet_explorer.vpn import GenericVpnManager


class IntentDiscoveryService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.persistence = MongoPersistence(config)
        self.llm = LLMClient(config)

    async def run(self, intent: str) -> RunSummary:
        run_id = f"run_{uuid.uuid4().hex}"
        summary = RunSummary(run_id=run_id, intent=intent, started_at=datetime.utcnow())
        telemetry = Telemetry(self.persistence, run_id=run_id, intent_id=f"intent_{uuid.uuid4().hex[:8]}")
        vpn_manager = GenericVpnManager(self.config)
        vpn_started_by_service = False
        self.persistence.create_run(
            summary,
            metadata={
                "tool_flow_env_path": str(self.config.tool_flow_env_path),
                "tool_flow_vpn_scripts": [str(path) for path in self.config.discovered_vpn_scripts],
                "selected_vpn_start_script": str(self.config.vpn_start_script) if self.config.vpn_start_script else None,
                "query_optimizer_ovpn_config": str(self.config.query_optimizer_ovpn_config) if self.config.query_optimizer_ovpn_config else None,
                "vpn_docdb_host": self.config.vpn_docdb_host,
                "vpn_docdb_port": self.config.vpn_docdb_port,
                "vpn_log_dir": str(self.config.vpn_log_dir),
                "eu_swarm_path": str(self.config.eu_swarm_path),
            },
        )

        baseline_domains = load_baseline_domains(self.config.baseline_domains_file)
        planner = StrategyPlanner(self.config, self.llm, telemetry)
        searcher = GoogleSearchCollector(self.config, telemetry)
        fetcher = AsyncWebFetcher(self.config)
        browser_manager = BrowserDelegationManager(self.config, telemetry, lambda fields: self.persistence.update_run(run_id, fields))
        evaluator = UrlEvaluator(self.config, self.llm, fetcher, telemetry, browser_manager)

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

            strategies = await planner.generate_strategies(intent)
            summary.strategy_count = len(strategies)
            queries = []
            for strategy in strategies:
                queries.extend(await planner.generate_queries(intent, strategy))
            summary.query_count = len(queries)
            self.persistence.update_run(
                run_id,
                {"strategy_count": summary.strategy_count, "query_count": summary.query_count},
            )

            raw_results = await searcher.collect_many(queries)
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
            self.persistence.update_run(
                run_id,
                {"raw_result_count": summary.raw_result_count, "unique_url_count": summary.unique_url_count},
            )

            evaluations = await self._evaluate_candidates(intent, deduped_candidates, evaluator)
            summary.evaluated_url_count = len(evaluations)
            summary.useful_url_count = sum(1 for evaluation in evaluations if evaluation.useful)
            summary.browser_peak_active = browser_manager.peak
            summary.finished_at = datetime.utcnow()
            summary.status = "completed"

            for evaluation in evaluations:
                self.persistence.upsert_url_summary(run_id, evaluation)
                telemetry.emit(
                    phase="db_write",
                    actor="system",
                    url_id=evaluation.url_id,
                    output_summary={"url_id": evaluation.url_id, "useful": evaluation.useful},
                    decision="url_summary_upserted",
                )

            self.persistence.update_run(run_id, summary.model_dump(mode="json"))
            return summary
        except Exception as exc:
            summary.status = "failed"
            summary.error = str(exc)
            summary.finished_at = datetime.utcnow()
            self.persistence.update_run(run_id, summary.model_dump(mode="json"))
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

    def _dedupe_results(self, raw_results, baseline_domains: set[str], telemetry: Telemetry) -> list[UrlCandidate]:
        deduped: dict[str, UrlCandidate] = {}
        for result in raw_results:
            canonical_url = canonicalize_url(result.url)
            domain = registrable_domain(canonical_url)
            if canonical_url in deduped:
                continue
            deduped[canonical_url] = UrlCandidate(
                url_id=f"url_{uuid.uuid4().hex[:10]}",
                strategy_id=result.strategy_id,
                query_id=result.query_id,
                raw_url=result.url,
                canonical_url=canonical_url,
                domain=domain,
                novelty=domain not in baseline_domains,
                source_title=result.title,
                source_snippet=result.snippet,
                serp_rank=result.rank,
                serp_page=result.serp_page,
            )
        candidates = list(deduped.values())
        telemetry.emit(
            phase="dedup",
            actor="system",
            input_payload={"raw_results": len(raw_results)},
            output_summary=[candidate.model_dump() for candidate in candidates[:50]],
            decision=f"deduped_to_{len(candidates)}",
            extra={"deduped_total": len(candidates)},
        )
        return candidates

    async def _evaluate_candidates(self, intent: str, candidates: list[UrlCandidate], evaluator: UrlEvaluator):
        semaphore = asyncio.Semaphore(self.config.max_url_concurrency) if self.config.max_url_concurrency > 0 else None

        async def run_one(candidate: UrlCandidate):
            if semaphore is None:
                return await evaluator.evaluate(intent=intent, candidate=candidate)
            async with semaphore:
                return await evaluator.evaluate(intent=intent, candidate=candidate)

        return await asyncio.gather(*(run_one(candidate) for candidate in candidates))
