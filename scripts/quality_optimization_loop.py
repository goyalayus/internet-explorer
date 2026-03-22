from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from internet_explorer.config import AppConfig
from internet_explorer.evaluator import _classify_scraping_path_quality
from internet_explorer.persistence import MongoPersistence
from internet_explorer.runtime_bootstrap import run_runtime_bootstrap
from internet_explorer.service import IntentDiscoveryService


@dataclass
class QualityMetrics:
    run_id: str
    evaluated_count: int
    useful_count: int
    strong_count: int
    strong_clear_path_count: int
    weak_useful_count: int

    @property
    def clear_path_ratio(self) -> float:
        if self.useful_count <= 0:
            return 0.0
        return self.strong_clear_path_count / self.useful_count

    @property
    def noise_useful_count(self) -> int:
        return max(0, self.useful_count - self.strong_clear_path_count)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated internet-explorer experiments and optimize for strong "
            "high-confidence sources with clear recurring scraping paths."
        )
    )
    parser.add_argument("--intent", required=True, help="Intent string for the run.")
    parser.add_argument("--max-runs", type=int, default=12, help="Maximum run attempts before forced stop.")
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=3,
        help="Stop when there is no quality-metric improvement for this many consecutive runs.",
    )
    parser.add_argument(
        "--experiment-dir",
        default="experiments/exp-003-quality-path-optimization",
        help="Directory where status/results artifacts are written.",
    )
    parser.add_argument(
        "--skip-runtime-up",
        action="store_true",
        help="Skip runtime bootstrap. Useful when VPN/Mongo are already known-good.",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def _close_persistence(persistence: MongoPersistence) -> None:
    close_fn = getattr(persistence, "close", None)
    if callable(close_fn):
        close_fn()
        return
    client = getattr(persistence, "client", None)
    if client is not None and hasattr(client, "close"):
        client.close()


def _is_mongo_connectivity_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = (
        "serverselectiontimeouterror",
        "networktimeout",
        "no servers found yet",
        "topology description",
        "timed out",
        "sockettimeoutms",
        "connecttimeoutms",
        "docdb",
        "mongodb",
    )
    return any(marker in text for marker in markers)


def _attempt_runtime_recovery(*, status_log_path: Path, reason: str) -> bool:
    _append_line(status_log_path, f"[{_utc_now()}] runtime recovery requested: {reason}")
    bootstrap = run_runtime_bootstrap(
        Path.cwd(),
        write_env=False,
        start_vpn=True,
        verify_mongo=True,
    )
    _append_line(
        status_log_path,
        (
            f"[{_utc_now()}] runtime recovery ready={bootstrap.ready} "
            f"vpn={bootstrap.vpn_status_after.get('message') if bootstrap.vpn_status_after else 'none'} "
            f"mongo={bootstrap.mongo_message}"
        ),
    )
    return bool(bootstrap.ready)


def _is_strong(doc: dict[str, Any]) -> bool:
    if not bool(doc.get("useful")):
        return False

    score = float(doc.get("relevance_score") or 0.0)
    if score < 0.8:
        return False

    evidence = doc.get("source_evidence") or []
    if not isinstance(evidence, list) or not evidence:
        return False

    notes = [str(item) for item in (doc.get("notes") or [])]
    if any(note.startswith("decision_fallback:") for note in notes):
        return False
    if any(note.startswith("evaluation_error:") for note in notes):
        return False

    return True


def _collect_metrics(config: AppConfig, run_id: str) -> QualityMetrics:
    persistence = MongoPersistence(config)
    try:
        summaries = list(
            persistence.url_summaries.find(
                {"run_id": run_id},
                {
                    "_id": 0,
                    "useful": 1,
                    "relevance_score": 1,
                    "reasoning": 1,
                    "source_evidence": 1,
                    "notes": 1,
                },
            )
        )
    finally:
        _close_persistence(persistence)

    evaluated_count = len(summaries)
    useful_count = 0
    strong_count = 0
    strong_clear_path_count = 0
    weak_useful_count = 0

    for doc in summaries:
        useful = bool(doc.get("useful"))
        if useful:
            useful_count += 1

        path_quality = _classify_scraping_path_quality(str(doc.get("reasoning") or ""))
        if useful and path_quality == "weak":
            weak_useful_count += 1

        strong = _is_strong(doc)
        if not strong:
            continue

        strong_count += 1
        if path_quality == "good":
            strong_clear_path_count += 1

    return QualityMetrics(
        run_id=run_id,
        evaluated_count=evaluated_count,
        useful_count=useful_count,
        strong_count=strong_count,
        strong_clear_path_count=strong_clear_path_count,
        weak_useful_count=weak_useful_count,
    )


def _latest_run_id_since(config: AppConfig, started_at_utc: datetime) -> str:
    persistence = MongoPersistence(config)
    try:
        row = (
            persistence.runs.find(
                {"started_at": {"$gte": started_at_utc}},
                {"_id": 0, "run_id": 1},
            )
            .sort("started_at", -1)
            .limit(1)
        )
        docs = list(row)
    finally:
        _close_persistence(persistence)
    if not docs:
        return ""
    return str(docs[0].get("run_id") or "")


def _run_status(config: AppConfig, run_id: str) -> dict[str, Any]:
    persistence = MongoPersistence(config)
    try:
        doc = persistence.runs.find_one(
            {"run_id": run_id},
            {"_id": 0, "status": 1, "error": 1, "evaluated_url_count": 1, "useful_url_count": 1},
        )
    finally:
        _close_persistence(persistence)
    return doc or {}


def _is_better(current: QualityMetrics, best: QualityMetrics | None) -> bool:
    if best is None:
        return True

    current_key = (
        current.strong_clear_path_count,
        -current.noise_useful_count,
        current.strong_count,
        -current.weak_useful_count,
    )
    best_key = (
        best.strong_clear_path_count,
        -best.noise_useful_count,
        best.strong_count,
        -best.weak_useful_count,
    )
    return current_key > best_key


def _check_query_budget(config: AppConfig) -> None:
    if config.strategy_count > 10:
        raise RuntimeError(
            f"Query-budget constraint violated: STRATEGY_COUNT={config.strategy_count} (>10)."
        )
    if config.queries_per_strategy > 5:
        raise RuntimeError(
            f"Query-budget constraint violated: QUERIES_PER_STRATEGY={config.queries_per_strategy} (>5)."
        )
    if config.serp_pages_per_query > 2:
        raise RuntimeError(
            f"Query-budget constraint violated: SERP_PAGES_PER_QUERY={config.serp_pages_per_query} (>2)."
        )


async def _run_once(intent: str) -> str:
    config = AppConfig.from_env(Path.cwd(), prefer_process_env=True)
    _check_query_budget(config)
    service = IntentDiscoveryService(config)
    summary = await service.run(intent)
    return summary.run_id


def main() -> None:
    args = _parse_args()
    experiment_dir = Path(args.experiment_dir).resolve()
    status_log_path = experiment_dir / "status-log.md"
    results_path = experiment_dir / "results.md"

    _append_line(status_log_path, f"[{_utc_now()}] quality-loop start intent={json.dumps(args.intent)}")

    if not args.skip_runtime_up:
        bootstrap = run_runtime_bootstrap(
            Path.cwd(),
            write_env=False,
            start_vpn=True,
            verify_mongo=True,
        )
        _append_line(
            status_log_path,
            (
                f"[{_utc_now()}] runtime-up ready={bootstrap.ready} "
                f"vpn={bootstrap.vpn_status_after.get('message') if bootstrap.vpn_status_after else 'none'} "
                f"mongo={bootstrap.mongo_message}"
            ),
        )
        if not bootstrap.ready:
            raise RuntimeError(f"runtime bootstrap failed: {bootstrap.error_type}: {bootstrap.error}")

    best: QualityMetrics | None = None
    best_iteration = 0
    plateau_count = 0
    run_rows: list[dict[str, Any]] = []

    for iteration in range(1, max(1, args.max_runs) + 1):
        config = AppConfig.from_env(Path.cwd(), prefer_process_env=True)
        iteration_started = datetime.now(timezone.utc)
        run_id = ""
        run_exception = ""
        try:
            run_id = asyncio.run(_run_once(args.intent))
        except BaseException as exc:  # noqa: BLE001
            run_exception = f"{type(exc).__name__}: {exc}"
            _append_line(status_log_path, f"[{_utc_now()}] iteration={iteration} run raised {run_exception}")
            if _is_mongo_connectivity_error(exc) and not args.skip_runtime_up:
                _attempt_runtime_recovery(
                    status_log_path=status_log_path,
                    reason=f"run_exception:{run_exception}",
                )
            try:
                run_id = _latest_run_id_since(config, iteration_started)
            except BaseException as latest_exc:  # noqa: BLE001
                _append_line(
                    status_log_path,
                    f"[{_utc_now()}] iteration={iteration} latest_run_id lookup failed: {type(latest_exc).__name__}: {latest_exc}",
                )
                if _is_mongo_connectivity_error(latest_exc) and not args.skip_runtime_up:
                    _attempt_runtime_recovery(
                        status_log_path=status_log_path,
                        reason=f"latest_run_id_lookup:{type(latest_exc).__name__}: {latest_exc}",
                    )
                run_id = ""
            if not run_id:
                plateau_count += 1
                _append_line(
                    status_log_path,
                    (
                        f"[{_utc_now()}] iteration={iteration} no run_id recovered; "
                        f"plateau_count={plateau_count}/{args.plateau_patience}"
                    ),
                )
                if plateau_count >= max(1, args.plateau_patience):
                    _append_line(status_log_path, f"[{_utc_now()}] plateau reached after run failures, stopping loop")
                    break
                continue

        metrics: QualityMetrics | None = None
        run_state: dict[str, Any] = {}
        metric_error = ""
        for attempt in (1, 2):
            try:
                metrics = _collect_metrics(config, run_id)
                run_state = _run_status(config, run_id)
                metric_error = ""
                break
            except BaseException as metrics_exc:  # noqa: BLE001
                metric_error = f"{type(metrics_exc).__name__}: {metrics_exc}"
                _append_line(
                    status_log_path,
                    (
                        f"[{_utc_now()}] iteration={iteration} metrics/status read failed "
                        f"(attempt={attempt}/2): {metric_error}"
                    ),
                )
                if attempt == 1 and _is_mongo_connectivity_error(metrics_exc) and not args.skip_runtime_up:
                    _attempt_runtime_recovery(
                        status_log_path=status_log_path,
                        reason=f"metrics_read:{metric_error}",
                    )
                    continue
                break

        if metrics is None:
            plateau_count += 1
            _append_line(
                status_log_path,
                (
                    f"[{_utc_now()}] iteration={iteration} metrics unavailable after retries; "
                    f"plateau_count={plateau_count}/{args.plateau_patience}"
                ),
            )
            if plateau_count >= max(1, args.plateau_patience):
                _append_line(status_log_path, f"[{_utc_now()}] plateau reached after metrics failures, stopping loop")
                break
            continue

        row = {
            "iteration": iteration,
            "run_id": metrics.run_id,
            "run_status": run_state.get("status"),
            "run_error": run_state.get("error") or run_exception or metric_error,
            "evaluated_count": metrics.evaluated_count,
            "useful_count": metrics.useful_count,
            "strong_count": metrics.strong_count,
            "strong_clear_path_count": metrics.strong_clear_path_count,
            "weak_useful_count": metrics.weak_useful_count,
            "noise_useful_count": metrics.noise_useful_count,
            "clear_path_ratio": round(metrics.clear_path_ratio, 4),
        }
        run_rows.append(row)
        _append_line(status_log_path, f"[{_utc_now()}] run={json.dumps(row, sort_keys=True)}")

        if _is_better(metrics, best):
            best = metrics
            best_iteration = iteration
            plateau_count = 0
            _append_line(status_log_path, f"[{_utc_now()}] improvement detected at iteration={iteration}")
        else:
            plateau_count += 1
            _append_line(
                status_log_path,
                f"[{_utc_now()}] no improvement plateau_count={plateau_count}/{args.plateau_patience}",
            )

        if plateau_count >= max(1, args.plateau_patience):
            _append_line(status_log_path, f"[{_utc_now()}] plateau reached, stopping loop")
            break

    final_payload = {
        "intent": args.intent,
        "max_runs": args.max_runs,
        "plateau_patience": args.plateau_patience,
        "runs": run_rows,
        "best_iteration": best_iteration,
        "best": {
            "run_id": best.run_id if best else "",
            "strong_clear_path_count": best.strong_clear_path_count if best else 0,
            "strong_count": best.strong_count if best else 0,
            "useful_count": best.useful_count if best else 0,
            "noise_useful_count": best.noise_useful_count if best else 0,
            "clear_path_ratio": round(best.clear_path_ratio, 4) if best else 0.0,
        },
        "finished_at_utc": _utc_now(),
    }
    results_path.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
    _append_line(status_log_path, f"[{_utc_now()}] quality-loop finished best_run={final_payload['best']['run_id']}")


if __name__ == "__main__":
    main()
