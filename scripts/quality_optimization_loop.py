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
        persistence.close()

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
        run_id = asyncio.run(_run_once(args.intent))
        config = AppConfig.from_env(Path.cwd(), prefer_process_env=True)
        metrics = _collect_metrics(config, run_id)

        row = {
            "iteration": iteration,
            "run_id": metrics.run_id,
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
