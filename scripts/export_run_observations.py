#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from internet_explorer.config import AppConfig
from internet_explorer.observation_report import build_run_observation_report
from internet_explorer.persistence import MongoPersistence


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a readable agent-path observation report for one run.")
    parser.add_argument("--run-id", required=True, help="Run id from Mongo.")
    parser.add_argument(
        "--output",
        default="",
        help="Output text path. Defaults to output/<run_id>_agent_path_observations.txt",
    )
    args = parser.parse_args()

    config = AppConfig.from_env(Path.cwd())
    persistence = MongoPersistence(config)
    try:
        db = persistence.db

        run_doc = db[config.mongodb_runs_collection].find_one({"run_id": args.run_id})
        if run_doc is None:
            raise SystemExit(f"Run not found: {args.run_id}")

        url_summaries = list(
            db[config.mongodb_url_summaries_collection]
            .find({"run_id": args.run_id})
            .sort([("useful", -1), ("domain", 1)])
        )
        events = list(db[config.mongodb_events_collection].find({"run_id": args.run_id}).sort("step_no", 1))

        report = build_run_observation_report(
            run_doc=run_doc,
            url_summaries=url_summaries,
            events=events,
        )
    finally:
        persistence.client.close()

    output_path = Path(args.output).expanduser().resolve() if args.output else (
        Path.cwd() / "output" / f"{args.run_id}_agent_path_observations.txt"
    ).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
