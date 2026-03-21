from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
from pathlib import Path

from internet_explorer.config import AppConfig
from internet_explorer.service import IntentDiscoveryService
from internet_explorer.vpn import GenericVpnManager


def _find_existing_workers() -> list[int]:
    this_pid = os.getpid()
    try:
        output = subprocess.run(  # noqa: S603
            ["ps", "-eo", "pid,args"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout or ""
    except Exception:
        return []

    existing: list[int] = []
    for line in output.splitlines()[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            continue
        pid_text, command = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == this_pid:
            continue
        lowered = command.lower()
        if "internet_explorer.cli" not in lowered:
            continue
        if "python" not in lowered:
            continue
        existing.append(pid)
    return existing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Intent-driven datasource discovery service.")
    parser.add_argument("--intent", help="Intent to discover new datasources for.")
    parser.add_argument("--print-config", action="store_true", help="Print resolved config and exit.")
    parser.add_argument("--vpn-start", action="store_true", help="Start the generic OpenVPN tunnel and exit.")
    parser.add_argument("--vpn-stop", action="store_true", help="Stop the generic OpenVPN tunnel and exit.")
    parser.add_argument("--vpn-status", action="store_true", help="Print generic OpenVPN status and exit.")
    parser.add_argument("--vpn-check-docdb", action="store_true", help="With --vpn-status, also test DocDB TCP reachability.")
    return parser


async def _run(intent: str, print_config: bool, vpn_start: bool, vpn_stop: bool, vpn_status: bool, vpn_check_docdb: bool) -> int:
    # CLI runtime flags and inline env vars should be able to override `.env`.
    config = AppConfig.from_env(Path.cwd(), prefer_process_env=True)
    if intent:
        config.intent = intent
    if print_config:
        print(json.dumps(config.model_dump(mode="json"), indent=2))
        return 0
    vpn_flags = [vpn_start, vpn_stop, vpn_status]
    if sum(1 for flag in vpn_flags if flag) > 1:
        raise SystemExit("Use only one of --vpn-start, --vpn-stop, or --vpn-status at a time.")
    if vpn_start or vpn_stop or vpn_status:
        manager = GenericVpnManager(config)
        if vpn_start:
            status = await asyncio.to_thread(manager.start)
        elif vpn_stop:
            status = await asyncio.to_thread(manager.stop)
        else:
            status = await asyncio.to_thread(manager.status, check_docdb=vpn_check_docdb)
        print(json.dumps(status.model_dump(mode="json"), indent=2))
        return 0
    if not config.allow_parallel_workers:
        existing_workers = _find_existing_workers()
        if existing_workers:
            worker_list = ", ".join(str(pid) for pid in existing_workers[:10])
            raise SystemExit(
                "Another internet-explorer worker is already running "
                f"(pid(s): {worker_list}). "
                "Stop existing workers or set ALLOW_PARALLEL_WORKERS=true to override."
            )
    if not config.intent:
        raise SystemExit("Intent is required. Pass --intent or set INTENT in env.")
    if config.auto_start_vpn:
        manager = GenericVpnManager(config)
        status = await asyncio.to_thread(manager.ensure_started)
        print(json.dumps({"auto_start_vpn": True, "vpn_status": status.model_dump(mode="json")}, indent=2))
    service = IntentDiscoveryService(config)
    summary = await service.run(config.intent)
    print(json.dumps(summary.model_dump(mode="json"), indent=2))
    return 0


def main() -> None:
    def _handle_sigterm(_signum, _frame):  # noqa: ANN001
        raise KeyboardInterrupt("SIGTERM received")

    signal.signal(signal.SIGTERM, _handle_sigterm)
    parser = build_parser()
    args = parser.parse_args()
    try:
        exit_code = asyncio.run(
            _run(
                intent=args.intent or "",
                print_config=args.print_config,
                vpn_start=args.vpn_start,
                vpn_stop=args.vpn_stop,
                vpn_status=args.vpn_status,
                vpn_check_docdb=args.vpn_check_docdb,
            )
        )
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
