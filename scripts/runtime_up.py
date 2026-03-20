#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from internet_explorer.runtime_bootstrap import run_runtime_bootstrap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare local VPN + Mongo runtime for internet-explorer.")
    parser.add_argument("--ovpn-config", default="", help="Explicit OVPN config path to use and persist into .env.")
    parser.add_argument("--no-write-env", action="store_true", help="Do not write resolved defaults back into .env.")
    parser.add_argument("--skip-vpn", action="store_true", help="Skip VPN startup and only verify environment + Mongo.")
    parser.add_argument("--skip-mongo", action="store_true", help="Skip Mongo ping after VPN setup.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    result = run_runtime_bootstrap(
        ROOT_DIR,
        explicit_ovpn=args.ovpn_config,
        write_env=not args.no_write_env,
        start_vpn=not args.skip_vpn,
        verify_mongo=not args.skip_mongo,
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0 if result.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
