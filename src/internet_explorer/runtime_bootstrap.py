from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel, Field

from internet_explorer.config import AppConfig
from internet_explorer.config import _candidate_ovpn_paths
from internet_explorer.config import _derive_docdb_endpoint_from_mongodb_uri
from internet_explorer.config import _resolve_ovpn_config_path
from internet_explorer.persistence import ping_mongo
from internet_explorer.vpn import GenericVpnManager


class RuntimeBootstrapResult(BaseModel):
    ready: bool = False
    env_file: str
    env_updates: dict[str, str] = Field(default_factory=dict)
    env_written: bool = False
    attempted_ovpn_paths: list[str] = Field(default_factory=list)
    resolved_ovpn_config: str | None = None
    resolved_docdb_host: str = ""
    resolved_docdb_port: int = 27017
    vpn_status_before: dict[str, Any] | None = None
    vpn_status_after: dict[str, Any] | None = None
    mongo_reachable: bool | None = None
    mongo_message: str = ""
    error_type: str = ""
    error: str = ""

def build_runtime_env_updates(root_dir: Path, env_values: dict[str, str], explicit_ovpn: str = "") -> dict[str, str]:
    updates: dict[str, str] = {}

    resolved_ovpn = _resolve_ovpn_config_path(
        root_dir,
        configured_path=explicit_ovpn.strip() or str(env_values.get("VPN_OVPN_CONFIG") or "").strip(),
        legacy_path=str(env_values.get("QUERY_OPTIMIZER_OVPN_CONFIG") or "").strip(),
    )
    current_vpn_ovpn = str(env_values.get("VPN_OVPN_CONFIG") or "").strip()
    if resolved_ovpn and current_vpn_ovpn != str(resolved_ovpn):
        updates["VPN_OVPN_CONFIG"] = str(resolved_ovpn)

    current_docdb_host = str(env_values.get("VPN_DOCDB_HOST") or "").strip()
    current_docdb_port = str(env_values.get("VPN_DOCDB_PORT") or "").strip()
    derived_host, derived_port = _derive_docdb_endpoint_from_mongodb_uri(str(env_values.get("MONGODB_URI") or ""))
    if not current_docdb_host:
        if derived_host:
            updates["VPN_DOCDB_HOST"] = derived_host
    if not current_docdb_port and derived_port:
        updates["VPN_DOCDB_PORT"] = str(derived_port)

    return updates


def apply_env_updates(env_path: Path, updates: dict[str, str]) -> bool:
    if not updates:
        return False

    original_text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    lines = original_text.splitlines()
    remaining = dict(updates)
    rewritten: list[str] = []

    for line in lines:
        replaced = False
        for key in list(remaining):
            if re.match(rf"^\s*{re.escape(key)}\s*=", line):
                rewritten.append(f"{key}={remaining.pop(key)}")
                replaced = True
                break
        if replaced:
            continue
        rewritten.append(line)

    if remaining:
        if rewritten and rewritten[-1].strip():
            rewritten.append("")
        for key, value in remaining.items():
            rewritten.append(f"{key}={value}")

    new_text = "\n".join(rewritten).rstrip() + "\n"
    if new_text == original_text:
        return False

    env_path.write_text(new_text, encoding="utf-8")
    return True

def run_runtime_bootstrap(
    root_dir: Path,
    *,
    explicit_ovpn: str = "",
    write_env: bool = True,
    start_vpn: bool = True,
    verify_mongo: bool = True,
) -> RuntimeBootstrapResult:
    env_path = (root_dir / ".env").resolve()
    env_values = {
        str(key): str(value or "")
        for key, value in dotenv_values(env_path).items()
        if key is not None
    }
    env_updates = build_runtime_env_updates(root_dir, env_values, explicit_ovpn=explicit_ovpn)
    env_written = apply_env_updates(env_path, env_updates) if write_env else False

    result = RuntimeBootstrapResult(
        env_file=str(env_path),
        env_updates=env_updates,
        env_written=env_written,
        attempted_ovpn_paths=[
            str(path)
            for path in _candidate_ovpn_paths(
                root_dir,
                configured_path=explicit_ovpn.strip() or str(env_values.get("VPN_OVPN_CONFIG") or "").strip(),
                legacy_path=str(env_values.get("QUERY_OPTIMIZER_OVPN_CONFIG") or "").strip(),
            )
        ],
    )

    try:
        config = AppConfig.from_env(root_dir)
        result.resolved_ovpn_config = str(config.vpn_ovpn_config) if config.vpn_ovpn_config else None
        result.resolved_docdb_host = config.vpn_docdb_host
        result.resolved_docdb_port = config.vpn_docdb_port

        manager = GenericVpnManager(config)
        result.vpn_status_before = manager.status(check_docdb=False).model_dump(mode="json")

        if start_vpn:
            if config.vpn_ovpn_config is None:
                raise RuntimeError(
                    "No local OVPN config could be resolved. "
                    f"Tried: {', '.join(result.attempted_ovpn_paths) or '<none>'}"
                )
            result.vpn_status_after = manager.ensure_started().model_dump(mode="json")
        else:
            result.vpn_status_after = result.vpn_status_before

        if verify_mongo:
            ping_mongo(config)
            result.mongo_message = "ping_ok"
            result.mongo_reachable = True
        else:
            result.mongo_message = "mongo_check_skipped"
            result.mongo_reachable = None

        result.ready = True
        return result
    except Exception as exc:
        result.error_type = type(exc).__name__
        result.error = str(exc)
        if result.mongo_reachable is None and verify_mongo and not result.mongo_message:
            result.mongo_message = "mongo_check_not_reached"
        return result
