from __future__ import annotations

import os
import re
import signal
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from pydantic import BaseModel, Field

from internet_explorer.config import AppConfig
from internet_explorer.config import _candidate_ovpn_paths
from internet_explorer.config import _derive_docdb_endpoint_from_mongodb_uri
from internet_explorer.config import _resolve_ovpn_config_path
from internet_explorer.persistence import _strip_tls_ca_file_from_uri
from internet_explorer.persistence import ping_mongo
from internet_explorer.vpn import GenericVpnManager

BROWSER_TMP_PREFIXES = (
    "browser-use-user-data-dir-",
    "browser-use-downloads-",
    "browser_use_agent_",
)
BROWSER_PROCESS_MARKERS = (
    "--user-data-dir=/tmp/browser-use-user-data-dir-",
    "--user-data-dir=/tmp/browser-use-downloads-",
    "/tmp/browser-use-user-data-dir-",
    "/tmp/browser-use-downloads-",
    "/tmp/browser_use_agent_",
)


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
    browser_process_cleanup: dict[str, Any] = Field(default_factory=dict)
    tmp_cleanup: dict[str, Any] = Field(default_factory=dict)
    error_type: str = ""
    error: str = ""


def _normalize_env_values(raw_values: dict[str, object] | None) -> dict[str, str]:
    if not raw_values:
        return {}

    normalized: dict[str, str] = {}
    for key, value in raw_values.items():
        normalized[str(key)] = str(value or "")
    return normalized


def _repo_relative_path(root_dir: Path, path: Path) -> str:
    resolved_root = root_dir.resolve()
    resolved_path = path.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(resolved_path)


def _resolve_env_path(root_dir: Path, raw_path: str) -> Path | None:
    cleaned = (raw_path or "").strip()
    if not cleaned:
        return None

    path = Path(cleaned).expanduser()
    if not path.is_absolute():
        path = root_dir / path
    return path.resolve()


def _is_ie_worker_running() -> bool:
    try:
        result = subprocess.run(  # noqa: S603
            ["pgrep", "-f", "^python -m internet_explorer.cli"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return False
    return bool((result.stdout or "").strip())


def _safe_dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for file_name in files:
            file_path = Path(root) / file_name
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_stale_browser_processes(*, skip_if_ie_worker_running: bool = True) -> dict[str, Any]:
    worker_running = _is_ie_worker_running() if skip_if_ie_worker_running else False
    result: dict[str, Any] = {
        "worker_running": worker_running,
        "matched_count": 0,
        "term_sent_count": 0,
        "kill_sent_count": 0,
        "already_gone_count": 0,
        "failed_count": 0,
        "skipped_reason": "",
        "error": "",
    }
    if worker_running:
        result["skipped_reason"] = "active_worker_detected"
        return result

    try:
        ps_output = subprocess.run(  # noqa: S603
            ["ps", "-eo", "pid,args"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        ).stdout or ""
    except Exception as exc:  # pragma: no cover - defensive branch
        result["skipped_reason"] = "ps_failed"
        result["error"] = str(exc)
        return result

    current_pid = os.getpid()
    candidate_pids: list[int] = []
    for line in ps_output.splitlines()[1:]:
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
        if pid <= 1 or pid == current_pid:
            continue
        lowered = command.lower()
        if "chrome" not in lowered and "chromium" not in lowered:
            continue
        if not any(marker in lowered for marker in BROWSER_PROCESS_MARKERS):
            continue
        candidate_pids.append(pid)

    result["matched_count"] = len(candidate_pids)

    for pid in candidate_pids:
        try:
            os.kill(pid, signal.SIGTERM)
            result["term_sent_count"] += 1
        except ProcessLookupError:
            result["already_gone_count"] += 1
        except Exception:
            result["failed_count"] += 1

    if candidate_pids:
        time.sleep(0.2)

    for pid in candidate_pids:
        if not _pid_exists(pid):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            result["kill_sent_count"] += 1
        except ProcessLookupError:
            result["already_gone_count"] += 1
        except Exception:
            result["failed_count"] += 1

    return result


def cleanup_stale_browser_tmp_dirs(
    *,
    tmp_root: Path = Path("/tmp"),
    older_than_minutes: int = 30,
    skip_if_ie_worker_running: bool = True,
) -> dict[str, Any]:
    normalized_minutes = max(0, int(older_than_minutes))
    worker_running = _is_ie_worker_running() if skip_if_ie_worker_running else False
    result: dict[str, Any] = {
        "tmp_root": str(tmp_root),
        "older_than_minutes": normalized_minutes,
        "worker_running": worker_running,
        "matched_count": 0,
        "removed_count": 0,
        "removed_bytes": 0,
        "failed_count": 0,
        "skipped_recent_count": 0,
        "skipped_reason": "",
    }
    if worker_running:
        result["skipped_reason"] = "active_worker_detected"
        return result
    if not tmp_root.exists():
        result["skipped_reason"] = "tmp_root_missing"
        return result

    cutoff_epoch = time.time() - (normalized_minutes * 60)
    for entry in tmp_root.iterdir():
        if not entry.is_dir():
            continue
        if not entry.name.startswith(BROWSER_TMP_PREFIXES):
            continue
        result["matched_count"] += 1
        try:
            mtime = entry.stat().st_mtime
        except OSError:
            result["failed_count"] += 1
            continue
        if mtime >= cutoff_epoch:
            result["skipped_recent_count"] += 1
            continue
        try:
            result["removed_bytes"] += _safe_dir_size_bytes(entry)
            shutil.rmtree(entry)
            result["removed_count"] += 1
        except Exception:
            result["failed_count"] += 1
    return result


def build_runtime_env_updates(root_dir: Path, env_values: dict[str, str], explicit_ovpn: str = "") -> dict[str, str]:
    updates: dict[str, str] = {}
    current_mongo_uri = str(env_values.get("MONGODB_URI") or "").strip()
    cleaned_mongo_uri = _strip_tls_ca_file_from_uri(current_mongo_uri)
    if cleaned_mongo_uri and cleaned_mongo_uri != current_mongo_uri:
        updates["MONGODB_URI"] = cleaned_mongo_uri

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

    repo_tls_ca = (root_dir / "certs/global-bundle.pem").resolve()
    current_tls_ca = str(env_values.get("MONGODB_TLS_CA_FILE") or "").strip()
    resolved_tls_ca = _resolve_env_path(root_dir, current_tls_ca) if current_tls_ca else None
    if repo_tls_ca.exists() and (not current_tls_ca or resolved_tls_ca is None or not resolved_tls_ca.exists()):
        updates["MONGODB_TLS_CA_FILE"] = _repo_relative_path(root_dir, repo_tls_ca)

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
    env_values = _normalize_env_values(dotenv_values(env_path) if env_path.exists() else {})
    env_updates = build_runtime_env_updates(root_dir, env_values, explicit_ovpn=explicit_ovpn)
    env_written = apply_env_updates(env_path, env_updates) if write_env else False
    resolved_env_values = dict(env_values)
    resolved_env_values.update(env_updates)

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
    result.browser_process_cleanup = cleanup_stale_browser_processes()
    result.tmp_cleanup = cleanup_stale_browser_tmp_dirs()

    try:
        config = AppConfig.from_env(
            root_dir,
            env_overrides=resolved_env_values,
            prefer_process_env=False,
        )
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
