from pathlib import Path

from internet_explorer.config import AppConfig
from internet_explorer.config import _parse_shell_default
from internet_explorer.vpn import GenericVpnManager, VpnStatus


def test_parse_shell_default_from_reference_script(tmp_path: Path) -> None:
    script = tmp_path / "vpn.sh"
    script.write_text(
        'DOCDB_HOST="${DOCDB_HOST:-docdb.example.internal}"\n'
        'DOCDB_PORT="${DOCDB_PORT:-27017}"\n'
        'REQUIRE_SPLIT_TUNNEL="${REQUIRE_SPLIT_TUNNEL:-true}"\n'
    )
    assert _parse_shell_default(script, "DOCDB_HOST") == "docdb.example.internal"
    assert _parse_shell_default(script, "DOCDB_PORT") == "27017"
    assert _parse_shell_default(script, "REQUIRE_SPLIT_TUNNEL") == "true"


def test_vpn_manager_uses_local_start_script_when_configured(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")
    known_tools = tmp_path / "known_tools.txt"
    known_tools.write_text("rapidapi\n")
    start_script = tmp_path / "vpn_start.sh"
    start_script.write_text("#!/usr/bin/env bash\n")
    ovpn = tmp_path / "client.ovpn"
    ovpn.write_text("client\n")

    config = AppConfig(
        workspace_root=tmp_path,
        mongodb_uri="mongodb://localhost:27017",
        baseline_domains_file=baseline,
        known_tools_file=known_tools,
        vpn_log_dir=tmp_path / ".vpn_logs",
        env_file_path=env_path,
        vpn_start_script=start_script,
        vpn_ovpn_config=ovpn,
    )
    manager = GenericVpnManager(config)

    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(manager, "_ensure_start_prerequisites", lambda: None)
    monkeypatch.setattr(manager, "_remove_stale_pid_file", lambda: None)
    monkeypatch.setattr(manager, "_wait_for_pid_and_tunnel", lambda: None)
    monkeypatch.setattr(manager, "_default_route", lambda: "default via 1.1.1.1 dev eth0")
    monkeypatch.setattr(manager, "status", lambda check_docdb=False: VpnStatus(running=True, pid=999, pid_file=str(manager.pid_file), log_file=str(manager.log_file), message="running"))

    import subprocess

    def _fake_run(command, check, capture_output, text):  # noqa: ANN001
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    manager.start()
    command = captured.get("command", [])
    assert command[:2] == ["bash", str(start_script)]
