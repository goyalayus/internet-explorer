from pathlib import Path

from internet_explorer.config import _derive_docdb_endpoint_from_mongodb_uri
from internet_explorer.runtime_bootstrap import apply_env_updates
from internet_explorer.runtime_bootstrap import build_runtime_env_updates
from internet_explorer.runtime_bootstrap import run_runtime_bootstrap


def test_derive_docdb_host_from_mongodb_uri() -> None:
    uri = (
        "mongodb://user:pass@docdb-recepto-staging.cluster.example.amazonaws.com:27017/"
        "?directConnection=true&tls=true"
    )

    assert (
        _derive_docdb_endpoint_from_mongodb_uri(uri)[0]
        == "docdb-recepto-staging.cluster.example.amazonaws.com"
    )


def test_build_runtime_env_updates_prefers_local_query_optimizer_ovpn(tmp_path: Path) -> None:
    root_dir = tmp_path / "internet-explorer"
    root_dir.mkdir()
    query_optimizer_repo = tmp_path / "query_optimizer_repo"
    query_optimizer_repo.mkdir()
    ovpn = query_optimizer_repo / "client-config-staging.ovpn"
    ovpn.write_text("client\n")

    updates = build_runtime_env_updates(
        root_dir,
        {
            "MONGODB_URI": "mongodb://user:pass@docdb.example.internal:27017/?tls=true",
            "QUERY_OPTIMIZER_OVPN_CONFIG": "/home/ubuntu/code/query_optimizer_repo/client-config-staging.ovpn",
            "VPN_DOCDB_HOST": "",
        },
    )

    assert updates == {
        "VPN_OVPN_CONFIG": str(ovpn.resolve()),
        "VPN_DOCDB_HOST": "docdb.example.internal",
        "VPN_DOCDB_PORT": "27017",
    }


def test_build_runtime_env_updates_sets_repo_local_tls_ca_when_missing(tmp_path: Path) -> None:
    root_dir = tmp_path / "internet-explorer"
    root_dir.mkdir()
    certs_dir = root_dir / "certs"
    certs_dir.mkdir()
    (certs_dir / "global-bundle.pem").write_text("bundle")

    updates = build_runtime_env_updates(
        root_dir,
        {
            "MONGODB_URI": "mongodb://user:pass@docdb.example.internal:27017/?tls=true",
            "MONGODB_TLS_CA_FILE": "",
        },
    )

    assert updates["MONGODB_TLS_CA_FILE"] == "certs/global-bundle.pem"


def test_apply_env_updates_rewrites_existing_keys_and_appends_missing(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "MONGODB_URI=mongodb://localhost:27017\n"
        "QUERY_OPTIMIZER_OVPN_CONFIG=/stale/path.ovpn\n"
        "VPN_DOCDB_HOST=\n"
    )

    changed = apply_env_updates(
        env_path,
        {
            "VPN_OVPN_CONFIG": "/new/path/client-config-staging.ovpn",
            "VPN_DOCDB_HOST": "docdb.example.internal",
        },
    )

    assert changed is True
    text = env_path.read_text()
    assert "VPN_OVPN_CONFIG=/new/path/client-config-staging.ovpn" in text
    assert "VPN_DOCDB_HOST=docdb.example.internal" in text
    assert "QUERY_OPTIMIZER_OVPN_CONFIG=/stale/path.ovpn" in text


def test_run_runtime_bootstrap_writes_env_and_uses_existing_modules(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "internet-explorer"
    root_dir.mkdir()
    (root_dir / ".env").write_text(
        "MONGODB_URI=mongodb://user:pass@docdb.example.internal:27017/?tls=true\n"
        "QUERY_OPTIMIZER_OVPN_CONFIG=/home/ubuntu/code/query_optimizer_repo/client-config-staging.ovpn\n"
        "AUTO_START_VPN=true\n"
    )
    (root_dir / "scripts").mkdir()
    (root_dir / "scripts" / "vpn_start.sh").write_text("#!/usr/bin/env bash\n")
    (root_dir / "data").mkdir()
    (root_dir / "certs").mkdir()
    (root_dir / "certs" / "global-bundle.pem").write_text("bundle")
    (root_dir / "data" / "tool_flow_baseline_domains.txt").write_text("")
    (root_dir / "data" / "known_tools.txt").write_text("rapidapi\n")

    query_optimizer_repo = tmp_path / "query_optimizer_repo"
    query_optimizer_repo.mkdir()
    ovpn = query_optimizer_repo / "client-config-staging.ovpn"
    ovpn.write_text("client\n")

    class _FakeManager:
        def __init__(self, config) -> None:  # noqa: ANN001
            self.config = config

        def status(self, *, check_docdb: bool = False):  # noqa: ARG002
            from internet_explorer.vpn import VpnStatus

            return VpnStatus(
                running=False,
                pid=None,
                ovpn_config=str(self.config.vpn_ovpn_config) if self.config.vpn_ovpn_config else None,
                pid_file=str(self.config.vpn_log_dir / "openvpn.pid"),
                log_file=str(self.config.vpn_log_dir / "openvpn.log"),
                docdb_host=self.config.vpn_docdb_host,
                docdb_port=self.config.vpn_docdb_port,
                message="stopped",
            )

        def ensure_started(self):
            from internet_explorer.vpn import VpnStatus

            return VpnStatus(
                running=True,
                pid=123,
                started_by_this_call=True,
                ovpn_config=str(self.config.vpn_ovpn_config) if self.config.vpn_ovpn_config else None,
                pid_file=str(self.config.vpn_log_dir / "openvpn.pid"),
                log_file=str(self.config.vpn_log_dir / "openvpn.log"),
                docdb_host=self.config.vpn_docdb_host,
                docdb_port=self.config.vpn_docdb_port,
                docdb_reachable=True,
                split_tunnel_ok=True,
                message="started",
            )

    monkeypatch.setattr("internet_explorer.runtime_bootstrap.GenericVpnManager", _FakeManager)
    monkeypatch.setattr("internet_explorer.runtime_bootstrap.ping_mongo", lambda config: "ping_ok")

    result = run_runtime_bootstrap(root_dir)

    assert result.ready is True
    assert result.env_written is True
    assert result.resolved_ovpn_config == str(ovpn.resolve())
    assert result.resolved_docdb_host == "docdb.example.internal"
    assert result.mongo_reachable is True
    env_text = (root_dir / ".env").read_text()
    assert f"VPN_OVPN_CONFIG={ovpn.resolve()}" in env_text
    assert "VPN_DOCDB_HOST=docdb.example.internal" in env_text
    assert "MONGODB_TLS_CA_FILE=certs/global-bundle.pem" in env_text


def test_run_runtime_bootstrap_uses_in_memory_updates_when_env_write_is_disabled(monkeypatch, tmp_path: Path) -> None:
    root_dir = tmp_path / "internet-explorer"
    root_dir.mkdir()
    (root_dir / ".env").write_text(
        "MONGODB_URI=mongodb://user:pass@docdb.example.internal:27017/?tls=true\n"
        "QUERY_OPTIMIZER_OVPN_CONFIG=/home/ubuntu/code/query_optimizer_repo/client-config-staging.ovpn\n"
    )
    (root_dir / "scripts").mkdir()
    (root_dir / "scripts" / "vpn_start.sh").write_text("#!/usr/bin/env bash\n")
    (root_dir / "data").mkdir()
    (root_dir / "certs").mkdir()
    (root_dir / "certs" / "global-bundle.pem").write_text("bundle")
    (root_dir / "data" / "tool_flow_baseline_domains.txt").write_text("")
    (root_dir / "data" / "known_tools.txt").write_text("rapidapi\n")

    query_optimizer_repo = tmp_path / "query_optimizer_repo"
    query_optimizer_repo.mkdir()
    ovpn = query_optimizer_repo / "client-config-staging.ovpn"
    ovpn.write_text("client\n")

    class _FakeManager:
        def __init__(self, config) -> None:  # noqa: ANN001
            self.config = config

        def status(self, *, check_docdb: bool = False):  # noqa: ARG002
            from internet_explorer.vpn import VpnStatus

            return VpnStatus(
                running=False,
                pid=None,
                ovpn_config=str(self.config.vpn_ovpn_config) if self.config.vpn_ovpn_config else None,
                pid_file=str(self.config.vpn_log_dir / "openvpn.pid"),
                log_file=str(self.config.vpn_log_dir / "openvpn.log"),
                docdb_host=self.config.vpn_docdb_host,
                docdb_port=self.config.vpn_docdb_port,
                message="stopped",
            )

        def ensure_started(self):
            from internet_explorer.vpn import VpnStatus

            return VpnStatus(
                running=True,
                pid=123,
                started_by_this_call=True,
                ovpn_config=str(self.config.vpn_ovpn_config) if self.config.vpn_ovpn_config else None,
                pid_file=str(self.config.vpn_log_dir / "openvpn.pid"),
                log_file=str(self.config.vpn_log_dir / "openvpn.log"),
                docdb_host=self.config.vpn_docdb_host,
                docdb_port=self.config.vpn_docdb_port,
                docdb_reachable=True,
                split_tunnel_ok=True,
                message="started",
            )

    monkeypatch.setattr("internet_explorer.runtime_bootstrap.GenericVpnManager", _FakeManager)
    monkeypatch.setattr("internet_explorer.runtime_bootstrap.ping_mongo", lambda config: "ping_ok")
    monkeypatch.setenv("QUERY_OPTIMIZER_OVPN_CONFIG", "/bad/process/path.ovpn")
    monkeypatch.setenv("MONGODB_URI", "mongodb://bad-process-value:27017")

    result = run_runtime_bootstrap(root_dir, write_env=False)

    assert result.ready is True
    assert result.env_written is False
    assert result.resolved_ovpn_config == str(ovpn.resolve())
    assert result.mongo_reachable is True
    env_text = (root_dir / ".env").read_text()
    assert "VPN_OVPN_CONFIG=" not in env_text
    assert "MONGODB_TLS_CA_FILE=" not in env_text
