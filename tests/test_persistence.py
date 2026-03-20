from pathlib import Path

from internet_explorer.config import AppConfig
from internet_explorer.persistence import _mongo_client_settings
from internet_explorer.persistence import _sanitize_bson
from internet_explorer.persistence import _strip_tls_ca_file_from_uri


def test_sanitize_bson_converts_out_of_range_integers() -> None:
    payload = {
        "ok": 42,
        "max_ok": 2**63 - 1,
        "min_ok": -(2**63),
        "too_big": 2**63,
        "too_small": -(2**63) - 1,
        "nested": [{"value": 10**40}],
        "bool_flag": True,
    }

    sanitized = _sanitize_bson(payload)

    assert sanitized["ok"] == 42
    assert sanitized["max_ok"] == 2**63 - 1
    assert sanitized["min_ok"] == -(2**63)
    assert sanitized["too_big"] == str(2**63)
    assert sanitized["too_small"] == str(-(2**63) - 1)
    assert sanitized["nested"][0]["value"] == str(10**40)
    assert sanitized["bool_flag"] is True


def test_strip_tls_ca_file_from_uri_removes_embedded_path() -> None:
    uri = (
        "mongodb://user:pass@example.com:27017/"
        "?directConnection=true&tls=true&tlsCAFile=/tmp/old.pem&serverSelectionTimeoutMS=20000"
    )

    cleaned = _strip_tls_ca_file_from_uri(uri)

    assert "tlsCAFile=" not in cleaned
    assert "directConnection=true" in cleaned
    assert "serverSelectionTimeoutMS=20000" in cleaned


def test_mongo_client_settings_prefers_explicit_tls_ca_file(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("")
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")
    known_tools = tmp_path / "known_tools.txt"
    known_tools.write_text("rapidapi\n")
    ca_file = tmp_path / "global-bundle.pem"
    ca_file.write_text("bundle")

    config = AppConfig(
        workspace_root=tmp_path,
        mongodb_uri=(
            "mongodb://localhost:27017/"
            "?directConnection=true&tls=true&tlsCAFile=/tmp/stale.pem"
        ),
        mongodb_tls_ca_file=ca_file,
        baseline_domains_file=baseline,
        known_tools_file=known_tools,
        vpn_log_dir=tmp_path / ".vpn_logs",
        env_file_path=env_path,
    )

    uri, client_kwargs = _mongo_client_settings(config)

    assert "tlsCAFile=" not in uri
    assert client_kwargs == {"tlsCAFile": str(ca_file)}
