from pathlib import Path

from internet_explorer.config import _parse_shell_default


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
