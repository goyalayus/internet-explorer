from pathlib import Path

from internet_explorer.tool_inventory import ToolInventory


def test_tool_inventory_loads_from_local_static_file(tmp_path: Path) -> None:
    known_tools = tmp_path / "known_tools.txt"
    known_tools.write_text(
        "# comment\n"
        "coresignal\n"
        "builtwith\n"
        "rapidapi\n"
        "linkedin_api\n"
    )

    inventory = ToolInventory.from_file(known_tools)

    assert "coresignal" in inventory.tool_names
    assert "builtwith" in inventory.tool_names
    assert "rapidapi" in inventory.tool_names
    assert "linkedin_api" in inventory.tool_names
    assert "db_utils" not in inventory.tool_names


def test_tool_inventory_matches_alias_terms() -> None:
    inventory = ToolInventory(["coresignal", "builtwith", "rapidapi"])

    match = inventory.match_terms(["Core Signal", "linkedin company profile"])

    assert match.duplicate_detected is True
    assert "coresignal" in match.matched_tools
    assert "rapidapi" in match.matched_tools
