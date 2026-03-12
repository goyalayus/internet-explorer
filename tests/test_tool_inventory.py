from pathlib import Path

from internet_explorer.tool_inventory import ToolInventory


def test_tool_inventory_discovers_tools_from_tool_flow_layout(tmp_path: Path) -> None:
    tool_flow = tmp_path / "tool-flow"
    utils_dir = tool_flow / "utils"
    utils_dir.mkdir(parents=True)

    (utils_dir / "coresignal.py").write_text("class CoreSignal: ...\n")
    (utils_dir / "builtwith.py").write_text("class BuiltWith: ...\n")
    (utils_dir / "db_utils.py").write_text("# ignored utility\n")

    provider_dir = utils_dir / "RapidApiProvider"
    provider_dir.mkdir(parents=True)
    (provider_dir / "linkedin_api.py").write_text("class LinkedInApi: ...\n")

    play = tool_flow / "a_generic_triggers" / "example.py"
    play.parent.mkdir(parents=True)
    play.write_text("from utils.rapidapi import RapidApi\nfrom utils.coresignal import CoreSignal\n")

    inventory = ToolInventory.from_tool_flow(tool_flow)

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
