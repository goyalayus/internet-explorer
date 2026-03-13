from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,}")
ALIAS_OVERRIDES: dict[str, set[str]] = {
    "coresignal": {"core signal"},
    "builtwith": {"built with", "tech stack"},
    "rapidapi": {"rapid api", "linkedin", "twitter", "reddit", "threads"},
    "theirstack": {"their stack"},
    "searchapi": {"search api"},
    "peopledatalabs": {"people data labs", "pdl"},
}
DEFAULT_TOOL_NAMES = [
    "apollo",
    "builtwith",
    "coresignal",
    "exa",
    "facebook_api",
    "intellizence",
    "linkedin_api",
    "peopledatalabs",
    "rapidapi",
    "reddit_api",
    "sales_nav_api",
    "searchapi",
    "sec_edgar",
    "semrush",
    "theirstack",
    "threads_api",
    "twitter_rapidapi",
    "twitter_v1_enterprise_api",
]


@dataclass(slots=True)
class ToolInventoryMatch:
    terms: list[str]
    matched_tools: list[str]
    reason: str

    @property
    def duplicate_detected(self) -> bool:
        return bool(self.matched_tools)


class ToolInventory:
    def __init__(self, tool_names: list[str]) -> None:
        ordered = sorted({name.strip().lower() for name in tool_names if name.strip()})
        self._tool_names = ordered
        self._tool_tokens: dict[str, set[str]] = {}
        for tool_name in ordered:
            aliases = {tool_name}
            aliases.update(ALIAS_OVERRIDES.get(tool_name, set()))
            tokens = set()
            for alias in aliases:
                tokens.update(TOKEN_PATTERN.findall(alias.lower()))
            tokens.add(tool_name.replace("_", ""))
            self._tool_tokens[tool_name] = {token for token in tokens if token}

    @property
    def tool_names(self) -> list[str]:
        return list(self._tool_names)

    @classmethod
    def from_file(cls, path: Path) -> "ToolInventory":
        names = _read_tool_names(path)
        if not names:
            names = list(DEFAULT_TOOL_NAMES)
        return cls(names)

    def match_terms(self, terms: list[str]) -> ToolInventoryMatch:
        normalized_terms = [term.strip().lower() for term in terms if term and term.strip()]
        tokens: set[str] = set()
        for term in normalized_terms:
            tokens.update(TOKEN_PATTERN.findall(term))
            compact = re.sub(r"[^a-z0-9]", "", term)
            if compact:
                tokens.add(compact)

        matched: list[str] = []
        for tool_name, tool_tokens in self._tool_tokens.items():
            if tokens.intersection(tool_tokens):
                matched.append(tool_name)

        matched.sort()
        reason = "matched_tool_terms" if matched else "no_tool_match"
        return ToolInventoryMatch(terms=normalized_terms, matched_tools=matched, reason=reason)


def _read_tool_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
        if isinstance(payload, dict):
            tools = payload.get("tools")
            if isinstance(tools, list):
                return [str(item).strip() for item in tools if str(item).strip()]
        return []

    names: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("#", 1)[0].strip().split(",", 1)[0].strip()
        if name:
            names.append(name)
    return names
