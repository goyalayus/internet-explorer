from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,}")
ALIAS_OVERRIDES: dict[str, set[str]] = {
    "coresignal": {"core signal"},
    "rapidapi": {"rapid api"},
    "theirstack": {"their stack"},
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


@dataclass(frozen=True, slots=True)
class _MatchPattern:
    normalized: str
    compact: str
    tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CandidateTerm:
    normalized: str
    compact: str
    tokens: frozenset[str]


class ToolInventory:
    def __init__(self, tool_names: list[str]) -> None:
        ordered = sorted({name.strip().lower() for name in tool_names if name.strip()})
        self._tool_names = ordered
        self._tool_patterns: dict[str, tuple[_MatchPattern, ...]] = {}
        for tool_name in ordered:
            aliases = {tool_name}
            aliases.update(ALIAS_OVERRIDES.get(tool_name, set()))
            patterns: list[_MatchPattern] = []
            for alias in aliases:
                pattern = _build_match_pattern(alias)
                if pattern is None:
                    continue
                patterns.append(pattern)
            self._tool_patterns[tool_name] = tuple(patterns)

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
        candidate_terms = [_build_candidate_term(term) for term in normalized_terms]

        matched: list[str] = []
        for tool_name, patterns in self._tool_patterns.items():
            if any(_term_matches_pattern(term, pattern) for term in candidate_terms for pattern in patterns):
                matched.append(tool_name)

        matched.sort()
        reason = "matched_tool_terms" if matched else "no_tool_match"
        return ToolInventoryMatch(terms=normalized_terms, matched_tools=matched, reason=reason)


def _build_match_pattern(value: str) -> _MatchPattern | None:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return None

    compact = re.sub(r"[^a-z0-9]", "", normalized)
    tokens = tuple(TOKEN_PATTERN.findall(normalized))
    if not compact and not tokens:
        return None

    return _MatchPattern(normalized=normalized, compact=compact, tokens=tokens)


def _build_candidate_term(value: str) -> _CandidateTerm:
    normalized = str(value or "").strip().lower()
    compact = re.sub(r"[^a-z0-9]", "", normalized)
    tokens = frozenset(TOKEN_PATTERN.findall(normalized))
    return _CandidateTerm(normalized=normalized, compact=compact, tokens=tokens)


def _term_matches_pattern(term: _CandidateTerm, pattern: _MatchPattern) -> bool:
    if not pattern.tokens and not pattern.compact:
        return False

    if len(pattern.tokens) <= 1:
        if pattern.compact and pattern.compact in term.tokens:
            return True
        return False

    if pattern.normalized and pattern.normalized in term.normalized:
        return True

    return set(pattern.tokens).issubset(term.tokens)


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
