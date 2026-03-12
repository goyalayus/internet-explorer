from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


IMPORT_PATTERN = re.compile(r"(?:from|import)\s+utils\.([A-Za-z0-9_\.]+)")
TOKEN_PATTERN = re.compile(r"[a-z0-9]{2,}")
IGNORED_TOOL_NAMES = {
    "__init__",
    "db_utils",
    "logger",
    "openai",
    "openai2",
    "openAI",
    "gemini",
    "aws",
}
ALIAS_OVERRIDES: dict[str, set[str]] = {
    "coresignal": {"core signal"},
    "builtwith": {"built with", "tech stack"},
    "rapidapi": {"rapid api", "linkedin", "twitter", "reddit", "threads"},
    "theirstack": {"their stack"},
    "searchapi": {"search api"},
    "peopledatalabs": {"people data labs", "pdl"},
}


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
    def from_tool_flow(cls, tool_flow_path: Path) -> "ToolInventory":
        tool_names: set[str] = set()

        utils_dir = tool_flow_path / "utils"
        if utils_dir.exists():
            for module_path in utils_dir.glob("*.py"):
                stem = module_path.stem
                if stem not in IGNORED_TOOL_NAMES:
                    tool_names.add(stem.lower())

        provider_dir = utils_dir / "RapidApiProvider"
        if provider_dir.exists():
            tool_names.add("rapidapi")
            for provider_path in provider_dir.glob("*.py"):
                stem = provider_path.stem
                if stem != "__init__":
                    tool_names.add(stem.lower())

        for py_path in cls._iter_python_files(tool_flow_path):
            try:
                text = py_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for match in IMPORT_PATTERN.finditer(text):
                module = match.group(1).split(".")[0]
                if module and module not in IGNORED_TOOL_NAMES:
                    tool_names.add(module.lower())

        return cls(sorted(tool_names))

    @classmethod
    def _iter_python_files(cls, root: Path):
        skip_parts = {".git", "venv", ".venv", "__pycache__", ".pytest_cache"}
        for path in root.rglob("*.py"):
            if any(part in skip_parts for part in path.parts):
                continue
            yield path

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
        reason = (
            "matched_tool_terms"
            if matched
            else "no_tool_match"
        )
        return ToolInventoryMatch(terms=normalized_terms, matched_tools=matched, reason=reason)
