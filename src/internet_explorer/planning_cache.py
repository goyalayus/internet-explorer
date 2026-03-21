from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from internet_explorer.config import AppConfig
from internet_explorer.models import QueryPlan, SearchResult, Strategy


class DiscoveryPlanningSnapshot(BaseModel):
    version: int = 1
    cache_key: str
    intent: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    strategies: list[Strategy] = Field(default_factory=list)
    queries: list[QueryPlan] = Field(default_factory=list)
    raw_results: list[SearchResult] = Field(default_factory=list)


class DiscoveryPlanningCacheStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cache_dir = config.discovery_cache_dir

    def key_for_intent(self, intent: str) -> str:
        explicit_key = (self.config.discovery_cache_key or "").strip()
        if explicit_key:
            return _safe_fragment(explicit_key)

        normalized_intent = " ".join((intent or "").split()).lower()
        fingerprint_payload = {
            "intent": normalized_intent,
            "strategy_count": self.config.strategy_count,
            "queries_per_strategy": self.config.queries_per_strategy,
            "serp_pages_per_query": self.config.serp_pages_per_query,
            "results_per_serp_page": self.config.results_per_serp_page,
        }
        digest = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
        hint = _safe_fragment(normalized_intent)[:40]
        if hint:
            return f"{hint}_{digest}"
        return digest

    def path_for_intent(self, intent: str) -> Path:
        return self.cache_dir / f"{self.key_for_intent(intent)}.json"

    def load(self, intent: str) -> DiscoveryPlanningSnapshot | None:
        path = self.path_for_intent(intent)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return DiscoveryPlanningSnapshot.model_validate(payload)

    def save(
        self,
        *,
        intent: str,
        strategies: list[Strategy],
        queries: list[QueryPlan],
        raw_results: list[SearchResult],
    ) -> DiscoveryPlanningSnapshot:
        snapshot = DiscoveryPlanningSnapshot(
            cache_key=self.key_for_intent(intent),
            intent=intent,
            strategies=strategies,
            queries=queries,
            raw_results=raw_results,
        )
        path = self.path_for_intent(intent)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(path)
        return snapshot


def _safe_fragment(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._-")
    if not cleaned:
        return "cache"
    return cleaned[:80]
