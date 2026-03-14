from __future__ import annotations

import asyncio
from typing import Any

import httpx

from internet_explorer.config import AppConfig
from internet_explorer.models import QueryPlan, SearchResult
from internet_explorer.telemetry import Telemetry


class GoogleSearchCollector:
    def __init__(self, config: AppConfig, telemetry: Telemetry) -> None:
        self.config = config
        self.telemetry = telemetry
        self.endpoint = "https://www.googleapis.com/customsearch/v1"
        self.client = httpx.AsyncClient(timeout=config.request_timeout_seconds)
        self.api_keys = _parse_api_keys(config.google_api_keys, fallback=config.google_api_key)
        self._key_cursor = 0

    async def collect(self, query_plan: QueryPlan) -> list[SearchResult]:
        all_results: list[SearchResult] = []
        for page_index in range(self.config.serp_pages_per_query):
            start = page_index * self.config.results_per_serp_page + 1
            started = self.telemetry.timed()
            raw_results = await self._search_page(
                query=query_plan.query,
                num=self.config.results_per_serp_page,
                start=start,
            )
            parsed = [
                SearchResult(
                    query_id=query_plan.query_id,
                    strategy_id=query_plan.strategy_id,
                    rank=start + idx,
                    serp_page=page_index + 1,
                    title=item.get("title", "") or "",
                    snippet=item.get("snippet", "") or "",
                    url=item.get("link", ""),
                )
                for idx, item in enumerate(raw_results)
                if item.get("link")
            ]
            all_results.extend(parsed)
            self.telemetry.emit(
                phase="serp_fetch",
                actor="system",
                strategy_id=query_plan.strategy_id,
                query_id=query_plan.query_id,
                input_payload={"query": query_plan.query, "start": start},
                output_summary=[result.model_dump() for result in parsed],
                decision=f"results_{len(parsed)}",
                latency_ms=self.telemetry.elapsed_ms(started),
            )
        return all_results

    async def collect_many(self, queries: list[QueryPlan]) -> list[SearchResult]:
        semaphore = asyncio.Semaphore(max(1, self.config.search_query_concurrency))

        async def run_query(query_plan: QueryPlan):
            async with semaphore:
                return await self.collect(query_plan)

        tasks = [run_query(query) for query in queries]
        batches = await asyncio.gather(*tasks)
        results: list[SearchResult] = []
        for batch in batches:
            results.extend(batch)
        return results

    async def close(self) -> None:
        await self.client.aclose()

    async def _search_page(self, *, query: str, num: int, start: int) -> list[dict[str, Any]]:
        if not self.api_keys:
            raise ValueError("GOOGLE_API_KEY is required for Google Custom Search.")
        if not self.config.google_search_engine_id:
            raise ValueError("GOOGLE_SEARCH_ENGINE_ID is required for Google Custom Search.")

        attempts = max(1, self.config.search_retry_attempts)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            base_index = self._next_key_index()
            for offset in range(len(self.api_keys)):
                key_index = (base_index + offset) % len(self.api_keys)
                key = self.api_keys[key_index]
                try:
                    response = await self.client.get(
                        self.endpoint,
                        params={
                            "key": key,
                            "cx": self.config.google_search_engine_id,
                            "q": query,
                            "num": num,
                            "start": start,
                        },
                    )
                except Exception as exc:
                    last_error = RuntimeError(f"Google search transport error ({type(exc).__name__}): {exc}")
                    continue

                if response.status_code < 400:
                    payload = response.json()
                    items = payload.get("items")
                    return items if isinstance(items, list) else []

                detail = response.text[:400]
                retryable = response.status_code in {403, 429, 500, 502, 503, 504}
                last_error = RuntimeError(f"Google search failed status={response.status_code}: {detail}")
                if retryable:
                    continue
                raise last_error

            if attempt < attempts:
                backoff = self.config.search_retry_base_backoff_seconds * (2 ** (attempt - 1))
                await asyncio.sleep(min(backoff, 8.0))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Google search failed without a captured error")

    def _next_key_index(self) -> int:
        if not self.api_keys:
            return 0
        index = self._key_cursor % len(self.api_keys)
        self._key_cursor += 1
        return index


def _parse_api_keys(raw: str, *, fallback: str = "") -> list[str]:
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    if keys:
        return keys
    if fallback.strip():
        return [fallback.strip()]
    return []
