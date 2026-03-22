from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from internet_explorer.config import AppConfig
from internet_explorer.llm import LLMClient
from internet_explorer.models import QueryPlan, Strategy
from internet_explorer.telemetry import Telemetry


class _StrategyItem(BaseModel):
    title: str
    concept: str


class _StrategyEnvelope(BaseModel):
    strategies: list[_StrategyItem] = Field(default_factory=list)


class _QueryEnvelope(BaseModel):
    queries: list[str] = Field(default_factory=list)


STRATEGY_SYSTEM_PROMPT = """
You design internet discovery strategies for finding new public or semi-public data sources.
Return only JSON.
"""

QUERY_SYSTEM_PROMPT = """
You write Google-only discovery queries for a single datasource-finding strategy.
Queries should help discover sites, portals, APIs, developer docs, datasets, procurement sources, and niche directories.
Return only JSON.
"""

PROCUREMENT_INTENT_MARKERS = (
    "rfp",
    "tender",
    "procurement",
    "solicitation",
    "bid",
    "bids",
    "contract opportunity",
    "contract opportunities",
)
PROCUREMENT_QUERY_MARKERS = (
    "rfp",
    "request for proposal",
    "tender",
    "procurement",
    "solicitation",
    "bid",
    "contract opportunities",
    "vendor portal",
)
QUERY_NOISE_SITE_EXCLUSIONS = (
    "-site:linkedin.com",
    "-site:facebook.com",
    "-site:instagram.com",
    "-site:x.com",
    "-site:twitter.com",
    "-site:youtube.com",
    "-site:github.com",
    "-site:readthedocs.io",
    "-site:medium.com",
)

FALLBACK_STRATEGY_TEMPLATES: list[tuple[str, str]] = [
    (
        "Government Procurement Portals",
        "Search national, state, and city procurement portals for tenders, RFPs, and contract notices tied to the intent.",
    ),
    (
        "Institutional Tender Boards",
        "Target universities, hospitals, and research institutions that publish procurement notices with domain-specific requirements.",
    ),
    (
        "Sector-Specific Bid Networks",
        "Find niche bid networks and industry directories that aggregate opportunities relevant to the intent domain.",
    ),
    (
        "Developer and API Discovery",
        "Look for developer portals, API docs, and integration catalogs that expose programmatic access to intent-relevant data.",
    ),
    (
        "Open Data Catalog Sweep",
        "Scan open-data platforms, catalog indexes, and public datasets that map directly to key intent entities.",
    ),
    (
        "RFP Software Footprint Search",
        "Use platform fingerprints from common RFP software to discover hidden procurement subdomains and opportunity pages.",
    ),
    (
        "Vendor Portal Backtracking",
        "Track vendor registration and supplier portals where opportunities are published before broad distribution.",
    ),
    (
        "Regulator and Agency Sources",
        "Identify regulatory agencies and public authorities that publish structured notices and source data in the intent area.",
    ),
    (
        "Geography-Localized Discovery",
        "Search by region, state, and city combinations to uncover local sources that do not rank in generic searches.",
    ),
    (
        "Long-Tail Keyword Expansion",
        "Use synonym and adjacent-term variants to surface non-obvious sources that still match the core intent.",
    ),
]

FALLBACK_QUERY_PATTERNS: list[str] = [
    '{intent} RFP "request for proposal"',
    '{intent} tender portal',
    '{intent} procurement site:gov',
    '"data annotation" RFP site:gov',
    '{intent} "vendor portal" "RFP"',
    '{intent} "open data" catalog',
    '{intent} "API documentation"',
    '{intent} "developers" "API"',
    '"{strategy}" {intent} opportunities',
    '{intent} "bids" "contract notice"',
]


class StrategyPlanner:
    def __init__(self, config: AppConfig, llm: LLMClient, telemetry: Telemetry) -> None:
        self.config = config
        self.llm = llm
        self.telemetry = telemetry

    async def generate_strategies(self, intent: str) -> list[Strategy]:
        started = self.telemetry.timed()
        fallback_error: Exception | None = None
        candidate_items: list[_StrategyItem] = []
        try:
            response = await self.llm.complete_json(
                system_prompt=STRATEGY_SYSTEM_PROMPT,
                user_prompt=(
                    "Generate exactly "
                    f"{self.config.strategy_count} distinct strategies for discovering new datasources for this intent.\n\n"
                    f"Intent:\n{intent}\n\n"
                    "Each strategy must include a short title and a 2-4 sentence concept. "
                    "Do not use an exhaustive or recursive loop. This is a fixed one-shot strategy set."
                ),
                schema=_StrategyEnvelope,
                temperature=0.3,
                max_completion_tokens=4096,
            )
            candidate_items = response.strategies
        except Exception as exc:
            fallback_error = exc

        items = [item for item in candidate_items if item.title.strip() and item.concept.strip()]
        llm_count = len(items)
        if len(items) < self.config.strategy_count:
            needed = self.config.strategy_count - len(items)
            items.extend(self._fallback_strategy_items(intent, needed))
        items = items[: self.config.strategy_count]

        strategies = [
            Strategy(
                strategy_id=f"strategy_{idx+1}_{uuid.uuid4().hex[:8]}",
                title=item.title.strip(),
                concept=item.concept.strip(),
            )
            for idx, item in enumerate(items)
        ]
        decision = (
            f"generated_{len(strategies)}_strategies"
            if fallback_error is None and llm_count >= self.config.strategy_count
            else f"fallback_generated_{len(strategies)}_strategies"
        )
        output_summary = [strategy.model_dump() for strategy in strategies]
        if fallback_error is not None:
            output_summary.append({"fallback_reason": str(fallback_error)})
        self.telemetry.emit(
            phase="strategy_gen",
            actor="normal_agent",
            input_payload={"intent": intent},
            output_summary=output_summary,
            decision=decision,
            error_code=type(fallback_error).__name__ if fallback_error is not None else None,
            latency_ms=self.telemetry.elapsed_ms(started),
        )
        return strategies

    async def generate_queries(self, intent: str, strategy: Strategy) -> list[QueryPlan]:
        started = self.telemetry.timed()
        fallback_error: Exception | None = None
        generated_queries: list[str] = []
        try:
            response = await self.llm.complete_json(
                system_prompt=QUERY_SYSTEM_PROMPT,
                user_prompt=(
                    "Generate exactly "
                    f"{self.config.queries_per_strategy} Google queries for this datasource discovery strategy.\n\n"
                    f"Intent:\n{intent}\n\n"
                    f"Strategy title:\n{strategy.title}\n\n"
                    f"Strategy concept:\n{strategy.concept}\n\n"
                    "Constraints:\n"
                    "- Google-only queries.\n"
                    "- Focus on novel sources, not known broad aggregators.\n"
                    "- Include API/docs angles when relevant.\n"
                    "- Include dataset/portal/RFP/search/catalog angles when relevant.\n"
                    "- Avoid exhaustive loops and avoid repeating the same shape.\n"
                ),
                schema=_QueryEnvelope,
                temperature=0.2,
                max_completion_tokens=2048,
            )
            generated_queries = [query.strip() for query in response.queries if query.strip()]
        except Exception as exc:
            fallback_error = exc

        llm_count = len(generated_queries)
        if len(generated_queries) < self.config.queries_per_strategy:
            needed = self.config.queries_per_strategy - len(generated_queries)
            generated_queries.extend(self._fallback_queries(intent, strategy, needed))
        generated_queries = generated_queries[: self.config.queries_per_strategy]
        generated_queries = _normalize_query_batch(generated_queries)
        if _intent_requires_procurement_focus(intent):
            generated_queries = [_ensure_procurement_query_focus(query) for query in generated_queries]
            generated_queries = _normalize_query_batch(generated_queries)

        queries = [
            QueryPlan(
                query_id=f"query_{strategy.strategy_id}_{idx+1}_{uuid.uuid4().hex[:6]}",
                strategy_id=strategy.strategy_id,
                query=query.strip(),
            )
            for idx, query in enumerate(generated_queries)
        ]
        decision = (
            f"generated_{len(queries)}_queries"
            if fallback_error is None and llm_count >= self.config.queries_per_strategy
            else f"fallback_generated_{len(queries)}_queries"
        )
        output_summary = [query.model_dump() for query in queries]
        if fallback_error is not None:
            output_summary.append({"fallback_reason": str(fallback_error)})
        self.telemetry.emit(
            phase="query_gen",
            actor="normal_agent",
            strategy_id=strategy.strategy_id,
            input_payload={"intent": intent, "strategy": strategy.model_dump()},
            output_summary=output_summary,
            decision=decision,
            error_code=type(fallback_error).__name__ if fallback_error is not None else None,
            latency_ms=self.telemetry.elapsed_ms(started),
        )
        return queries

    def _fallback_strategy_items(self, intent: str, needed: int) -> list[_StrategyItem]:
        items: list[_StrategyItem] = []
        for idx in range(needed):
            title, concept = FALLBACK_STRATEGY_TEMPLATES[idx % len(FALLBACK_STRATEGY_TEMPLATES)]
            items.append(
                _StrategyItem(
                    title=title,
                    concept=f"{concept} Focus intent anchor: {intent}.",
                )
            )
        return items

    def _fallback_queries(self, intent: str, strategy: Strategy, needed: int) -> list[str]:
        queries: list[str] = []
        for idx in range(needed):
            pattern = FALLBACK_QUERY_PATTERNS[idx % len(FALLBACK_QUERY_PATTERNS)]
            query = pattern.format(intent=intent, strategy=strategy.title).strip()
            if query:
                queries.append(query)
        return queries


def _intent_requires_procurement_focus(intent: str) -> bool:
    text = (intent or "").lower()
    return any(marker in text for marker in PROCUREMENT_INTENT_MARKERS)


def _normalize_query_batch(queries: list[str]) -> list[str]:
    cleaned: list[str] = []
    for query in queries:
        value = " ".join((query or "").split()).strip()
        if value:
            cleaned.append(value)
    return cleaned


def _ensure_procurement_query_focus(query: str) -> str:
    text = " ".join((query or "").split()).strip()
    if not text:
        return text

    lowered = text.lower()
    if not any(marker in lowered for marker in PROCUREMENT_QUERY_MARKERS):
        text = f'{text} ("RFP" OR tender OR procurement OR solicitation)'
        lowered = text.lower()

    for exclusion in QUERY_NOISE_SITE_EXCLUSIONS:
        if exclusion in lowered:
            continue
        text = f"{text} {exclusion}"
        lowered = text.lower()

    return " ".join(text.split()).strip()
