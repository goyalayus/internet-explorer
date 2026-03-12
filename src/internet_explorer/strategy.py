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


class StrategyPlanner:
    def __init__(self, config: AppConfig, llm: LLMClient, telemetry: Telemetry) -> None:
        self.config = config
        self.llm = llm
        self.telemetry = telemetry

    async def generate_strategies(self, intent: str) -> list[Strategy]:
        started = self.telemetry.timed()
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
        strategies = [
            Strategy(
                strategy_id=f"strategy_{idx+1}_{uuid.uuid4().hex[:8]}",
                title=item.title.strip(),
                concept=item.concept.strip(),
            )
            for idx, item in enumerate(response.strategies[: self.config.strategy_count])
        ]
        self.telemetry.emit(
            phase="strategy_gen",
            actor="normal_agent",
            input_payload={"intent": intent},
            output_summary=[strategy.model_dump() for strategy in strategies],
            decision=f"generated_{len(strategies)}_strategies",
            latency_ms=self.telemetry.elapsed_ms(started),
        )
        return strategies

    async def generate_queries(self, intent: str, strategy: Strategy) -> list[QueryPlan]:
        started = self.telemetry.timed()
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
        queries = [
            QueryPlan(
                query_id=f"query_{strategy.strategy_id}_{idx+1}_{uuid.uuid4().hex[:6]}",
                strategy_id=strategy.strategy_id,
                query=query.strip(),
            )
            for idx, query in enumerate(response.queries[: self.config.queries_per_strategy])
        ]
        self.telemetry.emit(
            phase="query_gen",
            actor="normal_agent",
            strategy_id=strategy.strategy_id,
            input_payload={"intent": intent, "strategy": strategy.model_dump()},
            output_summary=[query.model_dump() for query in queries],
            decision=f"generated_{len(queries)}_queries",
            latency_ms=self.telemetry.elapsed_ms(started),
        )
        return queries
