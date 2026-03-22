import asyncio
from types import SimpleNamespace

import pytest

from internet_explorer.models import QueryPlan, SearchResult, Strategy, UrlCandidate, UrlEvaluation
from internet_explorer.planning_cache import DiscoveryPlanningCacheStore
from internet_explorer.service import IntentDiscoveryService


class _TelemetryStub:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def emit(self, **kwargs) -> None:
        self.events.append(kwargs)


class _EvaluatorStub:
    def __init__(self, delays: dict[str, float]) -> None:
        self.delays = delays

    async def evaluate(self, intent: str, candidate: UrlCandidate) -> UrlEvaluation:
        await asyncio.sleep(self.delays[candidate.url_id])
        return UrlEvaluation(
            url_id=candidate.url_id,
            canonical_url=candidate.canonical_url,
            start_url=candidate.start_url,
            homepage_url=candidate.homepage_url,
            domain=candidate.domain,
            novelty=candidate.novelty,
            render_profile="static_ssr",
            outcome="data_on_site" if candidate.url_id == "url_2" else "irrelevant",
            useful=candidate.url_id == "url_2",
        )


class _PlannerFailStub:
    async def generate_strategies(self, intent: str):  # noqa: ARG002
        raise AssertionError("planner should not be called on cache hit")

    async def generate_queries(self, intent: str, strategy):  # noqa: ARG002
        raise AssertionError("planner should not be called on cache hit")


class _SearcherFailStub:
    async def collect_many(self, queries):  # noqa: ARG002
        raise AssertionError("search should not be called on cache hit")


class _PlannerStub:
    async def generate_strategies(self, intent: str):  # noqa: ARG002
        return [Strategy(strategy_id="strategy_1", title="Gov", concept="Gov sources")]

    async def generate_queries(self, intent: str, strategy: Strategy):  # noqa: ARG002
        return [QueryPlan(query_id="query_1", strategy_id=strategy.strategy_id, query="data annotation rfp site:gov")]


class _SearcherStub:
    async def collect_many(self, queries: list[QueryPlan]):
        assert len(queries) == 1
        return [
            SearchResult(
                query_id=queries[0].query_id,
                strategy_id=queries[0].strategy_id,
                rank=1,
                serp_page=1,
                title="RFP",
                snippet="Opportunity",
                url="https://example.gov/rfp",
            )
        ]


def _candidate(url_id: str, domain: str) -> UrlCandidate:
    return UrlCandidate(
        url_id=url_id,
        strategy_id="strategy_1",
        query_id="query_1",
        raw_url=f"https://{domain}/source",
        canonical_url=f"https://{domain}/source",
        start_url=f"https://{domain}/",
        homepage_url=f"https://{domain}/",
        domain=domain,
        novelty=True,
        serp_rank=1,
        serp_page=1,
    )


@pytest.mark.asyncio
async def test_evaluate_candidates_calls_on_result_as_each_result_finishes() -> None:
    service = object.__new__(IntentDiscoveryService)
    service.config = SimpleNamespace(url_batch_size=2, max_url_concurrency=0)
    telemetry = _TelemetryStub()
    evaluator = _EvaluatorStub(
        {
            "url_1": 0.03,
            "url_2": 0.0,
            "url_3": 0.0,
        }
    )
    seen: list[str] = []

    results = await IntentDiscoveryService._evaluate_candidates(
        service,
        intent="find sources",
        candidates=[
            _candidate("url_1", "example.com"),
            _candidate("url_2", "example.org"),
            _candidate("url_3", "example.net"),
        ],
        evaluator=evaluator,
        telemetry=telemetry,
        on_result=lambda evaluation: seen.append(evaluation.url_id),
    )

    assert len(results) == 3
    assert seen == ["url_2", "url_1", "url_3"]
    assert [event["decision"] for event in telemetry.events] == [
        "batch_start",
        "batch_done",
        "batch_start",
        "batch_done",
    ]


@pytest.mark.asyncio
async def test_resolve_discovery_inputs_uses_cache_on_hit(tmp_path) -> None:
    service = object.__new__(IntentDiscoveryService)
    service.config = SimpleNamespace(
        discovery_cache_mode="read_write",
        discovery_cache_dir=tmp_path / "cache",
        discovery_cache_key="shared-rfp-key",
        strategy_count=10,
        queries_per_strategy=5,
        serp_pages_per_query=2,
        results_per_serp_page=10,
    )
    telemetry = _TelemetryStub()
    cache_store = DiscoveryPlanningCacheStore(service.config)
    cache_store.save(
        intent="Find sources",
        strategies=[Strategy(strategy_id="strategy_1", title="Gov", concept="Gov sources")],
        queries=[QueryPlan(query_id="query_1", strategy_id="strategy_1", query="query")],
        raw_results=[
            SearchResult(
                query_id="query_1",
                strategy_id="strategy_1",
                rank=1,
                serp_page=1,
                title="Example",
                snippet="Example",
                url="https://example.gov/rfp",
            )
        ],
    )

    strategies, queries, raw_results, cache_info = await IntentDiscoveryService._resolve_discovery_inputs(
        service,
        intent="Find sources",
        planner=_PlannerFailStub(),
        searcher=_SearcherFailStub(),
        telemetry=telemetry,
    )

    assert len(strategies) == 1
    assert len(queries) == 1
    assert len(raw_results) == 1
    assert cache_info["status"] == "hit"
    assert any(event["decision"] == "cache_hit" for event in telemetry.events)


@pytest.mark.asyncio
async def test_resolve_discovery_inputs_writes_cache_on_miss(tmp_path) -> None:
    service = object.__new__(IntentDiscoveryService)
    service.config = SimpleNamespace(
        discovery_cache_mode="read_write",
        discovery_cache_dir=tmp_path / "cache",
        discovery_cache_key="shared-rfp-key",
        strategy_count=10,
        queries_per_strategy=5,
        serp_pages_per_query=2,
        results_per_serp_page=10,
    )
    telemetry = _TelemetryStub()

    strategies, queries, raw_results, cache_info = await IntentDiscoveryService._resolve_discovery_inputs(
        service,
        intent="Find sources",
        planner=_PlannerStub(),
        searcher=_SearcherStub(),
        telemetry=telemetry,
    )

    assert len(strategies) == 1
    assert len(queries) == 1
    assert len(raw_results) == 1
    assert cache_info["status"] == "written"
    assert (service.config.discovery_cache_dir / "shared-rfp-key.json").exists()
    assert any(event["decision"] == "cache_miss" for event in telemetry.events)
    assert any(event["decision"] == "cache_write" for event in telemetry.events)


def test_dedupe_results_selects_best_candidate_for_domain() -> None:
    service = object.__new__(IntentDiscoveryService)
    service.config = SimpleNamespace(candidate_start_mode="domain_homepage")
    telemetry = _TelemetryStub()

    raw_results = [
        SearchResult(
            query_id="query_1",
            strategy_id="strategy_1",
            rank=1,
            serp_page=1,
            title="Faculty profile",
            snippet="Department staff profile page",
            url="https://example.edu/people/jane-doe",
        ),
        SearchResult(
            query_id="query_2",
            strategy_id="strategy_2",
            rank=7,
            serp_page=1,
            title="Procurement opportunities",
            snippet="Open tenders and RFP opportunities",
            url="https://example.edu/procurement/opportunities",
        ),
    ]

    candidates = IntentDiscoveryService._dedupe_results(
        service,
        raw_results=raw_results,
        baseline_domains=set(),
        telemetry=telemetry,
    )

    assert len(candidates) == 1
    assert candidates[0].domain == "example.edu"
    assert candidates[0].canonical_url == "https://example.edu/procurement/opportunities"
    assert candidates[0].start_url == "https://example.edu/"


def test_dedupe_results_filters_known_noise_hosts() -> None:
    service = object.__new__(IntentDiscoveryService)
    service.config = SimpleNamespace(candidate_start_mode="domain_homepage")
    telemetry = _TelemetryStub()

    raw_results = [
        SearchResult(
            query_id="query_1",
            strategy_id="strategy_1",
            rank=1,
            serp_page=1,
            title="Company page",
            snippet="Professional network profile",
            url="https://www.linkedin.com/company/example-inc",
        ),
        SearchResult(
            query_id="query_1",
            strategy_id="strategy_1",
            rank=2,
            serp_page=1,
            title="Contract opportunities",
            snippet="State procurement RFP listing",
            url="https://state.gov/procurement/contracts",
        ),
    ]

    candidates = IntentDiscoveryService._dedupe_results(
        service,
        raw_results=raw_results,
        baseline_domains=set(),
        telemetry=telemetry,
    )

    assert len(candidates) == 1
    assert candidates[0].domain == "state.gov"


def test_dedupe_results_first_result_mode_uses_selected_canonical_start() -> None:
    service = object.__new__(IntentDiscoveryService)
    service.config = SimpleNamespace(candidate_start_mode="first_result_url")
    telemetry = _TelemetryStub()

    raw_results = [
        SearchResult(
            query_id="query_1",
            strategy_id="strategy_1",
            rank=1,
            serp_page=1,
            title="Team",
            snippet="Meet our team",
            url="https://agency.example.com/team",
        ),
        SearchResult(
            query_id="query_2",
            strategy_id="strategy_2",
            rank=8,
            serp_page=1,
            title="Vendor opportunities",
            snippet="Open solicitations and bids",
            url="https://agency.example.com/procurement/solicitations",
        ),
    ]

    candidates = IntentDiscoveryService._dedupe_results(
        service,
        raw_results=raw_results,
        baseline_domains=set(),
        telemetry=telemetry,
    )

    assert len(candidates) == 1
    assert candidates[0].canonical_url == "https://agency.example.com/procurement/solicitations"
    assert candidates[0].start_url == "https://agency.example.com/procurement/solicitations"
