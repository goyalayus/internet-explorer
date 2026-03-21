from types import SimpleNamespace

from internet_explorer.models import QueryPlan, SearchResult, Strategy
from internet_explorer.planning_cache import DiscoveryPlanningCacheStore


def _config(tmp_path, *, discovery_cache_key: str = ""):
    return SimpleNamespace(
        discovery_cache_dir=tmp_path / "cache",
        discovery_cache_key=discovery_cache_key,
        strategy_count=10,
        queries_per_strategy=5,
        serp_pages_per_query=2,
        results_per_serp_page=10,
    )


def test_discovery_planning_cache_roundtrip(tmp_path) -> None:
    config = _config(tmp_path)
    store = DiscoveryPlanningCacheStore(config)

    strategies = [Strategy(strategy_id="strategy_1", title="Gov", concept="Find government portals")]
    queries = [QueryPlan(query_id="query_1", strategy_id="strategy_1", query="data annotation rfp site:gov")]
    raw_results = [
        SearchResult(
            query_id="query_1",
            strategy_id="strategy_1",
            rank=1,
            serp_page=1,
            title="RFP Portal",
            snippet="Current opportunities",
            url="https://example.gov/rfp",
        )
    ]

    snapshot = store.save(intent="Find data annotation RFP sources", strategies=strategies, queries=queries, raw_results=raw_results)
    loaded = store.load("Find data annotation RFP sources")

    assert loaded is not None
    assert snapshot.cache_key == loaded.cache_key
    assert len(loaded.strategies) == 1
    assert len(loaded.queries) == 1
    assert len(loaded.raw_results) == 1
    assert loaded.raw_results[0].url == "https://example.gov/rfp"


def test_discovery_planning_cache_key_is_stable_for_same_input(tmp_path) -> None:
    config = _config(tmp_path)
    store = DiscoveryPlanningCacheStore(config)

    key_a = store.key_for_intent("Find data annotation RFP sources")
    key_b = store.key_for_intent("Find   data annotation   RFP sources")
    key_c = store.key_for_intent("Find synthetic data procurement sources")

    assert key_a == key_b
    assert key_a != key_c


def test_discovery_planning_cache_honors_explicit_shared_key(tmp_path) -> None:
    config = _config(tmp_path, discovery_cache_key="shared-rfp-key")
    store = DiscoveryPlanningCacheStore(config)

    key = store.key_for_intent("any intent text")
    path = store.path_for_intent("any intent text")

    assert key == "shared-rfp-key"
    assert path.name == "shared-rfp-key.json"
