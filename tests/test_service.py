import asyncio
from types import SimpleNamespace

import pytest

from internet_explorer.models import UrlCandidate, UrlEvaluation
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
