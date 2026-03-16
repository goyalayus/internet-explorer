from pathlib import Path

import pytest

from internet_explorer.config import AppConfig
from internet_explorer.evaluator import UrlEvaluator
from internet_explorer.models import FetchResult, UrlCandidate
from internet_explorer.tool_inventory import ToolInventory


class _TelemetryStub:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def timed(self) -> float:
        return 0.0

    def elapsed_ms(self, started_at: float) -> int:
        return 0

    def emit(self, **kwargs) -> None:
        self.events.append(kwargs)


class _FetcherStub:
    def __init__(self, responses: dict[str, FetchResult | Exception]) -> None:
        self.responses = responses

    async def fetch(self, url: str) -> FetchResult:
        if url not in self.responses:
            raise AssertionError(f"unexpected fetch for {url}")
        value = self.responses[url]
        if isinstance(value, Exception):
            raise value
        return value


class _BrowserManagerStub:
    @property
    def peak(self) -> int:
        return 0

    async def delegate(self, **kwargs):
        raise AssertionError("browser delegate should not be called in this test")


class _LLMStub:
    def __init__(
        self,
        *,
        tool_terms: list[str],
        decision_payload: dict | None = None,
        nav_payloads: list[dict] | None = None,
    ) -> None:
        self.nav_calls = 0
        self.tool_terms = tool_terms
        self.nav_payloads = nav_payloads or [
            {
                "reasoning": "Visit seed page first.",
                "action": "fetch_url",
                "target_url": "https://example.com/",
                "action_notes": ["seed"],
            },
            {
                "reasoning": "Enough evidence collected.",
                "action": "stop",
                "target_url": "",
                "action_notes": [],
            },
        ]
        self.decision_payload = decision_payload or {
            "useful": True,
            "relevance_score": 0.84,
            "outcome": "data_on_site",
            "reasoning": "Contains visible RFP listings and can be scraped from listing pages directly.",
            "api_stage": "none",
            "notes": [],
        }

    async def complete_json(self, *, schema, **kwargs):
        name = schema.__name__
        if name == "_NavigationPlanEnvelope":
            self.nav_calls += 1
            index = min(self.nav_calls - 1, len(self.nav_payloads) - 1)
            return schema.model_validate(self.nav_payloads[index])
        if name in {"_DecisionEnvelope", "_DecisionRawEnvelope"}:
            return schema.model_validate(self.decision_payload)
        if name == "_ToolTermEnvelope":
            return schema.model_validate({"terms": self.tool_terms, "reason": "stub"})
        raise AssertionError(f"unexpected schema: {name}")


def _config(tmp_path: Path) -> AppConfig:
    baseline = tmp_path / "baseline.txt"
    baseline.write_text("")
    env_path = tmp_path / ".env"
    env_path.write_text("")
    return AppConfig(
        workspace_root=tmp_path,
        mongodb_uri="mongodb://localhost:27017",
        eu_swarm_path=tmp_path,
        baseline_domains_file=baseline,
        known_tools_file=tmp_path / "known_tools.txt",
        vpn_log_dir=tmp_path / ".vpn_logs",
        env_file_path=env_path,
        max_internal_links=6,
        max_link_depth=1,
        max_site_graph_visits=3,
        max_site_graph_nodes=30,
        max_site_graph_frontier=6,
        max_sitemap_urls=20,
        max_sitemap_fetches=3,
    )


def _responses() -> dict[str, FetchResult]:
    empty = FetchResult(
        url="",
        final_url="",
        status_code=404,
        content_type="text/plain",
        body_text="",
        html="",
        text_excerpt="",
    )
    return {
        "https://example.com/robots.txt": empty.model_copy(update={"url": "https://example.com/robots.txt", "final_url": "https://example.com/robots.txt"}),
        "https://example.com/llms.txt": empty.model_copy(update={"url": "https://example.com/llms.txt", "final_url": "https://example.com/llms.txt"}),
        "https://example.com/llm.txt": empty.model_copy(update={"url": "https://example.com/llm.txt", "final_url": "https://example.com/llm.txt"}),
        "https://example.com/sitemap.xml": empty.model_copy(update={"url": "https://example.com/sitemap.xml", "final_url": "https://example.com/sitemap.xml"}),
        "https://example.com/sitemap_index.xml": empty.model_copy(update={"url": "https://example.com/sitemap_index.xml", "final_url": "https://example.com/sitemap_index.xml"}),
        "https://example.com/sitemap-index.xml": empty.model_copy(update={"url": "https://example.com/sitemap-index.xml", "final_url": "https://example.com/sitemap-index.xml"}),
        "https://example.com/": FetchResult(
            url="https://example.com/",
            final_url="https://example.com/",
            status_code=200,
            content_type="text/html",
            html="""
            <html><head><title>RFP Portal</title></head>
            <body>
              <h1>Current procurement RFP listings</h1>
              <a href=\"/rfp/current\">Open tenders</a>
            </body>
            </html>
            """,
            body_text="",
            text_excerpt="RFP listings and procurement updates",
        ),
    }


def _responses_with_fetch_failures() -> dict[str, FetchResult | Exception]:
    return {
        "https://example.com/robots.txt": TimeoutError("pool timeout"),
        "https://example.com/llms.txt": TimeoutError("pool timeout"),
        "https://example.com/llm.txt": TimeoutError("pool timeout"),
        "https://example.com/sitemap.xml": TimeoutError("pool timeout"),
        "https://example.com/sitemap_index.xml": TimeoutError("pool timeout"),
        "https://example.com/sitemap-index.xml": TimeoutError("pool timeout"),
        "https://example.com/": TimeoutError("pool timeout"),
    }


def _candidate(url_id: str, *, title: str = "Example") -> UrlCandidate:
    return UrlCandidate(
        url_id=url_id,
        strategy_id=f"strat_{url_id}",
        query_id=f"qry_{url_id}",
        raw_url="https://example.com/",
        canonical_url="https://example.com/",
        start_url="https://example.com/",
        homepage_url="https://example.com/",
        domain="example.com",
        novelty=True,
        source_title=title,
        source_snippet="",
        serp_rank=1,
        serp_page=1,
    )


@pytest.mark.asyncio
async def test_evaluator_keeps_one_line_visited_memory(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evaluator = UrlEvaluator(
        config,
        _LLMStub(tool_terms=["newsource"]),
        _FetcherStub(_responses()),
        _TelemetryStub(),
        _BrowserManagerStub(),
        tool_inventory=ToolInventory(["coresignal", "rapidapi", "builtwith"]),
    )
    candidate = _candidate("url_1")

    evaluation = await evaluator.evaluate(intent="find procurement data", candidate=candidate)

    assert evaluation.useful is True
    assert evaluation.outcome == "data_on_site"
    assert len(evaluation.visited_memory) == 1
    assert evaluation.visited_memory[0].step_no == 1
    assert len(evaluation.visited_memory[0].summary.splitlines()) == 1
    assert evaluation.site_graph is not None
    assert evaluation.tool_duplicate_signal.checked is True
    assert evaluation.tool_duplicate_signal.duplicate_detected is False


@pytest.mark.asyncio
async def test_evaluator_marks_duplicate_tool_sources(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evaluator = UrlEvaluator(
        config,
        _LLMStub(tool_terms=["core signal", "linkedin"]),
        _FetcherStub(_responses()),
        _TelemetryStub(),
        _BrowserManagerStub(),
        tool_inventory=ToolInventory(["coresignal", "rapidapi", "builtwith"]),
    )
    candidate = _candidate("url_2", title="Core Signal Style Source")

    evaluation = await evaluator.evaluate(intent="find procurement data", candidate=candidate)

    assert evaluation.useful is False
    assert evaluation.outcome == "irrelevant"
    assert evaluation.tool_duplicate_signal.duplicate_detected is True
    assert "coresignal" in evaluation.tool_duplicate_signal.matched_tools


@pytest.mark.asyncio
async def test_evaluator_normalizes_verdict_shape(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evaluator = UrlEvaluator(
        config,
        _LLMStub(
            tool_terms=["newsource"],
            decision_payload={
                "verdict": "data_on_site",
                "evidence": "RFP tenders are listed directly on page.",
            },
        ),
        _FetcherStub(_responses()),
        _TelemetryStub(),
        _BrowserManagerStub(),
        tool_inventory=ToolInventory(["coresignal", "rapidapi", "builtwith"]),
    )
    candidate = _candidate("url_3")

    evaluation = await evaluator.evaluate(intent="find procurement data", candidate=candidate)

    assert evaluation.useful is True
    assert evaluation.outcome == "data_on_site"
    assert evaluation.relevance_score >= 0.6
    assert all(not note.startswith("evaluation_error:") for note in evaluation.notes)


@pytest.mark.asyncio
async def test_evaluator_normalizes_category_reason_shape(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evaluator = UrlEvaluator(
        config,
        _LLMStub(
            tool_terms=["newsource"],
            decision_payload={
                "useful": False,
                "category": "irrelevant",
                "reason": "Not related to procurement or RFP data.",
            },
        ),
        _FetcherStub(_responses()),
        _TelemetryStub(),
        _BrowserManagerStub(),
        tool_inventory=ToolInventory(["coresignal", "rapidapi", "builtwith"]),
    )
    candidate = _candidate("url_4")

    evaluation = await evaluator.evaluate(intent="find procurement data", candidate=candidate)

    assert evaluation.useful is False
    assert evaluation.outcome == "irrelevant"
    assert "decision_shape_normalized" in evaluation.notes
    assert all(not note.startswith("evaluation_error:") for note in evaluation.notes)


@pytest.mark.asyncio
async def test_evaluator_accepts_string_source_evidence_urls(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evaluator = UrlEvaluator(
        config,
        _LLMStub(
            tool_terms=["newsource"],
            decision_payload={
                "useful": True,
                "outcome": "data_on_site",
                "reasoning": "Relevant procurement listings are visible.",
                "api_stage": "none",
                "source_evidence": [
                    "https://example.com/rfp/current",
                    "https://example.com/openapi.json",
                ],
            },
        ),
        _FetcherStub(_responses()),
        _TelemetryStub(),
        _BrowserManagerStub(),
        tool_inventory=ToolInventory(["coresignal", "rapidapi", "builtwith"]),
    )
    candidate = _candidate("url_5")

    evaluation = await evaluator.evaluate(intent="find procurement data", candidate=candidate)

    assert evaluation.useful is True
    evidence_by_url = {item.url: item.kind for item in evaluation.source_evidence}
    assert evidence_by_url["https://example.com/rfp/current"] == "page"
    assert evidence_by_url["https://example.com/openapi.json"] == "api"
    assert all(not note.startswith("evaluation_error:") for note in evaluation.notes)


@pytest.mark.asyncio
async def test_evaluator_ignores_delegate_action_when_browser_disabled(tmp_path: Path) -> None:
    config = _config(tmp_path).model_copy(update={"enable_browser_delegation": False})
    evaluator = UrlEvaluator(
        config,
        _LLMStub(
            tool_terms=["newsource"],
            nav_payloads=[
                {
                    "reasoning": "Use browser for the first page.",
                    "action": "delegate_browser",
                    "target_url": "https://example.com/",
                    "action_notes": [],
                },
                {
                    "reasoning": "Enough evidence collected.",
                    "action": "stop",
                    "target_url": "",
                    "action_notes": [],
                },
            ],
        ),
        _FetcherStub(_responses()),
        _TelemetryStub(),
        _BrowserManagerStub(),
        tool_inventory=ToolInventory(["coresignal", "rapidapi", "builtwith"]),
    )
    candidate = _candidate("url_6")

    evaluation = await evaluator.evaluate(intent="find procurement data", candidate=candidate)

    assert evaluation.useful is True
    assert len(evaluation.visited_memory) == 1
    assert evaluation.visited_memory[0].url == "https://example.com/"


@pytest.mark.asyncio
async def test_evaluator_marks_unknown_when_all_fetches_fail(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evaluator = UrlEvaluator(
        config,
        _LLMStub(tool_terms=["newsource"]),
        _FetcherStub(_responses_with_fetch_failures()),
        _TelemetryStub(),
        _BrowserManagerStub(),
        tool_inventory=ToolInventory(["coresignal", "rapidapi", "builtwith"]),
    )
    candidate = _candidate("url_5")

    evaluation = await evaluator.evaluate(intent="find procurement data", candidate=candidate)

    assert evaluation.useful is False
    assert evaluation.outcome == "unknown"
    assert "unknown_fetch_failure" in evaluation.notes
    assert evaluation.tool_duplicate_signal.checked is False
