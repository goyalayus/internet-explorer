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
    def __init__(self, responses: dict[str, FetchResult]) -> None:
        self.responses = responses

    async def fetch(self, url: str) -> FetchResult:
        if url not in self.responses:
            raise AssertionError(f"unexpected fetch for {url}")
        return self.responses[url]


class _BrowserManagerStub:
    @property
    def peak(self) -> int:
        return 0

    async def delegate(self, **kwargs):
        raise AssertionError("browser delegate should not be called in this test")


class _LLMStub:
    def __init__(self, *, tool_terms: list[str]) -> None:
        self.nav_calls = 0
        self.tool_terms = tool_terms

    async def complete_json(self, *, schema, **kwargs):
        name = schema.__name__
        if name == "_NavigationPlanEnvelope":
            self.nav_calls += 1
            if self.nav_calls == 1:
                return schema.model_validate(
                    {
                        "reasoning": "Visit seed page first.",
                        "action": "fetch_url",
                        "target_url": "https://example.com/",
                        "action_notes": ["seed"],
                    }
                )
            return schema.model_validate(
                {
                    "reasoning": "Enough evidence collected.",
                    "action": "stop",
                    "target_url": "",
                    "action_notes": [],
                }
            )
        if name == "_DecisionEnvelope":
            return schema.model_validate(
                {
                    "useful": True,
                    "relevance_score": 0.84,
                    "outcome": "data_on_site",
                    "why_useful": "Contains visible RFP listings.",
                    "how_to_use": "Scrape listing pages directly.",
                    "api_stage": "none",
                    "notes": [],
                }
            )
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
        query_optimizer_repo_path=tmp_path,
        tool_flow_path=tmp_path,
        baseline_domains_file=baseline,
        vpn_log_dir=tmp_path / ".vpn_logs",
        tool_flow_env_path=env_path,
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
    candidate = UrlCandidate(
        url_id="url_1",
        strategy_id="strat_1",
        query_id="qry_1",
        raw_url="https://example.com/",
        canonical_url="https://example.com/",
        domain="example.com",
        novelty=True,
        source_title="Example",
        source_snippet="",
        serp_rank=1,
        serp_page=1,
    )

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
    candidate = UrlCandidate(
        url_id="url_2",
        strategy_id="strat_2",
        query_id="qry_2",
        raw_url="https://example.com/",
        canonical_url="https://example.com/",
        domain="example.com",
        novelty=True,
        source_title="Core Signal Style Source",
        source_snippet="",
        serp_rank=1,
        serp_page=1,
    )

    evaluation = await evaluator.evaluate(intent="find procurement data", candidate=candidate)

    assert evaluation.useful is False
    assert evaluation.outcome == "irrelevant"
    assert evaluation.tool_duplicate_signal.duplicate_detected is True
    assert "coresignal" in evaluation.tool_duplicate_signal.matched_tools
