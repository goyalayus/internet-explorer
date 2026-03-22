import asyncio
from pathlib import Path

import pytest

from internet_explorer.config import AppConfig
from internet_explorer.evaluator import UrlEvaluator, _apply_quality_gates, _classify_scraping_path_quality
from internet_explorer.models import EvaluationDecision, FetchResult, SourceEvidenceItem, UrlCandidate
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
        decision_error: Exception | None = None,
    ) -> None:
        self.nav_calls = 0
        self.tool_terms = tool_terms
        self.decision_error = decision_error
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
        if name in {"_NavigationPlanEnvelope", "_NavigationRawPlanEnvelope"}:
            self.nav_calls += 1
            index = min(self.nav_calls - 1, len(self.nav_payloads) - 1)
            return schema.model_validate(self.nav_payloads[index])
        if name in {"_DecisionEnvelope", "_DecisionRawEnvelope"}:
            if self.decision_error is not None:
                raise self.decision_error
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
async def test_evaluator_seeds_tool_terms_when_llm_returns_empty_terms(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evaluator = UrlEvaluator(
        config,
        _LLMStub(tool_terms=[]),
        _FetcherStub(_responses()),
        _TelemetryStub(),
        _BrowserManagerStub(),
        tool_inventory=ToolInventory(["coresignal", "rapidapi", "builtwith"]),
    )
    candidate = _candidate("url_seeded", title="RapidAPI Marketplace Listing")

    evaluation = await evaluator.evaluate(intent="find procurement data", candidate=candidate)

    assert evaluation.tool_duplicate_signal.checked is True
    assert evaluation.tool_duplicate_signal.search_terms
    assert "rapidapi" in evaluation.tool_duplicate_signal.matched_tools
    assert evaluation.tool_duplicate_signal.duplicate_detected is True


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
async def test_evaluator_infers_outcome_when_useful_true_but_outcome_unknown(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evaluator = UrlEvaluator(
        config,
        _LLMStub(
            tool_terms=["newsource"],
            decision_payload={
                "useful": True,
                "reasoning": "This looks useful, but the model forgot to label the outcome.",
                "api_stage": "none",
            },
        ),
        _FetcherStub(_responses()),
        _TelemetryStub(),
        _BrowserManagerStub(),
        tool_inventory=ToolInventory(["coresignal", "rapidapi", "builtwith"]),
    )
    candidate = _candidate("url_shape_fix")

    evaluation = await evaluator.evaluate(intent="find procurement data", candidate=candidate)

    assert evaluation.useful is True
    assert evaluation.outcome == "data_on_site"
    assert "unknown_useful_outcome_inferred_from_evidence" in evaluation.notes
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
async def test_evaluator_normalizes_navigation_plan_shape(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evaluator = UrlEvaluator(
        config,
        _LLMStub(
            tool_terms=["newsource"],
            nav_payloads=[
                {
                    "reason": "Use the procurement page first.",
                    "next_action": "visit",
                    "url": "https://example.com/",
                    "notes": ["seed"],
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
    candidate = _candidate("url_nav_shape")

    evaluation = await evaluator.evaluate(intent="find procurement data", candidate=candidate)

    assert evaluation.useful is True
    assert evaluation.outcome == "data_on_site"
    assert len(evaluation.visited_memory) == 1
    assert evaluation.visited_memory[0].url == "https://example.com/"
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


@pytest.mark.asyncio
async def test_evaluator_uses_decision_fallback_when_llm_decision_fails(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evaluator = UrlEvaluator(
        config,
        _LLMStub(
            tool_terms=["newsource"],
            decision_error=ValueError("Could not parse JSON object from LLM response"),
        ),
        _FetcherStub(_responses()),
        _TelemetryStub(),
        _BrowserManagerStub(),
        tool_inventory=ToolInventory(["coresignal", "rapidapi", "builtwith"]),
    )
    candidate = _candidate("url_7")

    evaluation = await evaluator.evaluate(intent="find procurement data", candidate=candidate)

    assert evaluation.useful is False
    assert evaluation.outcome == "irrelevant"
    assert any(note.startswith("decision_fallback:") for note in evaluation.notes)
    assert "quality_gate:low_confidence_useful_demoted" in evaluation.notes
    assert all(not note.startswith("evaluation_error:") for note in evaluation.notes)


@pytest.mark.asyncio
async def test_evaluator_converts_internal_cancelled_error_to_unknown(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evaluator = UrlEvaluator(
        config,
        _LLMStub(
            tool_terms=["newsource"],
            decision_error=asyncio.CancelledError("agent task cancelled"),
        ),
        _FetcherStub(_responses()),
        _TelemetryStub(),
        _BrowserManagerStub(),
        tool_inventory=ToolInventory(["coresignal", "rapidapi", "builtwith"]),
    )
    candidate = _candidate("url_cancel")

    evaluation = await evaluator.evaluate(intent="find procurement data", candidate=candidate)

    assert evaluation.useful is False
    assert evaluation.outcome == "unknown"
    assert "evaluation_error:CancelledError" in evaluation.notes


def test_scraping_path_quality_classifier() -> None:
    good = _classify_scraping_path_quality(
        "Why useful: official procurement portal. Recurring path: navigate to bids section, filter by keyword, and monitor current tenders."
    )
    partial = _classify_scraping_path_quality(
        "Recurring path: use the procurement portal tenders feed for relevant notices."
    )
    weak = _classify_scraping_path_quality(
        "This is relevant and has useful information."
    )

    assert good == "good"
    assert partial == "partial"
    assert weak == "weak"


def test_quality_gate_clamps_weak_path_score() -> None:
    decision = EvaluationDecision(
        useful=True,
        relevance_score=0.91,
        outcome="data_on_site",
        reasoning="RFP notices are visible on the page.",
        api_stage="none",
        source_evidence=[
            SourceEvidenceItem(
                kind="page",
                url="https://example.com/rfp",
                summary="RFP listing snippet",
            )
        ],
        notes=[],
    )

    _apply_quality_gates(decision=decision)

    assert decision.useful is True
    assert decision.relevance_score <= 0.69
    assert "quality_gate:weak_scraping_path_score_clamped" in decision.notes


def test_quality_gate_demotes_weak_meta_path() -> None:
    decision = EvaluationDecision(
        useful=True,
        relevance_score=0.9,
        outcome="data_on_site",
        reasoning="This curated directory links to many procurement sources.",
        api_stage="none",
        source_evidence=[
            SourceEvidenceItem(
                kind="page",
                url="https://example.com/list",
                summary="Directory of procurement sources",
            )
        ],
        notes=[],
    )

    _apply_quality_gates(decision=decision)

    assert decision.useful is False
    assert decision.outcome == "irrelevant"
    assert "quality_gate:weak_scraping_path_meta_demoted" in decision.notes


def test_quality_gate_demotes_document_only_evidence() -> None:
    decision = EvaluationDecision(
        useful=True,
        relevance_score=0.9,
        outcome="data_on_site",
        reasoning="Why useful: report discusses annotation work. Recurring path: monitor this source.",
        api_stage="none",
        source_evidence=[
            SourceEvidenceItem(
                kind="pdf",
                url="https://example.com/report.pdf",
                summary="Procurement report",
            )
        ],
        notes=[],
    )

    _apply_quality_gates(decision=decision)

    assert decision.useful is False
    assert decision.outcome == "irrelevant"
    assert "quality_gate:document_only_evidence_demoted" in decision.notes


def test_quality_gate_demotes_low_confidence_useful() -> None:
    decision = EvaluationDecision(
        useful=True,
        relevance_score=0.7,
        outcome="data_on_site",
        reasoning="Why useful: related portal. Recurring path: search tenders by keyword in bids section.",
        api_stage="none",
        source_evidence=[
            SourceEvidenceItem(
                kind="page",
                url="https://example.com/bids",
                summary="Bids listing",
            )
        ],
        notes=[],
    )

    _apply_quality_gates(decision=decision)

    assert decision.useful is False
    assert decision.outcome == "irrelevant"
    assert "quality_gate:low_confidence_useful_demoted" in decision.notes


def test_quality_gate_demotes_off_domain_only_evidence() -> None:
    decision = EvaluationDecision(
        useful=True,
        relevance_score=0.9,
        outcome="data_on_site",
        reasoning=(
            "Why useful: this domain can lead to procurement results. "
            "Recurring path: use listings and filters."
        ),
        api_stage="none",
        source_evidence=[
            SourceEvidenceItem(
                kind="page",
                url="https://other-portal.example.org/rfps",
                summary="External procurement listing",
            ),
            SourceEvidenceItem(
                kind="pdf",
                url="https://external.example.org/tender.pdf",
                summary="Tender document",
            ),
        ],
        notes=[],
    )

    _apply_quality_gates(decision=decision, candidate_domain="example.com")

    assert decision.useful is False
    assert decision.outcome == "irrelevant"
    assert "quality_gate:off_domain_evidence_demoted" in decision.notes


def test_quality_gate_demotes_indirect_article_evidence_without_procurement_signals() -> None:
    decision = EvaluationDecision(
        useful=True,
        relevance_score=0.9,
        outcome="data_on_site",
        reasoning=(
            "Why useful: this research article references agency activity. "
            "Recurring path: monitor this source."
        ),
        api_stage="none",
        source_evidence=[
            SourceEvidenceItem(
                kind="page",
                url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12306506/",
                summary="Academic article about scientific workflows.",
            )
        ],
        notes=[],
    )

    _apply_quality_gates(decision=decision, candidate_domain="nih.gov")

    assert decision.useful is False
    assert decision.outcome == "irrelevant"
    assert "quality_gate:indirect_content_evidence_demoted" in decision.notes


def test_quality_gate_keeps_direct_procurement_pipeline_evidence() -> None:
    decision = EvaluationDecision(
        useful=True,
        relevance_score=0.9,
        outcome="data_on_site",
        reasoning=(
            "Why useful: official procurement source. "
            "Recurring path: use procurement pipeline listing and tender search."
        ),
        api_stage="none",
        source_evidence=[
            SourceEvidenceItem(
                kind="page",
                url="https://assets.publishing.service.gov.uk/media/pipeline/procurement-pipeline.xlsx",
                summary="UK procurement pipeline includes data annotation opportunities.",
            )
        ],
        notes=[],
    )

    _apply_quality_gates(decision=decision, candidate_domain="service.gov.uk")

    assert decision.useful is True
    assert decision.outcome == "data_on_site"
    assert "quality_gate:indirect_content_evidence_demoted" not in decision.notes


def test_quality_gate_demotes_indirect_candidate_page_without_surface_url() -> None:
    decision = EvaluationDecision(
        useful=True,
        relevance_score=0.9,
        outcome="data_on_site",
        reasoning=(
            "Why useful: this announcement mentions a solicitation. "
            "Recurring path: monitor this page."
        ),
        api_stage="none",
        source_evidence=[
            SourceEvidenceItem(
                kind="page",
                url="https://agency.gov/news/2026/03/solicitation-announcement",
                summary="Agency news update about solicitation plans.",
            )
        ],
        notes=[],
    )

    _apply_quality_gates(
        decision=decision,
        candidate_domain="agency.gov",
        candidate_canonical_url="https://agency.gov/news/2026/03/solicitation-announcement",
    )

    assert decision.useful is False
    assert decision.outcome == "irrelevant"
    assert "quality_gate:indirect_page_without_surface_demoted" in decision.notes


def test_quality_gate_keeps_indirect_candidate_when_surface_url_evidence_exists() -> None:
    decision = EvaluationDecision(
        useful=True,
        relevance_score=0.9,
        outcome="data_on_site",
        reasoning=(
            "Why useful: official source with clear recurring path. "
            "Recurring path: navigate to procurement bids list."
        ),
        api_stage="none",
        source_evidence=[
            SourceEvidenceItem(
                kind="page",
                url="https://agency.gov/news/2026/03/solicitation-announcement",
                summary="Announcement that links to procurement area.",
            ),
            SourceEvidenceItem(
                kind="page",
                url="https://agency.gov/procurement/bids/open",
                summary="Open bids listing page.",
            ),
        ],
        notes=[],
    )

    _apply_quality_gates(
        decision=decision,
        candidate_domain="agency.gov",
        candidate_canonical_url="https://agency.gov/news/2026/03/solicitation-announcement",
    )

    assert decision.useful is True
    assert decision.outcome == "data_on_site"
    assert "quality_gate:indirect_page_without_surface_demoted" not in decision.notes
