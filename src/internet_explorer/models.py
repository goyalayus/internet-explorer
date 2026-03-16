from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


OutcomeType = Literal[
    "data_on_site",
    "api_available",
    "contact_sales_only",
    "paywall",
    "irrelevant",
    "unknown",
]
RenderProfile = Literal["static_ssr", "hybrid", "csr_shell"]
SiteNodeStatus = Literal["unvisited", "fetched", "analyzed", "delegated_browser", "failed"]
CandidateStartMode = Literal["domain_homepage", "first_result_url"]
CandidateContentKind = Literal["html_page", "pdf", "binary_file", "unknown"]
EvidenceKind = Literal["page", "pdf", "api", "dataset", "contact_sales", "paywall", "browser_finding", "file"]


class Strategy(BaseModel):
    strategy_id: str
    title: str
    concept: str


class QueryPlan(BaseModel):
    query_id: str
    strategy_id: str
    query: str


class SearchResult(BaseModel):
    query_id: str
    strategy_id: str
    rank: int
    serp_page: int
    title: str = ""
    snippet: str = ""
    url: str


class UrlCandidate(BaseModel):
    url_id: str
    strategy_id: str
    query_id: str
    raw_url: str
    canonical_url: str
    start_url: str
    homepage_url: str
    domain: str
    novelty: bool
    start_mode: CandidateStartMode = "domain_homepage"
    source_title: str = ""
    source_snippet: str = ""
    serp_rank: int
    serp_page: int
    content_kind_hint: CandidateContentKind = "unknown"


class FetchResult(BaseModel):
    url: str
    final_url: str
    status_code: int | None = None
    content_type: str = ""
    content_disposition: str = ""
    content_length: int = 0
    is_binary: bool = False
    html: str = ""
    body_text: str = ""
    text_excerpt: str = ""
    headers: dict[str, str] = Field(default_factory=dict)


class BinaryFetchResult(BaseModel):
    url: str
    final_url: str
    status_code: int | None = None
    content_type: str = ""
    content_disposition: str = ""
    content_length: int = 0
    content_bytes: bytes = b""
    headers: dict[str, str] = Field(default_factory=dict)


class ApiSignal(BaseModel):
    detected: bool = False
    doc_links: list[str] = Field(default_factory=list)
    openapi_links: list[str] = Field(default_factory=list)
    graphql_hints: list[str] = Field(default_factory=list)
    auth_required: bool = False


class ApiProbeResult(BaseModel):
    attempted: bool = False
    command: list[str] = Field(default_factory=list)
    shell_command: str = ""
    url: str = ""
    status_code: int | None = None
    content_type: str = ""
    success: bool = False
    accessible: bool = False
    relevant_guess: bool = False
    viable_guess: bool = False
    planner_reason: str = ""
    planner_fallback_used: bool = False
    response_excerpt: str = ""
    error: str = ""


class PageEvidence(BaseModel):
    url: str
    title: str = ""
    content_type: str = ""
    content_kind: CandidateContentKind = "html_page"
    text_excerpt: str = ""
    html_excerpt: str = ""
    relevant_links: list[str] = Field(default_factory=list)
    api_signal: ApiSignal = Field(default_factory=ApiSignal)
    paywall_present: bool = False
    contact_sales_present: bool = False
    auth_required: bool = False
    captcha_present: bool = False
    data_signals: list[str] = Field(default_factory=list)


class SourceEvidenceItem(BaseModel):
    kind: EvidenceKind
    url: str
    title: str = ""
    summary: str = ""


class PdfVerificationResult(BaseModel):
    url: str
    final_url: str = ""
    title: str = ""
    status_code: int | None = None
    content_type: str = "application/pdf"
    relevant: bool = False
    reasoning: str = ""
    summary: str = ""
    extracted_signals: list[str] = Field(default_factory=list)
    fallback_urls: list[str] = Field(default_factory=list)
    source_evidence: SourceEvidenceItem | None = None
    error: str = ""


class BrowserStep(BaseModel):
    step_no: int
    action: str
    params: dict[str, Any] = Field(default_factory=dict)
    observations: dict[str, Any] = Field(default_factory=dict)


class BrowserDelegateResult(BaseModel):
    session_name: str
    classification: OutcomeType = "unknown"
    useful: bool = False
    reasoning: str = ""
    render_path: str = ""
    data_on_site: bool = False
    api_detected: bool = False
    api_accessible_guess: bool = False
    contact_sales_only: bool = False
    paywall_present: bool = False
    auth_required: bool = False
    captcha_present: bool = False
    relevant_links: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    source_evidence: list[SourceEvidenceItem] = Field(default_factory=list)
    recipe: list[BrowserStep] = Field(default_factory=list)
    confidence: float = 0.0
    raw_output: dict[str, Any] = Field(default_factory=dict)


class SiteGraphNode(BaseModel):
    canonical_url: str
    title: str = ""
    page_type_guess: str = ""
    discovered_via: list[str] = Field(default_factory=list)
    status: SiteNodeStatus = "unvisited"
    summary: str = ""
    signals: list[str] = Field(default_factory=list)
    depth: int = 0
    priority_score: float = 0.0
    last_render_profile: str = "unknown"
    last_visited_at: datetime | None = None


class SiteGraphEdge(BaseModel):
    from_url: str
    to_url: str
    discovered_via: str


class SiteGraphSnapshot(BaseModel):
    site_id: str
    root_url: str
    domain: str
    bootstrap_sources: list[str] = Field(default_factory=list)
    nodes: list[SiteGraphNode] = Field(default_factory=list)
    edges: list[SiteGraphEdge] = Field(default_factory=list)
    frontier: list[str] = Field(default_factory=list)


class NavigationMemoryEntry(BaseModel):
    step_no: int
    url: str
    summary: str


class ToolDuplicateSignal(BaseModel):
    checked: bool = False
    search_terms: list[str] = Field(default_factory=list)
    matched_tools: list[str] = Field(default_factory=list)
    duplicate_detected: bool = False
    reason: str = ""


class UrlEvaluation(BaseModel):
    url_id: str
    canonical_url: str
    start_url: str = ""
    homepage_url: str = ""
    domain: str
    novelty: bool
    render_profile: RenderProfile
    outcome: OutcomeType
    useful: bool
    relevance_score: float = 0.0
    reasoning: str = ""
    api_stage: Literal["none", "api_detected", "api_accessible", "api_relevant", "api_viable"] = "none"
    browser_delegated: bool = False
    data_on_site: bool = False
    api_signal: ApiSignal = Field(default_factory=ApiSignal)
    api_probe: ApiProbeResult | None = None
    contact_sales_only: bool = False
    paywall_present: bool = False
    auth_required: bool = False
    captcha_present: bool = False
    evidence: list[PageEvidence] = Field(default_factory=list)
    source_evidence: list[SourceEvidenceItem] = Field(default_factory=list)
    site_graph: SiteGraphSnapshot | None = None
    visited_memory: list[NavigationMemoryEntry] = Field(default_factory=list)
    tool_duplicate_signal: ToolDuplicateSignal = Field(default_factory=ToolDuplicateSignal)
    browser_result: BrowserDelegateResult | None = None
    notes: list[str] = Field(default_factory=list)


class RunSummary(BaseModel):
    run_id: str
    intent: str
    started_at: datetime
    finished_at: datetime | None = None
    strategy_count: int = 0
    query_count: int = 0
    raw_result_count: int = 0
    unique_url_count: int = 0
    evaluated_url_count: int = 0
    useful_url_count: int = 0
    unique_source_count: int = 0
    evaluated_source_count: int = 0
    useful_source_count: int = 0
    browser_peak_active: int = 0
    status: Literal["running", "completed", "failed"] = "running"
    error: str | None = None


class StrategyResponse(BaseModel):
    strategies: list[Strategy]


class QueryResponse(BaseModel):
    queries: list[str]


class EvaluationDecision(BaseModel):
    useful: bool
    relevance_score: float
    outcome: OutcomeType
    reasoning: str
    api_stage: Literal["none", "api_detected", "api_accessible", "api_relevant", "api_viable"]
    source_evidence: list[SourceEvidenceItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
