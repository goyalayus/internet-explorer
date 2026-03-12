from __future__ import annotations

import asyncio
import re
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

from pydantic import BaseModel, Field

from internet_explorer.canonicalize import canonical_domain, canonicalize_url, registrable_domain
from internet_explorer.config import AppConfig
from internet_explorer.fetcher import AsyncWebFetcher
from internet_explorer.models import (
    BrowserDelegateResult,
    PageEvidence,
    RenderProfile,
    SiteGraphEdge,
    SiteGraphNode,
    SiteGraphSnapshot,
    SiteNodeStatus,
)
from internet_explorer.telemetry import Telemetry

COMMON_SITEMAP_PATHS = ("/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml")
ROOT_DISCOVERY_PATHS = ("/robots.txt", "/llms.txt", "/llm.txt")
INTENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "for",
    "from",
    "how",
    "in",
    "into",
    "new",
    "of",
    "on",
    "or",
    "source",
    "sources",
    "that",
    "the",
    "this",
    "to",
    "with",
}
PAGE_TYPE_HINTS: list[tuple[str, tuple[str, ...]]] = [
    ("api_ref", ("openapi", "swagger", "graphql", "/api", "developer", "reference")),
    ("docs", ("docs", "documentation", "guide", "manual", "sdk", "reference")),
    ("dataset", ("dataset", "catalog", "portal", "data", "search-results")),
    ("rfp", ("rfp", "tender", "procurement", "bid", "solicitation")),
    ("pricing", ("pricing", "plans", "subscriptions")),
    ("contact", ("contact", "sales", "demo", "book-a-demo")),
    ("auth", ("login", "log-in", "signin", "sign-in", "signup", "register")),
]
PAGE_TYPE_WEIGHTS = {
    "api_ref": 5.0,
    "docs": 4.0,
    "dataset": 4.5,
    "rfp": 4.5,
    "pricing": 1.2,
    "contact": 1.5,
    "auth": 0.5,
}
DISCOVERY_BONUS = {
    "seed": 4.0,
    "root": 2.0,
    "sitemap": 2.0,
    "llms": 1.6,
    "robots": 1.2,
    "html_link": 1.0,
    "browser_click": 1.0,
}
SIGNAL_BONUS = {
    "api": 3.5,
    "api_docs": 2.0,
    "openapi": 3.0,
    "graphql": 2.5,
    "data": 2.5,
    "rfp": 3.0,
    "contact_sales": 1.0,
    "paywall": 0.8,
    "auth": 0.8,
    "captcha": -1.0,
}
ROBOTS_PATH_KEYWORDS = ("docs", "api", "data", "dataset", "rfp", "tender", "pricing", "contact", "developer", "reference")


class _TreeInput(BaseModel):
    max_nodes: int = 40


class _TreeOutput(BaseModel):
    tree: dict[str, Any]


class _FrontierInput(BaseModel):
    limit: int = 5


class _FrontierOutput(BaseModel):
    items: list[dict[str, Any]]


class _NodeInput(BaseModel):
    url: str


class _NodeOutput(BaseModel):
    node: dict[str, Any]


class _RecordPageInput(BaseModel):
    url: str
    title: str = ""
    page_type_guess: str = ""
    summary: str = ""
    signals: list[str] = Field(default_factory=list)
    status: SiteNodeStatus = "analyzed"
    relevant_links: list[str] = Field(default_factory=list)


class _RecordPageOutput(BaseModel):
    node: dict[str, Any]


class _AddLinksInput(BaseModel):
    from_url: str
    links: list[str] = Field(default_factory=list)
    discovered_via: str = "browser_click"


class _AddLinksOutput(BaseModel):
    added_urls: list[str]


class SiteGraph:
    def __init__(
        self,
        *,
        config: AppConfig,
        telemetry: Telemetry,
        url_id: str,
        intent: str,
        seed_url: str,
        domain: str,
    ) -> None:
        self.config = config
        self.telemetry = telemetry
        self.url_id = url_id
        self.intent = intent
        self.seed_url = canonicalize_url(seed_url)
        self.root_url = self._site_root(self.seed_url)
        self.domain = domain
        self.site_id = f"site_{canonical_domain(self.seed_url)}"
        self.intent_tokens = _intent_tokens(intent)
        self._nodes: dict[str, SiteGraphNode] = {}
        self._edges: dict[tuple[str, str], SiteGraphEdge] = {}
        self._bootstrap_sources: list[str] = []
        self._lock = threading.RLock()
        self._ensure_node(self.root_url, discovered_via="root", depth=0)
        self._ensure_node(self.seed_url, discovered_via="seed", depth=0)

    async def bootstrap(self, fetcher: AsyncWebFetcher) -> None:
        started = self.telemetry.timed()
        urls = [urljoin(self.root_url, path) for path in ROOT_DISCOVERY_PATHS]
        results = await asyncio.gather(*(fetcher.fetch(url) for url in urls), return_exceptions=True)
        sitemap_candidates: list[str] = []

        for path, result in zip(ROOT_DISCOVERY_PATHS, results):
            if isinstance(result, Exception):
                self._emit(
                    decision="bootstrap_fetch_failed",
                    output_summary={"path": path, "error": str(result)},
                    error_code=type(result).__name__,
                    latency_ms=self.telemetry.elapsed_ms(started),
                )
                continue
            if not result.status_code or result.status_code >= 400 or not result.body_text:
                continue
            self._note_bootstrap_source(path)
            if path == "/robots.txt":
                robots_sitemaps, robots_paths = _parse_robots(result.body_text, self.root_url)
                sitemap_candidates.extend(robots_sitemaps)
                for url in robots_paths:
                    self.add_url(url, discovered_via="robots", parent_url=self.root_url)
            else:
                for url in _parse_llm_manifest(result.body_text, self.root_url):
                    self.add_url(url, discovered_via="llms", parent_url=self.root_url)

        for path in COMMON_SITEMAP_PATHS:
            sitemap_candidates.append(urljoin(self.root_url, path))

        await self._bootstrap_sitemaps(fetcher, sitemap_candidates)
        self._emit(
            decision="bootstrapped",
            output_summary={
                "bootstrap_sources": self._bootstrap_sources,
                "node_count": len(self._nodes),
                "edge_count": len(self._edges),
            },
            latency_ms=self.telemetry.elapsed_ms(started),
        )

    def add_url(self, url: str, *, discovered_via: str, parent_url: str | None = None) -> str | None:
        with self._lock:
            parent_canonical = _safe_canonical_url(parent_url) if parent_url else ""
            source_depth = self._nodes.get(parent_canonical, SiteGraphNode(canonical_url="", depth=0)).depth if parent_canonical else 0
            depth = source_depth + 1 if parent_url else 0
            node = self._ensure_node(url, discovered_via=discovered_via, depth=depth)
            if node is None:
                return None
            if parent_url:
                self._add_edge(parent_url, node.canonical_url, discovered_via=discovered_via)
            node.priority_score = self._score_node(node)
            return node.canonical_url

    def add_links(self, from_url: str, links: list[str], *, discovered_via: str) -> list[str]:
        added: list[str] = []
        with self._lock:
            source = self._nodes.get(canonicalize_url(from_url))
            if source is None:
                source = self._ensure_node(from_url, discovered_via=discovered_via, depth=0)
            if source is None or source.depth >= self.config.max_link_depth:
                return added
            for link in links[: self.config.max_internal_links]:
                node = self._ensure_node(link, discovered_via=discovered_via, depth=source.depth + 1)
                if node is None:
                    continue
                self._add_edge(source.canonical_url, node.canonical_url, discovered_via=discovered_via)
                node.priority_score = self._score_node(node)
                added.append(node.canonical_url)
        if added:
            self._emit(
                decision="links_added",
                output_summary={"from_url": canonicalize_url(from_url), "added_urls": added[:20]},
            )
        return added

    def next_frontier(self, *, limit: int | None = None) -> list[SiteGraphNode]:
        with self._lock:
            frontier = [node for node in self._nodes.values() if node.status == "unvisited"]
            frontier.sort(key=lambda node: (-node.priority_score, node.depth, node.canonical_url))
            return [node.model_copy(deep=True) for node in frontier[: (limit or self.config.max_site_graph_frontier)]]

    def record_analysis(
        self,
        *,
        url: str,
        render_profile: RenderProfile,
        evidence: PageEvidence,
        status: SiteNodeStatus = "analyzed",
    ) -> SiteGraphNode:
        with self._lock:
            canonical = _safe_canonical_url(url)
            existing = self._nodes.get(canonical)
            node = self._ensure_node(url, discovered_via="html_link", depth=existing.depth if existing else 0)
            if node is None:
                raise ValueError(f"unable to record analysis for url={url}")
            node.title = evidence.title or node.title
            node.status = status
            node.last_render_profile = render_profile
            node.last_visited_at = datetime.now(timezone.utc)
            node.signals = _unique(node.signals + _signals_from_evidence(evidence))
            node.page_type_guess = self._infer_page_type(node.canonical_url, node.title, evidence.text_excerpt, node.signals, preferred=node.page_type_guess)
            node.summary = _build_summary(
                title=node.title,
                page_type=node.page_type_guess,
                render_profile=render_profile,
                evidence=evidence,
                signals=node.signals,
            )
            node.priority_score = self._score_node(node)
            return node.model_copy(deep=True)

    def record_browser_result(self, *, url: str, result: BrowserDelegateResult) -> SiteGraphNode:
        with self._lock:
            canonical = _safe_canonical_url(url)
            existing = self._nodes.get(canonical)
            node = self._ensure_node(url, discovered_via="browser_click", depth=existing.depth if existing else 0)
            if node is None:
                raise ValueError(f"unable to record browser result for url={url}")
            node.status = "delegated_browser"
            node.last_visited_at = datetime.now(timezone.utc)
            extra_signals = [signal for signal, present in (
                ("api", result.api_detected),
                ("data", result.data_on_site),
                ("contact_sales", result.contact_sales_only),
                ("paywall", result.paywall_present),
                ("auth", result.auth_required),
                ("captcha", result.captcha_present),
            ) if present]
            node.signals = _unique(node.signals + extra_signals)
            node.page_type_guess = self._infer_page_type(
                node.canonical_url,
                node.title,
                f"{result.why_useful} {result.how_to_use}",
                node.signals,
                preferred=_page_type_from_browser_classification(result.classification),
            )
            summary_bits = [result.why_useful.strip(), result.how_to_use.strip()]
            node.summary = " ".join(bit for bit in summary_bits if bit).strip() or node.summary
            node.priority_score = self._score_node(node)
        if result.relevant_links:
            self.add_links(url, result.relevant_links, discovered_via="browser_click")
        return node.model_copy(deep=True)

    def update_node_from_tool(
        self,
        *,
        url: str,
        title: str,
        page_type_guess: str,
        summary: str,
        signals: list[str],
        status: SiteNodeStatus,
        relevant_links: list[str],
    ) -> SiteGraphNode:
        with self._lock:
            canonical = _safe_canonical_url(url)
            existing = self._nodes.get(canonical)
            node = self._ensure_node(url, discovered_via="browser_click", depth=existing.depth if existing else 0)
            if node is None:
                raise ValueError(f"unable to update node for url={url}")
            if title:
                node.title = title
            if page_type_guess:
                node.page_type_guess = page_type_guess
            if summary:
                node.summary = summary.strip()
            if signals:
                node.signals = _unique(node.signals + [signal.strip() for signal in signals if signal.strip()])
            node.status = status
            node.last_visited_at = datetime.now(timezone.utc)
            node.priority_score = self._score_node(node)
        if relevant_links:
            self.add_links(url, relevant_links, discovered_via="browser_click")
        self._emit(
            decision="browser_tool_recorded_page",
            output_summary={"url": canonicalize_url(url), "status": status},
        )
        return node.model_copy(deep=True)

    def snapshot(self, *, max_nodes: int | None = None) -> SiteGraphSnapshot:
        with self._lock:
            nodes = sorted(self._nodes.values(), key=lambda node: (node.depth, -node.priority_score, node.canonical_url))
            limit = max_nodes or self.config.max_site_graph_nodes
            frontier = [node.canonical_url for node in self.next_frontier(limit=self.config.max_site_graph_frontier)]
            return SiteGraphSnapshot(
                site_id=self.site_id,
                root_url=self.root_url,
                domain=self.domain,
                bootstrap_sources=list(self._bootstrap_sources),
                nodes=[node.model_copy(deep=True) for node in nodes[:limit]],
                edges=[edge.model_copy(deep=True) for edge in sorted(self._edges.values(), key=lambda edge: (edge.from_url, edge.to_url))[:limit * 2]],
                frontier=frontier,
            )

    def prompt_context(self, *, max_nodes: int = 12) -> dict[str, Any]:
        snapshot = self.snapshot(max_nodes=max_nodes)
        visited_pages = [
            {
                "url": node.canonical_url,
                "title": node.title,
                "page_type_guess": node.page_type_guess,
                "status": node.status,
                "summary": node.summary,
                "signals": node.signals,
                "depth": node.depth,
            }
            for node in snapshot.nodes
            if node.status != "unvisited"
        ]
        frontier = [
            {
                "url": node.canonical_url,
                "page_type_guess": node.page_type_guess,
                "signals": node.signals,
                "priority_score": node.priority_score,
                "depth": node.depth,
            }
            for node in self.next_frontier(limit=min(self.config.max_site_graph_frontier, 5))
        ]
        return {
            "site_id": snapshot.site_id,
            "root_url": snapshot.root_url,
            "bootstrap_sources": snapshot.bootstrap_sources,
            "visited_pages": visited_pages,
            "frontier": frontier,
        }

    def build_browser_tools(self, tool_cls: type[Any]) -> list[Any]:
        return [
            tool_cls(
                name="sg_read_tree",
                description="Read the shared site graph snapshot and current node summaries.",
                fn=self._tool_read_tree,
                input_model=_TreeInput,
                output_model=_TreeOutput,
            ),
            tool_cls(
                name="sg_get_frontier",
                description="Get the highest-priority unvisited pages from the shared site graph.",
                fn=self._tool_get_frontier,
                input_model=_FrontierInput,
                output_model=_FrontierOutput,
            ),
            tool_cls(
                name="sg_get_node",
                description="Read the current stored node for a page in the shared site graph.",
                fn=self._tool_get_node,
                input_model=_NodeInput,
                output_model=_NodeOutput,
            ),
            tool_cls(
                name="sg_record_page",
                description="Store a concise summary of a page you explored so the shared site graph stays current.",
                fn=self._tool_record_page,
                input_model=_RecordPageInput,
                output_model=_RecordPageOutput,
            ),
            tool_cls(
                name="sg_add_links",
                description="Add newly discovered internal links to the shared site graph.",
                fn=self._tool_add_links,
                input_model=_AddLinksInput,
                output_model=_AddLinksOutput,
            ),
        ]

    async def _bootstrap_sitemaps(self, fetcher: AsyncWebFetcher, candidates: list[str]) -> None:
        queue = _unique(candidates)
        seen: set[str] = set()
        while queue and len(seen) < self.config.max_sitemap_fetches and len(self._nodes) < self.config.max_site_graph_nodes:
            sitemap_url = canonicalize_url(queue.pop(0))
            if sitemap_url in seen or registrable_domain(sitemap_url) != self.domain:
                continue
            seen.add(sitemap_url)
            try:
                result = await fetcher.fetch(sitemap_url)
            except Exception as exc:
                self._emit(
                    decision="sitemap_fetch_failed",
                    output_summary={"sitemap_url": sitemap_url, "error": str(exc)},
                    error_code=type(exc).__name__,
                )
                continue
            if not result.status_code or result.status_code >= 400 or not result.body_text:
                continue
            self._note_bootstrap_source(sitemap_url)
            urls, child_sitemaps = _parse_sitemap(result.body_text)
            for url in urls[: self.config.max_sitemap_urls]:
                if len(self._nodes) >= self.config.max_site_graph_nodes:
                    break
                self.add_url(url, discovered_via="sitemap", parent_url=self.root_url)
            for child in child_sitemaps:
                if child not in seen:
                    queue.append(child)

    def _ensure_node(self, url: str, *, discovered_via: str, depth: int) -> SiteGraphNode | None:
        canonical = _safe_canonical_url(url)
        if not canonical or registrable_domain(canonical) != self.domain:
            return None
        node = self._nodes.get(canonical)
        if node is None:
            if len(self._nodes) >= self.config.max_site_graph_nodes:
                return None
            page_type_guess = self._infer_page_type(canonical, "", "", [], preferred="")
            node = SiteGraphNode(
                canonical_url=canonical,
                page_type_guess=page_type_guess,
                discovered_via=[discovered_via],
                depth=depth,
                priority_score=0.0,
            )
            self._nodes[canonical] = node
        else:
            if discovered_via not in node.discovered_via:
                node.discovered_via.append(discovered_via)
            node.depth = min(node.depth, depth)
        node.priority_score = self._score_node(node)
        return node

    def _add_edge(self, from_url: str, to_url: str, *, discovered_via: str) -> None:
        source = _safe_canonical_url(from_url)
        target = _safe_canonical_url(to_url)
        if not source or not target or source == target:
            return
        key = (source, target)
        edge = self._edges.get(key)
        if edge is None:
            self._edges[key] = SiteGraphEdge(from_url=source, to_url=target, discovered_via=discovered_via)

    def _infer_page_type(
        self,
        canonical_url: str,
        title: str,
        summary_text: str,
        signals: list[str],
        *,
        preferred: str,
    ) -> str:
        if preferred:
            return preferred
        haystack = " ".join([canonical_url, title, summary_text, " ".join(signals)]).lower()
        for page_type, hints in PAGE_TYPE_HINTS:
            if any(hint in haystack for hint in hints):
                return page_type
        return ""

    def _score_node(self, node: SiteGraphNode) -> float:
        score = 0.0
        score += PAGE_TYPE_WEIGHTS.get(node.page_type_guess, 0.0)
        score += sum(DISCOVERY_BONUS.get(source, 0.4) for source in node.discovered_via)
        score += sum(SIGNAL_BONUS.get(signal, 0.3) for signal in node.signals)

        text = " ".join([node.canonical_url, node.title, node.summary]).lower()
        overlap = [token for token in self.intent_tokens if token in text]
        score += len(overlap) * 1.25

        if node.status != "unvisited":
            score -= 0.5
        score -= node.depth * 0.3
        return round(score, 3)

    def _site_root(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/"

    def _note_bootstrap_source(self, source: str) -> None:
        if source not in self._bootstrap_sources:
            self._bootstrap_sources.append(source)

    def _emit(
        self,
        *,
        decision: str,
        output_summary: Any,
        error_code: str | None = None,
        latency_ms: int | None = None,
    ) -> None:
        self.telemetry.emit(
            phase="site_graph",
            actor="system",
            url_id=self.url_id,
            output_summary=output_summary,
            decision=decision,
            error_code=error_code,
            latency_ms=latency_ms,
        )

    def _tool_read_tree(self, max_nodes: int = 40) -> dict[str, Any]:
        tree = self.snapshot(max_nodes=max_nodes).model_dump(mode="json")
        self.telemetry.emit(
            phase="site_graph_tool",
            actor="browser_agent",
            url_id=self.url_id,
            input_payload={"max_nodes": max_nodes},
            output_summary={"node_count": len(tree.get("nodes", []))},
            decision="sg_read_tree",
        )
        return {"tree": tree}

    def _tool_get_frontier(self, limit: int = 5) -> dict[str, Any]:
        items = [node.model_dump(mode="json") for node in self.next_frontier(limit=limit)]
        self.telemetry.emit(
            phase="site_graph_tool",
            actor="browser_agent",
            url_id=self.url_id,
            input_payload={"limit": limit},
            output_summary={"frontier_count": len(items)},
            decision="sg_get_frontier",
        )
        return {"items": items}

    def _tool_get_node(self, url: str) -> dict[str, Any]:
        canonical = _safe_canonical_url(url)
        with self._lock:
            node = self._nodes.get(canonical)
        payload = node.model_dump(mode="json") if node else {}
        self.telemetry.emit(
            phase="site_graph_tool",
            actor="browser_agent",
            url_id=self.url_id,
            input_payload={"url": url},
            output_summary={"found": bool(payload)},
            decision="sg_get_node",
        )
        return {"node": payload}

    def _tool_record_page(
        self,
        url: str,
        title: str = "",
        page_type_guess: str = "",
        summary: str = "",
        signals: list[str] | None = None,
        status: SiteNodeStatus = "analyzed",
        relevant_links: list[str] | None = None,
    ) -> dict[str, Any]:
        node = self.update_node_from_tool(
            url=url,
            title=title,
            page_type_guess=page_type_guess,
            summary=summary,
            signals=signals or [],
            status=status,
            relevant_links=relevant_links or [],
        )
        return {"node": node.model_dump(mode="json")}

    def _tool_add_links(self, from_url: str, links: list[str] | None = None, discovered_via: str = "browser_click") -> dict[str, Any]:
        added_urls = self.add_links(from_url, links or [], discovered_via=discovered_via)
        self.telemetry.emit(
            phase="site_graph_tool",
            actor="browser_agent",
            url_id=self.url_id,
            input_payload={"from_url": from_url, "count": len(links or [])},
            output_summary={"added_urls": added_urls[:20]},
            decision="sg_add_links",
        )
        return {"added_urls": added_urls}


def _safe_canonical_url(url: str | None) -> str:
    if not url:
        return ""
    try:
        return canonicalize_url(url)
    except Exception:
        return ""


def _intent_tokens(intent: str) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]{3,}", intent.lower())
        if token not in INTENT_STOPWORDS
    }
    return set(sorted(tokens)[:20])


def _parse_robots(text: str, root_url: str) -> tuple[list[str], list[str]]:
    sitemaps: list[str] = []
    paths: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if key == "sitemap":
            sitemaps.append(urljoin(root_url, value))
            continue
        if key not in {"allow", "disallow"}:
            continue
        if not value.startswith("/") or "*" in value or "$" in value or len(value) <= 1:
            continue
        lowered = value.lower()
        if any(keyword in lowered for keyword in ROBOTS_PATH_KEYWORDS):
            paths.append(urljoin(root_url, value))
    return _unique(sitemaps), _unique(paths)


def _parse_llm_manifest(text: str, root_url: str) -> list[str]:
    urls: list[str] = []
    for match in re.findall(r"https?://[^\s)>\"]+", text):
        urls.append(match)
    for match in re.findall(r"\((/[^)\s]+)\)", text):
        urls.append(urljoin(root_url, match))
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("/") and " " not in line:
            urls.append(urljoin(root_url, line))
    return _unique(urls)


def _parse_sitemap(text: str) -> tuple[list[str], list[str]]:
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return [], []
    tag = _strip_namespace(root.tag)
    urls: list[str] = []
    child_sitemaps: list[str] = []
    if tag == "urlset":
        for child in root:
            if _strip_namespace(child.tag) != "url":
                continue
            loc = _find_xml_text(child, "loc")
            if loc:
                urls.append(loc)
    elif tag == "sitemapindex":
        for child in root:
            if _strip_namespace(child.tag) != "sitemap":
                continue
            loc = _find_xml_text(child, "loc")
            if loc:
                child_sitemaps.append(loc)
    return _unique(urls), _unique(child_sitemaps)


def _find_xml_text(node: ET.Element, tag_name: str) -> str:
    for child in node:
        if _strip_namespace(child.tag) == tag_name and child.text:
            return child.text.strip()
    return ""


def _strip_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _signals_from_evidence(evidence: PageEvidence) -> list[str]:
    signals: list[str] = []
    if evidence.api_signal.detected:
        signals.append("api")
    if evidence.api_signal.doc_links:
        signals.append("api_docs")
    if evidence.api_signal.openapi_links:
        signals.append("openapi")
    if evidence.api_signal.graphql_hints:
        signals.append("graphql")
    if evidence.contact_sales_present:
        signals.append("contact_sales")
    if evidence.paywall_present:
        signals.append("paywall")
    if evidence.auth_required:
        signals.append("auth")
    if evidence.captcha_present:
        signals.append("captcha")
    for data_signal in evidence.data_signals:
        lowered = data_signal.lower()
        if "rfp" in lowered or "tender" in lowered or "bid" in lowered or "procurement" in lowered:
            signals.append("rfp")
        else:
            signals.append("data")
    return _unique(signals)


def _page_type_from_browser_classification(classification: str) -> str:
    return {
        "api_available": "api_ref",
        "data_on_site": "dataset",
        "contact_sales_only": "contact",
        "paywall": "pricing",
    }.get(classification, "")


def _build_summary(
    *,
    title: str,
    page_type: str,
    render_profile: RenderProfile,
    evidence: PageEvidence,
    signals: list[str],
) -> str:
    lead = title.strip() if title else ""
    body = _first_sentences(evidence.text_excerpt, limit=220)
    parts = [part for part in (lead, body) if part]
    if page_type:
        parts.append(f"type={page_type}")
    if signals:
        parts.append(f"signals={', '.join(signals[:5])}")
    parts.append(f"render={render_profile}")
    return " | ".join(parts)[:500]


def _first_sentences(text: str, *, limit: int) -> str:
    cleaned = " ".join(text.split())
    if not cleaned:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    output = ""
    for sentence in sentences:
        candidate = (output + " " + sentence).strip()
        if len(candidate) > limit:
            break
        output = candidate
        if len(output) >= limit * 0.7:
            break
    return output or cleaned[:limit]


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        value = item.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
