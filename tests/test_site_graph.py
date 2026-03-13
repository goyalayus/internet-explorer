from pathlib import Path

import pytest

from internet_explorer.config import AppConfig
from internet_explorer.models import ApiSignal, FetchResult, PageEvidence
from internet_explorer.site_graph import SiteGraph


class _TelemetryStub:
    def timed(self) -> float:
        return 0.0

    def elapsed_ms(self, started_at: float) -> int:
        return 0

    def emit(self, **kwargs) -> None:
        return None


class _FetcherStub:
    def __init__(self, responses: dict[str, FetchResult]) -> None:
        self.responses = responses

    async def fetch(self, url: str) -> FetchResult:
        if url not in self.responses:
            raise AssertionError(f"unexpected fetch for {url}")
        return self.responses[url]


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
        max_site_graph_visits=4,
        max_site_graph_nodes=30,
        max_site_graph_frontier=6,
        max_sitemap_urls=20,
        max_sitemap_fetches=3,
    )


@pytest.mark.asyncio
async def test_site_graph_bootstrap_collects_structure_sources(tmp_path: Path) -> None:
    config = _config(tmp_path)
    graph = SiteGraph(
        config=config,
        telemetry=_TelemetryStub(),
        url_id="url_1",
        intent="find procurement datasets and api docs",
        seed_url="https://example.com/start",
        domain="example.com",
    )
    fetcher = _FetcherStub(
        {
            "https://example.com/robots.txt": FetchResult(
                url="https://example.com/robots.txt",
                final_url="https://example.com/robots.txt",
                status_code=200,
                content_type="text/plain",
                body_text="Sitemap: https://example.com/sitemap.xml\nAllow: /docs\nDisallow: /pricing\n",
                text_excerpt="",
            ),
            "https://example.com/llms.txt": FetchResult(
                url="https://example.com/llms.txt",
                final_url="https://example.com/llms.txt",
                status_code=200,
                content_type="text/plain",
                body_text="- API reference (/developers/api)\n- Datasets (https://example.com/data/catalog)\n",
                text_excerpt="",
            ),
            "https://example.com/llm.txt": FetchResult(
                url="https://example.com/llm.txt",
                final_url="https://example.com/llm.txt",
                status_code=404,
                content_type="text/plain",
                body_text="",
                text_excerpt="",
            ),
            "https://example.com/sitemap.xml": FetchResult(
                url="https://example.com/sitemap.xml",
                final_url="https://example.com/sitemap.xml",
                status_code=200,
                content_type="application/xml",
                body_text=(
                    "<urlset>"
                    "<url><loc>https://example.com/rfp/current</loc></url>"
                    "<url><loc>https://example.com/pricing</loc></url>"
                    "</urlset>"
                ),
                text_excerpt="",
            ),
            "https://example.com/sitemap_index.xml": FetchResult(
                url="https://example.com/sitemap_index.xml",
                final_url="https://example.com/sitemap_index.xml",
                status_code=404,
                content_type="application/xml",
                body_text="",
                text_excerpt="",
            ),
            "https://example.com/sitemap-index.xml": FetchResult(
                url="https://example.com/sitemap-index.xml",
                final_url="https://example.com/sitemap-index.xml",
                status_code=404,
                content_type="application/xml",
                body_text="",
                text_excerpt="",
            ),
        }
    )

    await graph.bootstrap(fetcher)
    snapshot = graph.snapshot()
    urls = {node.canonical_url for node in snapshot.nodes}

    assert "https://example.com/start" in urls
    assert "https://example.com/docs" in urls
    assert "https://example.com/developers/api" in urls
    assert "https://example.com/data/catalog" in urls
    assert "https://example.com/rfp/current" in urls
    assert "/robots.txt" in snapshot.bootstrap_sources
    assert "/llms.txt" in snapshot.bootstrap_sources


def test_site_graph_record_analysis_stores_summary_and_signals(tmp_path: Path) -> None:
    config = _config(tmp_path)
    graph = SiteGraph(
        config=config,
        telemetry=_TelemetryStub(),
        url_id="url_2",
        intent="find procurement api docs",
        seed_url="https://example.com/start",
        domain="example.com",
    )
    node = graph.record_analysis(
        url="https://example.com/developers/api",
        render_profile="static_ssr",
        evidence=PageEvidence(
            url="https://example.com/developers/api",
            title="Developer API Reference",
            text_excerpt="Browse the procurement API reference and dataset endpoints for tenders and bids.",
            api_signal=ApiSignal(
                detected=True,
                doc_links=["https://example.com/developers/api"],
                openapi_links=["https://example.com/openapi.json"],
            ),
            data_signals=["procurement", "bid notice"],
        ),
    )

    assert node.status == "analyzed"
    assert "api" in node.signals
    assert "openapi" in node.signals
    assert node.page_type_guess == "api_ref"
    assert "Developer API Reference" in node.summary


def test_site_graph_frontier_prioritizes_relevant_pages(tmp_path: Path) -> None:
    config = _config(tmp_path)
    graph = SiteGraph(
        config=config,
        telemetry=_TelemetryStub(),
        url_id="url_3",
        intent="find api and rfp data",
        seed_url="https://example.com/start",
        domain="example.com",
    )
    graph.add_links(
        "https://example.com/start",
        [
            "https://example.com/pricing",
            "https://example.com/docs/api",
            "https://example.com/rfp/current",
        ],
        discovered_via="html_link",
    )
    graph.record_analysis(
        url="https://example.com/start",
        render_profile="static_ssr",
        evidence=PageEvidence(
            url="https://example.com/start",
            title="Home",
            text_excerpt="General company homepage.",
        ),
    )

    frontier = graph.next_frontier(limit=3)
    frontier_urls = {node.canonical_url for node in frontier}

    assert frontier[0].canonical_url in {"https://example.com/rfp/current", "https://example.com/docs/api"}
    assert "https://example.com/docs/api" in frontier_urls
    assert "https://example.com/rfp/current" in frontier_urls


def test_site_graph_allows_browser_added_links_from_direct_deep_page(tmp_path: Path) -> None:
    config = _config(tmp_path)
    graph = SiteGraph(
        config=config,
        telemetry=_TelemetryStub(),
        url_id="url_4",
        intent="find api docs",
        seed_url="https://example.com/",
        domain="example.com",
    )

    deep_url = "https://example.com/docs/v1/hi/okay.html"
    child_url = "https://example.com/docs/v1/openapi.json"

    added = graph.add_links(
        deep_url,
        [child_url],
        discovered_via="browser_click",
    )

    snapshot = graph.snapshot()
    urls = {node.canonical_url for node in snapshot.nodes}
    edges = {(edge.from_url, edge.to_url) for edge in snapshot.edges}

    assert added == [child_url]
    assert deep_url in urls
    assert child_url in urls
    assert (deep_url, child_url) in edges
