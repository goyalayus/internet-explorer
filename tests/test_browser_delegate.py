from pathlib import Path

from pydantic import BaseModel

from internet_explorer.browser_delegate import BrowserDelegationManager
from internet_explorer.config import AppConfig


class _TelemetryStub:
    def timed(self) -> float:
        return 0.0

    def elapsed_ms(self, started_at: float) -> int:
        return 0

    def emit(self, **kwargs) -> None:
        return None


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


class _AzureOpenAIProvider:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _SmartScraperPlan(BaseModel):
    planning_summary: str
    browser_task: str
    start_url: str
    max_steps: int
    assumptions: list[str]


class _PlannerAgent:
    def __init__(self, captured: dict[str, object]) -> None:
        self.captured = captured

    def execute(self, task: str):
        self.captured["planning_task"] = task
        return {
            "planning_summary": "use native browser-use",
            "browser_task": "inspect source deeply and return JSON",
            "start_url": "https://example.com/start",
            "max_steps": 11,
            "assumptions": ["test assumption"],
        }


class _Browser:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.started = False


class _History:
    def final_result(self):
        return (
            '{"classification":"api_available","useful":true,'
            '"reasoning":"API docs found and the domain can be queried through documented OpenAPI endpoints.",'
            '"api_detected":true,"api_accessible_guess":true,'
            '"relevant_links":["https://example.com/openapi.json"],"confidence":0.92}'
        )

    def is_successful(self):
        return True

    def action_history(self):
        return [[{"navigate": {"url": "https://example.com/start"}}, {"click": {"index": 3}}]]

    def urls(self):
        return ["https://example.com/start", "https://example.com/docs"]

    def errors(self):
        return []

    def extracted_content(self):
        return ["API reference and OpenAPI schema listed"]


class _BrowserUseAgent:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def run(self, max_steps: int):
        return _History()


class _HistoryAliasShape:
    def final_result(self):
        return (
            '{"classification":"not_useful","useful":false,'
            '"reasoning":"No relevant data here.",'
            '"render_path":["https://example.com/start","https://example.com/about"],'
            '"source_evidence":["https://example.com/about"],'
            '"confidence":"high"}'
        )

    def is_successful(self):
        return True

    def action_history(self):
        return []

    def urls(self):
        return ["https://example.com/start"]

    def errors(self):
        return []

    def extracted_content(self):
        return []


class _BrowserUseAgentAliasShape:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    async def run(self, max_steps: int):
        return _HistoryAliasShape()


def test_browser_delegate_uses_planner_and_native_browser_use(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    captured: dict[str, object] = {}

    def _create_agent(**kwargs):
        captured["planner_create_kwargs"] = kwargs
        return _PlannerAgent(captured)

    def _llm_by_name(name: str):
        captured["llm_name"] = name
        return object()

    monkeypatch.setattr(
        "internet_explorer.browser_delegate.load_eu_swarm_modules",
        lambda config: {
            "AzureOpenAIProvider": _AzureOpenAIProvider,
            "create_agent": _create_agent,
            "SmartScraperPlan": _SmartScraperPlan,
            "BrowserUseAgent": _BrowserUseAgent,
            "BrowserUseBrowser": _Browser,
            "get_browser_use_llm_by_name": _llm_by_name,
        },
    )

    manager = BrowserDelegationManager(config, _TelemetryStub(), lambda update: captured.setdefault("run_updates", []).append(update))
    result = manager._run_delegate_sync(
        "session_test",
        "https://example.com/start",
        "find procurement api docs",
        "url_1",
        ["https://example.com/docs", "https://example.com/pricing"],
    )

    planning_task = str(captured["planning_task"])
    assert "Initial links list" in planning_task
    assert "https://example.com/docs" in planning_task
    assert "https://example.com/pricing" in planning_task

    assert result.classification == "api_available"
    assert result.useful is True
    assert "OpenAPI" in result.reasoning or "openapi" in result.reasoning.lower()
    assert result.api_detected is True
    assert result.api_accessible_guess is True
    assert "https://example.com/openapi.json" in result.relevant_links
    assert result.recipe[0].action == "navigate"
    assert result.recipe[1].action == "click"


def test_browser_delegate_prefers_gemini_when_available(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.gemini_api_key = "gemini-key"
    config.azure_openai_api_key = "azure-key"
    config.azure_openai_endpoint = "https://example.openai.azure.com"
    captured: dict[str, object] = {}

    def _llm_by_name(name: str):
        captured["llm_name"] = name
        return object()

    monkeypatch.setattr(
        "internet_explorer.browser_delegate.load_eu_swarm_modules",
        lambda config: {
            "AzureOpenAIProvider": _AzureOpenAIProvider,
            "create_agent": lambda **kwargs: _PlannerAgent({}),
            "SmartScraperPlan": _SmartScraperPlan,
            "BrowserUseAgent": _BrowserUseAgent,
            "BrowserUseBrowser": _Browser,
            "get_browser_use_llm_by_name": _llm_by_name,
        },
    )

    manager = BrowserDelegationManager(config, _TelemetryStub(), lambda update: None)
    manager._create_browser_use_llm()

    assert captured["llm_name"] == "google_gemini_2_5_flash"


def test_browser_delegate_uses_configured_browser_use_model(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.browser_use_llm_model = "openai_gpt_4o_mini"
    config.gemini_api_key = "gemini-key"
    config.azure_openai_api_key = "azure-key"
    config.azure_openai_endpoint = "https://example.openai.azure.com"
    captured: dict[str, object] = {}

    def _llm_by_name(name: str):
        captured["llm_name"] = name
        return object()

    monkeypatch.setattr(
        "internet_explorer.browser_delegate.load_eu_swarm_modules",
        lambda config: {
            "AzureOpenAIProvider": _AzureOpenAIProvider,
            "create_agent": lambda **kwargs: _PlannerAgent({}),
            "SmartScraperPlan": _SmartScraperPlan,
            "BrowserUseAgent": _BrowserUseAgent,
            "BrowserUseBrowser": _Browser,
            "get_browser_use_llm_by_name": _llm_by_name,
        },
    )

    manager = BrowserDelegationManager(config, _TelemetryStub(), lambda update: None)
    manager._create_browser_use_llm()

    assert captured["llm_name"] == "openai_gpt_4o_mini"


def test_browser_delegate_normalizes_browser_output_shapes(monkeypatch, tmp_path: Path) -> None:
    config = _config(tmp_path)

    monkeypatch.setattr(
        "internet_explorer.browser_delegate.load_eu_swarm_modules",
        lambda config: {
            "AzureOpenAIProvider": _AzureOpenAIProvider,
            "create_agent": lambda **kwargs: _PlannerAgent({}),
            "SmartScraperPlan": _SmartScraperPlan,
            "BrowserUseAgent": _BrowserUseAgentAliasShape,
            "BrowserUseBrowser": _Browser,
            "get_browser_use_llm_by_name": lambda name: object(),
        },
    )

    manager = BrowserDelegationManager(config, _TelemetryStub(), lambda update: None)
    result = manager._run_delegate_sync(
        "session_alias",
        "https://example.com/start",
        "find procurement api docs",
        "url_alias",
        ["https://example.com/about"],
    )

    assert result.classification == "irrelevant"
    assert result.render_path == "https://example.com/start -> https://example.com/about"
    assert result.source_evidence[0].kind == "page"
    assert result.source_evidence[0].url == "https://example.com/about"
    assert result.confidence == 0.9
