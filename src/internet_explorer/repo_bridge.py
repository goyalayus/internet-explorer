from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from pydantic import BaseModel, Field

from internet_explorer.config import AppConfig


@contextmanager
def _prepend_sys_path(paths: list[Path]) -> Iterator[None]:
    resolved = [str(path.resolve()) for path in paths if path.exists()]
    for path in reversed(resolved):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield
    finally:
        for path in resolved:
            if path in sys.path:
                sys.path.remove(path)


class _FallbackSmartScraperPlan(BaseModel):
    planning_summary: str = Field(default="Fallback browser plan")
    browser_task: str = Field(default="Explore site and return strict JSON classification.")
    start_url: str
    max_steps: int = Field(default=25, ge=5, le=80)
    assumptions: list[str] = Field(default_factory=list)


def load_eu_swarm_modules(config: AppConfig) -> dict[str, object]:
    os.environ["AZURE_OPENAI_API_KEY"] = config.azure_openai_api_key
    os.environ["AZURE_OPENAI_ENDPOINT"] = config.azure_openai_endpoint
    os.environ["AZURE_OPENAI_API_VERSION"] = config.azure_openai_api_version
    os.environ["AZURE_OPENAI_DEPLOYMENT"] = config.azure_openai_model

    paths: list[Path] = []
    if config.eu_swarm_path is not None:
        paths.extend([config.eu_swarm_path, config.eu_swarm_path / "src"])
        browser_use_site_packages = _resolve_browser_use_site_packages(config.eu_swarm_path)
        if browser_use_site_packages is not None:
            paths.append(browser_use_site_packages)

    with _prepend_sys_path(paths):
        return _import_modules()


def _import_modules() -> dict[str, Any]:
    agent_module = importlib.import_module("smart_scraping_path_identifier.agent")
    create_agent = getattr(agent_module, "create_smart_scraping_path_identifier_agent", None)
    if create_agent is None:
        create_agent = getattr(agent_module, "create_agent", None)
    if create_agent is None:
        raise AttributeError("smart_scraping_path_identifier.agent does not export a compatible create agent function")

    smart_scraper_plan = getattr(agent_module, "SmartScraperPlan", None)
    if smart_scraper_plan is None:
        try:
            schemas_module = importlib.import_module("smart_scraping_path_identifier.schemas")
            smart_scraper_plan = getattr(schemas_module, "SmartScraperPlan", None)
        except Exception:
            smart_scraper_plan = None
    if smart_scraper_plan is None:
        smart_scraper_plan = _FallbackSmartScraperPlan

    return {
        "AzureOpenAIProvider": importlib.import_module("swarm.providers.azure_openai").AzureOpenAIProvider,
        "create_agent": create_agent,
        "SmartScraperPlan": smart_scraper_plan,
        "BrowserUseAgent": importlib.import_module("browser_use").Agent,
        "BrowserUseBrowser": importlib.import_module("browser_use").Browser,
        "get_browser_use_llm_by_name": importlib.import_module("browser_use.llm.models").get_llm_by_name,
    }


def _resolve_browser_use_site_packages(eu_swarm_path: Path) -> Path | None:
    venv_candidates = [
        eu_swarm_path / ".venv" / "lib",
        eu_swarm_path / ".venv" / "lib64",
        eu_swarm_path / ".venv311" / "lib",
        eu_swarm_path / ".venv311" / "lib64",
    ]
    for lib_dir in venv_candidates:
        if not lib_dir.exists():
            continue
        for site_packages in sorted(lib_dir.glob("python*/site-packages")):
            if (site_packages / "browser_use").exists():
                return site_packages
    return None
