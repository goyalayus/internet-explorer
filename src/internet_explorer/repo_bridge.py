from __future__ import annotations

import importlib
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from internet_explorer.config import AppConfig


@contextmanager
def _prepend_sys_path(paths: list[Path]) -> Iterator[None]:
    resolved = [str(path.resolve()) for path in paths]
    for path in reversed(resolved):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield
    finally:
        for path in resolved:
            if path in sys.path:
                sys.path.remove(path)


def load_google_search_client(config: AppConfig):
    os.environ["GOOGLE_API_KEY"] = config.google_api_key
    os.environ["GOOGLE_API_KEYS"] = config.google_api_keys or config.google_api_key
    os.environ["GOOGLE_SEARCH_ENGINE_ID"] = config.google_search_engine_id
    with _prepend_sys_path([config.query_optimizer_repo_path]):
        module = importlib.import_module("utils.google_search")
        return module.GoogleCustomSearch()


def load_eu_swarm_modules(config: AppConfig) -> dict[str, object]:
    os.environ["AZURE_OPENAI_API_KEY"] = config.azure_openai_api_key
    os.environ["AZURE_OPENAI_ENDPOINT"] = config.azure_openai_endpoint
    os.environ["AZURE_OPENAI_API_VERSION"] = config.azure_openai_api_version
    os.environ["AZURE_OPENAI_DEPLOYMENT"] = config.azure_openai_model
    browser_use_site_packages = _resolve_browser_use_site_packages(config.eu_swarm_path)
    paths = [config.eu_swarm_path, config.eu_swarm_path / "src"]
    if browser_use_site_packages is not None:
        paths.append(browser_use_site_packages)
    with _prepend_sys_path(paths):
        return {
            "AzureOpenAIProvider": importlib.import_module("swarm.providers.azure_openai").AzureOpenAIProvider,
            "create_agent": importlib.import_module("smart_scraping_path_identifier.agent").create_smart_scraping_path_identifier_agent,
            "SmartScraperPlan": importlib.import_module("smart_scraping_path_identifier.agent").SmartScraperPlan,
            "BrowserUseAgent": importlib.import_module("browser_use").Agent,
            "BrowserUseBrowser": importlib.import_module("browser_use").Browser,
            "get_browser_use_llm_by_name": importlib.import_module("browser_use.llm.models").get_llm_by_name,
        }


def _resolve_browser_use_site_packages(eu_swarm_path: Path) -> Path | None:
    venv_candidates = [
        eu_swarm_path / ".venv" / "lib",
        eu_swarm_path / ".venv311" / "lib",
    ]
    for lib_dir in venv_candidates:
        if not lib_dir.exists():
            continue
        for site_packages in sorted(lib_dir.glob("python*/site-packages")):
            if (site_packages / "browser_use").exists():
                return site_packages
    return None
