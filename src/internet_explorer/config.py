from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, Field, field_validator


def _parse_shell_default(script_path: Path | None, variable_name: str) -> str | None:
    if script_path is None or not script_path.exists():
        return None
    text = script_path.read_text()
    pattern = re.compile(rf'{re.escape(variable_name)}="\$\{{{re.escape(variable_name)}:-([^}}]+)\}}"')
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()


class AppConfig(BaseModel):
    workspace_root: Path
    intent: str = ""
    mongodb_uri: str
    mongodb_db: str = "web_crawler"
    mongodb_runs_collection: str = "ie_runs"
    mongodb_url_summaries_collection: str = "ie_url_summaries"
    mongodb_events_collection: str = "ie_events"
    google_api_key: str = ""
    google_api_keys: str = ""
    google_search_engine_id: str = ""
    searchapi_api_key: str = ""
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-12-01-preview"
    azure_openai_model: str = "gpt-4.1"
    eu_swarm_path: Path
    query_optimizer_repo_path: Path
    tool_flow_path: Path
    browser_mode: str = "chromium"
    browser_headed: bool = False
    temp_google_email: str = ""
    temp_google_password: str = ""
    temp_signup_email: str = ""
    temp_signup_password: str = ""
    baseline_domains_file: Path
    max_browser_concurrency: int = 0
    max_url_concurrency: int = 0
    vpn_start_script: Path | None = None
    auto_start_vpn: bool = True
    query_optimizer_ovpn_config: Path | None = None
    vpn_docdb_host: str = ""
    vpn_docdb_port: int = 27017
    vpn_require_split_tunnel: bool = True
    vpn_require_docdb_reachable: bool = True
    vpn_log_dir: Path
    strategy_count: int = 10
    queries_per_strategy: int = 5
    serp_pages_per_query: int = 2
    results_per_serp_page: int = 10
    max_internal_links: int = 12
    max_link_depth: int = 1
    max_site_graph_visits: int = 6
    max_site_graph_nodes: int = 80
    max_site_graph_frontier: int = 8
    max_sitemap_urls: int = 150
    max_sitemap_fetches: int = 6
    request_timeout_seconds: int = 20
    llm_max_retries: int = 2
    tool_flow_env_path: Path
    discovered_vpn_scripts: list[Path] = Field(default_factory=list)

    @field_validator("workspace_root", "eu_swarm_path", "query_optimizer_repo_path", "tool_flow_path", "baseline_domains_file", "tool_flow_env_path", "vpn_log_dir")
    @classmethod
    def _expand_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @classmethod
    def from_env(cls, root: Path | None = None) -> "AppConfig":
        root_dir = (root or Path.cwd()).resolve()
        tool_flow_path = (root_dir / "../tool-flow").resolve()
        query_optimizer_repo_path = (root_dir / "../query_optimizer_repo").resolve()
        eu_swarm_path = (root_dir / "../eu-swarm").resolve()
        tool_flow_env_path = (root_dir / ".env").resolve()
        local_env = dotenv_values(tool_flow_env_path) if tool_flow_env_path.exists() else {}

        def env_value(name: str, default: str = "") -> str:
            return (os.getenv(name) or str(local_env.get(name) or default)).strip()

        discovered_vpn_scripts = sorted((tool_flow_path / "scripts").glob("vpn_and_run_*.sh"))
        explicit_vpn = env_value("VPN_START_SCRIPT")
        vpn_start_script = Path(explicit_vpn).expanduser().resolve() if explicit_vpn else None
        if vpn_start_script and not vpn_start_script.exists():
            vpn_start_script = None
        reference_vpn_script = vpn_start_script or (discovered_vpn_scripts[0].resolve() if discovered_vpn_scripts else None)

        query_optimizer_ovpn = env_value("QUERY_OPTIMIZER_OVPN_CONFIG")
        if query_optimizer_ovpn:
            ovpn_path = Path(query_optimizer_ovpn).expanduser().resolve()
        else:
            ovpn_path = (query_optimizer_repo_path / "client-config-staging.ovpn").resolve()
        if not ovpn_path.exists():
            ovpn_path = None

        vpn_docdb_host = env_value("VPN_DOCDB_HOST") or _parse_shell_default(reference_vpn_script, "DOCDB_HOST") or ""
        vpn_docdb_port = int(env_value("VPN_DOCDB_PORT", _parse_shell_default(reference_vpn_script, "DOCDB_PORT") or "27017"))
        vpn_require_split_tunnel = env_value(
            "VPN_REQUIRE_SPLIT_TUNNEL",
            (_parse_shell_default(reference_vpn_script, "REQUIRE_SPLIT_TUNNEL") or "true"),
        ).lower() == "true"
        vpn_require_docdb_reachable = env_value("VPN_REQUIRE_DOCDB_REACHABLE", "true").lower() == "true"
        vpn_log_dir = Path(env_value("VPN_LOG_DIR", str(root_dir / ".vpn_logs")))

        mongo_uri = env_value("MONGODB_URI")
        if not mongo_uri:
            raise ValueError("MONGODB_URI is required in this repository env/config.")

        return cls(
            workspace_root=root_dir,
            intent=env_value("INTENT"),
            mongodb_uri=mongo_uri,
            mongodb_db=env_value("MONGODB_DB", "web_crawler"),
            mongodb_runs_collection=env_value("MONGODB_RUNS_COLLECTION", "ie_runs"),
            mongodb_url_summaries_collection=env_value("MONGODB_URL_SUMMARIES_COLLECTION", "ie_url_summaries"),
            mongodb_events_collection=env_value("MONGODB_EVENTS_COLLECTION", "ie_events"),
            google_api_key=env_value("GOOGLE_API_KEY"),
            google_api_keys=env_value("GOOGLE_API_KEYS"),
            google_search_engine_id=env_value("GOOGLE_SEARCH_ENGINE_ID"),
            searchapi_api_key=env_value("SEARCHAPI_API_KEY"),
            azure_openai_api_key=env_value("AZURE_OPENAI_API_KEY"),
            azure_openai_endpoint=env_value("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_version=env_value("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_openai_model=env_value("AZURE_OPENAI_MODEL", "gpt-4.1"),
            eu_swarm_path=Path(env_value("EU_SWARM_PATH", str(eu_swarm_path))),
            query_optimizer_repo_path=query_optimizer_repo_path,
            tool_flow_path=tool_flow_path,
            browser_mode=env_value("BROWSER_MODE", "chromium"),
            browser_headed=env_value("BROWSER_HEADED", "false").lower() == "true",
            temp_google_email=env_value("TEMP_GOOGLE_EMAIL"),
            temp_google_password=env_value("TEMP_GOOGLE_PASSWORD"),
            temp_signup_email=env_value("TEMP_SIGNUP_EMAIL"),
            temp_signup_password=env_value("TEMP_SIGNUP_PASSWORD"),
            baseline_domains_file=Path(env_value("BASELINE_DOMAINS_FILE", str(root_dir / "data/tool_flow_baseline_domains.txt"))),
            max_browser_concurrency=int(env_value("MAX_BROWSER_CONCURRENCY", "0")),
            max_url_concurrency=int(env_value("MAX_URL_CONCURRENCY", "0")),
            vpn_start_script=vpn_start_script,
            auto_start_vpn=env_value("AUTO_START_VPN", "true").lower() == "true",
            query_optimizer_ovpn_config=ovpn_path,
            vpn_docdb_host=vpn_docdb_host,
            vpn_docdb_port=vpn_docdb_port,
            vpn_require_split_tunnel=vpn_require_split_tunnel,
            vpn_require_docdb_reachable=vpn_require_docdb_reachable,
            vpn_log_dir=vpn_log_dir,
            strategy_count=int(env_value("STRATEGY_COUNT", "10")),
            queries_per_strategy=int(env_value("QUERIES_PER_STRATEGY", "5")),
            serp_pages_per_query=int(env_value("SERP_PAGES_PER_QUERY", "2")),
            results_per_serp_page=int(env_value("RESULTS_PER_SERP_PAGE", "10")),
            max_internal_links=int(env_value("MAX_INTERNAL_LINKS", "12")),
            max_link_depth=int(env_value("MAX_LINK_DEPTH", "1")),
            max_site_graph_visits=int(env_value("MAX_SITE_GRAPH_VISITS", "6")),
            max_site_graph_nodes=int(env_value("MAX_SITE_GRAPH_NODES", "80")),
            max_site_graph_frontier=int(env_value("MAX_SITE_GRAPH_FRONTIER", "8")),
            max_sitemap_urls=int(env_value("MAX_SITEMAP_URLS", "150")),
            max_sitemap_fetches=int(env_value("MAX_SITEMAP_FETCHES", "6")),
            request_timeout_seconds=int(env_value("REQUEST_TIMEOUT_SECONDS", "20")),
            llm_max_retries=int(env_value("LLM_MAX_RETRIES", "2")),
            tool_flow_env_path=tool_flow_env_path,
            discovered_vpn_scripts=discovered_vpn_scripts,
        )
