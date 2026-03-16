from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, field_validator


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
    search_query_concurrency: int = 8
    search_retry_attempts: int = 4
    search_retry_base_backoff_seconds: float = 1.0
    searchapi_api_key: str = ""
    gemini_api_key: str = ""
    gemini_api_keys: str = ""
    gemini_model: str = "gemini-3-flash-preview"
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-12-01-preview"
    azure_openai_model: str = "gpt-4.1"
    eu_swarm_path: Path | None = None
    browser_mode: str = "chromium"
    browser_headed: bool = False
    temp_google_email: str = ""
    temp_google_password: str = ""
    temp_signup_email: str = ""
    temp_signup_password: str = ""
    baseline_domains_file: Path
    known_tools_file: Path
    candidate_start_mode: str = "first_result_url"
    max_browser_concurrency: int = 0
    browser_delegate_timeout_seconds: int = 180
    max_url_concurrency: int = 0
    url_batch_size: int = 40
    vpn_start_script: Path | None = None
    auto_start_vpn: bool = True
    vpn_ovpn_config: Path | None = None
    vpn_defaults_file: Path | None = None
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
    pdf_inline_max_bytes: int = 8_000_000
    http_pool_timeout_seconds: int = 30
    http_max_connections: int = 300
    http_max_keepalive_connections: int = 100
    fetch_retry_attempts: int = 3
    fetch_retry_base_backoff_seconds: float = 0.5
    llm_max_retries: int = 2
    env_file_path: Path

    @field_validator("workspace_root", "baseline_domains_file", "known_tools_file", "env_file_path", "vpn_log_dir")
    @classmethod
    def _expand_required_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("eu_swarm_path", "vpn_start_script", "vpn_ovpn_config", "vpn_defaults_file")
    @classmethod
    def _expand_optional_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        return value.expanduser().resolve()

    @classmethod
    def from_env(cls, root: Path | None = None) -> "AppConfig":
        root_dir = (root or Path.cwd()).resolve()
        env_file_path = (root_dir / ".env").resolve()
        local_env = dotenv_values(env_file_path) if env_file_path.exists() else {}

        def env_value(name: str, default: str = "") -> str:
            return (os.getenv(name) or str(local_env.get(name) or default)).strip()

        eu_swarm_default = (root_dir / "../eu-swarm").resolve()
        eu_swarm_path_raw = env_value("EU_SWARM_PATH", str(eu_swarm_default))
        eu_swarm_path = Path(eu_swarm_path_raw).expanduser().resolve() if eu_swarm_path_raw else None

        vpn_defaults_raw = env_value("VPN_DEFAULTS_FILE", str(root_dir / "scripts/vpn_defaults.sh"))
        vpn_defaults_file = Path(vpn_defaults_raw).expanduser().resolve() if vpn_defaults_raw else None
        if vpn_defaults_file and not vpn_defaults_file.exists():
            vpn_defaults_file = None

        vpn_start_raw = env_value("VPN_START_SCRIPT", str(root_dir / "scripts/vpn_start.sh"))
        vpn_start_script = Path(vpn_start_raw).expanduser().resolve() if vpn_start_raw else None
        if vpn_start_script and not vpn_start_script.exists():
            vpn_start_script = None

        vpn_ovpn_raw = env_value("VPN_OVPN_CONFIG") or env_value("QUERY_OPTIMIZER_OVPN_CONFIG")
        if vpn_ovpn_raw:
            vpn_ovpn_config = Path(vpn_ovpn_raw).expanduser().resolve()
        else:
            default_ovpn = (root_dir / "vpn/client-config-staging.ovpn").resolve()
            vpn_ovpn_config = default_ovpn if default_ovpn.exists() else None
        if vpn_ovpn_config and not vpn_ovpn_config.exists():
            vpn_ovpn_config = None

        vpn_docdb_host = env_value("VPN_DOCDB_HOST") or _parse_shell_default(vpn_defaults_file, "DOCDB_HOST") or ""
        vpn_docdb_port = int(env_value("VPN_DOCDB_PORT", _parse_shell_default(vpn_defaults_file, "DOCDB_PORT") or "27017"))
        vpn_require_split_tunnel = env_value(
            "VPN_REQUIRE_SPLIT_TUNNEL",
            (_parse_shell_default(vpn_defaults_file, "REQUIRE_SPLIT_TUNNEL") or "true"),
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
            search_query_concurrency=int(env_value("SEARCH_QUERY_CONCURRENCY", "8")),
            search_retry_attempts=int(env_value("SEARCH_RETRY_ATTEMPTS", "4")),
            search_retry_base_backoff_seconds=float(env_value("SEARCH_RETRY_BASE_BACKOFF_SECONDS", "1.0")),
            searchapi_api_key=env_value("SEARCHAPI_API_KEY"),
            gemini_api_key=env_value("GEMINI_API_KEY"),
            gemini_api_keys=env_value("GEMINI_API_KEYS"),
            gemini_model=env_value("GEMINI_MODEL", "gemini-3-flash-preview"),
            azure_openai_api_key=env_value("AZURE_OPENAI_API_KEY"),
            azure_openai_endpoint=env_value("AZURE_OPENAI_ENDPOINT"),
            azure_openai_api_version=env_value("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_openai_model=env_value("AZURE_OPENAI_MODEL", "gpt-4.1"),
            eu_swarm_path=eu_swarm_path,
            browser_mode=env_value("BROWSER_MODE", "chromium"),
            browser_headed=env_value("BROWSER_HEADED", "false").lower() == "true",
            temp_google_email=env_value("TEMP_GOOGLE_EMAIL"),
            temp_google_password=env_value("TEMP_GOOGLE_PASSWORD"),
            temp_signup_email=env_value("TEMP_SIGNUP_EMAIL"),
            temp_signup_password=env_value("TEMP_SIGNUP_PASSWORD"),
            baseline_domains_file=Path(env_value("BASELINE_DOMAINS_FILE", str(root_dir / "data/tool_flow_baseline_domains.txt"))),
            known_tools_file=Path(env_value("KNOWN_TOOLS_FILE", str(root_dir / "data/known_tools.txt"))),
            candidate_start_mode=env_value("CANDIDATE_START_MODE", "first_result_url"),
            max_browser_concurrency=int(env_value("MAX_BROWSER_CONCURRENCY", "0")),
            browser_delegate_timeout_seconds=int(env_value("BROWSER_DELEGATE_TIMEOUT_SECONDS", "180")),
            max_url_concurrency=int(env_value("MAX_URL_CONCURRENCY", "0")),
            url_batch_size=int(env_value("URL_BATCH_SIZE", "40")),
            vpn_start_script=vpn_start_script,
            auto_start_vpn=env_value("AUTO_START_VPN", "true").lower() == "true",
            vpn_ovpn_config=vpn_ovpn_config,
            vpn_defaults_file=vpn_defaults_file,
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
            pdf_inline_max_bytes=int(env_value("PDF_INLINE_MAX_BYTES", "8000000")),
            http_pool_timeout_seconds=int(env_value("HTTP_POOL_TIMEOUT_SECONDS", "30")),
            http_max_connections=int(env_value("HTTP_MAX_CONNECTIONS", "300")),
            http_max_keepalive_connections=int(env_value("HTTP_MAX_KEEPALIVE_CONNECTIONS", "100")),
            fetch_retry_attempts=int(env_value("FETCH_RETRY_ATTEMPTS", "3")),
            fetch_retry_base_backoff_seconds=float(env_value("FETCH_RETRY_BASE_BACKOFF_SECONDS", "0.5")),
            llm_max_retries=int(env_value("LLM_MAX_RETRIES", "2")),
            env_file_path=env_file_path,
        )
