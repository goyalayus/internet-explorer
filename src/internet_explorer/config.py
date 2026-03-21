from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values
from pydantic import BaseModel, field_validator


def _normalize_env_values(raw_values: dict[str, object] | None) -> dict[str, str]:
    if not raw_values:
        return {}

    normalized: dict[str, str] = {}
    for key, value in raw_values.items():
        normalized[str(key)] = str(value or "")
    return normalized


def _parse_shell_default(script_path: Path | None, variable_name: str) -> str | None:
    if script_path is None or not script_path.exists():
        return None
    text = script_path.read_text()
    pattern = re.compile(rf'{re.escape(variable_name)}="\$\{{{re.escape(variable_name)}:-([^}}]+)\}}"')
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1).strip()


def _derive_docdb_endpoint_from_mongodb_uri(uri: str) -> tuple[str, int | None]:
    raw_uri = (uri or "").strip()
    if not raw_uri:
        return "", None

    try:
        parsed = urlsplit(raw_uri)
    except Exception:
        return "", None

    host = parsed.hostname or ""
    try:
        port = parsed.port
    except ValueError:
        port = None
    return host, port


def _candidate_ovpn_paths(root_dir: Path, configured_path: str = "", legacy_path: str = "") -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    raw_candidates = [
        configured_path,
        str(root_dir / "vpn/client-config-staging.ovpn"),
        str(root_dir / "client-config-staging.ovpn"),
        legacy_path,
        str(root_dir.parent / "query_optimizer_repo/client-config-staging.ovpn"),
        str(root_dir.parent / "query-optimzer/client-config-staging.ovpn"),
        str(root_dir.parent / "query_optimizer/client-config-staging.ovpn"),
    ]

    for raw_path in raw_candidates:
        cleaned = (raw_path or "").strip()
        if not cleaned:
            continue
        path = Path(cleaned).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        candidates.append(path)

    return candidates


def _resolve_ovpn_config_path(root_dir: Path, configured_path: str = "", legacy_path: str = "") -> Path | None:
    for path in _candidate_ovpn_paths(root_dir, configured_path, legacy_path):
        if path.exists():
            return path
    return None


class AppConfig(BaseModel):
    workspace_root: Path
    intent: str = ""
    mongodb_uri: str
    mongodb_tls_ca_file: Path | None = None
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
    browser_chromium_sandbox: bool = False
    browser_use_llm_model: str = ""
    enable_browser_delegation: bool = True
    temp_google_email: str = ""
    temp_google_password: str = ""
    temp_signup_email: str = ""
    temp_signup_password: str = ""
    baseline_domains_file: Path
    known_tools_file: Path
    candidate_start_mode: str = "domain_homepage"
    discovery_cache_mode: str = "off"
    discovery_cache_dir: Path = Path("data/discovery_cache")
    discovery_cache_key: str = ""
    max_browser_concurrency: int = 0
    browser_delegate_timeout_seconds: int = 120
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

    @field_validator(
        "workspace_root",
        "baseline_domains_file",
        "known_tools_file",
        "env_file_path",
        "vpn_log_dir",
        "discovery_cache_dir",
    )
    @classmethod
    def _expand_required_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("eu_swarm_path", "mongodb_tls_ca_file", "vpn_start_script", "vpn_ovpn_config", "vpn_defaults_file")
    @classmethod
    def _expand_optional_path(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        return value.expanduser().resolve()

    @field_validator("discovery_cache_mode")
    @classmethod
    def _validate_discovery_cache_mode(cls, value: str) -> str:
        mode = (value or "").strip().lower()
        allowed = {"off", "read_only", "read_write", "refresh"}
        if mode not in allowed:
            allowed_values = ", ".join(sorted(allowed))
            raise ValueError(f"DISCOVERY_CACHE_MODE must be one of: {allowed_values}")
        return mode

    @field_validator("candidate_start_mode")
    @classmethod
    def _validate_candidate_start_mode(cls, value: str) -> str:
        mode = (value or "").strip().lower()
        allowed = {"domain_homepage", "first_result_url"}
        if mode not in allowed:
            allowed_values = ", ".join(sorted(allowed))
            raise ValueError(f"CANDIDATE_START_MODE must be one of: {allowed_values}")
        return mode

    @classmethod
    def from_env(
        cls,
        root: Path | None = None,
        *,
        env_overrides: dict[str, object] | None = None,
        prefer_process_env: bool = False,
    ) -> "AppConfig":
        root_dir = (root or Path.cwd()).resolve()
        env_file_path = (root_dir / ".env").resolve()
        local_env = _normalize_env_values(dotenv_values(env_file_path) if env_file_path.exists() else {})
        local_env.update(_normalize_env_values(env_overrides))

        def env_value(name: str, default: str = "") -> str:
            local_value = str(local_env.get(name) or "").strip()
            process_value = str(os.getenv(name) or "").strip()

            if prefer_process_env:
                if process_value:
                    return process_value
                if local_value:
                    return local_value
                return default.strip()

            if local_value:
                return local_value
            if process_value:
                return process_value
            return default.strip()

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

        mongo_uri = env_value("MONGODB_URI")
        if not mongo_uri:
            raise ValueError("MONGODB_URI is required in this repository env/config.")
        derived_docdb_host, derived_docdb_port = _derive_docdb_endpoint_from_mongodb_uri(mongo_uri)

        configured_ovpn_path = env_value("VPN_OVPN_CONFIG")
        legacy_ovpn_path = env_value("QUERY_OPTIMIZER_OVPN_CONFIG")
        vpn_ovpn_config = _resolve_ovpn_config_path(
            root_dir,
            configured_path=configured_ovpn_path,
            legacy_path=legacy_ovpn_path,
        )

        vpn_docdb_host = (
            env_value("VPN_DOCDB_HOST")
            or _parse_shell_default(vpn_defaults_file, "DOCDB_HOST")
            or derived_docdb_host
            or ""
        )
        vpn_docdb_port = int(
            env_value(
                "VPN_DOCDB_PORT",
                _parse_shell_default(vpn_defaults_file, "DOCDB_PORT")
                or str(derived_docdb_port or 27017),
            )
        )
        vpn_require_split_tunnel = env_value(
            "VPN_REQUIRE_SPLIT_TUNNEL",
            (_parse_shell_default(vpn_defaults_file, "REQUIRE_SPLIT_TUNNEL") or "true"),
        ).lower() == "true"
        vpn_require_docdb_reachable = env_value("VPN_REQUIRE_DOCDB_REACHABLE", "true").lower() == "true"
        vpn_log_dir = Path(env_value("VPN_LOG_DIR", str(root_dir / ".vpn_logs")))

        mongodb_tls_ca_raw = env_value("MONGODB_TLS_CA_FILE")
        if mongodb_tls_ca_raw:
            mongodb_tls_ca_file = Path(mongodb_tls_ca_raw).expanduser().resolve()
        else:
            default_tls_ca = (root_dir / "certs/global-bundle.pem").resolve()
            mongodb_tls_ca_file = default_tls_ca if default_tls_ca.exists() else None
        if mongodb_tls_ca_file and not mongodb_tls_ca_file.exists():
            mongodb_tls_ca_file = None

        return cls(
            workspace_root=root_dir,
            intent=env_value("INTENT"),
            mongodb_uri=mongo_uri,
            mongodb_tls_ca_file=mongodb_tls_ca_file,
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
            browser_chromium_sandbox=env_value("BROWSER_CHROMIUM_SANDBOX", "false").lower() == "true",
            browser_use_llm_model=env_value("BROWSER_USE_LLM_MODEL"),
            enable_browser_delegation=env_value("ENABLE_BROWSER_DELEGATION", "true").lower() == "true",
            temp_google_email=env_value("TEMP_GOOGLE_EMAIL"),
            temp_google_password=env_value("TEMP_GOOGLE_PASSWORD"),
            temp_signup_email=env_value("TEMP_SIGNUP_EMAIL"),
            temp_signup_password=env_value("TEMP_SIGNUP_PASSWORD"),
            baseline_domains_file=Path(env_value("BASELINE_DOMAINS_FILE", str(root_dir / "data/tool_flow_baseline_domains.txt"))),
            known_tools_file=Path(env_value("KNOWN_TOOLS_FILE", str(root_dir / "data/known_tools.txt"))),
            candidate_start_mode=env_value("CANDIDATE_START_MODE", "domain_homepage"),
            discovery_cache_mode=env_value("DISCOVERY_CACHE_MODE", "off"),
            discovery_cache_dir=Path(env_value("DISCOVERY_CACHE_DIR", str(root_dir / "data/discovery_cache"))),
            discovery_cache_key=env_value("DISCOVERY_CACHE_KEY"),
            max_browser_concurrency=int(env_value("MAX_BROWSER_CONCURRENCY", "0")),
            browser_delegate_timeout_seconds=int(env_value("BROWSER_DELEGATE_TIMEOUT_SECONDS", "120")),
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
