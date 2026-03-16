from __future__ import annotations

import asyncio
import re
from html import unescape
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from internet_explorer.config import AppConfig
from internet_explorer.models import ApiSignal, BinaryFetchResult, FetchResult, PageEvidence, RenderProfile


API_LINK_KEYWORDS = ("api", "developer", "docs", "openapi", "swagger", "graphql", "reference")
EXPLORE_LINK_KEYWORDS = API_LINK_KEYWORDS + (
    "pricing",
    "plans",
    "data",
    "dataset",
    "portal",
    "catalog",
    "rfp",
    "tender",
    "procurement",
    "trial",
    "signup",
    "register",
    "contact",
    "sales",
)
PAYWALL_PATTERNS = (
    "upgrade to continue",
    "subscription required",
    "payment required",
    "start free trial",
    "choose a plan",
    "upgrade your plan",
)
CONTACT_SALES_PATTERNS = (
    "contact sales",
    "talk to sales",
    "book a demo",
    "request a demo",
)
AUTH_PATTERNS = ("sign in", "log in", "login", "authenticate")
CAPTCHA_PATTERNS = (
    "captcha",
    "recaptcha",
    "cloudflare turnstile",
    "i am human",
    "just a moment",
    "checking your browser before accessing",
    "verify you are human",
    "security check",
)
DATA_PATTERNS = (
    "request for proposal",
    "rfp",
    "tender",
    "dataset",
    "catalog",
    "search results",
    "procurement",
    "bid notice",
)
TRANSIENT_FETCH_ERRORS = (
    httpx.PoolTimeout,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
)


class AsyncWebFetcher:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        timeout = httpx.Timeout(
            connect=config.request_timeout_seconds,
            read=config.request_timeout_seconds,
            write=config.request_timeout_seconds,
            pool=config.http_pool_timeout_seconds,
        )
        limits = httpx.Limits(
            max_connections=config.http_max_connections,
            max_keepalive_connections=config.http_max_keepalive_connections,
        )
        self.client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            limits=limits,
            headers={
                "User-Agent": "internet-explorer/0.1 (+datasource-discovery)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        self.insecure_client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            limits=limits,
            verify=False,
            headers={
                "User-Agent": "internet-explorer/0.1 (+datasource-discovery)",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )

    async def fetch(self, url: str) -> FetchResult:
        attempts = max(1, self.config.fetch_retry_attempts)
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = await self.client.get(url)
                return _to_fetch_result(url=url, response=response)
            except TRANSIENT_FETCH_ERRORS as exc:
                if _is_ssl_verification_error(exc):
                    try:
                        insecure_response = await self.insecure_client.get(url)
                        return _to_fetch_result(url=url, response=insecure_response)
                    except TRANSIENT_FETCH_ERRORS as insecure_exc:
                        last_exc = insecure_exc
                        if attempt >= attempts:
                            break
                        backoff = self.config.fetch_retry_base_backoff_seconds * (2 ** (attempt - 1))
                        await asyncio.sleep(min(backoff, 8.0))
                        continue
                last_exc = exc
                if attempt >= attempts:
                    break
                backoff = self.config.fetch_retry_base_backoff_seconds * (2 ** (attempt - 1))
                await asyncio.sleep(min(backoff, 8.0))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("fetch failed without a captured exception")

    async def fetch_binary(self, url: str) -> BinaryFetchResult:
        attempts = max(1, self.config.fetch_retry_attempts)
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = await self.client.get(url)
                return _to_binary_fetch_result(url=url, response=response)
            except TRANSIENT_FETCH_ERRORS as exc:
                if _is_ssl_verification_error(exc):
                    try:
                        insecure_response = await self.insecure_client.get(url)
                        return _to_binary_fetch_result(url=url, response=insecure_response)
                    except TRANSIENT_FETCH_ERRORS as insecure_exc:
                        last_exc = insecure_exc
                        if attempt >= attempts:
                            break
                        backoff = self.config.fetch_retry_base_backoff_seconds * (2 ** (attempt - 1))
                        await asyncio.sleep(min(backoff, 8.0))
                        continue
                last_exc = exc
                if attempt >= attempts:
                    break
                backoff = self.config.fetch_retry_base_backoff_seconds * (2 ** (attempt - 1))
                await asyncio.sleep(min(backoff, 8.0))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("binary fetch failed without a captured exception")

    async def close(self) -> None:
        await self.client.aclose()
        await self.insecure_client.aclose()


def detect_render_profile(html: str) -> RenderProfile:
    lowered = html.lower()
    script_count = lowered.count("<script")
    body_text = _html_to_text(html)
    body_len = len(body_text.strip())
    app_root_markers = sum(
        1
        for marker in (
            'id="root"',
            "id='root'",
            'id="app"',
            "id='app'",
            "__next",
            "__nuxt",
            "data-reactroot",
        )
        if marker in lowered
    )
    if body_len < 300 and script_count > 10 and app_root_markers > 0:
        return "csr_shell"
    if body_len > 1200 and script_count < 25:
        return "static_ssr"
    return "hybrid"


def extract_relevant_links(base_url: str, html: str, limit: int) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    ranked: list[tuple[int, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        absolute = urljoin(base_url, href)
        anchor_text = f"{anchor.get_text(' ', strip=True)} {absolute}".lower()
        score = sum(1 for keyword in EXPLORE_LINK_KEYWORDS if keyword in anchor_text)
        if score <= 0:
            continue
        ranked.append((score, absolute))
    ranked.sort(key=lambda item: item[0], reverse=True)
    unique: list[str] = []
    seen: set[str] = set()
    for _, link in ranked:
        if link in seen:
            continue
        seen.add(link)
        unique.append(link)
        if len(unique) >= limit:
            break
    return unique


def analyze_page(fetch: FetchResult) -> PageEvidence:
    lowered = fetch.html.lower()
    links = extract_relevant_links(fetch.final_url, fetch.html, limit=12) if fetch.html else []
    api_signal = ApiSignal(
        detected=any(keyword in lowered for keyword in API_LINK_KEYWORDS),
        doc_links=[link for link in links if any(keyword in link.lower() for keyword in API_LINK_KEYWORDS)],
        openapi_links=[link for link in links if any(token in link.lower() for token in ("openapi", "swagger", ".json"))],
        graphql_hints=[token for token in ("graphql", "/graphql") if token in lowered],
        auth_required=any(pattern in lowered for pattern in AUTH_PATTERNS),
    )
    data_signals = [pattern for pattern in DATA_PATTERNS if pattern in lowered]
    title = _extract_title(fetch.html)
    return PageEvidence(
        url=fetch.final_url,
        title=title,
        content_type=fetch.content_type,
        content_kind="html_page",
        text_excerpt=fetch.text_excerpt[:1200],
        html_excerpt=fetch.html[:2000],
        relevant_links=links,
        api_signal=api_signal,
        paywall_present=any(pattern in lowered for pattern in PAYWALL_PATTERNS),
        contact_sales_present=any(pattern in lowered for pattern in CONTACT_SALES_PATTERNS),
        auth_required=any(pattern in lowered for pattern in AUTH_PATTERNS),
        captcha_present=any(pattern in lowered for pattern in CAPTCHA_PATTERNS),
        data_signals=data_signals,
    )


def _extract_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return unescape(match.group(1)).strip()


def _html_to_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for node in soup(["script", "style", "noscript"]):
        node.extract()
    return " ".join(soup.get_text(" ", strip=True).split())


def is_probable_pdf_url(url: str) -> bool:
    lowered = url.lower()
    return lowered.endswith(".pdf") or ".pdf?" in lowered


def is_pdf_fetch(fetch: FetchResult) -> bool:
    content_type = fetch.content_type.lower()
    disposition = fetch.content_disposition.lower()
    return "application/pdf" in content_type or ".pdf" in disposition or is_probable_pdf_url(fetch.final_url)


def _to_fetch_result(*, url: str, response: httpx.Response) -> FetchResult:
    content_type = response.headers.get("content-type", "")
    content_disposition = response.headers.get("content-disposition", "")
    is_binary = not _is_textual_content_type(content_type)
    body_text = response.text if not is_binary else ""
    html = body_text if ("html" in content_type or (not content_type and not is_binary)) else ""
    text = _html_to_text(html)[:4000] if html else body_text[:4000]
    return FetchResult(
        url=url,
        final_url=str(response.url),
        status_code=response.status_code,
        content_type=content_type,
        content_disposition=content_disposition,
        content_length=len(response.content),
        is_binary=is_binary,
        html=html,
        body_text=body_text,
        text_excerpt=text,
        headers=dict(response.headers),
    )


def _to_binary_fetch_result(*, url: str, response: httpx.Response) -> BinaryFetchResult:
    return BinaryFetchResult(
        url=url,
        final_url=str(response.url),
        status_code=response.status_code,
        content_type=response.headers.get("content-type", ""),
        content_disposition=response.headers.get("content-disposition", ""),
        content_length=len(response.content),
        content_bytes=response.content,
        headers=dict(response.headers),
    )


def _is_ssl_verification_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "certificate verify failed" in msg or ("ssl" in msg and "cert" in msg)


def _is_textual_content_type(content_type: str) -> bool:
    lowered = content_type.lower()
    if not lowered:
        return True
    return (
        "text/" in lowered
        or "html" in lowered
        or "json" in lowered
        or "xml" in lowered
        or "javascript" in lowered
        or "x-www-form-urlencoded" in lowered
    )
