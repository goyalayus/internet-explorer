from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


TRACKING_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "ref",
    "source",
    "mc_cid",
    "mc_eid",
}

COMMON_SECOND_LEVEL_SUFFIXES = {
    "co.uk",
    "org.uk",
    "gov.uk",
    "ac.uk",
    "co.in",
    "com.au",
    "co.jp",
}


def canonicalize_url(raw_url: str) -> str:
    parsed = urlparse(raw_url.strip())
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    if not netloc and parsed.path:
        parsed = urlparse(f"{scheme}://{parsed.path}")
        netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    filtered_query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if k.lower() not in TRACKING_QUERY_PARAMS
    ]
    query = urlencode(sorted(filtered_query))
    return urlunparse((scheme.lower(), netloc, path, "", query, ""))


def canonical_domain(url_or_domain: str) -> str:
    raw = url_or_domain.strip().lower()
    if "://" in raw:
        raw = urlparse(raw).netloc.lower()
    if raw.startswith("www."):
        raw = raw[4:]
    return raw


def registrable_domain(url_or_domain: str) -> str:
    domain = canonical_domain(url_or_domain)
    parts = [part for part in domain.split(".") if part]
    if len(parts) <= 2:
        return domain
    last_two = ".".join(parts[-2:])
    last_three = ".".join(parts[-3:])
    if last_two in COMMON_SECOND_LEVEL_SUFFIXES:
        return ".".join(parts[-3:])
    if last_three in COMMON_SECOND_LEVEL_SUFFIXES and len(parts) >= 4:
        return ".".join(parts[-4:])
    return ".".join(parts[-2:])


def load_baseline_domains(path: Path) -> set[str]:
    if not path.exists():
        return set()
    domains: set[str] = set()
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        domains.add(registrable_domain(stripped))
    return domains

