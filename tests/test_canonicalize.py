from internet_explorer.canonicalize import canonical_domain, canonicalize_url, registrable_domain


def test_canonicalize_url_removes_tracking_params() -> None:
    raw = "HTTPS://www.example.com/path/?utm_source=x&b=2&a=1#frag"
    assert canonicalize_url(raw) == "https://example.com/path?a=1&b=2"


def test_registrable_domain_handles_common_cases() -> None:
    assert registrable_domain("https://www.docs.example.com/path") == "example.com"
    assert registrable_domain("news.service.co.uk") == "service.co.uk"


def test_canonical_domain_strips_www() -> None:
    assert canonical_domain("https://www.example.com/test") == "example.com"
