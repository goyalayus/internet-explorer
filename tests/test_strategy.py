from internet_explorer.strategy import (
    _ensure_procurement_query_focus,
    _intent_requires_procurement_focus,
    _normalize_query_batch,
)


def test_intent_procurement_focus_detection() -> None:
    assert _intent_requires_procurement_focus("find RFP websites for data annotation") is True
    assert _intent_requires_procurement_focus("find API docs for weather data") is False


def test_ensure_procurement_query_focus_adds_anchor_and_noise_exclusions() -> None:
    query = "data annotation sources site:gov"
    hardened = _ensure_procurement_query_focus(query)

    assert "rfp" in hardened.lower()
    assert "-site:linkedin.com" in hardened.lower()
    assert "-site:github.com" in hardened.lower()
    assert "-site:youtube.com" in hardened.lower()
    assert "-inurl:profile" in hardened.lower()
    assert "-inurl:faculty" in hardened.lower()


def test_ensure_procurement_query_focus_does_not_duplicate_existing_anchor() -> None:
    query = "data annotation tender opportunities site:gov -site:linkedin.com"
    hardened = _ensure_procurement_query_focus(query)

    assert hardened.lower().count("tender") >= 1
    assert hardened.lower().count("-site:linkedin.com") == 1


def test_normalize_query_batch_removes_empty_items() -> None:
    raw = ["  one   query  ", "", "   ", "two query"]
    assert _normalize_query_batch(raw) == ["one query", "two query"]
