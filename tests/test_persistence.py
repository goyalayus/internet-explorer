from internet_explorer.persistence import _sanitize_bson


def test_sanitize_bson_converts_out_of_range_integers() -> None:
    payload = {
        "ok": 42,
        "max_ok": 2**63 - 1,
        "min_ok": -(2**63),
        "too_big": 2**63,
        "too_small": -(2**63) - 1,
        "nested": [{"value": 10**40}],
        "bool_flag": True,
    }

    sanitized = _sanitize_bson(payload)

    assert sanitized["ok"] == 42
    assert sanitized["max_ok"] == 2**63 - 1
    assert sanitized["min_ok"] == -(2**63)
    assert sanitized["too_big"] == str(2**63)
    assert sanitized["too_small"] == str(-(2**63) - 1)
    assert sanitized["nested"][0]["value"] == str(10**40)
    assert sanitized["bool_flag"] is True
