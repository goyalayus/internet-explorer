from pydantic import BaseModel

from internet_explorer.llm import _extract_json_payload, _normalize_payload_for_schema


class _Envelope(BaseModel):
    relevant: bool
    title: str = ""


def test_normalize_payload_for_schema_unwraps_single_item_list_dict() -> None:
    payload = [{"relevant": True, "title": "doc"}]

    normalized = _normalize_payload_for_schema(payload, schema=_Envelope)

    assert normalized == {"relevant": True, "title": "doc"}


def test_extract_json_payload_parses_python_literal_dict() -> None:
    payload = "{'relevant': True, 'title': 'doc'}"

    parsed = _extract_json_payload(payload)

    assert parsed == {"relevant": True, "title": "doc"}


def test_extract_json_payload_parses_python_literal_list() -> None:
    payload = "[{'relevant': True, 'title': 'doc'}]"

    parsed = _extract_json_payload(payload)

    assert parsed == [{"relevant": True, "title": "doc"}]
