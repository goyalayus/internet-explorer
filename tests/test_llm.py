from pydantic import BaseModel

from internet_explorer.llm import _normalize_payload_for_schema


class _Envelope(BaseModel):
    relevant: bool
    title: str = ""


def test_normalize_payload_for_schema_unwraps_single_item_list_dict() -> None:
    payload = [{"relevant": True, "title": "doc"}]

    normalized = _normalize_payload_for_schema(payload, schema=_Envelope)

    assert normalized == {"relevant": True, "title": "doc"}
