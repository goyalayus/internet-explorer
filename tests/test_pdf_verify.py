import pytest

from internet_explorer.models import BinaryFetchResult
from internet_explorer.pdf_verify import PdfVerifierService


class _FetcherStub:
    async def fetch_binary(self, url: str) -> BinaryFetchResult:
        return BinaryFetchResult(
            url=url,
            final_url=url,
            status_code=200,
            content_type="application/pdf",
            content_bytes=b"",
        )


class _LLMStub:
    async def complete_pdf_json(self, **kwargs):
        raise AssertionError("LLM should not be called for empty PDF bytes")


class _FetcherPdfStub:
    async def fetch_binary(self, url: str) -> BinaryFetchResult:
        return BinaryFetchResult(
            url=url,
            final_url=url,
            status_code=200,
            content_type="application/pdf",
            content_bytes=b"%PDF-1.7 fake",
        )


class _LLMNullSignalsStub:
    async def complete_pdf_json(self, **kwargs):
        schema = kwargs["schema"]
        return schema.model_validate(
            {
                "relevant": True,
                "title": "Example PDF",
                "summary": "Contains procurement references.",
                "reasoning": "Relevant to procurement discovery.",
                "extracted_signals": None,
            }
        )


class _TelemetryStub:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def timed(self) -> float:
        return 0.0

    def elapsed_ms(self, started_at: float) -> int:
        return 0

    def emit(self, **kwargs) -> None:
        self.events.append(kwargs)


@pytest.mark.asyncio
async def test_pdf_verify_empty_bytes_returns_unreadable_fallback() -> None:
    telemetry = _TelemetryStub()
    service = PdfVerifierService(_FetcherStub(), _LLMStub(), telemetry)

    result = await service.verify(
        url_id="url_1",
        intent="find data annotation tenders",
        pdf_url="https://example.com/docs/tender.pdf",
    )

    assert result.relevant is False
    assert result.error == "ValueError:pdf_empty_bytes"
    assert result.summary == "Empty PDF fallback used."
    assert result.fallback_urls
    assert any(event.get("decision") == "pdf_verify_unreadable_fallback" for event in telemetry.events)


@pytest.mark.asyncio
async def test_pdf_verify_null_extracted_signals_is_coerced_to_empty_list() -> None:
    telemetry = _TelemetryStub()
    service = PdfVerifierService(_FetcherPdfStub(), _LLMNullSignalsStub(), telemetry)

    result = await service.verify(
        url_id="url_2",
        intent="find data annotation tenders",
        pdf_url="https://example.com/docs/procurement.pdf",
    )

    assert result.relevant is True
    assert result.error in {None, ""}
    assert result.extracted_signals == []
    assert any(event.get("decision") == "pdf_verified" for event in telemetry.events)
