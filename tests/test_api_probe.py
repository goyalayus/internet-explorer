from internet_explorer.api_probe import ApiProbeService
from internet_explorer.models import ApiSignal, PageEvidence


class _TelemetryStub:
    def timed(self) -> float:
        return 0.0

    def elapsed_ms(self, started_at: float) -> int:
        return 0

    def emit(self, **kwargs) -> None:
        return None


class _LLMStub:
    async def complete_json(self, **kwargs):
        raise AssertionError("LLM planning should not run in these helper tests")


def test_probe_prefers_openapi_link() -> None:
    service = ApiProbeService(_TelemetryStub(), _LLMStub())
    evidence = [
        PageEvidence(
            url="https://example.com",
            api_signal=ApiSignal(
                detected=True,
                doc_links=["https://example.com/docs"],
                openapi_links=["https://example.com/openapi.json"],
            ),
        )
    ]
    result = service._merge_signal(evidence)
    assert service._pick_probe_url(result) == "https://example.com/openapi.json"


def test_probe_parses_headers_and_json_body() -> None:
    service = ApiProbeService(_TelemetryStub(), _LLMStub())
    stdout = "HTTP/2 200 \r\ncontent-type: application/json\r\n\r\n{\"items\":[1,2]}"
    result = service._to_probe_result(
        "https://example.com/openapi.json",
        ["bash", "-lc", "curl 'https://example.com/openapi.json'"],
        "curl 'https://example.com/openapi.json'",
        "",
        False,
        0,
        stdout,
        "",
    )
    assert result.status_code == 200
    assert result.content_type == "application/json"
    assert result.accessible is True
    assert result.relevant_guess is True


def test_probe_marks_auth_failures_inaccessible() -> None:
    service = ApiProbeService(_TelemetryStub(), _LLMStub())
    stdout = "HTTP/2 403 \r\ncontent-type: application/json\r\n\r\n{\"error\":\"forbidden\"}"
    result = service._to_probe_result(
        "https://example.com/openapi.json",
        ["bash", "-lc", "curl 'https://example.com/openapi.json'"],
        "curl 'https://example.com/openapi.json'",
        "",
        False,
        0,
        stdout,
        "",
    )
    assert result.status_code == 403
    assert result.accessible is False


def test_validate_shell_command_allows_read_only_pipelines() -> None:
    service = ApiProbeService(_TelemetryStub(), _LLMStub())
    service._validate_shell_command("curl -sS 'https://example.com/openapi.json' | jq '.paths' | head -c 500")


def test_validate_shell_command_rejects_control_operators() -> None:
    service = ApiProbeService(_TelemetryStub(), _LLMStub())
    try:
        service._validate_shell_command("curl 'https://example.com' && rm -rf /tmp/x")
    except ValueError as exc:
        assert "forbidden" in str(exc)
    else:
        raise AssertionError("expected command validation to fail")


def test_pick_probe_url_skips_document_links() -> None:
    service = ApiProbeService(_TelemetryStub(), _LLMStub())
    signal = ApiSignal(
        detected=True,
        doc_links=[
            "https://example.com/docs/api-guide.pdf",
            "https://example.com/developers/reference",
        ],
        openapi_links=["https://example.com/openapi-spec.pdf"],
    )

    assert service._pick_probe_url(signal) == "https://example.com/developers/reference"


def test_probe_result_treats_head_pipe_close_as_success() -> None:
    service = ApiProbeService(_TelemetryStub(), _LLMStub())
    stdout = "HTTP/1.1 200 OK\nContent-Type: application/json\n\n{\"openapi\":\"3.0.0\"}"
    result = service._to_probe_result(
        "https://example.com/openapi.json",
        ["bash", "-lc", "curl -sS https://example.com/openapi.json | head -c 8000"],
        "curl -sS https://example.com/openapi.json | head -c 8000",
        "fallback",
        True,
        23,
        stdout,
        "curl: (23) Failure writing output to destination\n",
    )

    assert result.success is True
    assert result.accessible is True
    assert result.error == ""


def test_pick_probe_url_skips_invalid_url() -> None:
    service = ApiProbeService(_TelemetryStub(), _LLMStub())
    signal = ApiSignal(
        detected=True,
        openapi_links=["https://[invalid"],
        doc_links=["https://example.com/developers/reference"],
    )

    assert service._pick_probe_url(signal) == "https://example.com/developers/reference"


def test_pick_probe_url_rejects_cross_domain_docs_link() -> None:
    service = ApiProbeService(_TelemetryStub(), _LLMStub())
    signal = ApiSignal(
        detected=True,
        doc_links=[
            "https://docs.google.com/forms/d/e/example/viewform",
            "https://example.com/developers/reference",
        ],
    )

    assert service._pick_probe_url(signal, candidate_domain="example.com") == "https://example.com/developers/reference"


def test_pick_probe_url_rejects_non_api_reference_pages() -> None:
    service = ApiProbeService(_TelemetryStub(), _LLMStub())
    signal = ApiSignal(
        detected=True,
        doc_links=["https://example.com/applicant-reference-library"],
    )

    assert service._pick_probe_url(signal, candidate_domain="example.com") == ""


def test_guess_content_type_handles_invalid_url() -> None:
    service = ApiProbeService(_TelemetryStub(), _LLMStub())

    content_type = service._guess_content_type("https://[invalid", "{\"openapi\":\"3.1.0\"}")

    assert content_type == "application/json"
