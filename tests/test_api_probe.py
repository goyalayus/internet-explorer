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
