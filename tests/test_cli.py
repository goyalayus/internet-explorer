import pytest

from internet_explorer import cli


class _ConfigStub:
    def __init__(self) -> None:
        self.intent = ""
        self.auto_start_vpn = False

    def model_dump(self, mode: str = "json"):  # noqa: ARG002
        return {"intent": self.intent}


@pytest.mark.asyncio
async def test_cli_prefers_process_env_overrides(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_from_env(root, *, env_overrides=None, prefer_process_env=False):  # noqa: ANN001, ARG001
        captured["prefer_process_env"] = prefer_process_env
        return _ConfigStub()

    monkeypatch.setattr(cli.AppConfig, "from_env", _fake_from_env)

    exit_code = await cli._run(
        intent="",
        print_config=True,
        vpn_start=False,
        vpn_stop=False,
        vpn_status=False,
        vpn_check_docdb=False,
    )

    assert exit_code == 0
    assert captured["prefer_process_env"] is True
