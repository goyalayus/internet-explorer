import sys

import pytest

from internet_explorer import cli


class _ConfigStub:
    def __init__(self) -> None:
        self.intent = ""
        self.auto_start_vpn = False
        self.allow_parallel_workers = False

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


def test_cli_exits_130_on_keyboard_interrupt(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["internet-explorer", "--print-config"])

    def _fake_asyncio_run(coro):  # noqa: ANN001
        coro.close()
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.asyncio, "run", _fake_asyncio_run)

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 130


@pytest.mark.asyncio
async def test_cli_blocks_when_another_worker_is_running(monkeypatch) -> None:
    def _fake_from_env(root, *, env_overrides=None, prefer_process_env=False):  # noqa: ANN001, ARG001
        config = _ConfigStub()
        config.intent = "find sources"
        return config

    monkeypatch.setattr(cli.AppConfig, "from_env", _fake_from_env)
    monkeypatch.setattr(cli, "_find_existing_workers", lambda: [12345])

    with pytest.raises(SystemExit, match="ALLOW_PARALLEL_WORKERS=true"):
        await cli._run(
            intent="find sources",
            print_config=False,
            vpn_start=False,
            vpn_stop=False,
            vpn_status=False,
            vpn_check_docdb=False,
        )


def test_find_existing_workers_ignores_bash_wrappers(monkeypatch) -> None:
    class _Result:
        stdout = (
            "338888 bash bash -c nohup .venv/bin/python -m internet_explorer.cli --intent test\n"
            "338901 python .venv/bin/python -m internet_explorer.cli --intent test\n"
            "338902 python3 python3 -m something_else\n"
        )

    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: _Result())
    monkeypatch.setattr(cli.os, "getpid", lambda: 999999)

    pids = cli._find_existing_workers()

    assert pids == [338901]
