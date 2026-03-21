from pathlib import Path

import pytest

from internet_explorer.config import AppConfig


def test_config_accepts_valid_candidate_start_mode(tmp_path: Path) -> None:
    cfg = AppConfig.from_env(
        root=tmp_path,
        env_overrides={
            "MONGODB_URI": "mongodb://localhost:27017",
            "CANDIDATE_START_MODE": "DOMAIN_HOMEPAGE",
        },
        prefer_process_env=True,
    )

    assert cfg.candidate_start_mode == "domain_homepage"


def test_config_rejects_invalid_candidate_start_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="CANDIDATE_START_MODE must be one of"):
        AppConfig.from_env(
            root=tmp_path,
            env_overrides={
                "MONGODB_URI": "mongodb://localhost:27017",
                "CANDIDATE_START_MODE": "domain_main_page",
            },
            prefer_process_env=True,
        )


def test_config_accepts_allow_parallel_workers_flag(tmp_path: Path) -> None:
    cfg = AppConfig.from_env(
        root=tmp_path,
        env_overrides={
            "MONGODB_URI": "mongodb://localhost:27017",
            "ALLOW_PARALLEL_WORKERS": "true",
        },
        prefer_process_env=True,
    )

    assert cfg.allow_parallel_workers is True


def test_config_accepts_browser_delegate_max_steps(tmp_path: Path) -> None:
    cfg = AppConfig.from_env(
        root=tmp_path,
        env_overrides={
            "MONGODB_URI": "mongodb://localhost:27017",
            "BROWSER_DELEGATE_MAX_STEPS": "9",
        },
        prefer_process_env=True,
    )

    assert cfg.browser_delegate_max_steps == 9


def test_config_rejects_invalid_browser_delegate_max_steps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="BROWSER_DELEGATE_MAX_STEPS must be >= 1"):
        AppConfig.from_env(
            root=tmp_path,
            env_overrides={
                "MONGODB_URI": "mongodb://localhost:27017",
                "BROWSER_DELEGATE_MAX_STEPS": "0",
            },
            prefer_process_env=True,
        )
