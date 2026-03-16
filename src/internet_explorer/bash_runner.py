from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class BashResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


class BashRunner:
    def __init__(self, *, timeout_seconds: int = 30) -> None:
        self.timeout_seconds = timeout_seconds

    def ensure_command(self, command_name: str) -> None:
        if shutil.which(command_name) is None:
            raise RuntimeError(f"Missing required command: {command_name}")

    def run(self, command: list[str]) -> BashResult:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
        )
        return BashResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def run_shell(self, command: str) -> BashResult:
        self.ensure_command("bash")
        return self.run(["bash", "-lc", command])
