from __future__ import annotations

import asyncio
import json
import shlex
from urllib.parse import urlparse

from pydantic import BaseModel

from internet_explorer.bash_runner import BashRunner
from internet_explorer.llm import LLMClient
from internet_explorer.models import ApiProbeResult, ApiSignal, PageEvidence
from internet_explorer.telemetry import Telemetry


ALLOWED_FILTER_COMMANDS = {"jq", "grep", "rg", "sed", "awk", "head", "tail", "cut", "tr", "sort", "uniq", "wc"}
FORBIDDEN_TOKENS = {";", "&&", "||", ">", ">>", "<", "<<", "&"}
FORBIDDEN_SUBSTRINGS = ("$(", "`")

API_PROBE_SYSTEM_PROMPT = """
You create a read-only bash command for API inspection.
The command must start with curl and may optionally pipe into read-only text-processing commands.
Never write files, never invoke shells inside shells, never use sudo, never mutate the system.
Return only JSON.
"""


class _ApiProbePlan(BaseModel):
    bash_command: str
    reason: str = ""


class ApiProbeService:
    def __init__(self, telemetry: Telemetry, llm: LLMClient, *, timeout_seconds: int = 30) -> None:
        self.telemetry = telemetry
        self.llm = llm
        self.runner = BashRunner(timeout_seconds=timeout_seconds)

    async def probe(self, *, url_id: str, intent: str, evidence: list[PageEvidence]) -> ApiProbeResult | None:
        api_signal = self._merge_signal(evidence)
        candidate_url = self._pick_probe_url(api_signal)
        if not candidate_url:
            return None

        shell_command, planner_reason, planner_fallback_used = await self._plan_shell_command(
            intent=intent,
            candidate_url=candidate_url,
            api_signal=api_signal,
        )
        started = self.telemetry.timed()
        try:
            shell_result = await asyncio.to_thread(self.runner.run_shell, shell_command)
        except Exception as exc:
            probe = ApiProbeResult(
                attempted=True,
                command=["bash", "-lc", shell_command],
                shell_command=shell_command,
                url=candidate_url,
                planner_reason=planner_reason,
                planner_fallback_used=planner_fallback_used,
                error=str(exc),
            )
            self.telemetry.emit(
                phase="api_verify",
                actor="normal_agent",
                url_id=url_id,
                input_payload={"intent": intent, "url": candidate_url, "shell_command": shell_command},
                output_summary=probe.model_dump(mode="json"),
                decision="probe_error",
                latency_ms=self.telemetry.elapsed_ms(started),
                error_code=type(exc).__name__,
            )
            return probe

        probe = self._to_probe_result(
            candidate_url,
            ["bash", "-lc", shell_command],
            shell_command,
            planner_reason,
            planner_fallback_used,
            shell_result.returncode,
            shell_result.stdout,
            shell_result.stderr,
        )
        self.telemetry.emit(
            phase="api_verify",
            actor="normal_agent",
            url_id=url_id,
            input_payload={"intent": intent, "url": candidate_url, "shell_command": shell_command},
            output_summary=probe.model_dump(mode="json"),
            decision="probe_success" if probe.success else "probe_failed",
            latency_ms=self.telemetry.elapsed_ms(started),
            error_code=None if probe.success else "bash_probe_failed",
        )
        return probe

    async def _plan_shell_command(self, *, intent: str, candidate_url: str, api_signal: ApiSignal) -> tuple[str, str, bool]:
        fallback_command = self._fallback_shell_command(candidate_url)
        try:
            response = await self.llm.complete_json(
                system_prompt=API_PROBE_SYSTEM_PROMPT,
                user_prompt=(
                    "Create a single read-only bash command to inspect whether this API-like URL is accessible and useful.\n\n"
                    f"Intent:\n{intent}\n\n"
                    f"Candidate URL:\n{candidate_url}\n\n"
                    f"API signal:\n{api_signal.model_dump(mode='json')}\n\n"
                    "Rules:\n"
                    "- Start the command with `curl`.\n"
                    "- You may pipe only into `jq`, `grep`, `rg`, `sed`, `awk`, `head`, `tail`, `cut`, `tr`, `sort`, `uniq`, or `wc`.\n"
                    "- No file writes, no redirects, no subshells, no sudo, no `&&`, no `||`, no `;`.\n"
                    "- Keep it to one command line.\n"
                    "- Prefer commands that leave enough output to judge whether the response looks machine-readable.\n"
                    "- Quote URLs.\n"
                    "- Return only JSON.\n"
                ),
                schema=_ApiProbePlan,
                temperature=0.0,
                max_completion_tokens=500,
            )
            shell_command = response.bash_command.strip()
            if not shell_command:
                raise ValueError("planner returned an empty bash command")
            self._validate_shell_command(shell_command)
            for command_name in self._extract_pipeline_commands(shell_command):
                self.runner.ensure_command(command_name)
            return shell_command, response.reason.strip(), False
        except Exception as exc:
            return fallback_command, f"fallback:{type(exc).__name__}:{exc}", True

    def _fallback_shell_command(self, candidate_url: str) -> str:
        return (
            f"curl -sS -L --max-time {self.runner.timeout_seconds} "
            f"-D - "
            f"-H {shlex.quote('Accept: application/json, application/yaml, text/plain, text/html;q=0.8, */*;q=0.5')} "
            f"{shlex.quote(candidate_url)} | head -c 8000"
        )

    def _validate_shell_command(self, shell_command: str) -> None:
        if not shell_command.strip():
            raise ValueError("empty shell command")
        if "\n" in shell_command or "\r" in shell_command:
            raise ValueError("shell command must be single-line")
        if any(token in shell_command for token in FORBIDDEN_SUBSTRINGS):
            raise ValueError("shell command contains forbidden shell expansion")

        tokens = self._tokenize(shell_command)
        if not tokens:
            raise ValueError("shell command could not be tokenized")
        if any(token in FORBIDDEN_TOKENS for token in tokens):
            raise ValueError("shell command contains forbidden control operators")

        pipeline_commands = self._extract_pipeline_commands_from_tokens(tokens)
        if not pipeline_commands:
            raise ValueError("no commands found in shell pipeline")
        if pipeline_commands[0] != "curl":
            raise ValueError("shell command must start with curl")
        for command_name in pipeline_commands[1:]:
            if command_name not in ALLOWED_FILTER_COMMANDS:
                raise ValueError(f"command `{command_name}` is not allowed in probe pipeline")

    def _tokenize(self, shell_command: str) -> list[str]:
        lexer = shlex.shlex(shell_command, posix=True, punctuation_chars="|&;<>")
        lexer.whitespace_split = True
        return list(lexer)

    def _extract_pipeline_commands(self, shell_command: str) -> list[str]:
        return self._extract_pipeline_commands_from_tokens(self._tokenize(shell_command))

    def _extract_pipeline_commands_from_tokens(self, tokens: list[str]) -> list[str]:
        commands: list[str] = []
        segment: list[str] = []
        for token in tokens + ["|"]:
            if token == "|":
                if not segment:
                    raise ValueError("invalid empty pipeline segment")
                commands.append(segment[0])
                segment = []
                continue
            segment.append(token)
        return commands

    def _pick_probe_url(self, api_signal: ApiSignal) -> str:
        for link in api_signal.openapi_links:
            if link:
                return link
        for link in api_signal.doc_links:
            if link:
                return link
        if api_signal.graphql_hints:
            return api_signal.graphql_hints[0]
        return ""

    def _merge_signal(self, evidence: list[PageEvidence]) -> ApiSignal:
        merged = ApiSignal()
        for page in evidence:
            signal = page.api_signal
            merged.detected = merged.detected or signal.detected
            merged.auth_required = merged.auth_required or signal.auth_required
            for link in signal.doc_links:
                if link not in merged.doc_links:
                    merged.doc_links.append(link)
            for link in signal.openapi_links:
                if link not in merged.openapi_links:
                    merged.openapi_links.append(link)
            for hint in signal.graphql_hints:
                if hint not in merged.graphql_hints:
                    merged.graphql_hints.append(hint)
        return merged

    def _to_probe_result(
        self,
        url: str,
        command: list[str],
        shell_command: str,
        planner_reason: str,
        planner_fallback_used: bool,
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> ApiProbeResult:
        headers, body = self._split_headers_and_body(stdout or "")
        body = body[:4000]
        content_type = self._guess_content_type(url, body)
        status_code = self._extract_status_code(headers)
        success = returncode == 0
        accessible = success and bool(body.strip()) and status_code not in {401, 403}
        relevant_guess = accessible and self._looks_machine_readable(body)
        viable_guess = relevant_guess and status_code not in {429}
        return ApiProbeResult(
            attempted=True,
            command=command,
            shell_command=shell_command,
            url=url,
            status_code=status_code,
            content_type=content_type,
            success=success,
            accessible=accessible,
            relevant_guess=relevant_guess,
            viable_guess=viable_guess,
            planner_reason=planner_reason,
            planner_fallback_used=planner_fallback_used,
            response_excerpt=body[:1200],
            error=stderr[:1200],
        )

    def _split_headers_and_body(self, stdout: str) -> tuple[str, str]:
        if "\r\n\r\n" in stdout:
            headers, body = stdout.rsplit("\r\n\r\n", 1)
            return headers, body
        if "\n\n" in stdout:
            headers, body = stdout.rsplit("\n\n", 1)
            return headers, body
        return "", stdout

    def _guess_content_type(self, url: str, body: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.lower()
        if path.endswith(".json"):
            return "application/json"
        if path.endswith(".yaml") or path.endswith(".yml"):
            return "application/yaml"
        stripped = body.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                json.loads(stripped)
                return "application/json"
            except json.JSONDecodeError:
                pass
        if "openapi:" in stripped[:200].lower():
            return "application/yaml"
        if "<html" in stripped[:200].lower():
            return "text/html"
        return "text/plain"

    def _extract_status_code(self, headers: str) -> int | None:
        for line in headers.splitlines():
            if line.startswith("HTTP/"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        return int(parts[1])
                    except ValueError:
                        return None
        return None

    def _looks_machine_readable(self, body: str) -> bool:
        stripped = body.strip()
        if not stripped:
            return False
        if stripped.startswith("{") or stripped.startswith("["):
            return True
        lowered = stripped[:800].lower()
        return "openapi" in lowered or "swagger" in lowered or "graphql" in lowered
