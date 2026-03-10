from __future__ import annotations

import asyncio
import uuid
from threading import Lock
from typing import Any

from internet_explorer.config import AppConfig
from internet_explorer.models import BrowserDelegateResult, BrowserStep
from internet_explorer.repo_bridge import load_eu_swarm_modules
from internet_explorer.site_graph import SiteGraph
from internet_explorer.telemetry import Telemetry

VALID_OUTCOMES = {
    "data_on_site",
    "api_available",
    "contact_sales_only",
    "paywall",
    "irrelevant",
    "unknown",
}


class BrowserDelegationManager:
    def __init__(self, config: AppConfig, telemetry: Telemetry, update_run_callback) -> None:
        self.config = config
        self.telemetry = telemetry
        self.update_run_callback = update_run_callback
        self.modules = load_eu_swarm_modules(config)
        self._lock = Lock()
        self._active = 0
        self._peak = 0
        self._semaphore = asyncio.Semaphore(config.max_browser_concurrency) if config.max_browser_concurrency > 0 else None

    @property
    def peak(self) -> int:
        return self._peak

    def _wrap_tools(self, agent: Any, session_name: str) -> None:
        ToolFunction = self.modules["ToolFunction"]
        wrapped_tools = []
        for tool in agent.tools:
            if not tool.name.startswith("bu_"):
                wrapped_tools.append(tool)
                continue

            def make_fn(original_tool):
                def _fn(**kwargs):
                    kwargs.setdefault("session_name", session_name)
                    if original_tool.name == "bu_open":
                        kwargs.setdefault("headed", self.config.browser_headed)
                        kwargs.setdefault("browser_mode", self.config.browser_mode)
                    return original_tool.fn(**kwargs)

                return _fn

            wrapped_tools.append(
                ToolFunction(
                    name=tool.name,
                    description=tool.description,
                    fn=make_fn(tool),
                    input_model=tool.input_model,
                    output_model=tool.output_model,
                )
            )
        agent.tools = wrapped_tools

    def _inc(self) -> int:
        with self._lock:
            self._active += 1
            self._peak = max(self._peak, self._active)
            peak = self._peak
        self.update_run_callback({"browser_peak_active": peak})
        return self._active

    def _dec(self) -> int:
        with self._lock:
            self._active = max(0, self._active - 1)
            return self._active

    async def delegate(self, *, url: str, intent: str, url_id: str, site_graph: SiteGraph | None = None) -> BrowserDelegateResult:
        if self._semaphore is not None:
            async with self._semaphore:
                return await self._delegate_unbounded(url=url, intent=intent, url_id=url_id, site_graph=site_graph)
        return await self._delegate_unbounded(url=url, intent=intent, url_id=url_id, site_graph=site_graph)

    async def _delegate_unbounded(self, *, url: str, intent: str, url_id: str, site_graph: SiteGraph | None) -> BrowserDelegateResult:
        session_name = f"ie_{url_id}_{uuid.uuid4().hex[:8]}"
        active = self._inc()
        self.telemetry.emit(
            phase="browser_delegate",
            actor="system",
            url_id=url_id,
            input_payload={"url": url, "intent": intent, "session_name": session_name},
            output_summary={"active_browser_sessions": active},
            decision="delegate_start",
        )
        try:
            result = await asyncio.to_thread(self._run_delegate_sync, session_name, url, intent, url_id, site_graph)
            self.telemetry.emit(
                phase="browser_delegate",
                actor="browser_agent",
                url_id=url_id,
                input_payload={"url": url},
                output_summary=result.model_dump(mode="json"),
                decision=result.classification,
            )
            for recipe_step in result.recipe:
                self.telemetry.emit(
                    phase="browser_step",
                    actor="browser_agent",
                    url_id=url_id,
                    input_payload=recipe_step.params,
                    output_summary=recipe_step.observations,
                    decision=recipe_step.action,
                    extra={"browser_step_no": recipe_step.step_no},
                )
            return result
        finally:
            active_after = self._dec()
            self.telemetry.emit(
                phase="browser_delegate",
                actor="system",
                url_id=url_id,
                output_summary={"active_browser_sessions": active_after},
                decision="delegate_end",
            )

    def _run_delegate_sync(self, session_name: str, url: str, intent: str, url_id: str, site_graph: SiteGraph | None) -> BrowserDelegateResult:
        AzureOpenAIProvider = self.modules["AzureOpenAIProvider"]
        create_agent = self.modules["create_agent"]
        Task = self.modules["Task"]
        close_session = self.modules["close_session"]
        ToolFunction = self.modules["ToolFunction"]

        provider = AzureOpenAIProvider(
            model=self.config.azure_openai_model,
            api_key=self.config.azure_openai_api_key,
            azure_endpoint=self.config.azure_openai_endpoint,
            api_version=self.config.azure_openai_api_version,
            temperature=0.0,
        )
        workspace = self.config.eu_swarm_path / "smart_scraping_path_identifier"
        extra_tools = site_graph.build_browser_tools(ToolFunction) if site_graph is not None else None
        agent = create_agent(
            provider=provider,
            workspace=workspace,
            debug_mode=False,
            max_iterations=18,
            extra_tools=extra_tools,
        )
        if extra_tools:
            existing = {tool.name for tool in agent.tools}
            agent.tools.extend(tool for tool in extra_tools if tool.name not in existing)
        self._wrap_tools(agent, session_name)
        task = Task(
            description=(
                "You are evaluating a website as a possible datasource for a user intent.\n\n"
                f"Intent:\n{intent}\n\n"
                f"Start URL:\n{url}\n\n"
                f"Hard requirements:\n"
                f"- Use the browser tools on session_name `{session_name}`.\n"
                "- Start with bu_open then bu_state.\n"
                "- Explore enough pages to decide whether this source is useful.\n"
                "- If the page is clearly dynamic/hard to inspect statically, keep exploring with browser tools.\n"
                "- Detect whether the source has data on site, an API, only contact-sales, a paywall, or is irrelevant.\n"
                "- If captcha or phone OTP appears, stop and report it.\n"
                "- If payment/paywall appears during signup or access, classify as paywall.\n"
                "- Do not buy anything.\n"
                "- Use sg_read_tree or sg_get_frontier when helpful to understand the current site structure.\n"
                "- When you inspect an important page, use sg_record_page with a concise summary instead of relying on long history.\n"
                "- If you discover important internal pages, use sg_add_links so the shared site graph stays current.\n"
            ),
            expected_output="JSON with task_result, recipe, evidence, assumptions, confidence",
            output_schema={
                "task_result": "object",
                "recipe": "array",
                "evidence": "array",
                "assumptions": "array",
                "confidence": "number",
            },
        )
        try:
            raw = task.execute(agent)
            task_result = raw.get("task_result", {}) if isinstance(raw, dict) else {}
            recipe = raw.get("recipe", []) if isinstance(raw, dict) else []
            parsed_recipe = [
                BrowserStep(
                    step_no=index + 1,
                    action=str(step.get("action", "")),
                    params=step.get("params", {}) if isinstance(step, dict) else {},
                    observations=step.get("observations", {}) if isinstance(step, dict) else {},
                )
                for index, step in enumerate(recipe if isinstance(recipe, list) else [])
            ]
            evidence_snippets = []
            for evidence in raw.get("evidence", []) if isinstance(raw, dict) else []:
                if isinstance(evidence, dict) and evidence.get("snippet"):
                    evidence_snippets.append(str(evidence["snippet"]))
            classification = str(task_result.get("classification", "unknown")).strip()
            if classification not in VALID_OUTCOMES:
                classification = "unknown"
            return BrowserDelegateResult(
                session_name=session_name,
                classification=classification,
                useful=bool(task_result.get("useful", False)),
                why_useful=str(task_result.get("why_useful", "")),
                how_to_use=str(task_result.get("how_to_use", "")),
                render_path=str(task_result.get("render_path", "")),
                data_on_site=bool(task_result.get("data_on_site", False)),
                api_detected=bool(task_result.get("api_detected", False)),
                api_accessible_guess=bool(task_result.get("api_accessible_guess", False)),
                contact_sales_only=bool(task_result.get("contact_sales_only", False)),
                paywall_present=bool(task_result.get("paywall_present", False)),
                auth_required=bool(task_result.get("auth_required", False)),
                captcha_present=bool(task_result.get("captcha_present", False)),
                relevant_links=[str(link) for link in task_result.get("relevant_links", []) if link],
                evidence_snippets=evidence_snippets,
                recipe=parsed_recipe,
                confidence=float(raw.get("confidence", 0.0)) if isinstance(raw, dict) else 0.0,
                raw_output=raw if isinstance(raw, dict) else {},
            )
        finally:
            try:
                close_session(session_name=session_name, all_sessions=False)
            except Exception:
                pass
