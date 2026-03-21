from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from threading import Lock
from typing import Any

from pydantic import BaseModel, Field

from internet_explorer.config import AppConfig
from internet_explorer.models import BrowserDelegateResult, BrowserStep, SourceEvidenceItem
from internet_explorer.pdf_verify import PdfVerifierService, pdf_verification_tool_payload
from internet_explorer.repo_bridge import load_eu_swarm_modules
from internet_explorer.telemetry import Telemetry

VALID_OUTCOMES = {
    "data_on_site",
    "api_available",
    "contact_sales_only",
    "paywall",
    "irrelevant",
    "unknown",
}
CLASSIFICATION_ALIASES = {
    "not_useful": "irrelevant",
    "not useful": "irrelevant",
    "not_relevant": "irrelevant",
    "relevant": "data_on_site",
}


class _VerifyPdfUrlInput(BaseModel):
    pdf_url: str = Field(..., min_length=1, description="Direct PDF URL to verify against the current intent.")


class BrowserDelegationManager:
    def __init__(
        self,
        config: AppConfig,
        telemetry: Telemetry,
        update_run_callback,
        pdf_verifier: PdfVerifierService | None = None,
    ) -> None:
        self.config = config
        self.telemetry = telemetry
        self.update_run_callback = update_run_callback
        self.pdf_verifier = pdf_verifier
        self.modules = load_eu_swarm_modules(config)
        self._lock = Lock()
        self._active = 0
        self._peak = 0
        self._semaphore = asyncio.Semaphore(config.max_browser_concurrency) if config.max_browser_concurrency > 0 else None

    @property
    def peak(self) -> int:
        return self._peak

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

    async def delegate(
        self,
        *,
        url: str,
        intent: str,
        url_id: str,
        initial_links: list[str] | None = None,
    ) -> BrowserDelegateResult:
        if self._semaphore is not None:
            async with self._semaphore:
                return await self._delegate_unbounded(url=url, intent=intent, url_id=url_id, initial_links=initial_links or [])
        return await self._delegate_unbounded(url=url, intent=intent, url_id=url_id, initial_links=initial_links or [])

    async def _delegate_unbounded(
        self,
        *,
        url: str,
        intent: str,
        url_id: str,
        initial_links: list[str],
    ) -> BrowserDelegateResult:
        session_name = f"ie_{url_id}_{uuid.uuid4().hex[:8]}"
        active = self._inc()
        self.telemetry.emit(
            phase="browser_delegate",
            actor="system",
            url_id=url_id,
            input_payload={"url": url, "intent": intent, "session_name": session_name, "initial_links_count": len(initial_links)},
            output_summary={"active_browser_sessions": active},
            decision="delegate_start",
        )
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(self._run_delegate_sync, session_name, url, intent, url_id, initial_links),
                timeout=max(1, self.config.browser_delegate_timeout_seconds),
            )
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
        except Exception as exc:
            fallback = self._delegate_error_result(
                session_name=session_name,
                url=url,
                intent=intent,
                error=exc,
            )
            self.telemetry.emit(
                phase="browser_delegate",
                actor="system",
                url_id=url_id,
                input_payload={"url": url},
                output_summary=fallback.model_dump(mode="json"),
                decision="delegate_failed_fallback",
                error_code=type(exc).__name__,
            )
            return fallback
        finally:
            active_after = self._dec()
            self.telemetry.emit(
                phase="browser_delegate",
                actor="system",
                url_id=url_id,
                output_summary={"active_browser_sessions": active_after},
                decision="delegate_end",
            )

    def _run_delegate_sync(
        self,
        session_name: str,
        url: str,
        intent: str,
        url_id: str,
        initial_links: list[str],
    ) -> BrowserDelegateResult:
        try:
            plan = self._plan_browser_task(intent=intent, start_url=url, url_id=url_id, initial_links=initial_links)
        except Exception as exc:
            plan = self._fallback_plan(start_url=url, intent=intent, error=exc)
            self.telemetry.emit(
                phase="browser_delegate",
                actor="normal_agent",
                url_id=url_id,
                input_payload={"intent": intent, "start_url": url},
                output_summary={"plan": plan.model_dump(mode="json"), "planner_error": _exception_to_text(exc)},
                decision="planner_fallback",
                error_code=type(exc).__name__,
            )
        native_result = asyncio.run(
            self._run_browser_use_native(
                browser_task=self._build_browser_task(intent=intent, start_url=plan.start_url, initial_links=initial_links, plan_task=plan.browser_task),
                start_url=plan.start_url,
                intent=intent,
                url_id=url_id,
                max_steps=plan.max_steps,
            )
        )
        return self._to_delegate_result(
            session_name=session_name,
            intent=intent,
            start_url=plan.start_url,
            plan=plan.model_dump(mode="json"),
            native_result=native_result,
        )

    def _plan_browser_task(self, *, intent: str, start_url: str, url_id: str, initial_links: list[str]):
        AzureOpenAIProvider = self.modules["AzureOpenAIProvider"]
        create_agent = self.modules["create_agent"]
        SmartScraperPlan = self.modules["SmartScraperPlan"]

        provider = AzureOpenAIProvider(
            model=self.config.azure_openai_model,
            api_key=self.config.azure_openai_api_key,
            azure_endpoint=self.config.azure_openai_endpoint,
            api_version=self.config.azure_openai_api_version,
            temperature=0.0,
        )
        planner = create_agent(
            provider=provider,
            debug_mode=False,
            max_iterations=18,
        )
        links_block = "\n".join(f"- {link}" for link in initial_links[:40]) if initial_links else "- (none)"
        planning_task = (
            "You are planning browser automation for datasource evaluation.\n\n"
            f"Intent:\n{intent}\n\n"
            f"Start URL:\n{start_url}\n\n"
            f"Initial links list:\n{links_block}\n\n"
            "Output structured plan only."
        )
        raw_plan = planner.execute(planning_task)
        plan = SmartScraperPlan.model_validate(raw_plan)
        self.telemetry.emit(
            phase="browser_delegate",
            actor="normal_agent",
            url_id=url_id,
            input_payload={"intent": intent, "start_url": start_url},
            output_summary=plan.model_dump(mode="json"),
            decision="planner_output",
        )
        return plan

    def _fallback_plan(self, *, start_url: str, intent: str, error: Exception):
        SmartScraperPlan = self.modules["SmartScraperPlan"]
        payload = {
            "planning_summary": "Fallback plan due to planner error.",
            "browser_task": (
                "Open the start URL, inspect RFP/data/API/PDF signals, and return strict JSON classification "
                "for datasource usefulness."
            ),
            "start_url": start_url,
            "max_steps": 20,
            "assumptions": [
                "Planner unavailable; using direct browser exploration fallback.",
                f"Planner error: {_exception_to_text(error)}",
                f"Intent: {intent}",
            ],
        }
        return SmartScraperPlan.model_validate(payload)

    async def _run_browser_use_native(
        self,
        *,
        browser_task: str,
        start_url: str,
        intent: str,
        url_id: str,
        max_steps: int,
    ) -> dict[str, Any]:
        BrowserUseAgent = self.modules["BrowserUseAgent"]
        BrowserUseBrowser = self.modules["BrowserUseBrowser"]
        llm = self._create_browser_use_llm()
        tools = self._build_browser_use_tools(intent=intent, url_id=url_id)

        browser = BrowserUseBrowser(
            headless=not self.config.browser_headed,
            keep_alive=False,
            chromium_sandbox=self.config.browser_chromium_sandbox,
        )
        await browser.start()
        try:
            agent_kwargs: dict[str, Any] = {
                "task": browser_task,
                "browser": browser,
                "llm": llm,
                "directly_open_url": True,
            }
            if tools is not None:
                agent_kwargs["tools"] = tools
                agent_kwargs["controller"] = tools
            agent = BrowserUseAgent(**agent_kwargs)
            history = await agent.run(max_steps=max_steps)
            return {
                "final_result": history.final_result() if hasattr(history, "final_result") else None,
                "is_successful": history.is_successful() if hasattr(history, "is_successful") else None,
                "action_history": history.action_history() if hasattr(history, "action_history") else [],
                "urls": history.urls() if hasattr(history, "urls") else [],
                "errors": history.errors() if hasattr(history, "errors") else [],
                "extracted_content": history.extracted_content() if hasattr(history, "extracted_content") else [],
            }
        finally:
            await browser.stop()

    def _create_browser_use_llm(self):
        get_llm_by_name = self.modules["get_browser_use_llm_by_name"]
        candidates: list[str] = []

        if self.config.browser_use_llm_model:
            candidates.append(self.config.browser_use_llm_model.strip())

        if any(
            [
                self.config.gemini_api_key,
                self.config.gemini_api_keys,
                os.getenv("GEMINI_API_KEY"),
                os.getenv("GOOGLE_API_KEY"),
            ]
        ):
            candidates.append("google_gemini_2_5_flash")

        if self.config.azure_openai_api_key and self.config.azure_openai_endpoint:
            candidates.append("azure_gpt_4_1_mini")

        if os.getenv("OPENAI_API_KEY"):
            candidates.append("openai_gpt_4o_mini")

        for candidate in candidates:
            if not candidate:
                continue
            try:
                return get_llm_by_name(candidate)
            except Exception:
                continue

        return None

    def _build_browser_use_tools(self, *, intent: str, url_id: str):
        if self.pdf_verifier is None:
            return None

        BrowserUseTools = self.modules["BrowserUseTools"]
        BrowserUseActionResult = self.modules["BrowserUseActionResult"]
        tools = BrowserUseTools()

        @tools.action(
            "Verify a direct PDF URL against the current intent and return whether it is relevant evidence.",
            param_model=_VerifyPdfUrlInput,
        )
        async def verify_pdf_url(params: _VerifyPdfUrlInput):
            result = await self.pdf_verifier.verify(url_id=url_id, intent=intent, pdf_url=params.pdf_url)
            payload = pdf_verification_tool_payload(result)
            memory = json.dumps(payload, ensure_ascii=True)
            return BrowserUseActionResult(
                extracted_content=memory,
                long_term_memory=memory,
                metadata={"pdf_verification": payload},
            )

        return tools

    def _build_browser_task(self, *, intent: str, start_url: str, initial_links: list[str], plan_task: str) -> str:
        links_block = "\n".join(f"- {link}" for link in initial_links[:80]) if initial_links else "- (none)"
        return (
            f"{plan_task}\n\n"
            "Datasource evaluation requirements:\n"
            f"- Intent: {intent}\n"
            f"- Start URL: {start_url}\n"
            "- You can navigate to other URLs in this domain when useful.\n"
            "- Use the initial links list as hints; you are not restricted to it.\n"
            "- Stay anchored to URLs you actually opened in this session. Do not mention or reason about a different domain unless you really navigated there.\n"
            "- If you hit a direct PDF and the `verify_pdf_url` tool is available, use it.\n"
            "- The PDF tool returns `fallback_urls`; after verifying a direct PDF, use those fallback URLs to continue exploration on the parent page, section, or homepage when needed.\n"
            "- If a page looks blank or empty, do one short wait and at most one reload or direct re-open.\n"
            "- If the page is still empty after that, stop wasting steps on it. Move to a better sibling/homepage/initial-link URL in the same domain, or return a clean failure/unknown judgment.\n"
            "- Do not spend more than 2 consecutive actions trying to revive the same empty page.\n"
            "- A loaded search or listing page with visible controls is not a blank-page failure state.\n"
            "- If the page shows search inputs, filters, tabs, result counts, pagination, or messages like 'no results' or 'set search criteria', stay on the domain and use that workflow before giving up.\n"
            "- Before leaving a loaded search/listing workflow page or calling `done`, do at least one manual on-page query reformulation and one filter/category/listing interaction if visible.\n"
            "- If a prefilled search URL returns no hits, do not immediately conclude failure. Try broader on-page terms such as `data annotation`, `data labeling`, `machine learning`, `procurement`, `tender`, or other domain-relevant variants.\n"
            "- Do not jump to Google or DuckDuckGo just because one internal search or listing path returned no results.\n"
            "- Use external search only if the start page is inaccessible, or after at least two internal navigation/search paths failed.\n"
            "- If external search leads to CAPTCHA, report blocked and stop. Do not spend steps trying to solve the CAPTCHA.\n"
            "- If captcha appears, stop and report captcha.\n"
            "- If payment/paywall appears for access, classify paywall.\n"
            "- If source has mostly contact-sales flow, classify contact_sales_only.\n"
            "- Example: if a procurement portal says 'Set Your Search Criteria', use the visible search bar and at least one filter before concluding failure.\n"
            "- Example: if a site search page loads with weak or zero results, reformulate the query on-site before leaving the domain.\n"
            "- Example: if the homepage is accessible but the right section is unclear, try menu/footer/site-search paths like procurement, tenders, opportunities, or vendor pages before using a generic search engine.\n"
            "- Your reasoning must explain both why the source matches the intent and the rough recurring access path for this domain.\n"
            "- Return STRICT JSON with keys:\n"
            "  classification, useful, reasoning, render_path,\n"
            "  data_on_site, api_detected, api_accessible_guess,\n"
            "  contact_sales_only, paywall_present, auth_required, captcha_present,\n"
            "  relevant_links, evidence_snippets, source_evidence, confidence.\n\n"
            f"Initial links list:\n{links_block}"
        )

    def _to_delegate_result(
        self,
        *,
        session_name: str,
        intent: str,
        start_url: str,
        plan: dict[str, Any],
        native_result: dict[str, Any],
    ) -> BrowserDelegateResult:
        parsed = _extract_json_dict(native_result.get("final_result"))
        classification = str((parsed or {}).get("classification", "unknown")).strip()
        classification = CLASSIFICATION_ALIASES.get(classification.lower(), classification)
        if classification not in VALID_OUTCOMES:
            classification = "unknown"

        urls = [str(item) for item in native_result.get("urls", []) if item]
        extracted = [str(item) for item in native_result.get("extracted_content", []) if item]
        relevant_links = [str(item) for item in (parsed or {}).get("relevant_links", []) if item]
        if not relevant_links:
            relevant_links = urls[:20]
        evidence_snippets = _coerce_string_list((parsed or {}).get("evidence_snippets")) or extracted[:10]
        confidence = _coerce_confidence((parsed or {}).get("confidence"), succeeded=bool(native_result.get("is_successful")))

        source_evidence = _coerce_source_evidence((parsed or {}).get("source_evidence"))
        if not source_evidence:
            for snippet in evidence_snippets[:3]:
                source_evidence.append(SourceEvidenceItem(kind="browser_finding", url=start_url, summary=snippet[:800]))

        return BrowserDelegateResult(
            session_name=session_name,
            classification=classification,
            useful=bool((parsed or {}).get("useful", False)),
            reasoning=str((parsed or {}).get("reasoning", "")).strip(),
            render_path=_normalize_render_path((parsed or {}).get("render_path")),
            data_on_site=bool((parsed or {}).get("data_on_site", False)),
            api_detected=bool((parsed or {}).get("api_detected", False)),
            api_accessible_guess=bool((parsed or {}).get("api_accessible_guess", False)),
            contact_sales_only=bool((parsed or {}).get("contact_sales_only", False)),
            paywall_present=bool((parsed or {}).get("paywall_present", False)),
            auth_required=bool((parsed or {}).get("auth_required", False)),
            captcha_present=bool((parsed or {}).get("captcha_present", False)),
            relevant_links=relevant_links,
            evidence_snippets=evidence_snippets,
            source_evidence=source_evidence,
            recipe=_history_to_recipe(native_result.get("action_history", [])),
            confidence=confidence,
            raw_output={
                "intent": intent,
                "start_url": start_url,
                "plan": plan,
                "native": native_result,
                "parsed_final_result": parsed,
            },
        )

    def _delegate_error_result(
        self,
        *,
        session_name: str,
        url: str,
        intent: str,
        error: Exception,
    ) -> BrowserDelegateResult:
        return BrowserDelegateResult(
            session_name=session_name,
            classification="unknown",
            useful=False,
            reasoning="Browser delegation failed before completing exploration.",
            render_path="browser_delegate_fallback",
            data_on_site=False,
            api_detected=False,
            api_accessible_guess=False,
            contact_sales_only=False,
            paywall_present=False,
            auth_required=False,
            captcha_present=False,
            relevant_links=[url],
            evidence_snippets=[],
            source_evidence=[],
            recipe=[],
            confidence=0.0,
            raw_output={
                "intent": intent,
                "start_url": url,
                "error": _exception_to_text(error),
            },
        )


def _extract_json_dict(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _history_to_recipe(action_history: Any) -> list[BrowserStep]:
    if not isinstance(action_history, list):
        return []
    recipe: list[BrowserStep] = []
    step_no = 1
    for item in action_history:
        actions = item if isinstance(item, list) else [item]
        for action in actions:
            if isinstance(action, dict):
                if len(action) == 1:
                    action_name = next(iter(action))
                    params = action.get(action_name, {})
                else:
                    action_name = str(action.get("action", "browser_action"))
                    params = {key: value for key, value in action.items() if key != "action"}
                observations = {}
            else:
                action_name = str(action)
                params = {}
                observations = {}
            recipe.append(
                BrowserStep(
                    step_no=step_no,
                    action=action_name,
                    params=params if isinstance(params, dict) else {"value": params},
                    observations=observations,
                )
            )
            step_no += 1
    return recipe


def _coerce_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _coerce_confidence(value: Any, *, succeeded: bool) -> float:
    if value is None or value == "":
        return 0.8 if succeeded else 0.4
    try:
        return max(0.0, min(float(value), 1.0))
    except Exception:
        lowered = str(value).strip().lower()
        if lowered in {"high", "very high"}:
            return 0.9 if succeeded else 0.7
        if lowered in {"medium", "moderate"}:
            return 0.6 if succeeded else 0.4
        if lowered in {"low", "very low"}:
            return 0.3 if succeeded else 0.1
        return 0.8 if succeeded else 0.4


def _coerce_source_evidence(value: Any) -> list[SourceEvidenceItem]:
    if not isinstance(value, list):
        return []
    items: list[SourceEvidenceItem] = []
    for raw in value:
        coerced = _coerce_source_evidence_item(raw)
        if coerced is not None:
            items.append(coerced)
    return items


def _coerce_source_evidence_item(value: Any) -> SourceEvidenceItem | None:
    if isinstance(value, SourceEvidenceItem):
        return value
    if isinstance(value, dict):
        try:
            return SourceEvidenceItem.model_validate(value)
        except Exception:
            return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if raw.startswith(("http://", "https://")):
            return SourceEvidenceItem(kind="page", url=raw)
        return SourceEvidenceItem(kind="browser_finding", url="", summary=raw[:800])
    return None


def _normalize_render_path(value: Any) -> str:
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return " -> ".join(parts) if parts else "browser_use_native"
    text = str(value or "").strip()
    return text or "browser_use_native"


def _exception_to_text(exc: Exception) -> str:
    text = str(exc).strip()
    if text:
        return text
    return type(exc).__name__
