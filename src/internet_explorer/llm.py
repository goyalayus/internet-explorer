from __future__ import annotations

import ast
import asyncio
import base64
import json
import re
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

from internet_explorer.config import AppConfig

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient(timeout=180)
        self.endpoint_template = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.api_keys = _parse_api_keys(
            config.gemini_api_keys,
            fallback=config.gemini_api_key or config.google_api_key,
        )
        self._key_cursor = 0

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        temperature: float = 0.1,
        max_completion_tokens: int = 4096,
    ) -> T:
        if not self.api_keys:
            raise ValueError("GEMINI_API_KEY or GEMINI_API_KEYS is required for LLM requests.")

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_completion_tokens,
                "responseMimeType": "application/json",
            },
        }
        text = await self._call_gemini(payload=payload)
        return await self._decode_schema_payload(
            raw_text=text,
            schema=schema,
            max_completion_tokens=max_completion_tokens,
        )

    async def complete_pdf_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        pdf_bytes: bytes,
        schema: type[T],
        temperature: float = 0.1,
        max_completion_tokens: int = 4096,
    ) -> T:
        if not self.api_keys:
            raise ValueError("GEMINI_API_KEY or GEMINI_API_KEYS is required for LLM requests.")
        if not pdf_bytes:
            raise ValueError("pdf_bytes is required")
        if len(pdf_bytes) > self.config.pdf_inline_max_bytes:
            raise ValueError(
                f"pdf_too_large_for_inline_gemini:{len(pdf_bytes)}>{self.config.pdf_inline_max_bytes}"
            )

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_prompt},
                        {
                            "inlineData": {
                                "mimeType": "application/pdf",
                                "data": base64.b64encode(pdf_bytes).decode("ascii"),
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_completion_tokens,
                "responseMimeType": "application/json",
            },
        }
        text = await self._call_gemini(payload=payload)
        return await self._decode_schema_payload(
            raw_text=text,
            schema=schema,
            max_completion_tokens=max_completion_tokens,
        )

    async def _call_gemini(self, *, payload: dict) -> str:
        attempts = max(1, self.config.llm_max_retries + 1)
        last_error: Exception | None = None
        url = self.endpoint_template.format(model=self.config.gemini_model)
        for attempt in range(1, attempts + 1):
            base_index = self._next_key_index()
            for offset in range(len(self.api_keys)):
                key_index = (base_index + offset) % len(self.api_keys)
                key = self.api_keys[key_index]
                try:
                    response = await self.client.post(url, params={"key": key}, json=payload)
                except Exception as exc:
                    last_error = RuntimeError(f"Gemini transport error ({type(exc).__name__}): {exc}")
                    continue

                if response.status_code >= 400:
                    detail = response.text[:500]
                    retryable = response.status_code in {408, 409, 429, 500, 502, 503, 504}
                    last_error = RuntimeError(f"Gemini request failed status={response.status_code}: {detail}")
                    if retryable:
                        continue
                    raise last_error

                text = _extract_text_from_gemini_response(response.json())
                if text:
                    return text
                last_error = RuntimeError("Gemini response did not contain text content")

            if attempt < attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 8))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini request failed without a captured error")

    def _next_key_index(self) -> int:
        if not self.api_keys:
            return 0
        index = self._key_cursor % len(self.api_keys)
        self._key_cursor += 1
        return index

    async def _decode_schema_payload(
        self,
        *,
        raw_text: str,
        schema: type[T],
        max_completion_tokens: int,
    ) -> T:
        last_exc: Exception | None = None
        candidate_text = raw_text
        for attempt in range(2):
            try:
                parsed = _extract_json_payload(candidate_text)
                parsed = _normalize_payload_for_schema(parsed, schema=schema)
                return schema.model_validate(parsed)
            except Exception as exc:
                last_exc = exc
                if attempt >= 1:
                    break
                candidate_text = await self._repair_json_output(
                    raw_text=candidate_text,
                    schema=schema,
                    error=type(exc).__name__,
                    max_completion_tokens=max_completion_tokens,
                )
        if last_exc is not None:
            raise last_exc
        raise ValueError("Could not decode schema payload")

    async def _repair_json_output(
        self,
        *,
        raw_text: str,
        schema: type[BaseModel],
        error: str,
        max_completion_tokens: int,
    ) -> str:
        schema_keys = list(schema.model_fields.keys())
        payload = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            "You repair malformed model output into strict JSON. "
                            "Return only a JSON object that follows the expected keys."
                        )
                    }
                ]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "Repair the following output into strict JSON only.\n\n"
                                f"Expected keys: {schema_keys}\n"
                                f"Observed error: {error}\n\n"
                                f"Output to repair:\n{raw_text}\n"
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": min(max(max_completion_tokens, 512), 2048),
                "responseMimeType": "application/json",
            },
        }
        return await self._call_gemini(payload=payload)


def _extract_text_from_gemini_response(payload: dict) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        return ""
    for candidate in candidates:
        content = candidate.get("content") if isinstance(candidate, dict) else None
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        for part in parts:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    return text
    return ""


def _extract_json_payload(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return {}
    parsed = _try_parse_json_or_literal(text)
    if parsed is not None:
        return parsed

    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence_match:
        parsed = _try_parse_json_or_literal(fence_match.group(1))
        if parsed is not None:
            return parsed

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", text):
        try:
            parsed, _ = decoder.raw_decode(text[match.start() :])
        except Exception:
            parsed = _try_parse_json_or_literal(text[match.start() :])
            if parsed is None:
                continue
        return parsed

    raise ValueError("Could not parse JSON object from LLM response")


def _normalize_payload_for_schema(payload: Any, *, schema: type[BaseModel]) -> Any:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        if len(payload) == 1 and isinstance(payload[0], dict):
            return payload[0]
        fields = list(schema.model_fields.keys())
        if len(fields) == 1:
            return {fields[0]: payload}
    return payload


def _parse_api_keys(raw: str, *, fallback: str = "") -> list[str]:
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    if keys:
        return keys
    if fallback.strip():
        return [fallback.strip()]
    return []


def _try_parse_json_or_literal(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return None
    if isinstance(parsed, (dict, list)):
        return parsed
    return None
