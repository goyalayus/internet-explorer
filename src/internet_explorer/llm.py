from __future__ import annotations

import json
from typing import TypeVar

from openai import AsyncAzureOpenAI
from pydantic import BaseModel

from internet_explorer.config import AppConfig

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = AsyncAzureOpenAI(
            azure_endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            timeout=180,
        )

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        temperature: float = 0.1,
        max_completion_tokens: int = 4096,
    ) -> T:
        response = await self.client.chat.completions.create(
            model=self.config.azure_openai_model,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        return schema.model_validate(parsed)

