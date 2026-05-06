"""Google Gemini SDK wrapper.

The external interface (`LLMClient.complete(system_blocks, user_message) -> str`)
is unchanged from the prior Anthropic / Gemini 2.0 / Groq implementations, so
callers do not know the provider changed. The Anthropic-shaped `cache_control`
field on system blocks is silently ignored — Gemini does not consume it.

The three system blocks (rules, corpus, per-request prediction context) are
concatenated into a single `system_instruction` on the GenerateContentConfig.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        max_tokens: int = 1024,
    ) -> None:
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._client = genai.Client(api_key=key)
        self.model_name = model
        self.max_tokens = max_tokens

    async def complete(
        self,
        system_blocks: list[dict],
        user_message: str,
    ) -> str:
        system_text = "\n\n".join(
            block["text"] for block in system_blocks if block.get("type") == "text"
        )
        response = await self._client.aio.models.generate_content(
            model=self.model_name,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_text,
                max_output_tokens=self.max_tokens,
            ),
        )
        text = response.text
        if text is None:
            return ""
        return text.strip()
