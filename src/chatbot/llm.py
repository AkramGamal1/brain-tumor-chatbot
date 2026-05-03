"""Groq SDK wrapper.

The external interface (`LLMClient.complete(system_blocks, user_message) -> str`)
is unchanged from the prior Anthropic / Gemini implementations, so callers do
not know the provider changed. The Anthropic-shaped `cache_control` field on
system blocks is silently ignored (Groq's free tier does not expose prompt
caching, and the corpus is small).

The three system blocks (rules, corpus, per-request prediction context) are
concatenated into a single system message at the front of the chat completion's
messages list (OpenAI-compatible message shape).
"""

from __future__ import annotations

import os

from groq import AsyncGroq


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 1024,
    ) -> None:
        key = api_key or os.environ.get("GROQ_API_KEY")
        self._client = AsyncGroq(api_key=key)
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
        response = await self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_message},
            ],
            max_tokens=self.max_tokens,
        )
        choice = response.choices[0] if response.choices else None
        if choice is None or choice.message is None or choice.message.content is None:
            return ""
        return choice.message.content.strip()
