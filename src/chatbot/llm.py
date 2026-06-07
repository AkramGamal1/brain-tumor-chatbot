"""Google Gemini SDK wrapper.

The external interface (`LLMClient.complete(system_blocks, user_message) -> str`)
is unchanged from the prior Anthropic / Gemini 2.0 / Groq implementations, so
callers do not know the provider changed. The Anthropic-shaped `cache_control`
field on system blocks is silently ignored — Gemini does not consume it.

The three system blocks (rules, corpus, per-request prediction context) are
concatenated into a single `system_instruction` on the GenerateContentConfig.

Upstream errors are translated to provider-agnostic exceptions
(`LLMRateLimited`, `LLMUpstreamUnavailable`, `LLMServiceError`) so the API
layer can return clean JSON responses without importing google.genai.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import errors as genai_errors
from google.genai import types


class LLMServiceError(Exception):
    """Generic upstream LLM failure with a user-renderable message."""

    def __init__(self, user_message: str, *, status_code: int | None = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.status_code = status_code


class LLMRateLimited(LLMServiceError):
    """429 RESOURCE_EXHAUSTED — quota exhausted, retry later."""


class LLMUpstreamUnavailable(LLMServiceError):
    """503 UNAVAILABLE — the upstream model is temporarily down."""


def _translate(exc: genai_errors.APIError) -> LLMServiceError:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return LLMRateLimited(
            "The chatbot has reached its daily request limit. Please try again later.",
            status_code=429,
        )
    if code == 503:
        return LLMUpstreamUnavailable(
            "The chatbot is temporarily unavailable. Please try again in a moment.",
            status_code=503,
        )
    return LLMServiceError(
        "The chatbot ran into an unexpected upstream error. Please try again.",
        status_code=code,
    )


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        fallback_model: str = "gemini-2.0-flash-lite",
        max_tokens: int = 1024,
    ) -> None:
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._client = genai.Client(api_key=key)
        self.model_name = model
        self.fallback_model = fallback_model
        self.max_tokens = max_tokens

    async def complete(
        self,
        system_blocks: list[dict],
        user_message: str,
    ) -> str:
        system_text = "\n\n".join(
            block["text"] for block in system_blocks if block.get("type") == "text"
        )
        for model in (self.model_name, self.fallback_model):
            try:
                response = await self._client.aio.models.generate_content(
                    model=model,
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=system_text,
                        max_output_tokens=self.max_tokens,
                    ),
                )
                text = response.text
                return text.strip() if text else ""
            except genai_errors.APIError as exc:
                err = _translate(exc)
                if isinstance(err, LLMRateLimited) and model == self.model_name:
                    continue   # try fallback
                raise err from exc
        # both exhausted
        raise LLMRateLimited(
            "The chatbot has reached its daily request limit on all available models. "
            "Please try again tomorrow.",
            status_code=429,
        )
