"""
OpenAI client wrappers.

Two surfaces:
  - `LLMClassifierCall` — sync call returning structured JSON for the
    classifier. Wired into `classifier.classify` as the `llm` argument.
  - `LLMNarrator` — async streaming for agent narrative. Has a `stream()`
    method that yields token chunks.

Both lazy-construct the OpenAI client. If `OPENAI_API_KEY` is missing,
they remain unusable but DON'T crash on import — that's important for
running tests with no key set.
"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from .config import settings

log = logging.getLogger(__name__)


class _OpenAIClientHolder:
    _client: Any = None
    _async_client: Any = None

    @classmethod
    def sync(cls):
        if cls._client is not None:
            return cls._client
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        import openai
        cls._client = openai.OpenAI(api_key=settings.openai_api_key)
        return cls._client

    @classmethod
    def async_(cls):
        if cls._async_client is not None:
            return cls._async_client
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        import openai
        cls._async_client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        return cls._async_client


# ---------------------------------------------------------------------------
# Classifier callable
# ---------------------------------------------------------------------------
class LLMClassifierCall:
    """Implements the `LLMCall` protocol used by classifier.llm.

    Calling it like `instance(system, user)` returns parsed JSON.
    """

    def __init__(self, model: str | None = None, timeout_s: float | None = None):
        self.model = model or settings.model_dev
        self.timeout = timeout_s if timeout_s is not None else settings.classifier_llm_timeout_s

    def __call__(self, system: str, user: str) -> dict[str, Any] | None:
        client = _OpenAIClientHolder.sync()
        # Use JSON mode (response_format json_object) — supported on both
        # gpt-4o-mini and gpt-4.1.
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=self.timeout,
            )
        except Exception as e:
            log.warning("LLM classifier call failed: %s", e)
            return None
        try:
            content = resp.choices[0].message.content or ""
            return json.loads(content)
        except (json.JSONDecodeError, IndexError, AttributeError) as e:
            log.warning("LLM classifier returned malformed JSON: %s", e)
            return None


# ---------------------------------------------------------------------------
# Streaming narrator
# ---------------------------------------------------------------------------
class LLMNarrator:
    """Streams plain-text narrative tokens. Used by agents for human-friendly
    summaries on top of structured output."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.model_dev

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        client = _OpenAIClientHolder.async_()
        try:
            stream = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a concise wealth-management assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None)
                if text:
                    yield text
        except Exception as e:
            log.warning("LLM narrator stream failed: %s", e)
            return
