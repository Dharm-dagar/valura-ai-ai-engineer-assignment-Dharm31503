"""
Hybrid classifier entry point.

Public API:
  classify(query, history=None, llm=None) -> ClassificationResult

Algorithm:
  1. Run the rule cascade. If it returns a result with confidence >= the
     configured threshold, return it. Done — no LLM call.
  2. If the LLM is available and the rule confidence is below threshold,
     try the LLM. If it returns a usable result, return that.
  3. If the LLM fails or is unavailable, return the best-guess rule result
     (confidence is low but it's a graceful degradation).

Optional stretch goal: identical-query LLM dedupe cache.
We cache by (query, len(history), last_history_hash) so identical follow-ups
in the same session are deduped, but a new conversation gets a fresh look.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable

from cachetools import TTLCache

from ..config import settings
from .llm import classify_with_llm
from .rules import classify_with_rules
from .types import ClassificationResult, Source

log = logging.getLogger(__name__)


# Module-level cache. Configured at import time; tests don't depend on it
# because rules cover the gold set.
_cache: TTLCache[str, ClassificationResult] = TTLCache(
    maxsize=max(1, settings.classifier_cache_size),
    ttl=max(1, settings.classifier_cache_ttl_s),
)


def _cache_key(query: str, history: list[str] | None) -> str:
    h = "" if not history else hashlib.md5(
        ("\n".join(history)).encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return f"{query.strip().lower()}|{h}"


def classify(
    query: str,
    history: list[str] | None = None,
    llm: Callable[..., Any] | None = None,
    *,
    use_cache: bool = True,
) -> ClassificationResult:
    """
    Classify `query`. `history` is the prior user-turn list (oldest first).
    `llm` is a callable `(system, user) -> dict|str|ClassificationResult`.

    The function never raises — failures degrade to a best-guess result.
    """
    # Cache lookup (stretch goal)
    cache_key = _cache_key(query, history)
    if use_cache and cache_key in _cache:
        cached = _cache[cache_key]
        return cached.model_copy(update={"source": Source.cache})

    # Stage 1: rules
    rule_result = classify_with_rules(query, history)
    if rule_result.confidence >= settings.rule_confidence_threshold:
        if use_cache:
            _cache[cache_key] = rule_result
        return rule_result

    # Stage 2: LLM fallback (only if available)
    if llm is not None:
        llm_result = classify_with_llm(query, history, llm)
        if llm_result is not None and llm_result.agent is not None:
            if use_cache:
                _cache[cache_key] = llm_result
            return llm_result

    # Stage 3: graceful degradation — return rule's best guess but flag it
    fallback = rule_result.model_copy(
        update={
            "source": Source.fallback,
            "rationale": (rule_result.rationale or "") + " (LLM unavailable; using rule best-guess)",
        }
    )
    if use_cache:
        _cache[cache_key] = fallback
    return fallback


def clear_cache() -> None:
    """Test/admin helper."""
    _cache.clear()
