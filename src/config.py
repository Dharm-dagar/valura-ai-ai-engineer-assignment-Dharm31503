"""
Application settings.

Pulled from environment variables (via python-dotenv at app startup).
Keep this thin — no logic, just typed config.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # ----- LLM -----
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    # Two model tiers, per ASSIGNMENT.md: dev = gpt-4o-mini, eval = gpt-4.1.
    # We expose both so the per-tenant model selection stretch goal is trivial.
    model_dev: str = field(default_factory=lambda: os.environ.get("MODEL_DEV", "gpt-4o-mini"))
    model_premium: str = field(default_factory=lambda: os.environ.get("MODEL_PREMIUM", "gpt-4.1"))

    # ----- Pipeline -----
    pipeline_timeout_s: float = field(default_factory=lambda: _get_float("PIPELINE_TIMEOUT_S", 12.0))
    classifier_llm_timeout_s: float = field(default_factory=lambda: _get_float("CLASSIFIER_LLM_TIMEOUT_S", 4.0))
    rule_confidence_threshold: float = field(
        default_factory=lambda: _get_float("RULE_CONFIDENCE_THRESHOLD", 0.7)
    )

    # ----- Cache (stretch: identical-query LLM dedupe) -----
    classifier_cache_size: int = field(default_factory=lambda: _get_int("CLASSIFIER_CACHE_SIZE", 256))
    classifier_cache_ttl_s: int = field(default_factory=lambda: _get_int("CLASSIFIER_CACHE_TTL_S", 300))

    # ----- Session memory -----
    session_history_max_turns: int = field(
        default_factory=lambda: _get_int("SESSION_HISTORY_MAX_TURNS", 10)
    )

    # ----- Market data -----
    market_data_provider: str = field(
        default_factory=lambda: os.environ.get("MARKET_DATA_PROVIDER", "yfinance")
    )

    @property
    def llm_available(self) -> bool:
        return bool(self.openai_api_key)


# Singleton settings — read once at import time. Re-import is rare in practice;
# tests can construct Settings() directly without env vars.
settings = Settings()
