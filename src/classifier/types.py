"""
Classifier types.

`ClassificationResult` is the structured output the assignment requires from
the single LLM call. We use it as the canonical shape both for the
rules-engine output and for the LLM output, so the rest of the pipeline
doesn't have to care which path produced it.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# The taxonomy in fixtures/test_queries/intent_classification.json.
# Kept as a string Enum so it serializes cleanly to SSE JSON and is matched
# by string-equality against the gold set.
class Agent(str, Enum):
    portfolio_health = "portfolio_health"
    market_research = "market_research"
    investment_strategy = "investment_strategy"
    financial_planning = "financial_planning"
    financial_calculator = "financial_calculator"
    risk_assessment = "risk_assessment"
    product_recommendation = "product_recommendation"
    predictive_analysis = "predictive_analysis"
    customer_support = "customer_support"
    general_query = "general_query"


# Loose entity bag. Pydantic gives us validation for free. Every field is
# optional because user queries vary widely. We do NOT enforce closed
# vocabularies here (e.g. via Literal types) because the LLM should be free
# to extract novel sectors / topics, and the matcher in tests/ is what
# enforces the canonical vocabulary.
class Entities(BaseModel):
    tickers: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    amount: float | None = None
    currency: str | None = None
    rate: float | None = None
    period_years: int | None = None
    frequency: str | None = None       # daily | weekly | monthly | yearly
    horizon: str | None = None         # 6_months | 1_year | 5_years | ...
    time_period: str | None = None     # today | this_week | this_month | this_year
    index: str | None = None           # S&P 500 | FTSE 100 | NIKKEI 225 | MSCI World
    action: str | None = None          # buy | sell | hold | hedge | rebalance
    goal: str | None = None            # retirement | education | house | FIRE | emergency_fund
    intent: str | None = None          # free-form sub-intent hint (e.g. "comparison")

    def as_clean_dict(self) -> dict[str, Any]:
        """Drop None and empty-list fields. Useful for SSE payloads."""
        out: dict[str, Any] = {}
        for k, v in self.model_dump().items():
            if v is None:
                continue
            if isinstance(v, list) and not v:
                continue
            out[k] = v
        return out


# Source of the classification — useful for telemetry and for the README's
# "how I measured cost" defence.
class Source(str, Enum):
    rules = "rules"
    llm = "llm"
    cache = "cache"
    fallback = "fallback"


class ClassificationResult(BaseModel):
    agent: Agent
    entities: Entities = Field(default_factory=Entities)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: Source = Source.rules
    # Per ASSIGNMENT.md: "an informational safety verdict — all in one
    # structured output." This is INFORMATIONAL ONLY; it does not block.
    safety_note: str | None = None
    # Free-form rationale for telemetry / debugging (don't show to user).
    rationale: str | None = None
