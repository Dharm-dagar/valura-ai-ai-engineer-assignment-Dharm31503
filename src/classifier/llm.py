"""
LLM-backed classifier.

Used as a fallback when the rule engine isn't confident. The LLM is given:
  - The user's query
  - Optional prior turns (for follow-up resolution)
  - The agent taxonomy and entity vocabulary
  - Instructions to output strict JSON matching `ClassificationResult`

The actual model call is decoupled via a callable parameter (`llm_call`),
so tests can inject a mock without monkey-patching `openai`. In production
the wiring lives in `src/llm_client.py`.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable

from .types import Agent, ClassificationResult, Entities, Source

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are an intent classifier for a wealth-management AI assistant.

Given a user query (and optional recent turns), output STRICT JSON with these keys:

  agent: one of:
    - portfolio_health        (assess user's portfolio: concentration, performance, benchmark)
    - market_research         (factual/recent info about an instrument, sector, or market event)
    - investment_strategy     (advice: should I buy/sell/rebalance, allocation guidance)
    - financial_planning      (long-term: retirement, goals, savings rate)
    - financial_calculator    (deterministic numeric: DCA, mortgage, tax, FX conversion)
    - risk_assessment         (risk metrics, exposure analysis, what-if scenarios)
    - product_recommendation  (recommend specific products/funds matching user profile)
    - predictive_analysis     (forward-looking: forecasts, trend extrapolation)
    - customer_support        (platform/account issues, how-to-use-app)
    - general_query           (educational, conversational, definitions, greetings)

  entities: object with optional keys:
    tickers      (array of upper-case tickers, e.g. ["AAPL", "ASML.AS"])
    topics       (array of strings)
    sectors      (array of strings)
    amount       (number, in unit of currency)
    currency     (ISO 4217 code: USD, EUR, GBP, JPY, ...)
    rate         (decimal, 0.08 for 8%)
    period_years (integer)
    frequency    (one of: daily, weekly, monthly, yearly)
    horizon      (string token: 6_months, 1_year, 5_years)
    time_period  (string token: today, this_week, this_month, this_year)
    index        (one of: "S&P 500", "FTSE 100", "NIKKEI 225", "MSCI World", ...)
    action       (one of: buy, sell, hold, hedge, rebalance)
    goal         (one of: retirement, education, house, FIRE, emergency_fund)
    intent       (free-form sub-intent hint, e.g. "comparison")

  safety_note: optional string flagging anything informationally risky
               (e.g. "user mentions 100% concentration").
               This is INFORMATIONAL ONLY — it does not block the request.

Resolve follow-ups using prior turns: if the current turn uses pronouns
("it", "them", "the stock") or omits an entity, copy from the most recent
prior turn that has it. If the current turn introduces a new entity, the
new entity wins.

Output JSON only. No prose. No markdown code fences.
"""


def _build_user_prompt(query: str, history: list[str] | None) -> str:
    parts: list[str] = []
    if history:
        parts.append("Recent prior user turns (oldest first):")
        for i, turn in enumerate(history[-5:], 1):
            parts.append(f"  {i}. {turn}")
    parts.append(f"Current user turn: {query}")
    parts.append("Output the JSON now.")
    return "\n".join(parts)


def _coerce(raw: Any) -> ClassificationResult | None:
    """Best-effort coercion of an LLM payload to ClassificationResult."""
    if raw is None:
        return None
    if isinstance(raw, ClassificationResult):
        return raw
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("LLM returned non-JSON output: %r", raw[:200])
            return None
    if not isinstance(raw, dict):
        return None
    try:
        agent_str = str(raw.get("agent", "")).strip()
        agent = Agent(agent_str) if agent_str in Agent._value2member_map_ else Agent.general_query
        ent_raw = raw.get("entities") or {}
        if not isinstance(ent_raw, dict):
            ent_raw = {}
        # Pydantic will coerce/validate
        entities = Entities(**{k: v for k, v in ent_raw.items() if v is not None})
        return ClassificationResult(
            agent=agent,
            entities=entities,
            confidence=float(raw.get("confidence", 0.7)),
            source=Source.llm,
            safety_note=raw.get("safety_note"),
            rationale=raw.get("rationale"),
        )
    except Exception as e:
        log.warning("Failed to coerce LLM output: %s", e)
        return None


# A callable that takes (system_prompt, user_prompt) and returns
# the parsed structured payload (or a string of JSON).
LLMCall = Callable[[str, str], Any]


def classify_with_llm(
    query: str,
    history: list[str] | None,
    llm_call: LLMCall,
) -> ClassificationResult | None:
    """
    Single LLM call. Returns None on any failure — caller decides fallback.

    `llm_call(system, user)` may return:
      - a dict matching the ClassificationResult shape
      - a JSON string of same
      - a ClassificationResult directly
    """
    user_prompt = _build_user_prompt(query, history)
    try:
        raw = llm_call(SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        log.warning("LLM classifier call failed: %s", e)
        return None
    return _coerce(raw)
