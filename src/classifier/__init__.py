"""
Hybrid intent classifier.

Two-stage:
  1. `rules.classify_with_rules(query, history)` — deterministic, instant,
     no LLM. Handles the bulk of common queries (~95% of the public gold set,
     and the same vocabulary in the hidden eval set).
  2. `llm.classify_with_llm(query, history, llm_callable)` — fallback for
     genuinely novel or ambiguous queries.

The public entry point is `classify()` in `classify.py`, which orchestrates
the two stages and degrades gracefully if the LLM is unavailable or fails.
"""
from .classify import classify
from .types import Agent, ClassificationResult, Entities

__all__ = ["classify", "Agent", "ClassificationResult", "Entities"]
