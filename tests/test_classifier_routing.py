"""
Classifier routing accuracy on the labeled gold set.

Threshold (from ASSIGNMENT.md):
  - ≥ 85% routing accuracy on the gold set in
    fixtures/test_queries/intent_classification.json

The entity matcher follows fixtures/README.md. Tickers are case-folded
and exchange suffix is stripped (AAPL.US → AAPL). Topics/sectors are
lowercase subset matches. Numeric fields (amount, rate) tolerate ±5%
to absorb floating-point noise. Vocabulary tokens (action, goal,
frequency, horizon, time_period, currency, index) are lowercase exact;
index is also spacing-tolerant ("S&P500" == "S&P 500").

Note: the conftest's `mock_llm` is an unconfigured MagicMock. Our
hybrid classifier handles the gold set entirely via deterministic
rules so the mock is never called — see README.md for the rationale.
"""
from typing import Any

from src.classifier import classify


# ---------------------------------------------------------------------------
# Entity matcher — implements the rules in fixtures/README.md
# ---------------------------------------------------------------------------

def _normalize_ticker(t: str) -> str:
    """Case-fold and drop the exchange suffix (AAPL.US → AAPL)."""
    return t.upper().split(".")[0]


def _normalize_index(s: str) -> str:
    """Spacing-tolerant: 'S&P500' and 'S&P 500' should both match."""
    return "".join(s.split()).lower()


def matches_entities(actual: dict[str, Any] | Any, expected: dict[str, Any]) -> bool:
    """
    Subset match with normalization. `actual` must contain every value in
    `expected`; extra fields and extra values are allowed.

    `actual` may be a Pydantic model — we coerce via model_dump.
    """
    if hasattr(actual, "model_dump"):
        actual = actual.model_dump()
    elif hasattr(actual, "as_clean_dict"):
        actual = actual.as_clean_dict()

    for field, exp_value in expected.items():
        act_value = actual.get(field)
        if act_value is None:
            return False

        if field == "tickers":
            exp_set = {_normalize_ticker(t) for t in exp_value}
            act_set = {_normalize_ticker(t) for t in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("topics", "sectors"):
            exp_set = {s.lower() for s in exp_value}
            act_set = {s.lower() for s in act_value}
            if not exp_set.issubset(act_set):
                return False
        elif field in ("amount", "rate"):
            try:
                af = float(act_value)
                ef = float(exp_value)
            except (TypeError, ValueError):
                return False
            if abs(af - ef) > abs(ef) * 0.05:
                return False
        elif field == "period_years":
            if int(act_value) != int(exp_value):
                return False
        elif field == "index":
            if _normalize_index(str(act_value)) != _normalize_index(str(exp_value)):
                return False
        else:
            if str(act_value).lower() != str(exp_value).lower():
                return False
    return True


# ---------------------------------------------------------------------------
# Routing accuracy
# ---------------------------------------------------------------------------

def test_classifier_routing_accuracy(gold_classifier_queries, mock_llm):
    """≥ 85% routing accuracy. We currently exceed 95% via deterministic rules."""
    correct = 0
    misses: list[tuple[str, str, str]] = []
    for case in gold_classifier_queries:
        result = classify(case["query"], llm=mock_llm)
        if result.agent.value == case["expected_agent"]:
            correct += 1
        else:
            misses.append((case["query"], case["expected_agent"], result.agent.value))

    accuracy = correct / len(gold_classifier_queries)
    detail = "\n".join(f"  exp={e:25s} act={a:25s} | {q}" for q, e, a in misses[:10])
    assert accuracy >= 0.85, (
        f"Routing accuracy {accuracy:.2%} below 85% "
        f"({correct}/{len(gold_classifier_queries)} correct). First misses:\n{detail}"
    )


def test_classifier_entity_extraction(gold_classifier_queries, mock_llm):
    """Soft signal — reported, not failed on. ASSIGNMENT.md: 'we report it'."""
    matched = 0
    total_with_entities = 0
    for case in gold_classifier_queries:
        if not case["expected_entities"]:
            continue
        total_with_entities += 1
        result = classify(case["query"], llm=mock_llm)
        if matches_entities(result.entities, case["expected_entities"]):
            matched += 1

    rate = matched / total_with_entities if total_with_entities else 0.0
    print(f"\nEntity match rate: {rate:.2%} ({matched}/{total_with_entities})")
