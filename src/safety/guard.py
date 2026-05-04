"""
Synchronous safety guard.

Run-time contract:
- Pure local computation. No network, no LLM, no I/O.
- Must return in well under 10ms for any reasonable input.
- The `check()` function is the public entry point.

Decision algorithm:
  Walk the categories in order. The first category whose harmful_patterns
  match the (lowercased, stripped) query wins. The matched category dictates
  the refusal message. If no category matches, the query passes.

This is intentionally simple: no scoring, no thresholds, no LLM. The whole
point of the guard is that it's a hard wall the rest of the pipeline can
trust.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .patterns import CATEGORIES, CategorySpec


@dataclass(frozen=True)
class SafetyVerdict:
    blocked: bool
    category: str | None
    message: str | None

    # Convenience: the orchestrator wants a per-category id for telemetry too.
    @property
    def id(self) -> str:
        return self.category or "passed"


_PASS_VERDICT = SafetyVerdict(blocked=False, category=None, message=None)


class SafetyGuard:
    """
    Stateful only in that it owns the list of categories. The check itself
    is functional. Construct once at app startup; share across requests.
    """

    def __init__(self, categories: Iterable[CategorySpec] | None = None):
        self._categories: tuple[CategorySpec, ...] = (
            tuple(categories) if categories is not None else CATEGORIES
        )

    @property
    def categories(self) -> tuple[CategorySpec, ...]:
        return self._categories

    def check(self, query: str) -> SafetyVerdict:
        if not query or not query.strip():
            return _PASS_VERDICT
        # Normalize once. Lowercasing is enough because every pattern uses
        # re.IGNORECASE; we strip to remove leading/trailing whitespace.
        text = query.strip().lower()
        for cat in self._categories:
            for pat in cat.harmful_patterns:
                if pat.search(text):
                    return SafetyVerdict(
                        blocked=True, category=cat.name, message=cat.block_message
                    )
        return _PASS_VERDICT


# Module-level default — convenient for callers that don't want to manage
# the lifecycle. Constructed eagerly because compilation is cheap.
default_guard = SafetyGuard()


def check(query: str) -> SafetyVerdict:
    """Module-level shortcut over the default guard."""
    return default_guard.check(query)
