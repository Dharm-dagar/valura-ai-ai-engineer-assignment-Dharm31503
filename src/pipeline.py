"""
Request orchestrator.

Public surface: `process_request()` — an async generator that yields
`AgentEvent`s. The HTTP layer maps these to SSE frames.

Pipeline stages, in order:
  1. Safety guard (synchronous, no LLM, <10ms)
       - If blocked: yield a `blocked` event and stop. No classifier, no agent.
  2. Intent classifier (rules-first, LLM fallback)
       - Single classification call; structured output containing intent +
         entities + safety_note.
       - History-aware for follow-ups.
  3. Agent dispatch (registry lookup → portfolio_health or stub)
       - Streams the agent's events through to the caller.
  4. Done event with simple metrics.

The whole pipeline is bounded by `settings.pipeline_timeout_s` to prevent
runaway streams. On timeout we yield an `error` event and stop cleanly.

Why an async generator (not an HTTP-coupled function): so the pipeline can
be unit-tested without spinning up FastAPI, AND so the same generator can
later drive a websocket / gRPC / batch transport without rewriting.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Callable

from .agents import AgentEvent, AgentRequest, EventKind
from .agents.registry import get_agent
from .classifier import classify
from .config import settings
from .market_data import MarketDataProvider
from .safety import check as safety_check
from .session import SessionStore, default_store

log = logging.getLogger(__name__)


async def process_request(
    *,
    query: str,
    user: dict[str, Any] | None = None,
    session_id: str | None = None,
    session_store: SessionStore | None = None,
    market_data: MarketDataProvider | None = None,
    classifier_llm: Callable[..., Any] | None = None,
    narrator_llm: Any = None,
    timeout_s: float | None = None,
) -> AsyncIterator[AgentEvent]:
    """
    Run the full pipeline for one user query.

    Parameters
    ----------
    query
        The current user turn (raw text).
    user
        The user fixture / record. Required for any agent that needs
        portfolio context (e.g. portfolio_health). May be None for purely
        informational queries — the stub agents tolerate it.
    session_id
        If provided, history is tracked for follow-up resolution.
    session_store
        Override for the default in-memory store. Useful for tests.
    market_data
        Injectable provider — defaults to None (agent falls back to cost-basis).
    classifier_llm
        The callable the classifier uses when the rules engine is uncertain.
        If None, we run rules-only.
    narrator_llm
        The streaming LLM client passed to agents for narrative.
    timeout_s
        Hard wall-clock cap. None → use settings.pipeline_timeout_s.
    """
    store = session_store or default_store
    deadline = time.monotonic() + (timeout_s or settings.pipeline_timeout_s)

    # ----- Stage 1: safety guard -----
    verdict = safety_check(query)
    if verdict.blocked:
        yield AgentEvent(
            kind=EventKind.blocked,
            data={
                "category": verdict.category,
                "message": verdict.message,
                "query": query,
            },
        )
        yield AgentEvent(
            kind=EventKind.done,
            data={"blocked": True, "category": verdict.category},
        )
        return

    # ----- Stage 2: classifier (history-aware) -----
    history = store.history(session_id) if session_id else []
    try:
        classification = classify(query, history=history, llm=classifier_llm)
    except Exception as e:
        # Classifier promises not to raise; if it does, surface a structured
        # error and stop. Tests should never hit this path.
        log.exception("Classifier raised unexpectedly: %s", e)
        yield AgentEvent(
            kind=EventKind.error,
            data={"stage": "classifier", "message": str(e)},
        )
        yield AgentEvent(kind=EventKind.done, data={"error": True})
        return

    # Track history AFTER classification so the current turn doesn't pollute
    # its own follow-up resolution.
    if session_id:
        store.add_user_turn(session_id, query)

    yield AgentEvent(
        kind=EventKind.meta,
        data={
            "agent": classification.agent.value,
            "confidence": classification.confidence,
            "source": classification.source.value,
            "entities": classification.entities.as_clean_dict(),
            "safety_note": classification.safety_note,
        },
    )

    # ----- Stage 3: agent dispatch -----
    agent_fn = get_agent(classification.agent)
    request = AgentRequest(
        user=user or {},
        classification=classification,
        query=query,
        history=history,
        market_data=market_data,
        llm=narrator_llm,
    )

    try:
        async for ev in _with_deadline(agent_fn(request), deadline):
            # Don't double-emit the agent's own done event; we'll add our own
            # at the end with overall pipeline metrics.
            if ev.kind is EventKind.done:
                continue
            yield ev
    except asyncio.TimeoutError:
        yield AgentEvent(
            kind=EventKind.error,
            data={
                "stage": "agent",
                "message": "Agent exceeded timeout; partial response above.",
                "agent": classification.agent.value,
            },
        )

    # ----- Stage 4: pipeline-level done -----
    yield AgentEvent(
        kind=EventKind.done,
        data={
            "agent": classification.agent.value,
            "elapsed_s": round(
                settings.pipeline_timeout_s - max(0.0, deadline - time.monotonic()),
                3,
            ),
        },
    )


async def _with_deadline(
    gen: AsyncIterator[AgentEvent], deadline: float
) -> AsyncIterator[AgentEvent]:
    """Yield from `gen` while remaining time > 0. Raises asyncio.TimeoutError
    if the deadline passes mid-stream. We use per-event waits rather than
    a single wait_for so partial output is preserved."""
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError("pipeline deadline exceeded")
        try:
            ev = await asyncio.wait_for(_anext(gen), timeout=remaining)
        except StopAsyncIteration:
            return
        yield ev


async def _anext(it: AsyncIterator[AgentEvent]) -> AgentEvent:
    return await it.__anext__()
