"""
Agent base types.

An agent is an async generator over `AgentEvent`s. The orchestrator
collects events and forwards them to the SSE response stream.

Why async-generator:
  - Native streaming. Each token / structured payload is yielded as it's
    available.
  - Clean cancellation. If the client disconnects, the generator is
    GC'd and any in-flight LLM call is cancelled.
  - Trivial to test: collect events into a list and assert.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Awaitable, Callable

from ..classifier.types import ClassificationResult


class EventKind(str, Enum):
    meta = "meta"             # classification result, agent name, etc.
    token = "token"           # streaming text token from LLM narrative
    structured = "structured" # the agent's structured output (single payload)
    done = "done"             # marks end of stream + summary metrics
    error = "error"           # structured error event
    blocked = "blocked"       # safety guard blocked the request


@dataclass
class AgentEvent:
    kind: EventKind
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentRequest:
    """Input to an agent."""
    user: dict[str, Any]                       # full user fixture/context
    classification: ClassificationResult       # intent + entities + safety_note
    query: str                                 # the raw user query
    history: list[str] = field(default_factory=list)
    # Optional dependencies — agents pull what they need from the kwargs on
    # construction or at call time. We pass them on the request to avoid
    # forcing each agent to know how to build them.
    market_data: Any = None                    # MarketDataProvider | None
    llm: Any = None                            # callable | None — for streaming narrative


# An agent is just an async generator factory.
AgentFn = Callable[[AgentRequest], AsyncIterator[AgentEvent]]


# Convenience: run an agent and return the full event list (testing helper).
async def collect(agent_fn: AgentFn, request: AgentRequest) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    async for ev in agent_fn(request):
        events.append(ev)
    return events


# The orchestrator's tiny dispatch helper. Real routing happens in
# pipeline.py; this helper is here so unit tests can drive an agent directly.
async def run(
    agent_fn: AgentFn,
    request: AgentRequest,
    on_event: Callable[[AgentEvent], Awaitable[None]] | None = None,
) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    async for ev in agent_fn(request):
        events.append(ev)
        if on_event is not None:
            await on_event(ev)
    return events
