"""Specialist agents.

Each agent is a callable that takes a `Request` and yields `AgentEvent`s
(streamed). The orchestrator routes based on the classifier's `agent` field.
"""
from .base import AgentEvent, AgentRequest, EventKind, run as run_agent
from .registry import REGISTRY

__all__ = ["AgentEvent", "AgentRequest", "EventKind", "run_agent", "REGISTRY"]
