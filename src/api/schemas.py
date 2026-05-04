"""
HTTP request/response models.

The single endpoint accepts a `QueryRequest` and streams SSE events. The
event payloads themselves are loose `dict`s — the schema is documented
in README.md and enforced informally by the agents.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """One conversational turn from a client."""

    query: str = Field(..., min_length=1, max_length=4000, description="The user's message.")
    user_id: str | None = Field(
        None,
        description=(
            "If provided AND a fixture exists for this user_id, the agents "
            "are given access to the user's portfolio context. Otherwise "
            "the agents run with empty context."
        ),
    )
    session_id: str | None = Field(
        None,
        description=(
            "Opaque per-conversation key. If provided, prior turns are "
            "tracked in-memory for follow-up resolution. Omit for one-shot "
            "queries."
        ),
    )
    # Inline user payload — useful for callers that don't have a stored
    # fixture (e.g. multi-tenant production where the user record lives
    # in someone else's DB). Either user_id or user is fine; user wins.
    user: dict[str, Any] | None = Field(
        None,
        description="Inline user record. Overrides user_id lookup if provided.",
    )


class HealthResponse(BaseModel):
    status: str
    llm_available: bool
    agents_loaded: list[str]
