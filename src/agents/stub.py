"""
Stub agent.

For agents listed in the taxonomy but not implemented in this build (every
specialist except portfolio_health), the stub returns a structured "not
implemented" response that includes the classified intent, the extracted
entities, and which agent would have handled the query.

Per ASSIGNMENT.md: "Do not crash. Do not return errors. The router's job is
to route correctly even when the destination is a stub."

The stub still streams (single token + structured event + done) so the
client experience is consistent.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .base import AgentEvent, AgentRequest, EventKind


_USER_FRIENDLY_NAMES = {
    "market_research": "Market Research",
    "investment_strategy": "Investment Strategy",
    "financial_planning": "Financial Planning",
    "financial_calculator": "Financial Calculator",
    "risk_assessment": "Risk Assessment",
    "product_recommendation": "Product Recommendation",
    "predictive_analysis": "Predictive Analysis",
    "customer_support": "Customer Support",
    "general_query": "General Query",
}


def _stub_message(agent_name: str) -> str:
    pretty = _USER_FRIENDLY_NAMES.get(agent_name, agent_name)
    return (
        f"The {pretty} agent isn't implemented in this build. "
        f"Your query has been classified and the relevant entities extracted "
        f"so the rest of the pipeline can be tested end-to-end. "
        f"This is a known limitation of the demo scope (only Portfolio Health "
        f"is fully implemented)."
    )


async def stub_agent(req: AgentRequest) -> AsyncIterator[AgentEvent]:
    agent_name = req.classification.agent.value

    yield AgentEvent(
        kind=EventKind.structured,
        data={
            "status": "not_implemented",
            "agent": agent_name,
            "intent": req.classification.rationale or "n/a",
            "entities": req.classification.entities.as_clean_dict(),
            "message": _stub_message(agent_name),
            "disclaimer": (
                "This response is a stub. No real analysis was performed. "
                "Implementation of this specialist is on the roadmap."
            ),
        },
    )
    # Stream one token chunk so the client gets the SSE-streaming code path
    # exercised even for stubs. Tiny await so we don't busy-yield.
    await asyncio.sleep(0)
    yield AgentEvent(
        kind=EventKind.token,
        data={"text": _stub_message(agent_name)},
    )
    yield AgentEvent(kind=EventKind.done, data={"agent": agent_name, "stub": True})
