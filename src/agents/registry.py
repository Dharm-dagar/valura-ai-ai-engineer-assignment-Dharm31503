"""
Agent registry.

Maps the classifier's Agent enum value to a callable async-generator
implementation. Unimplemented agents are wired to the stub — the router
*always* finds something callable so it never crashes on a valid intent.
"""
from __future__ import annotations

from ..classifier.types import Agent
from .base import AgentFn
from .portfolio_health import portfolio_health_agent
from .stub import stub_agent


REGISTRY: dict[Agent, AgentFn] = {
    Agent.portfolio_health: portfolio_health_agent,
    # Stubs for everything else
    Agent.market_research: stub_agent,
    Agent.investment_strategy: stub_agent,
    Agent.financial_planning: stub_agent,
    Agent.financial_calculator: stub_agent,
    Agent.risk_assessment: stub_agent,
    Agent.product_recommendation: stub_agent,
    Agent.predictive_analysis: stub_agent,
    Agent.customer_support: stub_agent,
    Agent.general_query: stub_agent,
}


def get_agent(agent: Agent) -> AgentFn:
    return REGISTRY.get(agent, stub_agent)
