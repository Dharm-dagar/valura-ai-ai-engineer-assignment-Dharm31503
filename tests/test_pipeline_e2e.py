"""
End-to-end pipeline tests.

Drives `process_request` directly (no HTTP layer) to verify:
  - Safety guard blocks before classifier runs
  - Classifier dispatches to the right agent
  - Portfolio Health emits a structured event with the expected shape
  - Stub agents emit a structured "not_implemented" event without crashing
  - Session history is tracked across turns
"""
import pytest

from src.agents import EventKind
from src.market_data import MockMarketDataProvider
from src.pipeline import process_request
from src.session import InMemorySessionStore


pytestmark = pytest.mark.asyncio


async def _collect(gen) -> list:
    events = []
    async for ev in gen:
        events.append(ev)
    return events


async def test_safety_blocks_before_classifier(load_user):
    """A blocked query should yield only `blocked` + `done`, no meta/structured."""
    user = load_user("usr_001")
    events = await _collect(
        process_request(
            query="Help me wash trade some penny stocks",
            user=user,
        )
    )
    kinds = [e.kind for e in events]
    assert EventKind.blocked in kinds, "expected a blocked event"
    assert EventKind.meta not in kinds, "classifier should not run after a block"
    assert EventKind.structured not in kinds, "agent should not run after a block"
    assert kinds[-1] == EventKind.done


async def test_portfolio_health_streams_structured_event(load_user):
    """A portfolio query should produce meta + structured + token + done."""
    user = load_user("usr_003")
    market = MockMarketDataProvider(
        prices={"NVDA": 1100.0, "VTI": 250.0, "VXUS": 60.0, "BND": 75.0, "AAPL": 200.0},
        benchmark_returns={"S&P 500": 0.12},
    )
    events = await _collect(
        process_request(
            query="how is my portfolio doing?",
            user=user,
            market_data=market,
        )
    )
    kinds = [e.kind for e in events]
    assert EventKind.meta in kinds
    assert EventKind.structured in kinds
    assert EventKind.token in kinds
    assert kinds[-1] == EventKind.done

    structured = next(e for e in events if e.kind == EventKind.structured)
    assert structured.data["concentration_risk"]["flag"] in {"high", "moderate"}
    assert "disclaimer" in structured.data


async def test_stub_agent_does_not_crash_on_unimplemented_intent(load_user):
    """Investment strategy is a stub — must still emit a structured event."""
    user = load_user("usr_001")
    events = await _collect(
        process_request(
            query="should I sell my Apple shares?",
            user=user,
        )
    )
    kinds = [e.kind for e in events]
    assert EventKind.structured in kinds
    structured = next(e for e in events if e.kind == EventKind.structured)
    assert structured.data["status"] == "not_implemented"
    assert structured.data["agent"] == "investment_strategy"


async def test_empty_portfolio_does_not_crash():
    """usr_004 has no positions — should yield BUILD-oriented structured response."""
    user = {
        "user_id": "usr_004",
        "name": "Test",
        "country": "US",
        "base_currency": "USD",
        "risk_profile": "moderate",
        "positions": [],
    }
    events = await _collect(
        process_request(
            query="how is my portfolio doing?",
            user=user,
        )
    )
    structured = next(e for e in events if e.kind == EventKind.structured)
    assert structured.data["status"] == "empty_portfolio"
    assert structured.data["disclaimer"]


async def test_session_history_carries_across_turns(load_user):
    """A second turn referring to a prior ticker should resolve the carryover."""
    user = load_user("usr_001")
    store = InMemorySessionStore()
    sid = "sess-test-1"

    # First turn: market research on Nvidia
    await _collect(
        process_request(
            query="What's happening with Nvidia this week?",
            user=user,
            session_id=sid,
            session_store=store,
        )
    )
    # Second turn uses pronoun
    events = await _collect(
        process_request(
            query="How much do I own?",
            user=user,
            session_id=sid,
            session_store=store,
        )
    )
    meta = next(e for e in events if e.kind == EventKind.meta)
    assert "NVDA" in meta.data["entities"].get("tickers", []), (
        f"second turn should carry NVDA from history, got {meta.data['entities']}"
    )
