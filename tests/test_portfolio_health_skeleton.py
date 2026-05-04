"""
Portfolio Health agent tests.

Verifies the structured contract:
  - Empty portfolios get a BUILD-oriented response (no crash, has disclaimer).
  - Concentrated portfolios surface the concentration risk.
  - Every response carries the regulatory disclaimer.
"""
import pytest

from src.agents.portfolio_health import run


def test_portfolio_health_does_not_crash_on_empty_portfolio(load_user, mock_llm):
    """user_004 has no positions. Agent must not crash."""
    user = load_user("usr_004")
    response = run(user, llm=mock_llm)

    assert response is not None
    assert "disclaimer" in response
    # The empty-portfolio response should be a BUILD-oriented one, not an error
    assert response.get("status") == "empty_portfolio"
    # Structured shape must be uniform with the populated case
    assert "concentration_risk" in response
    assert "performance" in response
    assert "benchmark_comparison" in response
    assert "observations" in response
    # Empty portfolio should still produce useful suggestions
    assert response["observations"], "empty portfolio should produce some observations"


def test_portfolio_health_flags_concentration(load_user, mock_llm):
    """user_003 has high NVDA concentration. Agent must surface this."""
    user = load_user("usr_003")
    response = run(user, llm=mock_llm)

    assert response["concentration_risk"]["flag"] in {"high", "moderate", "warning"}
    assert response["concentration_risk"]["top_position_pct"] is not None
    assert response["concentration_risk"]["top_position_pct"] > 25.0


def test_portfolio_health_includes_disclaimer(load_user, mock_llm):
    """ASSIGNMENT.md: every Portfolio Health response carries the disclaimer."""
    user = load_user("usr_001")
    response = run(user, llm=mock_llm)
    assert response["disclaimer"]
    assert "not investment advice" in response["disclaimer"].lower()


def test_portfolio_health_observations_are_bounded(load_user, mock_llm):
    """Surface the one or two things that matter most — cap observations."""
    user = load_user("usr_003")
    response = run(user, llm=mock_llm)
    assert len(response["observations"]) <= 5, "observations should be ranked + bounded"


@pytest.mark.parametrize("user_id", ["usr_001", "usr_003", "usr_004", "usr_006", "usr_008"])
def test_portfolio_health_works_for_every_fixture(user_id, load_user, mock_llm):
    """Smoke test across all fixture users — none should crash."""
    user = load_user(user_id)
    response = run(user, llm=mock_llm)
    assert response is not None
    assert "disclaimer" in response
    assert "concentration_risk" in response
    assert "performance" in response
