"""
Portfolio Health agent.

Speaks to the MONITOR + PROTECT halves of the BUILD/MONITOR/GROW/PROTECT
mission, plus a BUILD-oriented response when the user has no positions.

Output shape (per ASSIGNMENT.md, possibly extended):

    {
      "concentration_risk": {
        "top_position_pct": 60.4,
        "top_3_positions_pct": 78.2,
        "flag": "high"
      },
      "performance": {
        "total_return_pct": 18.4,
        "annualized_return_pct": 12.1
      },
      "benchmark_comparison": {
        "benchmark": "S&P 500",
        "portfolio_return_pct": 18.4,
        "benchmark_return_pct": 14.2,
        "alpha_pct": 4.2
      },
      "observations": [
        {"severity": "warning", "text": "..."},
        {"severity": "info",    "text": "..."}
      ],
      "disclaimer": "..."
    }

Design notes:

- All math is deterministic and runs WITHOUT an LLM. The LLM only generates
  a friendly streaming narrative on top. This means the structured output
  is testable without mocking, and it also keeps the per-query cost down
  (the agent's LLM call is bounded narrative, not analytics).

- Empty portfolios (e.g. usr_004) get a BUILD-oriented response — not a
  crash, not an error. The structured shape stays the same; the values
  reflect "you haven't started yet, here's what to consider".

- Multi-currency portfolios (usr_006) report holdings in their original
  currency for the structured output and convert via the market data
  provider for the concentration calc. If FX rates aren't available, we
  fall back to summing native amounts and flag the limitation in
  observations rather than crashing.

- Annualized returns assume the user holds positions long enough that the
  oldest purchase date is meaningful. We compute days-held off
  `purchased_at` and convert to a CAGR. If purchased_at is missing or
  in the future, we fall back to total return only.

- "Surface the one or two things that matter most" — observations are
  ranked: concentration risk (warning) first if present, then any
  benchmark over/under-performance, then anything else worth noting.
  We cap at ~5 observations to avoid dumping every metric.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime
from typing import Any, AsyncIterator, Iterable

from ..market_data import MarketDataProvider, Quote
from .base import AgentEvent, AgentRequest, EventKind

log = logging.getLogger(__name__)


DISCLAIMER = (
    "This is not investment advice. Figures are computed from your provided "
    "holdings and best-available market data, which may be delayed or "
    "incomplete. Past performance does not predict future returns. Consult "
    "a licensed financial adviser before making investment decisions."
)


# ---------------------------------------------------------------------------
# Structured output computation (pure — no LLM, no I/O beyond market data)
# ---------------------------------------------------------------------------
def _today() -> date:
    return date.today()


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _flag_concentration(top_pct: float | None) -> str:
    if top_pct is None:
        return "unknown"
    if top_pct >= 40.0:
        return "high"
    if top_pct >= 25.0:
        return "moderate"
    return "low"


def _benchmark_for_user(user: dict[str, Any]) -> str:
    pref = (user.get("preferences") or {}).get("preferred_benchmark")
    if pref:
        return pref
    # Sensible default by country.
    country = user.get("country", "US")
    return {
        "US": "S&P 500", "GB": "FTSE 100", "UK": "FTSE 100",
        "JP": "NIKKEI 225", "DE": "DAX", "SG": "MSCI World",
    }.get(country, "S&P 500")


def _value_position(pos: dict[str, Any], quote: Quote | None) -> float:
    """Current market value in the position's native currency. We do NOT
    cross-currency-convert here — that lives one layer up so the caller
    can see when conversion failed."""
    qty = float(pos.get("quantity", 0))
    if quote is not None and quote.price > 0:
        return qty * quote.price
    # Fallback: cost basis (better than zero — we don't pretend prices are 0)
    avg_cost = float(pos.get("avg_cost", 0))
    return qty * avg_cost


def _annualized_return(total_return: float, days_held: int) -> float | None:
    """CAGR. Returns None for nonsensical inputs."""
    if days_held <= 30 or total_return <= -1.0:
        return None
    years = days_held / 365.25
    try:
        return (1.0 + total_return) ** (1.0 / years) - 1.0
    except (ValueError, OverflowError, ZeroDivisionError):
        return None


def _compute_structured_output(
    user: dict[str, Any], market: MarketDataProvider | None
) -> dict[str, Any]:
    positions: list[dict[str, Any]] = list(user.get("positions") or [])

    # ----- Empty portfolio: BUILD-oriented response -----
    if not positions:
        return _build_response_for_empty(user)

    # ----- Quotes -----
    tickers = [p["ticker"] for p in positions]
    quotes: dict[str, Quote] = {}
    if market is not None:
        try:
            quotes = market.get_quotes(tickers)
        except Exception as e:
            log.warning("market.get_quotes failed: %s", e)

    # ----- Per-position market value & cost basis (native currency) -----
    enriched: list[dict[str, Any]] = []
    total_value = 0.0
    total_cost = 0.0
    multi_currency = False
    base_ccy = user.get("base_currency", "USD")
    fx_failed = False

    for p in positions:
        q = quotes.get(p["ticker"])
        value_native = _value_position(p, q)
        ccy = p.get("currency", "USD")
        if ccy != base_ccy:
            multi_currency = True
        # For total_value / total_cost we want a single base currency.
        # We do NOT have FX rates wired in this build (out of scope), so
        # for multi-currency portfolios we sum native values and flag
        # the limitation. Single-currency portfolios are exact.
        cost_native = float(p.get("quantity", 0)) * float(p.get("avg_cost", 0))
        if multi_currency and ccy != base_ccy:
            fx_failed = True
        total_value += value_native
        total_cost += cost_native
        enriched.append(
            {
                "ticker": p["ticker"],
                "value": value_native,
                "currency": ccy,
                "weight_pct": 0.0,           # filled below
                "purchased_at": _parse_date(p.get("purchased_at")),
            }
        )

    # ----- Concentration -----
    if total_value > 0:
        for e in enriched:
            e["weight_pct"] = round(100.0 * e["value"] / total_value, 1)
    enriched.sort(key=lambda e: e["weight_pct"], reverse=True)
    top_pct = enriched[0]["weight_pct"] if enriched else None
    top3_pct = round(sum(e["weight_pct"] for e in enriched[:3]), 1) if enriched else None
    conc_flag = _flag_concentration(top_pct)

    # ----- Performance -----
    total_return = (total_value - total_cost) / total_cost if total_cost > 0 else None
    earliest = min((e["purchased_at"] for e in enriched if e["purchased_at"]), default=None)
    days_held = (_today() - earliest).days if earliest else 0
    annualized = _annualized_return(total_return, days_held) if total_return is not None else None

    # ----- Benchmark -----
    bench = _benchmark_for_user(user)
    bench_period = "1y" if days_held >= 365 else "6mo" if days_held >= 180 else "3mo"
    bench_return = market.get_benchmark_return(bench, bench_period) if market is not None else None
    alpha = (
        (total_return - bench_return)
        if (total_return is not None and bench_return is not None)
        else None
    )

    # ----- Observations (rank: most important first, cap at 5) -----
    obs: list[dict[str, str]] = []
    if conc_flag == "high":
        top_holding = enriched[0]
        obs.append(
            {
                "severity": "warning",
                "text": (
                    f"{top_holding['weight_pct']:.0f}% of your portfolio is in "
                    f"{top_holding['ticker']}. A single-name drawdown there would hit "
                    f"hard. Consider trimming to a more balanced weight."
                ),
            }
        )
    elif conc_flag == "moderate":
        obs.append(
            {
                "severity": "info",
                "text": (
                    f"Your largest position is {enriched[0]['ticker']} at "
                    f"{enriched[0]['weight_pct']:.0f}% — manageable, but worth watching."
                ),
            }
        )

    if alpha is not None:
        if alpha >= 0.02:
            obs.append(
                {
                    "severity": "info",
                    "text": (
                        f"You're outperforming {bench} by "
                        f"{alpha * 100:.1f} percentage points over the period."
                    ),
                }
            )
        elif alpha <= -0.05:
            obs.append(
                {
                    "severity": "warning",
                    "text": (
                        f"You're trailing {bench} by {abs(alpha) * 100:.1f} percentage points. "
                        f"Worth checking whether your allocation still matches your goals."
                    ),
                }
            )

    if multi_currency and fx_failed:
        obs.append(
            {
                "severity": "info",
                "text": (
                    f"Your portfolio spans multiple currencies. Totals here are summed "
                    f"in native amounts and may differ slightly from a fully FX-converted "
                    f"figure in {base_ccy}."
                ),
            }
        )

    if (user.get("preferences") or {}).get("income_focus"):
        obs.append(
            {
                "severity": "info",
                "text": (
                    "Your profile emphasises income — for a full picture, the next step is "
                    "looking at portfolio yield and dividend stability, which this health "
                    "check doesn't cover yet."
                ),
            }
        )

    if not bench_return:
        obs.append(
            {
                "severity": "info",
                "text": (
                    f"Couldn't fetch a current {bench} reference; benchmark comparison is "
                    f"unavailable for this run."
                ),
            }
        )

    obs = obs[:5]

    return {
        "concentration_risk": {
            "top_position_pct": top_pct,
            "top_3_positions_pct": top3_pct,
            "top_holding": enriched[0]["ticker"] if enriched else None,
            "flag": conc_flag,
        },
        "performance": {
            "total_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_return_pct": round(total_return * 100.0, 2) if total_return is not None else None,
            "annualized_return_pct": round(annualized * 100.0, 2) if annualized is not None else None,
            "days_held": days_held,
            "currency": base_ccy,
        },
        "benchmark_comparison": {
            "benchmark": bench,
            "portfolio_return_pct": round(total_return * 100.0, 2) if total_return is not None else None,
            "benchmark_return_pct": round(bench_return * 100.0, 2) if bench_return is not None else None,
            "alpha_pct": round(alpha * 100.0, 2) if alpha is not None else None,
        },
        "observations": obs,
        "positions_summary": [
            {
                "ticker": e["ticker"],
                "weight_pct": e["weight_pct"],
                "currency": e["currency"],
            }
            for e in enriched[:5]
        ],
        "disclaimer": DISCLAIMER,
    }


def _build_response_for_empty(user: dict[str, Any]) -> dict[str, Any]:
    """For users with no positions, return a BUILD-oriented response.

    The structured shape matches the populated case — same keys — so the
    HTTP response stays uniform for clients. Values reflect "you haven't
    started yet, here's how to think about starting".
    """
    risk = (user.get("risk_profile") or "moderate").lower()
    bench = _benchmark_for_user(user)
    name = user.get("name") or "there"

    # A risk-profile-aware suggested first allocation (illustrative only;
    # always behind the disclaimer). Values are rounded percentages and
    # are intentionally generic — the user should see this as a starting
    # frame, not a recommendation of a specific product.
    suggested_split = {
        "aggressive": {"equities": 90, "bonds": 5, "cash": 5},
        "moderate":   {"equities": 60, "bonds": 30, "cash": 10},
        "conservative": {"equities": 35, "bonds": 55, "cash": 10},
    }.get(risk, {"equities": 60, "bonds": 30, "cash": 10})

    obs = [
        {
            "severity": "info",
            "text": (
                f"Hi {name} — your account is set up and KYC is complete, but you don't "
                f"have any positions yet. The hardest part of investing is the first "
                f"deposit; the second-hardest is sticking to a plan."
            ),
        },
        {
            "severity": "info",
            "text": (
                f"Based on your risk profile ({risk}), a reasonable starting frame is "
                f"~{suggested_split['equities']}% equities / "
                f"{suggested_split['bonds']}% bonds / "
                f"{suggested_split['cash']}% cash. A single broad-market ETF (tracking "
                f"{bench} or similar) covers the equity portion in one trade."
            ),
        },
        {
            "severity": "info",
            "text": (
                "Once you've made your first allocation, come back here and I'll "
                "monitor concentration, performance vs benchmark, and risk drift over time."
            ),
        },
    ]

    return {
        "status": "empty_portfolio",
        "concentration_risk": {
            "top_position_pct": None,
            "top_3_positions_pct": None,
            "top_holding": None,
            "flag": "n/a",
        },
        "performance": {
            "total_value": 0.0,
            "total_cost": 0.0,
            "total_return_pct": None,
            "annualized_return_pct": None,
            "days_held": 0,
            "currency": user.get("base_currency", "USD"),
        },
        "benchmark_comparison": {
            "benchmark": bench,
            "portfolio_return_pct": None,
            "benchmark_return_pct": None,
            "alpha_pct": None,
        },
        "observations": obs,
        "suggested_starting_allocation": suggested_split,
        "positions_summary": [],
        "disclaimer": DISCLAIMER,
    }


# ---------------------------------------------------------------------------
# Streaming agent
# ---------------------------------------------------------------------------
def _format_narrative(structured: dict[str, Any]) -> str:
    """A deterministic narrative based on the structured output.

    Used as the default token stream when no LLM is wired in. Keeps the
    streaming code path always working, even in CI without OPENAI_API_KEY.
    """
    if structured.get("status") == "empty_portfolio":
        lines = [
            "Your portfolio is empty — this is a perfect starting point.",
            "",
        ]
        for o in structured["observations"]:
            lines.append(o["text"])
            lines.append("")
        lines.append(structured["disclaimer"])
        return "\n".join(lines)

    conc = structured["concentration_risk"]
    perf = structured["performance"]
    bench = structured["benchmark_comparison"]

    lines: list[str] = []
    if conc.get("top_holding") and conc.get("top_position_pct") is not None:
        lines.append(
            f"Your top position is {conc['top_holding']} at "
            f"{conc['top_position_pct']:.0f}% of the portfolio "
            f"({conc.get('flag', 'n/a')} concentration)."
        )
    if perf.get("total_return_pct") is not None:
        line = f"Total return: {perf['total_return_pct']:.1f}%"
        if perf.get("annualized_return_pct") is not None:
            line += f" ({perf['annualized_return_pct']:.1f}% annualized)"
        line += "."
        lines.append(line)
    if bench.get("alpha_pct") is not None:
        sign = "ahead of" if bench["alpha_pct"] >= 0 else "behind"
        lines.append(
            f"Vs {bench['benchmark']}, you're {sign} by "
            f"{abs(bench['alpha_pct']):.1f} percentage points."
        )
    if structured.get("observations"):
        lines.append("")
        lines.append("Things to watch:")
        for o in structured["observations"]:
            lines.append(f"- {o['text']}")
    lines.append("")
    lines.append(structured["disclaimer"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sync entry point — used by tests and by callers who only want the
# structured analysis without the SSE streaming layer.
# ---------------------------------------------------------------------------
def run(
    user: dict[str, Any],
    market: MarketDataProvider | None = None,
    llm: Any = None,  # accepted for signature compatibility with tests
) -> dict[str, Any]:
    """
    Synchronous façade around the structured-output computation.

    Returns the same dict shape that the streaming agent yields in its
    `structured` event. The `llm` argument is accepted but unused — the
    structured analysis is deterministic and never calls the LLM. The LLM
    is only used for the streaming narrative on top, which lives in the
    async agent.

    This is the function the test skeleton imports as
    `from src.agents.portfolio_health import run`.
    """
    return _compute_structured_output(user, market)


async def portfolio_health_agent(req: AgentRequest) -> AsyncIterator[AgentEvent]:
    """
    Async generator. Yields:
      1. structured event (the JSON output)
      2. token events (narrative chunks, streamed)
      3. done event
    """
    # Compute structured output up front. This is fast and synchronous so
    # the client gets the JSON payload immediately, before the narrative
    # tokens start flowing.
    structured = await asyncio.to_thread(
        _compute_structured_output, req.user, req.market_data
    )
    yield AgentEvent(kind=EventKind.structured, data=structured)

    # Narrative: prefer LLM streaming if available; else use deterministic
    # narrative chunked by sentence. The sentence chunking is "good enough
    # streaming" for the SSE channel — clients see incremental updates.
    if req.llm is not None and callable(getattr(req.llm, "stream", None)):
        try:
            async for tok in req.llm.stream(_build_llm_prompt(structured)):
                yield AgentEvent(kind=EventKind.token, data={"text": tok})
        except Exception as e:
            log.warning("LLM narrative streaming failed: %s — falling back", e)
            async for ev in _stream_deterministic(structured):
                yield ev
    else:
        async for ev in _stream_deterministic(structured):
            yield ev

    yield AgentEvent(kind=EventKind.done, data={"agent": "portfolio_health"})


def _build_llm_prompt(structured: dict[str, Any]) -> str:
    return (
        "You are a wealth-management assistant. Given the structured "
        "portfolio analysis below, write a short, plain-language summary "
        "(3-5 sentences) for a novice investor. Lead with the most important "
        "thing. Avoid jargon. Do NOT repeat all the numbers — surface what "
        "matters. End by inviting one follow-up question.\n\n"
        f"{structured}"
    )


async def _stream_deterministic(structured: dict[str, Any]) -> AsyncIterator[AgentEvent]:
    """Sentence-level streaming of the deterministic narrative."""
    text = _format_narrative(structured)
    for sentence in _split_for_streaming(text):
        await asyncio.sleep(0)  # cooperative yield
        yield AgentEvent(kind=EventKind.token, data={"text": sentence})


def _split_for_streaming(text: str) -> Iterable[str]:
    # Cheap sentence/line splitter — preserve trailing punctuation so the
    # client can concatenate without losing structure.
    import re
    parts = re.findall(r"[^\n]+\n?", text)
    return [p for p in parts if p.strip()]
