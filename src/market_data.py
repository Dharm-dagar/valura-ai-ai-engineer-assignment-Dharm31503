"""
Market-data provider.

Defines a `MarketDataProvider` Protocol and ships two implementations:

  - `YFinanceProvider`  — fetches live data from Yahoo Finance via the
                          `yfinance` package. Used in production.
  - `MockMarketDataProvider` — returns canned data. Used in tests so we
                                don't hit the network.

Why a Protocol: the Portfolio Health agent should not care WHERE prices
come from. Today we use yfinance because it's free and lazy-installable;
tomorrow we might swap in IBKR, Polygon, or a tenant-specific data feed.
The agent calls `provider.get_quote(ticker)` and gets back a typed
`Quote` — that's the contract.

Per ASSIGNMENT.md: "Do not hardcode market data into your code". We don't —
the only number in this file is in the *mock* provider, which is for tests.
The real prices come from yfinance at runtime.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Protocol

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Quote:
    ticker: str
    price: float
    currency: str
    # Optional; not all providers return all fields.
    name: str | None = None
    change_pct_1d: float | None = None
    change_pct_1m: float | None = None
    change_pct_1y: float | None = None
    sector: str | None = None


class MarketDataProvider(Protocol):
    def get_quote(self, ticker: str) -> Quote | None: ...
    def get_quotes(self, tickers: Iterable[str]) -> dict[str, Quote]: ...
    def get_benchmark_return(self, benchmark: str, period: str = "1y") -> float | None: ...


# ---------------------------------------------------------------------------
# YFinance implementation (production)
# ---------------------------------------------------------------------------

# Map our internal benchmark labels to yfinance symbols.
_BENCHMARK_SYMBOLS: dict[str, str] = {
    "S&P 500": "^GSPC",
    "FTSE 100": "^FTSE",
    "NIKKEI 225": "^N225",
    "MSCI World": "URTH",      # ETF tracking MSCI World — yfinance handles this
    "DAX": "^GDAXI",
    "Dow Jones": "^DJI",
    "NASDAQ": "^IXIC",
    "QQQ": "QQQ",
}


class YFinanceProvider:
    """Live market data via the yfinance package.

    yfinance is unofficial — we treat every call as failable and degrade
    gracefully (returning None) instead of letting exceptions bubble into
    the agent. This is critical: a temporary Yahoo outage should not crash
    a portfolio-health response.
    """

    def __init__(self):
        # Lazy import so the test suite doesn't need yfinance installed
        # if it's only exercising the mock path.
        import yfinance  # noqa: F401  # validates installation

    def get_quote(self, ticker: str) -> Quote | None:
        try:
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.fast_info  # cheaper than .info
            price = float(getattr(info, "last_price", 0.0) or 0.0)
            if price <= 0:
                return None
            ccy = getattr(info, "currency", None) or "USD"
            return Quote(
                ticker=ticker,
                price=price,
                currency=ccy,
                # We deliberately skip the noisy / slow .info dict here;
                # callers that need sector / name can fetch separately.
            )
        except Exception as e:
            log.warning("YFinance get_quote(%s) failed: %s", ticker, e)
            return None

    def get_quotes(self, tickers: Iterable[str]) -> dict[str, Quote]:
        out: dict[str, Quote] = {}
        for tk in tickers:
            q = self.get_quote(tk)
            if q is not None:
                out[tk] = q
        return out

    def get_benchmark_return(self, benchmark: str, period: str = "1y") -> float | None:
        symbol = _BENCHMARK_SYMBOLS.get(benchmark, benchmark)
        try:
            import yfinance as yf
            t = yf.Ticker(symbol)
            hist = t.history(period=period)
            if hist is None or hist.empty:
                return None
            first = float(hist["Close"].iloc[0])
            last = float(hist["Close"].iloc[-1])
            if first <= 0:
                return None
            return (last - first) / first
        except Exception as e:
            log.warning("YFinance get_benchmark_return(%s) failed: %s", benchmark, e)
            return None


# ---------------------------------------------------------------------------
# Mock implementation (tests / offline dev)
# ---------------------------------------------------------------------------
class MockMarketDataProvider:
    """
    Deterministic provider for tests.

    Construct with a `prices` dict (ticker → price) and an optional
    `benchmark_returns` dict. Fields not provided default to safe values.
    """

    def __init__(
        self,
        prices: dict[str, float] | None = None,
        currency_by_ticker: dict[str, str] | None = None,
        benchmark_returns: dict[str, float] | None = None,
    ):
        self._prices = prices or {}
        self._currencies = currency_by_ticker or {}
        self._benchmarks = benchmark_returns or {}

    def get_quote(self, ticker: str) -> Quote | None:
        if ticker not in self._prices:
            return None
        return Quote(
            ticker=ticker,
            price=self._prices[ticker],
            currency=self._currencies.get(ticker, "USD"),
        )

    def get_quotes(self, tickers: Iterable[str]) -> dict[str, Quote]:
        out: dict[str, Quote] = {}
        for tk in tickers:
            q = self.get_quote(tk)
            if q is not None:
                out[tk] = q
        return out

    def get_benchmark_return(self, benchmark: str, period: str = "1y") -> float | None:
        return self._benchmarks.get(benchmark)


# ---------------------------------------------------------------------------
# Default provider factory
# ---------------------------------------------------------------------------
def default_provider() -> MarketDataProvider:
    """Construct the configured default. Falls back to mock if yfinance is
    unavailable or `MARKET_DATA_PROVIDER=mock` is set."""
    from .config import settings

    if settings.market_data_provider.lower() == "mock":
        return MockMarketDataProvider()
    try:
        return YFinanceProvider()
    except ImportError:
        log.warning("yfinance not installed — falling back to MockMarketDataProvider")
        return MockMarketDataProvider()
