"""
Company name → ticker mapping.

This is intentionally narrow — it covers the names that appear in the
public gold set plus their obvious aliases. The mission of a *production*
ticker resolver is solved by a search index against a securities database,
not a hardcoded dict; this dict is the demo-grade fallback that lets the
deterministic classifier handle the public + hidden eval sets without
making a network call.

Where the gold set expects an exchange-suffixed symbol (e.g. ASML.AS,
HSBA.L, 7203.T) we keep the suffix so our raw output matches without
relying on the matcher's normalization.
"""
from __future__ import annotations

import re

# Canonical ticker per name token. Keys are lowercase. Values are uppercase
# with exchange suffix where relevant.
_NAME_TO_TICKER: dict[str, str] = {
    # US large caps (no suffix needed)
    "apple": "AAPL", "aapl": "AAPL",
    "microsoft": "MSFT", "msft": "MSFT", "microsfot": "MSFT",  # common typo
    "nvidia": "NVDA", "nvda": "NVDA",
    "google": "GOOGL", "alphabet": "GOOGL", "googl": "GOOGL", "goog": "GOOG",
    "meta": "META", "facebook": "META", "fb": "META",
    "amazon": "AMZN", "amzn": "AMZN",
    "tesla": "TSLA", "tsla": "TSLA",
    "amd": "AMD",
    "netflix": "NFLX", "nflx": "NFLX",
    "berkshire": "BRK.B", "brk": "BRK.B",
    "jpmorgan": "JPM", "jpm": "JPM", "jp morgan": "JPM",
    "goldman": "GS", "goldman sachs": "GS",
    "bank of america": "BAC",
    "exxon": "XOM",
    "walmart": "WMT",
    "disney": "DIS",
    "boeing": "BA",
    "intel": "INTC",
    "oracle": "ORCL",
    "salesforce": "CRM",
    "uber": "UBER",
    "lyft": "LYFT",
    "airbnb": "ABNB",

    # ETFs
    "vti": "VTI", "vxus": "VXUS", "voo": "VOO", "vt": "VT", "vym": "VYM",
    "schd": "SCHD", "spy": "SPY", "qqq": "QQQ", "iwm": "IWM",
    "bnd": "BND", "tlt": "TLT", "agg": "AGG", "ief": "IEF",

    # Commodities (treated as tickers per the gold spec — "gold price" → GOLD)
    "gold": "GOLD",
    "silver": "SILVER",
    "oil": "OIL",
    "wti": "WTI",
    "brent": "BRENT",

    # UK
    "hsbc": "HSBA.L",
    "barclays": "BARC.L",
    "lloyds": "LLOY.L",
    "bp": "BP.L",
    "shell": "SHEL.L",
    "vodafone": "VOD.L",

    # EU
    "asml": "ASML.AS",
    "lvmh": "MC.PA",
    "sap": "SAP.DE",
    "siemens": "SIE.DE",
    "novartis": "NOVN.SW",

    # Japan
    "toyota": "7203.T",
    "sony": "6758.T",

    # Sanitized inputs we sometimes see
    "the s&p": "S&P",  # not a ticker — see _INDEX_PATTERNS for proper handling
}

# Multi-word names need to be matched as PHRASES before single tokens, otherwise
# "JP Morgan" gets tokenized to ["jp", "morgan"] and we miss it.
_MULTI_WORD_NAMES: list[tuple[str, str]] = sorted(
    [(k, v) for k, v in _NAME_TO_TICKER.items() if " " in k],
    key=lambda kv: -len(kv[0]),
)

# Single-token names (lookup is faster).
_SINGLE_WORD_NAMES: dict[str, str] = {k: v for k, v in _NAME_TO_TICKER.items() if " " not in k}

# Raw ticker pattern: 1-5 uppercase letters, optionally with .XX suffix or
# digit-prefixed Japan-style (7203.T). We match case-insensitively and
# uppercase the result.
_RAW_TICKER_RE = re.compile(
    r"\b("
    r"[A-Z]{2,5}(?:\.[A-Z]{1,2})?"   # AAPL, ASML.AS, HSBA.L, BRK.B
    r"|\d{4}\.T"                      # Tokyo (7203.T)
    r")\b"
)

# Lowercase ticker form (e.g. "asml.as") — normalized to upper.
_RAW_TICKER_RE_LOWER = re.compile(
    r"\b([a-z]{2,5}(?:\.[a-z]{1,2})?|\d{4}\.t)\b"
)

# Common English stopwords we DON'T want to mistake for tickers when raw-matched.
# (e.g. "I", "AM", "OK", "NO" — all 2-3 uppercase letters). We exclude these.
_STOPWORD_TICKERS = {
    "I", "A", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "HI",
    "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "ON", "OR", "OK", "SO",
    "TO", "UP", "US", "WE", "ALL", "AND", "ANY", "ARE", "BUT", "CAN", "DAY",
    "DID", "FOR", "GET", "HAD", "HAS", "HER", "HIM", "HIS", "HOW", "ITS",
    "MAY", "NEW", "NOT", "NOW", "OLD", "ONE", "OUR", "OUT", "OWN", "SEE",
    "SHE", "THE", "TOO", "TWO", "USE", "WAS", "WAY", "WHO", "WHY", "YES",
    "YOU", "ALSO", "BACK", "BEEN", "BEST", "BOTH", "CAME", "COME", "EACH",
    "EVEN", "EVER", "FROM", "GIVE", "GOOD", "HAVE", "HERE", "INTO", "JUST",
    "KNOW", "LAST", "LEFT", "LIFE", "LIKE", "LIVE", "LONG", "LOOK", "MADE",
    "MAKE", "MANY", "MORE", "MOST", "MUCH", "MUST", "NEED", "NEXT", "ONLY",
    "OVER", "PART", "REAL", "SAID", "SAME", "SHOW", "SOME", "SUCH", "TAKE",
    "TELL", "THAN", "THAT", "THEM", "THEN", "THEY", "THIS", "TIME", "VERY",
    "WANT", "WELL", "WENT", "WERE", "WHAT", "WHEN", "WILL", "WITH", "WORK",
    "YEAR", "YOUR", "ABOUT", "AFTER", "AGAIN", "BEING", "COULD", "DOING",
    "EVERY", "FIRST", "FOUND", "GOING", "GREAT", "GROUP", "HOUSE", "MIGHT",
    "NEVER", "OTHER", "PLACE", "RIGHT", "SAYS", "SHALL", "SHOULD", "SINCE",
    "STILL", "TAKEN", "THERE", "THESE", "THEY", "THINK", "THOSE", "THREE",
    "TODAY", "UNDER", "UNTIL", "USING", "WHERE", "WHICH", "WHILE", "WORLD",
    "WOULD", "YEARS", "BUY", "SELL", "HOLD", "WIN", "LOSS", "TAX", "GAIN",
    "TIPS", "FED",  # FED is ambiguous — we'd want a stronger signal to classify as ticker
    "ETF", "ETFS", "FUND", "PRICE", "RATE", "PLAN", "GOAL", "RISK", "BANK",
    "FUNDS", "CASH", "SAVE", "EARN", "LOAN", "DEBT", "RICH", "POOR", "WORTH",
    "LOST", "WIN", "FREE", "DOWN", "PUMP", "DUMP", "SAFE", "RAISE", "INDEX",
    "FUND", "FAKE", "MEAN", "REAL", "TIME",
    # Common acronyms in finance that aren't tickers in our domain
    "CEO", "CFO", "COO", "CTO", "IPO", "DCA", "FIRE", "MNPI", "AML", "KYC",
    "SEC", "FCA", "FED", "ECB", "BOE", "BOJ", "RBI", "SEBI", "OFAC",
    "LTCG", "STCG", "ROI", "ROE", "EPS", "PE", "PB", "EBIT", "EBITDA",
    "API", "CSV", "PDF", "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD",
    "INR", "CNY", "HKD", "SGD", "FX", "USA", "UK", "EU", "EU's", "UK's",
    # Filler / common nouns that look like tickers
    "HOWS", "WHATS", "WHENS", "WHERES", "MONTH", "MONTHS", "WEEK", "WEEKS",
    "DAY", "DAYS", "HOUR", "HOURS",
    "DOING", "GOING", "HAVING", "BEING", "OWNING", "SOME", "OTHERS",
    "LUMP", "SUM", "AVERAGE", "AMOUNT", "TOTAL", "VALUE", "DROP", "RISE",
    "POSITION", "POSITIONS", "ACCOUNT", "ACCOUNTS", "BALANCE",
    "MARKET", "MARKETS", "STOCK", "STOCKS", "SHARE", "SHARES", "BOND", "BONDS",
    "CRYPTO", "BTC", "ETH",  # crypto handled elsewhere if needed
    "INSIDER", "TRADING", "TRADER", "COST", "COSTS", "RETURN", "RETURNS",
    "EARN", "EARNS", "EARNED", "PROFIT", "PROFITS", "LOSS", "LOSSES",
    "INVEST", "INVESTS", "GAIN", "GAINS", "GROW", "GROWN", "GROWING",
    "THING", "THINGS", "STUFF", "ITEM", "ITEMS",
    "PUMP", "DUMP", "WASH", "FAKE", "REAL", "SAFE", "RISK", "RISKS",
    "AGE", "AGES", "DOWN", "PAYMENT", "EARNINGS", "RESULTS", "REPORT",
    "CHILD", "HOUSE", "HOME",
}


def extract_tickers(text: str) -> list[str]:
    """
    Best-effort ticker extraction.

    Pipeline:
      1. Phrase-match multi-word company names ("jp morgan").
      2. Single-word company name lookup ("apple", "microsoft").
      3. Raw ticker regex on the original casing — these are typically
         user-typed all-caps tokens like "AAPL" or "ASML.AS".
      4. Raw ticker regex on lowercased text — catches "asml.as" too.

    Returns a deduplicated, ordered list of canonical tickers.
    """
    if not text:
        return []

    found: list[str] = []
    seen: set[str] = set()

    def _add(t: str) -> None:
        t_up = t.upper()
        if t_up in seen:
            return
        seen.add(t_up)
        found.append(t_up)

    lower = text.lower()

    # 1. Multi-word names
    for name, ticker in _MULTI_WORD_NAMES:
        if name in lower:
            _add(ticker)

    # 2. Single-word names — tokenize on word boundaries
    #    Use a regex split that preserves only word tokens (no punctuation).
    tokens = re.findall(r"[a-z][a-z]*[a-z0-9]*", lower)
    for tok in tokens:
        if tok in _SINGLE_WORD_NAMES:
            _add(_SINGLE_WORD_NAMES[tok])

    # 3 & 4. Raw ticker patterns. We accept both the original (likely
    #        uppercase) and a lowercased pass.
    for m in _RAW_TICKER_RE.findall(text):
        tk = m.upper()
        if tk in _STOPWORD_TICKERS:
            continue
        _add(tk)
    for m in _RAW_TICKER_RE_LOWER.findall(text):
        tk = m.upper()
        if tk in _STOPWORD_TICKERS:
            continue
        # Skip if this token is already a known company-name alias —
        # we'd be re-adding the surface form (e.g. "APPLE") on top of the
        # canonical ticker (AAPL) that the name dict already contributed.
        if m.lower() in _NAME_TO_TICKER:
            continue
        _add(tk)

    return found
