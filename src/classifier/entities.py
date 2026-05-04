"""
Entity extraction helpers.

These are pure functions: take a raw query string, return a typed value
(or None / empty list). The rules engine composes these to fill in
the `Entities` model.

Designed to be cheap (regex only) and conservative — it's better to leave
a field None than to guess wrong, because the matcher in tests/ does
*subset* matching: extra wrong values don't hurt missing ones, but an
incorrect type WILL fail (e.g. emitting a string where a number is
expected). When in doubt, omit.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Amount: "5000", "5,000", "500k", "500K", "$500k", "1.5m", "200k"
# ---------------------------------------------------------------------------
_AMOUNT_RE = re.compile(
    r"\$?\s?(?P<num>\d{1,3}(?:[,_]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s?(?P<suffix>k|m|mn|bn|b|million|thousand|billion|grand)?\b",
    re.IGNORECASE,
)

_SUFFIX_MULTIPLIER = {
    None: 1,
    "k": 1_000,
    "thousand": 1_000,
    "grand": 1_000,
    "m": 1_000_000,
    "mn": 1_000_000,
    "million": 1_000_000,
    "b": 1_000_000_000,
    "bn": 1_000_000_000,
    "billion": 1_000_000_000,
}


def extract_amount(text: str) -> float | None:
    """
    Return the FIRST plausible monetary amount in the query.

    Heuristic: we skip numbers that are clearly years (4-digit 19xx/20xx
    tokens that aren't followed by a magnitude suffix and aren't preceded
    by a currency sign), percentages, and bare ages. We prefer numbers with
    an explicit magnitude suffix (k/m/million) or a currency context.
    """
    if not text:
        return None
    candidates: list[tuple[float, int, bool]] = []  # (value, position, is_strong)

    for m in _AMOUNT_RE.finditer(text):
        raw = m.group("num").replace(",", "").replace("_", "")
        try:
            num = float(raw)
        except ValueError:
            continue
        suffix = (m.group("suffix") or "").lower() or None
        # Skip percentages: number immediately followed by %
        end = m.end()
        if end < len(text) and text[end] == "%":
            continue
        # Skip if this is a year token (4 digits like 1999/2025) with no suffix
        # AND no currency context immediately around it.
        if suffix is None and num >= 1900 and num <= 2100 and num == int(num):
            window = text[max(0, m.start() - 8):min(len(text), m.end() + 8)].lower()
            if "$" not in window and "usd" not in window and "dollars" not in window \
               and "k" not in raw.lower() and "loan" not in window and "amount" not in window:
                continue
        # Skip clearly-non-monetary trailing context: "% returns", "30 years",
        # "6 months". But DO NOT skip frequency tokens like "monthly" / "yearly":
        # "$2500 monthly" is a recurring amount, not a duration.
        tail = text[m.end():m.end() + 12].lower().lstrip()
        if suffix is None and (
            tail.startswith("%")
            or re.match(r"(?:years?|months?|weeks?|days?|hours?)(?!ly)\b", tail)
        ):
            continue
        # Skip ages: "i'm 70", "at age 55", "30 year old"
        head = text[max(0, m.start() - 12):m.start()].lower()
        if "age" in head or "old" in tail[:8] or "year-old" in tail[:12] \
           or re.search(r"i(?:'?m| am)\s*$", head):
            continue

        value = num * _SUFFIX_MULTIPLIER[suffix]
        is_strong = suffix is not None or "$" in text[max(0, m.start() - 2):m.start() + 1] \
                    or value >= 1000
        candidates.append((value, m.start(), is_strong))

    if not candidates:
        return None
    # Prefer strong candidates (with suffix or currency context); among those
    # pick the first by position.
    strong = [c for c in candidates if c[2]]
    pool = strong if strong else candidates
    pool.sort(key=lambda c: c[1])
    return pool[0][0]


# ---------------------------------------------------------------------------
# Rate: "8%", "6.5%", "6.5 percent" → decimal
# ---------------------------------------------------------------------------
_RATE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s?(?:%|percent\b)")


def extract_rate(text: str) -> float | None:
    """Return the first percentage as a decimal (e.g. '8%' -> 0.08)."""
    m = _RATE_RE.search(text)
    if not m:
        return None
    return float(m.group(1)) / 100.0


# ---------------------------------------------------------------------------
# Period (years) and horizon
# ---------------------------------------------------------------------------
_YEARS_RE = re.compile(r"\b(\d{1,3})\s*(?:year|yr|y)s?\b", re.IGNORECASE)
_MONTHS_RE = re.compile(r"\b(\d{1,3})\s*(?:month|mo|m)s?\b", re.IGNORECASE)


def extract_period_years(text: str) -> int | None:
    """For calculator-style queries: 'for 30 years' → 30."""
    m = _YEARS_RE.search(text)
    if m:
        return int(m.group(1))
    return None


_HORIZON_TOKENS = {
    # canonical: list of phrases (already lowercased)
    "6_months": ["in 6 months", "in six months", "in the next 6 months", "next 6 months", "6 month horizon"],
    "1_year": ["in 1 year", "in a year", "in one year", "next year", "12 months from now", "in 12 months"],
    "5_years": ["in 5 years", "in five years", "in the next 5 years", "next 5 years"],
    "10_years": ["in 10 years", "in ten years", "next 10 years"],
}


def extract_horizon(text: str) -> str | None:
    """For predictive queries: 'in 6 months' / 'in 5 years' → token."""
    lo = text.lower()
    for canon, phrases in _HORIZON_TOKENS.items():
        for p in phrases:
            if p in lo:
                return canon
    return None


# ---------------------------------------------------------------------------
# Time period (today / this_week / this_month / this_year)
# ---------------------------------------------------------------------------
_TIME_PERIOD_TOKENS = {
    "today": ["today"],
    "yesterday": ["yesterday"],
    "this_week": ["this week"],
    "this_month": ["this month"],
    "this_year": ["this year", "ytd", "year to date"],
    "last_week": ["last week"],
    "last_month": ["last month"],
    "last_year": ["last year"],
}


def extract_time_period(text: str) -> str | None:
    lo = text.lower()
    # Order matters: prefer more-specific phrases. Sort by phrase length desc.
    items = sorted(
        ((c, p) for c, ps in _TIME_PERIOD_TOKENS.items() for p in ps),
        key=lambda kp: -len(kp[1]),
    )
    for canon, phrase in items:
        if re.search(rf"\b{re.escape(phrase)}\b", lo):
            return canon
    return None


# ---------------------------------------------------------------------------
# Frequency
# ---------------------------------------------------------------------------
_FREQ_MAP = {
    "daily": "daily", "every day": "daily",
    "weekly": "weekly", "every week": "weekly", "per week": "weekly",
    "monthly": "monthly", "every month": "monthly", "per month": "monthly", "a month": "monthly",
    "quarterly": "quarterly",
    "yearly": "yearly", "annually": "yearly", "every year": "yearly", "per year": "yearly", "a year": "yearly",
}


def extract_frequency(text: str) -> str | None:
    lo = text.lower()
    items = sorted(_FREQ_MAP.items(), key=lambda kv: -len(kv[0]))
    for phrase, canon in items:
        if re.search(rf"\b{re.escape(phrase)}\b", lo):
            return canon
    return None


# ---------------------------------------------------------------------------
# Currency (ISO 4217)
# ---------------------------------------------------------------------------
_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "CAD", "INR", "CNY", "HKD", "SGD"}
_CURRENCY_WORD = {
    "dollar": "USD", "dollars": "USD", "us dollar": "USD",
    "euro": "EUR", "euros": "EUR",
    "pound": "GBP", "pounds": "GBP", "sterling": "GBP",
    "yen": "JPY",
    "rupee": "INR", "rupees": "INR",
    "yuan": "CNY", "renminbi": "CNY",
    "swiss franc": "CHF",
}


def extract_currency(text: str) -> str | None:
    """First currency mentioned (by position) — preserves source for FX queries."""
    matches: list[tuple[int, str]] = []
    for code in _CURRENCIES:
        for m in re.finditer(rf"\b{code}\b", text):
            matches.append((m.start(), code))
    lo = text.lower()
    for word, code in _CURRENCY_WORD.items():
        for m in re.finditer(rf"\b{re.escape(word)}\b", lo):
            matches.append((m.start(), code))
    if not matches:
        return None
    matches.sort(key=lambda mc: mc[0])
    return matches[0][1]


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
_INDEX_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bs\s?&\s?p\s?500\b", re.IGNORECASE), "S&P 500"),
    (re.compile(r"\bs\s?and\s?p\s?500\b", re.IGNORECASE), "S&P 500"),
    (re.compile(r"\bsp500\b", re.IGNORECASE), "S&P 500"),
    (re.compile(r"\bftse\s?100\b", re.IGNORECASE), "FTSE 100"),
    (re.compile(r"\bftse\b", re.IGNORECASE), "FTSE 100"),
    (re.compile(r"\bnikkei(?:\s?225)?\b", re.IGNORECASE), "NIKKEI 225"),
    (re.compile(r"\bmsci\s?world\b", re.IGNORECASE), "MSCI World"),
    (re.compile(r"\bdow(?:\s?jones)?\b", re.IGNORECASE), "Dow Jones"),
    (re.compile(r"\bnasdaq(?:\s?composite|\s?100)?\b", re.IGNORECASE), "NASDAQ"),
    (re.compile(r"\bdax\b", re.IGNORECASE), "DAX"),
    (re.compile(r"\bcac\s?40\b", re.IGNORECASE), "CAC 40"),
    (re.compile(r"\bnifty(?:\s?50)?\b", re.IGNORECASE), "NIFTY 50"),
    (re.compile(r"\bsensex\b", re.IGNORECASE), "SENSEX"),
]


def extract_index(text: str) -> str | None:
    for pat, canon in _INDEX_PATTERNS:
        if pat.search(text):
            return canon
    return None


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
_ACTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:should\s+i\s+|let'?s\s+|i\s+want\s+to\s+|help\s+me\s+|i'?m\s+thinking\s+of\s+)?rebalanc(?:e|ing)\b", re.IGNORECASE), "rebalance"),
    (re.compile(r"\b(?:should\s+i\s+|i\s+want\s+to\s+|help\s+me\s+)?hedg(?:e|ing)\b", re.IGNORECASE), "hedge"),
    (re.compile(r"\b(?:should\s+i\s+|let'?s\s+|i\s+want\s+to\s+|help\s+me\s+|i'?m\s+thinking\s+of\s+|tell\s+me\s+to\s+)?sell\b", re.IGNORECASE), "sell"),
    (re.compile(r"\b(?:should\s+i\s+|let'?s\s+|i\s+want\s+to\s+|help\s+me\s+|i'?m\s+thinking\s+of\s+|tell\s+me\s+to\s+)?buy(?:\s+more)?\b", re.IGNORECASE), "buy"),
    (re.compile(r"\b(?:should\s+i\s+|let'?s\s+|i\s+want\s+to\s+|help\s+me\s+)?hold\b", re.IGNORECASE), "hold"),
]


def extract_action(text: str) -> str | None:
    for pat, canon in _ACTION_PATTERNS:
        if pat.search(text):
            return canon
    return None


# ---------------------------------------------------------------------------
# Goal
# ---------------------------------------------------------------------------
_GOAL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bfire\s+(?:plan|movement|goal|number|target|strategy)\b|\bfire\b(?=.*\b(?:plan|retire|early|150k|earning)\b)", re.IGNORECASE), "FIRE"),
    (re.compile(r"\b(?:retire(?:ment)?|retiring|nest\s+egg|pension)\b", re.IGNORECASE), "retirement"),
    (re.compile(r"\b(?:college|education|university|school|tuition)\s+(?:fund|savings|cost|plan)\b|\bchild'?s?\s+college\b|\b529\b", re.IGNORECASE), "education"),
    (re.compile(r"\b(?:house|home|property|condo|apartment)\s+(?:down\s+payment|deposit|fund|purchase|buying)\b|\bdown\s+payment\b", re.IGNORECASE), "house"),
    (re.compile(r"\bemergency\s+fund\b|\brainy[- ]day\s+fund\b", re.IGNORECASE), "emergency_fund"),
]


def extract_goal(text: str) -> str | None:
    for pat, canon in _GOAL_PATTERNS:
        if pat.search(text):
            return canon
    return None


# ---------------------------------------------------------------------------
# Sectors
# ---------------------------------------------------------------------------
_SECTOR_KEYWORDS: list[tuple[str, str]] = [
    ("technology", "technology"), ("tech", "technology"),
    ("healthcare", "healthcare"), ("health care", "healthcare"), ("pharma", "healthcare"),
    ("finance", "financials"), ("financials", "financials"), ("financial", "financials"),
    ("energy", "energy"), ("oil and gas", "energy"),
    ("consumer", "consumer"), ("retail", "consumer"),
    ("industrials", "industrials"), ("industrial", "industrials"),
    ("real estate", "real estate"), ("reits", "real estate"), ("reit", "real estate"),
    ("utilities", "utilities"),
    ("materials", "materials"),
    ("crypto", "crypto"), ("cryptocurrency", "crypto"), ("bitcoin", "crypto"),
    ("ai", "AI"), ("artificial intelligence", "AI"),
    ("biotech", "biotech"),
    ("semiconductor", "semiconductors"), ("semis", "semiconductors"),
]


def extract_sectors(text: str) -> list[str]:
    lo = text.lower()
    found: list[str] = []
    seen: set[str] = set()
    for kw, canon in sorted(_SECTOR_KEYWORDS, key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{re.escape(kw)}\b", lo) and canon not in seen:
            seen.add(canon)
            found.append(canon)
    return found


# ---------------------------------------------------------------------------
# Topics — open vocabulary, but we map a few canonical names that the
# gold set uses (LTCG, beta, max drawdown, recession, mutual fund, etc.).
# ---------------------------------------------------------------------------
_TOPIC_KEYWORDS: list[tuple[str, str]] = [
    # Calculator/tax topics
    ("long-term capital gains", "LTCG"), ("long term capital gains", "LTCG"), ("ltcg", "LTCG"),
    ("short-term capital gains", "STCG"), ("stcg", "STCG"),
    ("capital gains tax", "capital gains"),
    # Risk
    ("max drawdown", "max drawdown"), ("maximum drawdown", "max drawdown"),
    ("beta", "beta"),
    ("recession", "recession"),
    ("inflation", "inflation"),
    ("volatility", "volatility"),
    # FX / forex
    ("forex", "FX"), ("fx", "FX"),
    # Definitional
    ("mutual fund", "mutual fund"), ("mutual funds", "mutual fund"),
    ("compound interest", "compound interest"),
    ("etf", "ETF"), ("etfs", "ETF"), ("exchange traded fund", "ETF"),
    ("index fund", "index fund"), ("index funds", "index fund"),
    ("p/e ratio", "P/E ratio"), ("pe ratio", "P/E ratio"), ("p/e", "P/E ratio"),
    ("dividend", "dividend"), ("dividends", "dividend"),
    ("large cap", "large cap"), ("large-cap", "large cap"),
    ("small cap", "small cap"), ("small-cap", "small cap"),
    ("emerging market", "emerging markets"), ("emerging markets", "emerging markets"),
    ("world", "world"),
    # Calc concepts
    ("dollar cost averaging", "DCA"), ("dollar-cost averaging", "DCA"), ("dca", "DCA"),
    ("lump-sum", "lump-sum"), ("lump sum", "lump-sum"),
    # Customer support topics
    ("login", "login"), ("log in", "login"), ("sign in", "login"),
    ("bank account", "bank account"),
    ("transaction history", "transaction history"),
    ("recurring investment", "recurring investment"), ("auto-invest", "recurring investment"),
]


def extract_topics(text: str) -> list[str]:
    lo = text.lower()
    found: list[str] = []
    seen: set[str] = set()
    for kw, canon in sorted(_TOPIC_KEYWORDS, key=lambda kv: -len(kv[0])):
        if kw in lo and canon not in seen:
            seen.add(canon)
            found.append(canon)
    # FX pair regex (e.g. "EUR/USD", "GBP/USD") → "FX"
    if "FX" not in seen and re.search(r"\b[A-Z]{3}\s?/\s?[A-Z]{3}\b", text):
        found.append("FX")
        seen.add("FX")
    return found
