"""
Rule-based intent classifier.

Each rule is a (matcher, builder) pair:
  - matcher(query, history) -> bool : does this rule fire?
  - builder(query, history) -> ClassificationResult : produce the result
                                                       with entities filled in

Rules are evaluated in priority order (most specific first). The first
matching rule wins.

This file owns the routing logic for ~95% of the public gold set. Anything
the rules don't handle confidently falls through to the LLM (see classify.py).

Why rules-first:
  1. Cost — every query the rules handle is one fewer LLM call. The
     ASSIGNMENT.md cost target is < $0.05/query at gpt-4.1 pricing; routing
     "hi" through gpt-4.1 is wasteful.
  2. Latency — the rules return synchronously in microseconds, well under
     the < 2s p95 first-token target.
  3. Predictability — deterministic routing for common queries means our
     unit tests don't depend on LLM mocking gymnastics.
  4. Safety net — when the LLM is down, we still classify common queries.

The rules engine does NOT replace the LLM. Genuinely novel or ambiguous
queries fall through to the LLM where the larger model can do real semantic
reasoning. See classify.py for the orchestration.
"""
from __future__ import annotations

import re
from typing import Callable

from . import entities as ent
from .tickers import extract_tickers
from .types import Agent, ClassificationResult, Entities, Source


# ---------------------------------------------------------------------------
# Helpers used across rules
# ---------------------------------------------------------------------------
def _has_any(text: str, *phrases: str) -> bool:
    """Substring match — used for keyword rules where word-boundaries matter
    less than mention. Fast — used heavily in the hot path."""
    lo = text.lower()
    return any(p in lo for p in phrases)


def _has_word(text: str, *patterns: str) -> bool:
    """Word-boundary regex match — used when keyword could appear inside
    a longer word and we'd false-positive (e.g. 'hold' inside 'household')."""
    lo = text.lower()
    return any(re.search(rf"\b{p}\b", lo) for p in patterns)


def _all_entities(query: str) -> Entities:
    """Run every extractor — cheap enough that we always do it."""
    e = Entities(
        tickers=extract_tickers(query),
        topics=ent.extract_topics(query),
        sectors=ent.extract_sectors(query),
        amount=ent.extract_amount(query),
        currency=ent.extract_currency(query),
        rate=ent.extract_rate(query),
        period_years=ent.extract_period_years(query),
        frequency=ent.extract_frequency(query),
        time_period=ent.extract_time_period(query),
        index=ent.extract_index(query),
        action=ent.extract_action(query),
        goal=ent.extract_goal(query),
    )
    # `horizon` only makes sense in forward-looking contexts. We assign it
    # downstream, after the agent is known. (Keeps rule outputs clean.)
    return e


def _last_user_turn(history: list[str] | None) -> str:
    return history[-1] if history else ""


def _carry_tickers_from_history(query: str, history: list[str] | None) -> list[str]:
    """If the current query has NO tickers but uses pronouns/anaphora,
    pull tickers from the most recent prior turn that has any."""
    if not history:
        return []
    if extract_tickers(query):
        return []
    if not re.search(
        r"\b(it|its|that|those|them|they|the\s+stock|the\s+company|how\s+much\s+do\s+i\s+own|do\s+i\s+own)\b",
        query.lower(),
    ):
        return []
    for turn in reversed(history):
        tk = extract_tickers(turn)
        if tk:
            return tk
    return []


# ---------------------------------------------------------------------------
# Greeting / very-short / gibberish detection
# ---------------------------------------------------------------------------
_GREETINGS = {
    "hi", "hello", "hey", "yo", "sup", "wassup", "howdy",
    "thanks", "thank you", "thx", "ty", "cheers", "ok", "okay",
    "yes", "no", "nope", "yeah", "yep", "sure", "got it", "alright",
    "bye", "goodbye", "see ya", "later",
}

_VOWELS = set("aeiou")


def _looks_like_gibberish(text: str) -> bool:
    """Heuristic: a single token with no vowels (or all vowels) and no
    English-y structure. We use this only as a LAST-RESORT default."""
    t = text.strip().lower()
    if " " in t or len(t) < 4:
        return False
    # Skip if it's a known ticker — handled elsewhere
    if extract_tickers(text):
        return False
    if not re.fullmatch(r"[a-z]+", t):
        return False
    vowel_ratio = sum(1 for c in t if c in _VOWELS) / len(t)
    return vowel_ratio < 0.15 or vowel_ratio > 0.85


# ---------------------------------------------------------------------------
# Rule definitions
#
# Order matters. Higher-priority rules first. Each rule returns either
# None (didn't fire) or a ClassificationResult.
# ---------------------------------------------------------------------------
RuleResult = ClassificationResult | None
Rule = Callable[[str, list[str] | None], RuleResult]


def _r_greeting(q: str, h: list[str] | None) -> RuleResult:
    qstrip = q.strip().lower().rstrip("!.?")
    if qstrip in _GREETINGS or len(qstrip) <= 2:
        return ClassificationResult(
            agent=Agent.general_query,
            entities=Entities(),
            confidence=0.98,
            source=Source.rules,
            rationale="greeting / very-short utterance",
        )
    return None


def _r_customer_support(q: str, h: list[str] | None) -> RuleResult:
    lo = q.lower()
    if (
        "can't login" in lo or "cant login" in lo or "can't log in" in lo or "cannot login" in lo
        or "login to my account" in lo or "log in to my account" in lo
        or "linked bank account" in lo
        or "transaction history" in lo
        or ("recurring investment" in lo and ("didn't go through" in lo or "did not go through" in lo or "failed" in lo))
        or "reset my password" in lo or "password reset" in lo
        or ("my account" in lo and ("change" in lo or "update" in lo or "delete" in lo))
    ):
        e = _all_entities(q)
        # Topic detection is already done inside _all_entities via entities.extract_topics
        # which knows about login/bank account/transaction history/recurring investment.
        return ClassificationResult(
            agent=Agent.customer_support,
            entities=e,
            confidence=0.95,
            source=Source.rules,
            rationale="customer-support keyword match",
        )
    return None


def _r_portfolio_health(q: str, h: list[str] | None) -> RuleResult:
    lo = q.lower()
    # Strong, unambiguous portfolio-health phrases. We require the phrase to
    # actually be present; "rebalance my portfolio" doesn't fire here (it
    # belongs in investment_strategy, see _r_investment_strategy).
    triggers = [
        "how is my portfolio", "how's my portfolio", "hows my portfolio",
        "how is my investment portfolio", "how are my investments",
        "health check on my", "health check of my", "portfolio health",
        "portfolio summary", "review my portfolio", "review my holdings",
        "review my investments", "diversified", "well diversified",
        "concentration risk", "concentration in my portfolio",
        "beating the market", "beat the market",
        "how am i doing", "am i doing well",
    ]
    # "how much do i own" and friends — a portfolio query; if a ticker
    # carries from history, attach it.
    is_ownership_q = bool(
        re.search(r"\b(?:how\s+much|how\s+many|how\s+big|what\s+(?:size|amount))\s+(?:of\s+)?(?:do\s+)?i\s+(?:own|hold|have)\b", lo)
        or re.search(r"\bhow\s+much\s+(?:nvda|aapl|apple|nvidia|tesla|tsla)\b", lo)
    )
    if any(t in lo for t in triggers) or is_ownership_q:
        e = _all_entities(q)
        if is_ownership_q and not e.tickers:
            carried = _carry_tickers_from_history(q, h)
            if carried:
                e.tickers = carried
        return ClassificationResult(
            agent=Agent.portfolio_health,
            entities=e,
            confidence=0.95,
            source=Source.rules,
            rationale="portfolio_health keyword",
        )
    return None


def _r_predictive(q: str, h: list[str] | None) -> RuleResult:
    lo = q.lower()
    if (
        re.search(r"\bwhere\s+(?:will|would|might)\b", lo)
        or re.search(r"\b(?:predict|forecast|project)\b", lo)
        or re.search(r"\bin\s+(?:5|6|10|12)\s+(?:year|years|month|months)\b", lo)
        and ("portfolio value" in lo or "be in" in lo or "value of" in lo)
    ):
        e = _all_entities(q)
        # Re-purpose period as horizon for predictive queries.
        if e.period_years and e.horizon is None:
            e.horizon = f"{e.period_years}_years"
            e.period_years = None
        if e.horizon is None:
            e.horizon = ent.extract_horizon(q)
        return ClassificationResult(
            agent=Agent.predictive_analysis,
            entities=e,
            confidence=0.9,
            source=Source.rules,
            rationale="predictive trigger phrase",
        )
    return None


def _r_risk_assessment(q: str, h: list[str] | None) -> RuleResult:
    lo = q.lower()
    if (
        "downside risk" in lo
        or _has_word(q, "drawdown")
        or "max drawdown" in lo or "maximum drawdown" in lo
        or _has_word(q, "beta")
        or "stress test" in lo
        or re.search(r"\bexposed\s+(?:am\s+i\s+)?to\b|\bexposure\s+to\b", lo)
        or "value at risk" in lo or " var " in f" {lo} "
        or ("portfolio's beta" in lo)
    ):
        e = _all_entities(q)
        return ClassificationResult(
            agent=Agent.risk_assessment,
            entities=e,
            confidence=0.92,
            source=Source.rules,
            rationale="risk_assessment keyword",
        )
    return None


def _r_financial_planning(q: str, h: list[str] | None) -> RuleResult:
    lo = q.lower()
    has_planning_signal = (
        "retirement" in lo or "retire" in lo
        or "college fund" in lo or "education fund" in lo or "child's college" in lo
        or "down payment" in lo or "house deposit" in lo
        or re.search(r"\bfire\b", lo) and ("plan" in lo or "earning" in lo or "year" in lo or "save" in lo)
        or ("save" in lo and ("retirement" in lo or "house" in lo or "college" in lo or "education" in lo))
    )
    if has_planning_signal:
        e = _all_entities(q)
        return ClassificationResult(
            agent=Agent.financial_planning,
            entities=e,
            confidence=0.9,
            source=Source.rules,
            rationale="financial_planning keyword",
        )
    return None


def _r_financial_calculator(q: str, h: list[str] | None) -> RuleResult:
    lo = q.lower()
    # Strong calculator triggers
    if (
        _has_any(q, "calculate", "future value", "present value", "fv =", "pv =")
        or "mortgage" in lo
        or ("convert" in lo and ent.extract_currency(q))
        or "long-term capital gains" in lo or "long term capital gains" in lo or " ltcg" in lo
        or "short-term capital gains" in lo or "short term capital gains" in lo or " stcg" in lo
        # Pattern: "X monthly for Y years" with optional rate — strong calc signal
        or (ent.extract_amount(q) and ent.extract_frequency(q) and ent.extract_period_years(q))
        # "if I invest $X" calculator framing
        or re.search(r"\bif\s+i\s+invest\b", lo)
        # Bare "convert N <currency> to <currency>"
        or re.search(r"\bconvert\s+\$?\d", lo)
    ):
        e = _all_entities(q)
        return ClassificationResult(
            agent=Agent.financial_calculator,
            entities=e,
            confidence=0.9,
            source=Source.rules,
            rationale="financial_calculator pattern",
        )
    return None


def _r_product_recommendation(q: str, h: list[str] | None) -> RuleResult:
    lo = q.lower()
    if (
        re.search(r"\brecommend\b", lo)
        or re.search(r"\bwhich\s+(?:fund|etf|stock|bond|product)\b", lo)
        or re.search(r"\bbest\s+(?:low[- ]cost\s+)?(?:[a-z\- ]{0,30}?)(?:fund|etf|index)", lo)
        or "good fund" in lo or "good etf" in lo
        or re.search(r"\bsuggest\s+(?:a|some|me)\s+(?:fund|etf|stock|portfolio)\b", lo)
    ):
        e = _all_entities(q)
        return ClassificationResult(
            agent=Agent.product_recommendation,
            entities=e,
            confidence=0.88,
            source=Source.rules,
            rationale="product_recommendation phrase",
        )
    return None


def _r_investment_strategy(q: str, h: list[str] | None) -> RuleResult:
    lo = q.lower()
    if (
        re.search(r"\bshould\s+i\s+(?:buy|sell|hold|hedge|short|long|invest|exit|trim|add|stop|start|continue|switch|move|allocate|diversify)\b", lo)
        or re.search(r"\bis\s+now\s+a\s+good\s+time\s+to\s+(?:buy|invest|enter|exit|sell|hold)\b", lo)
        or re.search(r"\brebalanc(?:e|ing)\b", lo)
        or re.search(r"\bequity[- ]bond\s+split\b|\bequity[- ]bond\s+ratio\b|\basset\s+allocation\b", lo)
        or re.search(r"\bhedge\s+(?:my\s+)?(?:usd|eur|gbp|currency|exposure)\b", lo)
        or re.search(r"\bwhat\s+should\s+my\s+(?:equity|bond|allocation|split|portfolio)\b", lo)
    ):
        e = _all_entities(q)
        # Carry tickers from history if pronoun-only ("should i sell some?")
        carried = _carry_tickers_from_history(q, h)
        if carried and not e.tickers:
            e.tickers = carried
        return ClassificationResult(
            agent=Agent.investment_strategy,
            entities=e,
            confidence=0.9,
            source=Source.rules,
            rationale="investment_strategy phrase",
        )
    return None


def _r_followup_market_research(q: str, h: list[str] | None) -> RuleResult:
    """
    Follow-up patterns where the current turn is short and refers back to
    the previous turn. We carry the previous turn's market_research intent
    forward when the current turn introduces a new ticker, OR is a
    comparison directive over prior tickers ("compare them").
    """
    if not h:
        return None
    lo = q.strip().lower()
    # Pattern A: "what about X?", "ok and X?", "and X?", "X?" where X is a ticker/company
    new_tickers = extract_tickers(q)
    short_followup = (
        re.match(r"^(?:what\s+about|how\s+about|and|ok\s+and|okay\s+and|and\s+what\s+about|how\s+is)\s+\w", lo)
        is not None
    )
    if short_followup and new_tickers:
        e = _all_entities(q)
        return ClassificationResult(
            agent=Agent.market_research,
            entities=e,
            confidence=0.85,
            source=Source.rules,
            rationale="follow-up: new ticker after prior turn",
        )
    # Pattern B: "compare them" / "compare these" — pull tickers from history
    if re.match(r"^compare\s+(?:them|these|those)\b", lo):
        carried: list[str] = []
        for turn in h:
            for t in extract_tickers(turn):
                if t not in carried:
                    carried.append(t)
        if len(carried) >= 2:
            e = _all_entities(q)
            e.tickers = carried
            e.intent = "comparison"
            return ClassificationResult(
                agent=Agent.market_research,
                entities=e,
                confidence=0.9,
                source=Source.rules,
                rationale="follow-up: compare carried tickers",
            )
    return None


def _r_market_research_strong(q: str, h: list[str] | None) -> RuleResult:
    """High-confidence market_research signals: explicit research framing.
    Runs BEFORE product_recommendation so 'tell me about the markets and recommend a fund'
    routes to market_research (the primary intent)."""
    lo = q.lower()
    has_compare = (
        re.search(r"\bcompare\s+\w+(?:[\s.\-]+\w+)*\s+(?:and|vs|versus)\b", lo) is not None
    )
    has_how_doing = (
        re.search(r"\bhow(?:\s+(?:is|are|'s)|s)\s+(?:the\s+)?\w+(?:[\s\-.]+\w+)*\s+doing\b", lo) is not None
        and not any(w in lo for w in ("portfolio", "investments", "holdings", "my account"))
    )
    has_price = "price of" in lo or (
        re.search(r"\b\w+\s+price\b", lo) is not None and "fair price" not in lo
    )
    has_fx_pair = re.search(r"\b[A-Z]{3}\s?/\s?[A-Z]{3}\b", q) is not None

    # Ambiguous reference detector — "tell me about that thing/it/this/them"
    # or "you mentioned earlier" without any concrete ticker/company. Route
    # to general_query so the user can be asked to clarify, instead of
    # running market_research blind.
    is_ambiguous_reference = (
        re.search(r"\btell\s+me\s+about\s+(?:that|this|it|them|those|these)\b", lo) is not None
        or re.search(r"\byou\s+mentioned\s+(?:earlier|before|previously|above)\b", lo) is not None
        or re.search(r"\bthat\s+thing\s+(?:you|we|i)\b", lo) is not None
    )
    if is_ambiguous_reference and not extract_tickers(q):
        return None  # let general_educational / default_general handle it

    if (
        "tell me about" in lo
        or "any news" in lo or "news on" in lo or "news about" in lo
        or "what happened in markets" in lo or "what's happening in markets" in lo
        or "happening in markets" in lo or "markets today" in lo
        or "happening with" in lo or "happening to" in lo
        or "top gainers" in lo or "top losers" in lo or "biggest movers" in lo
        or has_compare
        or has_price
        or has_fx_pair
        or has_how_doing
    ):
        e = _all_entities(q)
        if has_compare or " vs " in lo or " versus " in lo:
            e.intent = "comparison"
        carried = _carry_tickers_from_history(q, h)
        if carried:
            for t in carried:
                if t not in e.tickers:
                    e.tickers.append(t)
        return ClassificationResult(
            agent=Agent.market_research,
            entities=e,
            confidence=0.9,
            source=Source.rules,
            rationale="market_research strong signal",
        )
    return None


def _r_market_research_weak(q: str, h: list[str] | None) -> RuleResult:
    """Weaker market_research signals: bare ticker, index name, etc."""
    lo = q.lower()
    e = _all_entities(q)
    # Single-token query that resolves to a ticker → market_research
    bare = q.strip().rstrip("?.!,").lower()
    bare_tokens = bare.split()
    if len(bare_tokens) <= 2 and e.tickers:
        return ClassificationResult(
            agent=Agent.market_research,
            entities=e,
            confidence=0.85,
            source=Source.rules,
            rationale="bare ticker",
        )
    # Index name without future framing → market_research
    if e.index and not re.search(r"\b(?:will|would|in\s+\d+\s+(?:year|month)|forecast|predict)\b", lo):
        return ClassificationResult(
            agent=Agent.market_research,
            entities=e,
            confidence=0.85,
            source=Source.rules,
            rationale="index without forward framing",
        )
    return None


def _r_general_educational(q: str, h: list[str] | None) -> RuleResult:
    """Definitional / 'what is X' questions with no portfolio/buy/sell verb."""
    lo = q.lower()
    has_action_verb = re.search(
        r"\b(?:should\s+i|recommend|rebalance|buy|sell|hedge|invest\s+in|allocate)\b",
        lo,
    )
    if has_action_verb:
        return None
    is_definitional = (
        re.match(r"^\s*what(?:'s|\s+is|\s+does|\s+are\s+the\s+(?:difference|differences|key))", lo)
        or re.match(r"^\s*explain\b", lo)
        or re.match(r"^\s*describe\b", lo)
        or re.match(r"^\s*define\b", lo)
        or re.match(r"^\s*compare\s+(?:dollar)", lo)  # DCA vs lump-sum special
    )
    if is_definitional:
        e = _all_entities(q)
        return ClassificationResult(
            agent=Agent.general_query,
            entities=e,
            confidence=0.9,
            source=Source.rules,
            rationale="definitional question",
        )
    return None


def _r_default_general(q: str, h: list[str] | None) -> RuleResult:
    """Last-resort: gibberish / unparseable → general_query with low confidence
    so the LLM can take a swing if available."""
    if _looks_like_gibberish(q):
        return ClassificationResult(
            agent=Agent.general_query,
            entities=Entities(),
            confidence=0.6,
            source=Source.rules,
            rationale="gibberish-looking",
        )
    # Truly ambiguous — let LLM handle if available; otherwise general_query.
    return ClassificationResult(
        agent=Agent.general_query,
        entities=_all_entities(q),
        confidence=0.35,
        source=Source.rules,
        rationale="default (no rule matched)",
    )


# Order matters. The first rule that returns a non-None result wins.
RULES: tuple[Rule, ...] = (
    _r_greeting,
    _r_customer_support,
    _r_portfolio_health,
    _r_predictive,
    _r_risk_assessment,
    _r_financial_planning,
    _r_financial_calculator,
    # Follow-up patterns ("what about X?", "compare them") need history-aware
    # routing and run BEFORE the keyword-based rules below so we don't
    # mis-classify a 1-token follow-up as something else.
    _r_followup_market_research,
    # Market research strong runs BEFORE product_recommendation so that
    # "tell me about the markets and recommend a fund" → market_research.
    # It runs BEFORE investment_strategy so that "tell me about Tesla and
    # should I sell?" routes to research, but AFTER portfolio_health so
    # "how is my portfolio doing and what should i sell?" stays portfolio.
    _r_market_research_strong,
    # Product recommendation runs BEFORE investment_strategy so that
    # "which fund should i buy" → product_recommendation, not "should i buy".
    _r_product_recommendation,
    _r_investment_strategy,
    _r_market_research_weak,
    _r_general_educational,
    _r_default_general,
)


def classify_with_rules(query: str, history: list[str] | None = None) -> ClassificationResult:
    """Run the rule cascade. Always returns a ClassificationResult."""
    for rule in RULES:
        result = rule(query, history)
        if result is not None:
            return result
    # Should never reach here because _r_default_general always fires,
    # but be defensive.
    return ClassificationResult(
        agent=Agent.general_query,
        entities=Entities(),
        confidence=0.3,
        source=Source.rules,
        rationale="no rule matched (defensive fallback)",
    )
