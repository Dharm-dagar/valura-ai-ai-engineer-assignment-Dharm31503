"""
Safety patterns.

Each category lists `harmful_patterns` (regexes that, if matched, block the
query). Patterns are written to be CONTEXTUAL: e.g. "wash trade" alone does
NOT block, but "help me wash trade" does. This is what lets us hit the
≥95% recall AND ≥90% educational-passthrough targets simultaneously.

Each category also has a `block_message` — a distinct, professional refusal
that names the regulatory concern. Per ASSIGNMENT.md: "Each blocked category
returns a distinct, professional response — not a generic refusal."

We do NOT keep an explicit "educational allow-list". Educational queries pass
because they don't match the harmful patterns, not because we explicitly
whitelist them. This is more robust to novel educational phrasings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern


def _compile(*patterns: str) -> tuple[Pattern[str], ...]:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


@dataclass(frozen=True)
class CategorySpec:
    name: str
    block_message: str
    harmful_patterns: tuple[Pattern[str], ...]


# ---------------------------------------------------------------------------
# Insider trading
# ---------------------------------------------------------------------------
INSIDER_TRADING = CategorySpec(
    name="insider_trading",
    block_message=(
        "I can't help with trading on material non-public information. "
        "Acting on inside information is a securities-law violation in every "
        "major jurisdiction (SEC in the US, FCA in the UK, SEBI in India, "
        "ESMA in the EU) and carries criminal penalties. "
        "If you'd like, I can explain how disclosure rules work, or what "
        "constitutes MNPI vs public information."
    ),
    harmful_patterns=_compile(
        # "i work at X and know about an unannounced acquisition"
        r"\bi\s+(?:work|am\s+working|am\s+employed)\s+at\s+\w+.{0,40}\b(?:know|have\s+heard|got|got\s+wind)\b",
        # "my friend at microsoft gave me a tip about earnings"
        r"\b(?:friend|colleague|contact|insider|source|buddy|cousin|brother|sister|relative)\s+at\s+\w+.{0,40}\b(?:gave|told|tipped|sent|shared|leaked|forwarded)\s+me\b",
        # "trade on this confidential merger news"
        r"\b(?:trade|act|profit|capitalize|move|position\s+myself)\s+(?:on|with|using)\s+(?:this|the|a|these|my|that|those)?\s*(?:confidential|inside|insider|non[- ]public|undisclosed|leaked|secret|unreleased|unannounced)\b",
        # "i need to know the earnings before tomorrow's announcement"
        r"\b(?:i\s+(?:need|want)\s+to|help\s+me|let\s+me)\s+know\s+(?:the\s+)?(?:earnings|results|numbers|guidance|figures)\s+before\b",
        # "load up before the earnings call"
        r"\b(?:load\s+up|stack\s+up|stock\s+up|go\s+long|buy\s+heavily|pile\s+in)\s+(?:on\s+\w+\s+)?(?:before|ahead\s+of)\s+(?:the\s+|this\s+|tomorrow'?s\s+|today'?s\s+|next\s+week'?s\s+)?(?:earnings|announcement|release|merger|acquisition|deal|call|filing|report)\b",
        # "buy shares before the announcement" — used as the guidance request
        r"\bwhen\s+should\s+i\s+buy\s+(?:shares|stock|the\s+stock|in)\b.{0,80}\b(?:unannounced|undisclosed|before|merger|acquisition|deal)\b",
        # "unannounced acquisition" / "unreleased earnings" — strong signal of MNPI
        r"\b(?:unannounced|unreleased|undisclosed|leaked|stolen)\s+(?:acquisition|merger|deal|earnings|results|guidance|filing|announcement|takeover|buyout)\b",
    ),
)

# ---------------------------------------------------------------------------
# Market manipulation
# ---------------------------------------------------------------------------
MARKET_MANIPULATION = CategorySpec(
    name="market_manipulation",
    block_message=(
        "I can't help with market manipulation. Pump-and-dump schemes, "
        "wash trading, spoofing, layering, and coordinated volume schemes are "
        "all illegal under securities-fraud statutes. They distort price "
        "discovery and harm other investors. "
        "If you want to understand how regulators detect these patterns or "
        "what penalties apply, I'm happy to explain."
    ),
    harmful_patterns=_compile(
        # "how can i pump up the price of a small cap"
        r"\b(?:how\s+can\s+i|how\s+do\s+i|i\s+(?:want|need)\s+to|help\s+me|let\s+me|teach\s+me\s+to|tell\s+me\s+how\s+to)\s+pump\s+(?:up\s+)?(?:the\s+)?(?:price|stock|share|small\s+cap|penny\s+stock|coin|token)\b",
        # "design a coordinated buying scheme"
        r"\b(?:design|create|set\s+up|coordinate|run|organize|orchestrate|plan)\s+(?:a\s+|an\s+)?coordinated\s+(?:buying|selling|trading|pump)\b",
        # "move this stock 20%"
        r"\bmove\s+(?:this|the|that)\s+(?:stock|share|price|ticker)\s+(?:up\s+|down\s+)?(?:by\s+)?\d+\s?%",
        # "help me wash trade between two accounts"
        r"\b(?:help\s+me|let\s+me|how\s+do\s+i|how\s+can\s+i|i\s+(?:want|need)\s+to|teach\s+me\s+to)\s+wash\s+(?:trade|trading|sale)\b",
        # "do spoofing" / "execute layering"
        r"\b(?:help\s+me|let\s+me|how\s+do\s+i|how\s+can\s+i|i\s+(?:want|need)\s+to)\s+(?:do|execute|run|engage\s+in|perform)\s+(?:spoofing|layering|painting\s+the\s+tape|marking\s+the\s+close)\b",
        # "create fake volume between accounts"
        r"\b(?:create|generate|fabricate|fake|inflate)\s+(?:fake\s+|artificial\s+)?(?:trading\s+)?volume\b.{0,40}\b(?:account|wallet|exchange)",
    ),
)

# ---------------------------------------------------------------------------
# Money laundering
# ---------------------------------------------------------------------------
MONEY_LAUNDERING = CategorySpec(
    name="money_laundering",
    block_message=(
        "I can't help with money laundering, structuring, or tax evasion. "
        "These are serious financial crimes under AML statutes (BSA in the "
        "US, MLR in the UK, the EU AML Directives, PMLA in India), and "
        "brokerages are legally required to report suspicious activity. "
        "If you want to understand the AML reporting framework or how "
        "structuring is detected, I can walk you through that."
    ),
    harmful_patterns=_compile(
        # "move 500k in cash through stock trading without reporting it"
        r"\b(?:move|transfer|send|push|funnel|route|run|wash|shift)\s+(?:\$?[\d,.]+\s?(?:k|m|mn|bn|million|thousand)?|cash|funds|money)\s+(?:in\s+)?(?:cash|money|funds|usd|dollars|gbp|eur|jpy)?\b.{0,80}\bwithout\s+(?:report|trac|detect|disclos|declar|notic|flag)",
        # "structure deposits to avoid the reporting threshold"
        r"\bstructur(?:e|ing)\s+(?:my\s+|the\s+|cash\s+|deposit\s+)?(?:deposits?|withdrawals?|transactions?|payments?)\s+(?:to\s+(?:avoid|evade|stay\s+under|stay\s+below|fall\s+under)|under|below|to\s+keep\s+under)\b",
        # "how can i layer my trades to obscure the source of these funds"
        r"\b(?:layer|obscure|hide|conceal|disguise|mask|launder|cleanse)\b.{0,60}\b(?:source|origin|trail|provenance|history)\s+of\s+(?:my\s+|these\s+|the\s+|those\s+)?(?:funds|money|profits|trades|cash|proceeds|capital|deposits)\b",
        # "help me hide trading profits from the tax authorities"
        r"\b(?:hide|conceal|evade|disguise|mask|underreport|fail\s+to\s+report)\b.{0,40}\b(?:profits|gains|income|earnings|taxes)\s+from\s+(?:the\s+)?(?:tax|irs|hmrc|cra|ato|fisc|government|authorit|revenue)",
        # "avoid the 10k reporting threshold"
        r"\bavoid\s+(?:the\s+)?(?:\$?\d+\s?k?|cash|aml|kyc)\s*(?:reporting|aml|kyc|disclosure|currency)?\s*(?:threshold|requirement|limit|trigger|rule)\b",
    ),
)

# ---------------------------------------------------------------------------
# Guaranteed returns
# ---------------------------------------------------------------------------
GUARANTEED_RETURNS = CategorySpec(
    name="guaranteed_returns",
    block_message=(
        "I won't promise guaranteed returns, certain outcomes, or "
        "risk-free profits — they don't exist in real markets, and any "
        "claim of one is a hallmark of fraud. "
        "If you're trying to estimate plausible long-term returns, I can "
        "help you reason about historical averages, ranges, and the "
        "uncertainty around them."
    ),
    harmful_patterns=_compile(
        # "guarantee me 30% returns"
        r"\bguarantee\s+(?:me\s+|that\s+|us\s+)?\d+\s?%",
        # "promise me my money will double in a year"
        r"\bpromise\s+(?:me\s+|that\s+|us\s+)?(?:my\s+money|i'?ll|we'?ll|my\s+investment|my\s+capital|the\s+investment)\b.{0,40}\b(?:double|triple|grow|return|multipl|2x|3x|5x|10x|100\s?%|\d+\s?%)",
        # "tell me a stock that's 100% certain to go up"
        r"\b100\s?%\s+(?:certain|sure|guaranteed|safe|reliable)\b",
        # "what's a foolproof way to make 50% in a month"
        r"\bfoolproof\s+(?:way|strategy|method|trick|plan|system|approach)\b",
        # "risk-free 20% return"
        r"\b(?:risk[- ]free|no[- ]risk|zero[- ]risk|can'?t[- ]lose|cant[- ]lose)\s+(?:way|method|strategy|trade|investment|return|profit|gain)\b",
        # "how do i make 50% in a month guaranteed"
        r"\b(?:make|earn|profit|gain|return)\s+\d+\s?%\b.{0,30}\b(?:guaranteed|certain|sure|risk[- ]free|every\s+month|every\s+week|every\s+day)\b",
    ),
)

# ---------------------------------------------------------------------------
# Reckless advice
# ---------------------------------------------------------------------------
RECKLESS_ADVICE = CategorySpec(
    name="reckless_advice",
    block_message=(
        "I won't recommend that. Putting an entire emergency fund, "
        "retirement balance, or borrowed money into a single risky position "
        "is a recipe for ruin and isn't something I can in good conscience "
        "endorse. "
        "I'm happy to discuss the trade-offs of allocation, leverage, and "
        "concentration in general terms — and to suggest a position size "
        "that's defensible for your profile."
    ),
    harmful_patterns=_compile(
        # "put all my retirement savings in crypto, do it"
        r"\b(?:put|invest|move|throw|dump|allocate|stick|place|drop|pour)\s+(?:all\s+(?:of\s+)?|every\s+penny\s+of\s+|every\s+cent\s+of\s+|the\s+entire\s+|my\s+entire\s+|my\s+whole\s+|all\s+of\s+my\s+|100\s?%\s+of\s+my?\s+)?(?:my\s+)?(?:retirement\s+(?:savings|fund|account|money|nest\s+egg)?|savings|emergency\s+fund|emergency\s+savings|401\s?k|ira|pension|nest\s+egg|life\s+savings)\b",
        # "tell me to take a margin loan"
        r"\b(?:tell\s+me\s+to|advise\s+me\s+to|recommend\s+(?:i|that\s+i|me)|help\s+me)\s+(?:take\s+(?:out\s+)?(?:a\s+)?|get\s+(?:a\s+)?|use\s+)?margin\s+(?:loan|debt|leverage|borrowing|credit)\b",
        # "margin loan to buy more nvidia"
        r"\bmargin\s+(?:loan|debt|leverage|borrowing|credit)\s+to\s+(?:buy|purchase|invest|trade|get|acquire|stack)\b",
        # "put my entire emergency fund into options"
        r"\b(?:put|invest|throw|move|dump|allocate|drop)\s+(?:my\s+)?(?:entire|whole|all|complete|full|total)\s+(?:emergency\s+fund|emergency\s+savings|nest\s+egg|life\s+savings|rainy[- ]day\s+fund|safety\s+net)\s+(?:in|into|on|toward)",
        # "tell me which stock to mortgage my house for"
        r"\bmortgag(?:e|ing)\s+(?:my|the|our|his|her|their)\s+(?:house|home|property|condo|apartment)\b",
        # "i want to put all my retirement savings in crypto, do it"  — also covered above
        # "yolo my entire account into options"
        r"\byolo(?:'?ing|\s+my)\s+(?:entire|whole|all|every|my)\b",
    ),
)

# ---------------------------------------------------------------------------
# Sanctions evasion
# ---------------------------------------------------------------------------
SANCTIONS_EVASION = CategorySpec(
    name="sanctions_evasion",
    block_message=(
        "I can't help with bypassing sanctions or transacting with "
        "sanctioned parties. OFAC (US), HMT (UK), and EU sanctions regimes "
        "carry severe civil and criminal penalties, and brokerages are "
        "required to screen counterparties. "
        "If you'd like, I can explain how sanctions screening works or "
        "what compliance obligations brokers have."
    ),
    harmful_patterns=_compile(
        # "shell company to bypass OFAC sanctions"
        r"\b(?:via\s+|through\s+|using\s+|by\s+way\s+of\s+|with\s+(?:a\s+)?)\s*shell\s+(?:company|corp|corporation|entity|firm|llc)\b",
        # "bypass OFAC" / "evade sanctions"
        r"\b(?:bypass|evade|circumvent|skirt|dodge|get\s+around|work\s+around)\s+(?:ofac|the\s+ofac|sanctions|the\s+sanctions|the\s+embargo|us\s+sanctions|eu\s+sanctions|uk\s+sanctions)\b",
        # "invest in sanctioned russian company without it being traced"
        r"\bsanctioned\b.{0,100}\bwithout\s+(?:it\s+)?(?:being\s+|getting\s+|me\s+being\s+)?(?:traced|detected|caught|noticed|flagged|reported|seen|found)",
        # "buy stock in a sanctioned entity"
        r"\b(?:invest|trade|buy|purchase|sell|move\s+money)\s+(?:in|with|to|from|via)\s+(?:a\s+|the\s+|any\s+)?sanctioned\s+(?:russian|iranian|north\s+korean|cuban|syrian|venezuelan)?\s*(?:company|entity|firm|bank|individual|oligarch|corp|institution)\b",
    ),
)

# ---------------------------------------------------------------------------
# Fraud (residual catch — fake documents, etc.)
# ---------------------------------------------------------------------------
FRAUD = CategorySpec(
    name="fraud",
    block_message=(
        "I can't help with fabricating documents or financial records. "
        "Forging contract notes, tax forms, or brokerage statements is "
        "fraud, exposes you to criminal liability, and undermines the "
        "integrity of regulated markets. "
        "If you have a legitimate accounting question, I'm happy to help."
    ),
    harmful_patterns=_compile(
        # "draft a fake contract note to claim losses"
        r"\b(?:draft|create|generate|fabricate|forge|write|prepare|cook\s+up|whip\s+up|make\s+up|produce)\s+(?:a\s+|an\s+|me\s+a\s+|some\s+)?(?:fake|forged|false|fraudulent|sham|bogus|phony|fictitious|bogus|dummy|fudged|cooked)\s+(?:contract|document|note|invoice|statement|tax\s+form|tax\s+return|form|receipt|bill|broker\s+note|brokerage\s+statement|trade\s+confirmation)\b",
    ),
)


CATEGORIES: tuple[CategorySpec, ...] = (
    INSIDER_TRADING,
    MARKET_MANIPULATION,
    MONEY_LAUNDERING,
    GUARANTEED_RETURNS,
    RECKLESS_ADVICE,
    SANCTIONS_EVASION,
    FRAUD,
)
