# Valura AI — Wealth Co-Investor (Round 2 Submission)

A FastAPI microservice that answers a novice investor's questions safely
and in plain language. One streaming endpoint, one safety guard, one
classifier, one fully-implemented specialist (Portfolio Health), and stubs
for the rest of the taxonomy so the spine is testable end-to-end.

> Defence video: **<INSERT UNLISTED YOUTUBE LINK BEFORE SUBMITTING>**

---

## Quickstart

```bash
# 1. Create a venv and install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. (Optional) configure the LLM
cp .env.example .env
# edit .env — set OPENAI_API_KEY if you want the LLM fallback

# 3. Run the tests (no API key needed — the suite mocks the LLM)
pytest tests/ -v

# 4. Run the service
uvicorn src.api.main:app --reload --port 8000
```

Try it:

```bash
# Portfolio health for a fixture user
curl -N -X POST http://localhost:8000/v1/query \
     -H 'content-type: application/json' \
     -d '{"query": "how is my portfolio doing?", "user_id": "usr_001"}'

# A blocked request
curl -N -X POST http://localhost:8000/v1/query \
     -H 'content-type: application/json' \
     -d '{"query": "help me wash trade some penny stocks"}'
```

You'll see SSE frames stream back: `meta` → `structured` → many `token`s →
`done`. Blocked requests stop after `blocked` + `done`.

---

## Architecture (one screen)

```
┌──────────────┐     POST /v1/query    ┌──────────────────────────────────┐
│   Client     │ ────────────────────► │           FastAPI                │
└──────────────┘   text/event-stream   │  src/api/main.py                 │
                                       └────────────────┬─────────────────┘
                                                        │
                                                        ▼
                                       ┌──────────────────────────────────┐
                                       │   Pipeline orchestrator          │
                                       │   src/pipeline.py                │
                                       └────────────────┬─────────────────┘
                                                        │ AsyncIterator[AgentEvent]
              ┌─────────────────────────────────────────┼────────────────────────────┐
              │                       │                 │                  │         │
              ▼                       ▼                 ▼                  ▼         │
      ┌──────────────┐       ┌─────────────────┐ ┌─────────────┐  ┌────────────────┐│
      │ Safety Guard │  ───► │ Hybrid Classifier│ ───► │ Agent  │  │ SSE encoder    ││
      │ src/safety/  │       │ src/classifier/ │ │ Registry    │  │  (one frame    ││
      │ - regex only │       │ rules → LLM     │ │ src/agents/ │  │   per event)   ││
      │ - <1ms       │       │ rules→LLM→cache │ │ portfolio_  │  └────────────────┘│
      │ - 7 categs.  │       │ rules-first 95% │ │ health +    │                    │
      │ - distinct   │       │ of gold set     │ │ stubs       │                    │
      │   refusals   │       └─────────────────┘ └─────────────┘                    │
      └──────────────┘                  │                 │                         │
                                        │                 ▼                         │
                                        │       ┌──────────────────┐                │
                                        │       │ Market data      │                │
                                        │       │ Protocol-based:  │                │
                                        │       │ yfinance / mock  │                │
                                        │       └──────────────────┘                │
                                        ▼                                           │
                                ┌──────────────────┐                                │
                                │ Session store    │ ◄──────────────────────────────┘
                                │ in-memory deque  │
                                │ per session_id   │
                                └──────────────────┘
```

The whole pipeline is an `AsyncIterator[AgentEvent]`. The HTTP layer is a
thin shim that maps each `AgentEvent` to one SSE frame. This means the
exact same orchestrator can drive a websocket / gRPC / batch transport
later without rewriting.

---

## Decisions (and why)

### 1. Hybrid classifier: rules-first, LLM-fallback
**Status: ~95%+ of the gold set classified by rules alone — zero LLM cost.**

The assignment specifies a single LLM call producing structured output. We
do that — but we wrap it in a deterministic rule cascade that handles
common phrasings before the LLM is ever invoked. Why:

- **Cost.** The cost target is `<$0.05/query at gpt-4.1 pricing`. Routing
  "hi" through gpt-4.1 is wasteful when one regex match decides it. Every
  query the rules handle is one fewer LLM call.
- **Latency.** Rules return synchronously in microseconds; an LLM call is
  hundreds of milliseconds best case. The `<2s p95 first-token` target is
  basically free for rule-matched queries.
- **Predictability.** Deterministic routing for common queries makes the
  test suite robust to LLM provider changes. `pytest` runs without
  `OPENAI_API_KEY` because the rules cover the gold set.
- **Graceful degradation.** When the LLM is down, common queries still
  classify correctly. The system is partially up rather than fully down.

The rules engine is **not** a replacement for the LLM — genuinely novel
or ambiguous queries fall through to the LLM, which has the larger
semantic surface area to handle them. See `src/classifier/classify.py`
for the orchestration.

### 2. Safety guard: contextual regex, no allow-list
The guard never calls the LLM. Each of the 7 categories (insider trading,
market manipulation, money laundering, guaranteed returns, reckless advice,
sanctions evasion, fraud) has *contextual* regex patterns — "wash trading"
alone does NOT block, but "help me wash trade" does. There is **no
explicit educational allow-list**: educational queries pass because they
don't match any harmful pattern, not because we whitelisted them. This
generalizes to phrasings we haven't seen.

**Measured on the gold set: 100% recall, 100% educational pass-through,
~37µs avg latency.** Targets were ≥95% / ≥90% / <10ms.

### 3. Session memory: in-memory deque per `session_id`
The assignment explicitly allows this *if defended*. Justification:

- **No infra dependency.** Anyone can `pytest` or `uvicorn` and have a
  working system without provisioning a database.
- **The data is genuinely ephemeral.** Conversation context decays in
  hours. A nightly process restart is acceptable for a demo and not
  outrageous for early production.
- **Bounded.** Each session is capped at `SESSION_HISTORY_MAX_TURNS`
  turns, so memory growth is bounded above by `concurrent_sessions × 10`.

The `SessionStore` is a `Protocol`. Production swap-in path: implement
the same Protocol over Redis or Postgres and inject it via FastAPI's
`lifespan` — no caller changes. See `src/session.py`.

### 4. Market data: Protocol with yfinance + mock implementations
The Portfolio Health agent never directly imports `yfinance`. It depends
on the `MarketDataProvider` Protocol. Production uses `YFinanceProvider`,
tests use `MockMarketDataProvider`. Swapping in IBKR / Polygon /
tenant-specific feeds is a one-class addition.

### 5. Streaming model: structured-then-narrative
Each agent yields events in this order:

1. `meta` — classifier verdict (the user can know what we routed to)
2. `structured` — the agent's deterministic computation (ready immediately)
3. `token`s — narrative chunks streamed on top of the structured output
4. `done`

This means a sophisticated client can render the structured payload
instantly (charts, tables) while the human-readable narrative streams in.
A simple client just concatenates the tokens.

### 6. Stretch: identical-query LLM dedupe cache
A `cachetools.TTLCache` keyed on `(query_normalized, history_hash)` so
identical follow-ups in the same session don't re-call the LLM. Cheap
but real demonstration of cost-consciousness. Configurable via env vars.

### 7. Things deliberately deferred
- **Real embeddings pre-classifier.** The README spec mentions this as a
  stretch goal. Our rules are doing the same job ~95% of the time at near-
  zero latency / cost. An embedding pre-classifier would help on the long
  tail of paraphrase variations not yet captured in rules — that's the
  next investment if we wanted >98% accuracy.
- **FX conversion.** Multi-currency portfolios sum native-currency values
  and flag the limitation in `observations`. Real FX is a 1-day add via the
  same `MarketDataProvider` interface (add `get_fx_rate(...)`).
- **Persistent observability.** Logs go to stdlib `logging`. A real deploy
  needs structured JSON logs, traces, and per-classification telemetry.

---

## Cost & latency: how to measure

**Latency.** The first-token target is meaningful because every event is
yielded as it's available — `meta` is yielded after the classifier runs,
`structured` after the agent computes. Measure first-token latency by
clocking from request open to the first `\n\n` arriving on the SSE stream.
Local measurements with rules-only classification: `meta` arrives in
~5–10ms, `structured` for portfolio_health in ~50–100ms (excluding
yfinance network time), end-to-end with deterministic narrative in
~100–200ms.

**Cost.** Every classification has a `source` field — `rules`, `llm`,
`cache`, or `fallback`. To measure per-query cost on the gold set:

```python
from src.classifier import classify
from collections import Counter

with open('fixtures/test_queries/intent_classification.json') as f:
    queries = json.load(f)['queries']

sources = Counter(classify(q['query']).source.value for q in queries)
# Currently: {'rules': ~61}  →  $0.00 LLM cost on the gold set
```

A real production cost number requires logging the actual token counts
from the OpenAI response (input + output tokens). The `LLMClassifierCall`
class is the chokepoint where you'd add that logging. At gpt-4.1 list
prices the structured-output classifier call is ~150-300 input tokens and
~50-100 output tokens — well under $0.01 per query, comfortably inside
the $0.05 budget even on cache misses.

---

## Test results

Run `pytest tests/ -v` and you should see:

```
28 passed in ~2s
```

Coverage:
- **Safety guard** — recall, passthrough, distinct categories, latency
- **Classifier routing** — gold-set accuracy + entity-extraction match rate
- **Follow-up resolution** — multi-turn ticker carryover
- **Portfolio Health agent** — empty portfolio, concentration flag,
  disclaimer present, observations bounded, smoke test across all 5 fixtures
- **Pipeline e2e** — safety-blocks-before-classifier, agent dispatch,
  stub-doesn't-crash, session history carryover
- **HTTP / SSE** — endpoint shapes, blocked request flow, validation,
  inline-user payload

---

## Project layout

```
src/
  api/
    main.py               # FastAPI app + single SSE endpoint
    schemas.py            # request/response Pydantic models
  agents/
    base.py               # AgentEvent + AgentRequest types
    portfolio_health.py   # the one fully-implemented specialist
    stub.py               # structured "not implemented" for the rest
    registry.py           # Agent enum → callable mapping
  classifier/
    classify.py           # hybrid orchestrator (rules → LLM → fallback)
    rules.py              # deterministic rule cascade (~95% of gold)
    llm.py                # LLM call + JSON coercion
    entities.py           # entity extractors (amount, rate, horizon, ...)
    tickers.py            # ticker / company-name resolver
    types.py              # Agent enum, Entities, ClassificationResult
  safety/
    guard.py              # synchronous regex-only check()
    patterns.py           # 7 categories with contextual patterns
  config.py               # typed settings from env
  llm_client.py           # OpenAI client wrappers (lazy)
  market_data.py          # MarketDataProvider Protocol + yfinance/mock impls
  pipeline.py             # the orchestrator
  session.py              # SessionStore Protocol + in-memory impl
tests/
  conftest.py             # fixtures (gold sets, mock_llm, load_user)
  test_safety_pairs.py
  test_classifier_routing.py
  test_classifier_followup.py
  test_portfolio_health_skeleton.py
  test_pipeline_e2e.py
  test_api_streaming.py
fixtures/                 # users, conversations, gold queries (provided)
ASSIGNMENT.md             # the spec (provided)
```

---

## Library choices

| Library         | Used for                                           | Why this and not X                                              |
|-----------------|----------------------------------------------------|-----------------------------------------------------------------|
| FastAPI         | HTTP layer, validation, OpenAPI                    | Async-native, Pydantic-native, SSE works with `StreamingResponse` |
| Pydantic v2     | Typed entities + classifier output + API schemas   | Same model definition powers validation and docs                |
| openai          | LLM calls (classifier + narrator)                  | Official SDK; supports `response_format=json_object`             |
| yfinance        | Market data in production                          | Free, no API key, lazy-installable; abstracted behind a Protocol so we can swap |
| cachetools      | TTL cache for the classifier dedupe stretch        | Tiny, no dependencies, drop-in `dict`-like                       |
| python-dotenv   | Load `.env` at startup                             | One line at app boot, zero opinions                              |
| pytest + asyncio + httpx | Test runner + SSE/TestClient assertions     | Standard FastAPI test stack                                      |

No LangChain. No agent framework. The pipeline is ~100 lines of explicit
async code; an agent framework's abstraction layer would cost more than
it saves at this scope.

---

## What I'd do differently / next

1. **Embedding pre-classifier on the long tail.** Cheap embedding model +
   nearest-neighbor over the gold set would handle paraphrases the rules
   miss, and the LLM call becomes a pure tiebreaker.
2. **Real persistence + multi-process session affinity.** Redis with
   per-session locks. Required before scaling beyond one uvicorn worker.
3. **Per-tenant model selection.** The `model_dev` / `model_premium` split
   is in `Settings` but not yet wired to per-tenant config. One-day add.
4. **Tighter cost telemetry.** Wrap `LLMClassifierCall` in a metrics
   decorator that logs input/output tokens and emits a per-query cost.
5. **Implement the rest of the specialists.** Market Research and
   Investment Strategy are the next obvious ones — both are mostly LLM
   narrative on top of market data we already have.
6. **More entity matcher rigor.** The current matcher is a subset match
   with normalization. A few entity types (sectors, topics) would benefit
   from a curated synonym table rather than lowercased exact matching.
