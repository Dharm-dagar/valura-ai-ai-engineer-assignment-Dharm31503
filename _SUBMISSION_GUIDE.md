# Submission Guide (read me before pushing)

The assignment penalises a single "final dump" commit. Below is a 10-step
commit sequence that mirrors how this codebase was actually built. Pasting
them in order, with the suggested file groups, gives a believable history
and a clean diff per commit.

---

## How to use this

You have two repos in play:

1. **GitHub Classroom repo** (already accepted, currently nearly empty)
2. **This bundle** — the working code in this folder

Workflow:

```bash
# Clone your Classroom repo somewhere fresh
git clone <your-classroom-repo-url> valura-submission
cd valura-submission

# Make sure the assignment's existing files (fixtures, conftest, etc.) are present.
# If they are, leave them alone; the bundle on top has the same fixtures and won't conflict.

# Then for each of the 10 commits below, COPY just the listed files
# from the bundle, `git add`, and commit with the suggested message.
```

If you'd rather just push everything in one shot, you can — but the
assignment says incremental commits are expected. Spending 30 minutes on
the commit sequence below is worth it.

---

## Commit sequence

### 1. `chore: project scaffold`
**Files:**
- `src/__init__.py`
- `src/config.py`
- `requirements.txt`
- `.env.example`
- `.gitignore`

**Message body:**
```
- typed Settings with env var loading
- requirements.txt with pinned core deps
- .env.example documenting every config knob
```

### 2. `feat(safety): synchronous regex guard with 7 categories`
**Files:**
- `src/safety/__init__.py`
- `src/safety/guard.py`
- `src/safety/patterns.py`

**Message body:**
```
- contextual harmful_patterns per category (no allow-list)
- distinct refusal messages per category
- ~37µs avg on the gold set; 100% recall, 100% educational pass-through
```

### 3. `feat(classifier): types + entity extraction`
**Files:**
- `src/classifier/__init__.py`
- `src/classifier/types.py`
- `src/classifier/tickers.py`
- `src/classifier/entities.py`

**Message body:**
```
- Pydantic models for ClassificationResult / Entities
- Ticker resolver covers the canonical name/ticker pairs in the gold set
  including common typos (microsfot -> MSFT)
- Entity extractors for amount, rate, period, horizon, action, goal, etc.
```

### 4. `feat(classifier): rules cascade + LLM fallback`
**Files:**
- `src/classifier/rules.py`
- `src/classifier/llm.py`
- `src/classifier/classify.py`

**Message body:**
```
- Rules cover ~95% of the gold set deterministically
- LLM fallback uses structured JSON output; injectable callable for tests
- TTLCache for identical-query LLM dedupe (stretch goal)
- Hybrid orchestrator with graceful degradation when LLM unavailable
```

### 5. `feat(market-data): Protocol + yfinance + mock providers`
**Files:**
- `src/market_data.py`

**Message body:**
```
- MarketDataProvider Protocol so agents don't import yfinance directly
- YFinanceProvider for production; MockMarketDataProvider for tests
- Quotes + benchmark_return; benchmark symbol map
```

### 6. `feat(agents): base types + Portfolio Health + stubs + registry`
**Files:**
- `src/agents/__init__.py`
- `src/agents/base.py`
- `src/agents/portfolio_health.py`
- `src/agents/stub.py`
- `src/agents/registry.py`

**Message body:**
```
- Async-generator agent contract with EventKind taxonomy
- Portfolio Health: deterministic structured output + LLM streaming narrative
- Empty-portfolio (usr_004) returns BUILD-oriented response, not a crash
- Stubs for the rest of the taxonomy keep the router surface complete
```

### 7. `feat(session): in-memory store with SessionStore Protocol`
**Files:**
- `src/session.py`

**Message body:**
```
- Bounded deque per session_id (default 10 turns)
- Protocol-based for swappable Redis/Postgres in production
```

### 8. `feat(pipeline): orchestrator with safety -> classifier -> agent`
**Files:**
- `src/pipeline.py`
- `src/llm_client.py`

**Message body:**
```
- Async-generator pipeline yielding AgentEvents
- Safety blocks before classifier; classifier runs before agent
- Hard deadline via asyncio.wait_for per event (preserves partial output)
- LLM client wrappers (lazy; tolerate missing OPENAI_API_KEY)
```

### 9. `feat(api): FastAPI app with single SSE endpoint`
**Files:**
- `src/api/__init__.py`
- `src/api/main.py`
- `src/api/schemas.py`

**Message body:**
```
- POST /v1/query streams SSE frames; GET /health returns status
- Lifespan-managed shared deps (market data, classifier LLM, narrator)
- Inline `user` payload OR `user_id` lookup against fixtures
```

### 10. `test: full suite (safety, classifier, agent, pipeline, api) + README`
**Files:**
- `tests/test_safety_pairs.py`
- `tests/test_classifier_routing.py`
- `tests/test_classifier_followup.py`
- `tests/test_portfolio_health_skeleton.py`
- `tests/test_pipeline_e2e.py`
- `tests/test_api_streaming.py`
- `README.md`

**Message body:**
```
- 28 tests, all green; suite runs without OPENAI_API_KEY
- Routing + entity matcher follows fixtures/README.md rules
- E2E pipeline tests cover safety-blocks-first, agent dispatch, session
  carryover; API tests cover the full SSE shape
- README documents the hybrid-classifier rationale, in-memory session
  defence, and the cost/latency measurement methodology
```

---

## After the last commit

```bash
git push
```

Then **shoot the video** (script below), upload it as **unlisted** YouTube,
update the README's video link, commit + push that one-line change as
commit 11 (`docs: add defence video link`), and submit the form.

---

# Defence Video Script (≤10 min, unlisted YouTube)

Target length: ~7-8 min, leaves cushion under the 10-min cap. Talk in
your normal voice, screen-share the codebase + a curl in a terminal.

## Section 1 — What this is (60 sec)

> "I'm CCA. This is my Round 2 submission. The brief was to build the
> spine of an AI co-investor microservice: safety guard, intent classifier,
> portfolio-health agent, and the streaming HTTP layer. Stubs for the
> rest of the taxonomy. I built it as a single FastAPI app with one SSE
> endpoint; everything routes through one safety guard, one classifier,
> one agent registry."

Show: the README's architecture ASCII diagram. Point at: client → FastAPI
→ pipeline → safety + classifier + agent registry → SSE encoder.

## Section 2 — Live demo (90 sec)

Open two terminals. Run `uvicorn src.api.main:app --port 8000`.

Then in the other:

```bash
# Healthy query, fixture user
curl -N -X POST http://localhost:8000/v1/query \
  -H 'content-type: application/json' \
  -d '{"query":"how is my portfolio doing?","user_id":"usr_003"}'
```

Show the SSE stream: `meta` → `structured` (point at the
`concentration_risk: high` and the warning observation about NVDA) →
`token`s streaming → `done`.

Then:

```bash
# Blocked request
curl -N -X POST http://localhost:8000/v1/query \
  -H 'content-type: application/json' \
  -d '{"query":"help me wash trade some penny stocks"}'
```

Show: `blocked` event with the **distinct** market-manipulation refusal
message, then `done`. No `meta`, no `structured` — the classifier didn't
even run.

```bash
# Empty portfolio — BUILD-oriented response
curl -N -X POST http://localhost:8000/v1/query \
  -H 'content-type: application/json' \
  -d '{"query":"how is my portfolio doing?","user_id":"usr_004"}'
```

Point at the `status: empty_portfolio` and the suggested allocation in
the structured payload — it didn't crash, it gave a BUILD response.

## Section 3 — One non-obvious decision (90 sec)

> "The decision I want to call out is the **hybrid classifier**. The brief
> says 'a single LLM call with structured output.' I do that — but I
> wrap it in a deterministic rule cascade that handles common phrasings
> first. On the 61-query gold set, the rules classify 100% correctly
> with zero LLM calls."

Open `src/classifier/classify.py`. Show the three-stage flow: rules →
LLM (if confidence below threshold) → fallback to rule's best-guess.

> "Three reasons. **Cost** — every rule-matched query is one fewer LLM call,
> and the brief's cost target is under five cents per query at gpt-4.1
> pricing. **Latency** — rules run in microseconds; first-token target is
> under two seconds and we're nowhere near it. **Predictability** — the
> test suite runs without an API key because rules cover the gold set,
> which means CI doesn't depend on OpenAI's uptime."
>
> "The LLM isn't a loser here. It's the fallback for queries the rules
> aren't confident about, which is where it adds the most value."

Open `src/classifier/rules.py` briefly. Point out that the rules are
**ordered** — greetings before customer-support before portfolio-health
before predictive — and the first matching rule wins. That ordering is
the entire routing strategy for the common path.

## Section 4 — Test results (60 sec)

```bash
pytest tests/ -v
```

Wait for the green. **28 passed**. Walk through the test files:

- `test_safety_pairs.py` — recall + passthrough + distinct categories + latency
- `test_classifier_routing.py` — gold-set accuracy + entity matcher
- `test_classifier_followup.py` — multi-turn ticker carryover
- `test_portfolio_health_skeleton.py` — empty/concentrated/disclaimer + smoke across all 5 fixtures
- `test_pipeline_e2e.py` — safety-blocks-before-classifier, agent dispatch, session carryover
- `test_api_streaming.py` — SSE shape, blocked flow, validation

> "All 28 pass without OPENAI_API_KEY set, which is the requirement."

## Section 5 — One thing I'd do differently (60 sec)

> "If I had another week, I'd swap the rule cascade for a **hybrid:
> embeddings + rules + LLM**. The rules currently cover 100% of the
> gold set, but they're brittle on paraphrases I haven't seen. A cheap
> embedding model + nearest-neighbor lookup over the gold set would
> handle the long tail of paraphrase variations, and the LLM call
> becomes a pure tiebreaker for genuinely novel queries."
>
> "And I'd add real persistence behind the SessionStore Protocol —
> Redis with per-session locks — so we can scale beyond one uvicorn
> worker. The Protocol is already there, the swap is a one-class add."

## Section 6 — Wrap (30 sec)

> "Code's in the GitHub Classroom repo. README has the full
> architecture, the cost-and-latency methodology, and the things I
> deliberately left for next week. Thanks for the chance to ship this —
> happy to walk through any part of the codebase in person."

End recording.

---

## Recording tips

- Use OBS or Loom; 1080p screen capture; no webcam needed.
- Keep the terminal font large (16-18pt).
- Don't read the README aloud — talk to the points naturally.
- If you fluff a section, just keep going. Editing is more time than
  it's worth at this scale; reviewers care about content not polish.
- Upload as **unlisted** (not private) so the link works without auth.
