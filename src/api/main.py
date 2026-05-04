"""
FastAPI application.

Single streaming endpoint at POST /v1/query.

SSE event format (server-sent events):
    event: <event_kind>
    data: <json payload>

Event kinds (mirrors `EventKind` in src/agents/base.py):
    meta        — classifier verdict (intent + entities + confidence)
    structured  — agent's structured payload (single per response)
    token       — incremental narrative text
    blocked     — safety guard refused the request
    error       — pipeline-level error event
    done        — terminal event with summary metrics

Why SSE and not a JSON fallback: per ASSIGNMENT.md, "Streaming only — no
JSON fallback." We honor that. Clients that can't parse SSE can use the
`stream_to_list()` helper in tests/ as a reference implementation.

Why a single endpoint: the entire spec is "answer the user's question";
multiplying endpoints (one per agent) would leak the routing decision
into the URL, which (a) breaks if we add agents, (b) breaks the safety
guarantee that EVERY request goes through the same guard, and (c) breaks
the per-session history model. One door, one guard, one history.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from ..agents import EventKind
from ..agents.registry import REGISTRY
from ..config import settings
from ..market_data import MockMarketDataProvider, default_provider
from ..pipeline import process_request
from ..session import default_store
from .schemas import HealthResponse, QueryRequest

log = logging.getLogger(__name__)

# Resolve fixtures relative to the repo root. We allow this in dev/demo so a
# caller can hit the API with `user_id=usr_001` and get a working response.
# In a production deployment the inline `user` payload would be the only
# supported path.
FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures" / "users"


# ---------------------------------------------------------------------------
# App lifespan: lazy-construct shared deps (market data provider, narrator).
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Construct shared singletons once. The narrator client lazy-connects
    # on first call; if OPENAI_API_KEY is unset, we leave it as None and
    # the agents fall back to deterministic narrative.
    state: dict = {}
    try:
        state["market_data"] = default_provider()
    except Exception as e:
        log.warning("Market data provider init failed: %s — using mock", e)
        state["market_data"] = MockMarketDataProvider()

    if settings.llm_available:
        # Lazy-import so a missing openai package doesn't break the import.
        try:
            from ..llm_client import LLMClassifierCall, LLMNarrator
            state["classifier_llm"] = LLMClassifierCall()
            state["narrator_llm"] = LLMNarrator()
        except Exception as e:
            log.warning("LLM client init failed: %s — running rules-only", e)
            state["classifier_llm"] = None
            state["narrator_llm"] = None
    else:
        state["classifier_llm"] = None
        state["narrator_llm"] = None

    app.state.deps = state
    log.info(
        "App started. llm=%s market_data=%s",
        bool(state["classifier_llm"]),
        type(state["market_data"]).__name__,
    )
    try:
        yield
    finally:
        # Nothing to clean up for in-memory store. If you swap in Redis,
        # close the connection pool here.
        pass


app = FastAPI(
    title="Valura AI Wealth Co-Investor",
    version="0.1.0",
    description=(
        "AI co-investor microservice. Single streaming endpoint at POST /v1/query. "
        "Routes user queries through a safety guard + intent classifier + "
        "specialist agent. Portfolio Health is the only fully-implemented "
        "specialist in this build; the rest return structured stubs."
    ),
    lifespan=lifespan,
)

# CORS — open in dev; lock down per-tenant in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_user_by_id(user_id: str) -> dict | None:
    """Look up a fixture user by id. Returns None if not found."""
    if not user_id or not FIXTURES_DIR.exists():
        return None
    for path in FIXTURES_DIR.glob("*.json"):
        try:
            with open(path, encoding="utf-8") as f:
                u = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if u.get("user_id") == user_id:
            return u
    return None


def _sse_frame(kind: EventKind, data: dict) -> bytes:
    """Encode one SSE frame. Keeps payload encoding consistent (JSON, UTF-8)."""
    payload = json.dumps(data, ensure_ascii=False, default=str)
    # Note: `data:` lines must not contain raw newlines; we replace them
    # with the SSE multi-line continuation form.
    safe = payload.replace("\n", "\\n")
    return f"event: {kind.value}\ndata: {safe}\n\n".encode("utf-8")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        llm_available=settings.llm_available,
        agents_loaded=sorted(a.value for a in REGISTRY.keys()),
    )


@app.post("/v1/query")
async def query(req: QueryRequest):
    """
    Single streaming endpoint.

    Returns a `text/event-stream` response. Clients should consume frames
    incrementally — see README.md for a `curl` example.
    """
    # Resolve user context: inline payload wins over user_id lookup.
    user = req.user
    if user is None and req.user_id:
        user = _load_user_by_id(req.user_id)
    # We deliberately do NOT 404 if the user_id is unknown — the agents
    # tolerate missing user context. Instead we proceed with an empty user.

    deps = app.state.deps

    async def stream() -> AsyncIterator[bytes]:
        try:
            async for ev in process_request(
                query=req.query,
                user=user,
                session_id=req.session_id,
                session_store=default_store,
                market_data=deps.get("market_data"),
                classifier_llm=deps.get("classifier_llm"),
                narrator_llm=deps.get("narrator_llm"),
            ):
                yield _sse_frame(ev.kind, ev.data)
        except Exception as e:
            # Last-resort: surface a structured error frame so the client
            # always sees a terminal event. Should be unreachable.
            log.exception("Unhandled exception in stream: %s", e)
            yield _sse_frame(EventKind.error, {"stage": "stream", "message": str(e)})
            yield _sse_frame(EventKind.done, {"error": True})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # disable nginx buffering on streamed responses
            "Connection": "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exc_handler(_, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail or "request failed"},
    )
