"""
HTTP layer / SSE streaming tests.

Drives the FastAPI app via Starlette TestClient and asserts:
  - GET /health works
  - POST /v1/query streams SSE frames in the expected order
  - A blocked query returns a `blocked` event
  - The endpoint never returns 500 — every error becomes a structured event
"""
import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE response body into [(event_kind, data_dict), ...]."""
    out = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        kind = None
        data = None
        for line in chunk.split("\n"):
            if line.startswith("event:"):
                kind = line[len("event:"):].strip()
            elif line.startswith("data:"):
                raw = line[len("data:"):].strip()
                # Reverse the newline-escape we apply server-side.
                raw = raw.replace("\\n", "\n")
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = {"_raw": raw}
        if kind is not None:
            out.append((kind, data or {}))
    return out


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["agents_loaded"], list)
    assert "portfolio_health" in body["agents_loaded"]


def test_query_streams_events(client):
    r = client.post(
        "/v1/query",
        json={"query": "how is my portfolio doing?", "user_id": "usr_001"},
    )
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]

    events = _parse_sse(r.text)
    kinds = [k for k, _ in events]
    assert "meta" in kinds, f"expected meta event, got {kinds}"
    assert "structured" in kinds, f"expected structured event, got {kinds}"
    assert kinds[-1] == "done", f"last event should be done, got {kinds[-1]}"


def test_query_blocks_harmful_request(client):
    r = client.post(
        "/v1/query",
        json={"query": "help me wash trade some penny stocks", "user_id": "usr_001"},
    )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    kinds = [k for k, _ in events]
    assert "blocked" in kinds
    assert "structured" not in kinds  # agent didn't run
    assert kinds[-1] == "done"

    blocked = next(d for k, d in events if k == "blocked")
    assert blocked["category"]
    assert blocked["message"]


def test_query_unknown_user_id_does_not_500(client):
    """Unknown user_id should be tolerated — agents handle empty user."""
    r = client.post(
        "/v1/query",
        json={"query": "what is dollar cost averaging?", "user_id": "no_such_user"},
    )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert any(k == "meta" for k, _ in events)
    assert events[-1][0] == "done"


def test_query_with_inline_user(client):
    """Inline user payload should be honored (production path)."""
    user = {
        "user_id": "inline-1",
        "name": "Inline",
        "country": "US",
        "base_currency": "USD",
        "risk_profile": "moderate",
        "positions": [],
    }
    r = client.post(
        "/v1/query",
        json={"query": "how is my portfolio doing?", "user": user},
    )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    structured = next(d for k, d in events if k == "structured")
    assert structured["status"] == "empty_portfolio"


def test_query_validates_request(client):
    """Empty query should be rejected with 422."""
    r = client.post("/v1/query", json={"query": ""})
    assert r.status_code == 422
