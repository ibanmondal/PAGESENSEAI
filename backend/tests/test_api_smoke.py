"""End-to-end smoke test: ingest tabs -> query -> get cited answer.

Uses FastAPI's TestClient so no server process is needed. The LLM is stubbed
via an injected answer_fn so the test runs offline — it asserts the RETRIEVAL
and CITATION plumbing, which is the part we can deterministically verify.
Generation quality is evaluated separately via the eval harness + a held-out set.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app.main import app
from app.agents.graph import AgentState, build_graph, run_query
from app.agents.llm_client import ModelRouter
from app.models.schemas import QueryRequest


client = TestClient(app)

TABS = [
    {
        "tab_id": "bm25_paper",
        "url": "https://example.com/bm25",
        "title": "BM25 Ranking",
        "text": (
            "Okapi BM25 is a probabilistic ranking function. It scores documents "
            "using term frequency, inverse document frequency, and document length "
            "normalization. Parameters k1 and b tune term frequency saturation."
        ),
        "source_type": "web",
    },
    {
        "tab_id": "cross_encoder",
        "url": "https://example.com/crossencoder",
        "title": "Cross-Encoder Re-ranking",
        "text": (
            "A cross-encoder jointly encodes a query and a document to produce a "
            "relevance score. It is more accurate than bi-encoders but slower, so "
            "it is used to re-rank the top candidates from a first-stage retriever."
        ),
        "source_type": "web",
    },
]


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ingest_then_query_offline():
    # 1. Ingest
    r = client.post("/api/ingest", json={"session_id": "s1", "tabs": TABS})
    assert r.status_code == 200
    body = r.json()
    assert body["indexed_tabs"] == 2
    assert body["ingested_chunks"] >= 2

    # 2. Query with a stubbed LLM so the test is fully offline.
    def fake_answer(system: str, user: str) -> str:
        return "BM25 ranks documents using term frequency and length normalization [1]."

    class FakeClient:
        name = "fake"
        model = "fake-test-model"
        def complete(self, system, user, temperature=0.2):
            return fake_answer(system, user)

    router = ModelRouter(fast=FakeClient(), strong=FakeClient())
    graph = build_graph(
        retrieve_fn=lambda req: __import__(
            "app.rag.store", fromlist=["store"]
        ).store.retrieve(req),
        router=router,
    )
    state = AgentState(request=QueryRequest(session_id="s1", question="How does BM25 rank documents?"))
    resp = run_query(state, graph)

    assert "BM25" in resp.answer
    assert resp.model_used == "fake-test-model"
    assert len(resp.citations) >= 1
    # The top citation should be the BM25 doc, not the cross-encoder doc.
    assert resp.citations[0].tab_id == "bm25_paper"
    assert 0.0 <= resp.confidence <= 1.0


def test_query_without_ingest_errors():
    r = client.post("/api/query", json={
        "session_id": "empty-session", "question": "anything",
    })
    assert r.status_code == 400
