"""HTTP routes. Thin layer over the retrieval store + agent graph.

Every endpoint is intentionally small — business logic lives in rag/ and agents/
so it stays unit-testable without spinning up the server.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..agents.graph import AgentState, build_graph, run_query
from ..agents.llm_client import build_router
from ..core.logging import get_logger
from ..models.schemas import (
    IngestRequest, IngestResponse, QueryRequest, QueryResponse,
)
from ..rag.store import store

log = get_logger(__name__)
router = APIRouter()

# Build the model router once at import. If no keys are set, routing still works
# structurally; generation will only fail if a query actually needs an LLM.
_router = build_router()


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "sessions": len(store._sessions)}


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    if not req.tabs:
        raise HTTPException(status_code=400, detail="No tabs provided")
    n_chunks, n_tabs = store.ingest(req.session_id, req.tabs)
    log.info("Ingested session=%s tabs=%d chunks=%d", req.session_id, n_tabs, n_chunks)
    return IngestResponse(
        session_id=req.session_id, ingested_chunks=n_chunks, indexed_tabs=n_tabs,
    )


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    sess = store.get_or_create(req.session_id)
    if not sess.chunks:
        raise HTTPException(
            status_code=400,
            detail="No ingested content for this session. POST /ingest first.",
        )
    # Reuse a per-session graph; retrieval closure binds to the store.
    graph = build_graph(
        retrieve_fn=lambda r: store.retrieve(r),
        router=_router,
    )
    state = AgentState(request=req)
    return run_query(state, graph)
