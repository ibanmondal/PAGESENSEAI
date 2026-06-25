"""Typed data structures that flow through the RAG pipeline.

Pydantic models for API boundaries + lightweight dataclasses for internal flow.
Keeping these explicit is what lets the eval harness assert against real outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


# ---------------------------------------------------------------------------
# Ingestion payloads (what the extension sends)
# ---------------------------------------------------------------------------

class TabPayload(BaseModel):
    """A single cleaned tab extracted by the extension content script."""
    tab_id: str = Field(..., description="Stable id, e.g. url hash")
    url: HttpUrl
    title: str
    text: str = Field(..., description="Main-content text, boilerplate already removed")
    source_type: Literal["web", "pdf", "github", "youtube", "doc"] = "web"


class IngestRequest(BaseModel):
    """Batch ingest from the extension — one or more open tabs."""
    session_id: str
    tabs: list[TabPayload]


class IngestResponse(BaseModel):
    session_id: str
    ingested_chunks: int
    indexed_tabs: int


# ---------------------------------------------------------------------------
# Query + answer
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    session_id: str
    question: str
    mode: Literal["ask", "compare", "summarize"] = "ask"
    # When mode == "compare", the caller may pin specific tabs to compare across.
    compare_tab_ids: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    """One piece of supporting evidence attached to an answer."""
    tab_id: str
    url: str
    title: str
    snippet: str
    # Relevance score from the re-ranker (0..1 after sigmoid), for transparency.
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    # Which model handled the query — exposes the routing layer to the UI/eval.
    model_used: str
    # 0..1 confidence, derived from re-ranker scores + fact-check signal.
    confidence: float
    # Short human-readable flag when the fact-check pass disagrees.
    warning: str | None = None


# ---------------------------------------------------------------------------
# Internal pipeline types (not serialized over the wire)
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A unit of retrieved text. The atom of the whole system."""
    chunk_id: str
    tab_id: str
    url: str
    title: str
    text: str
    # Position in the source doc — helps citations point to the right spot.
    ordinal: int


@dataclass
class ScoredChunk:
    """A chunk with an attached relevance score from some retriever."""
    chunk: Chunk
    score: float

    def __lt__(self, other: "ScoredChunk") -> bool:
        return self.score < other.score


@dataclass
class RetrievalResult:
    """Output of the retrieval stage, input to the generation stage."""
    query: str
    top_chunks: list[ScoredChunk] = field(default_factory=list)
    # Provenance: which retrievers contributed, for debugging/eval.
    trace: dict = field(default_factory=dict)


@dataclass
class EvalRow:
    """One labeled example for the retrieval eval harness.

    `relevant_chunk_ids` is the ground-truth set: chunks that *should* be
    retrieved for this query. Used to compute nDCG / MRR / Recall@k.
    """
    query: str
    pool_doc_ids: list[str]  # candidate docs (each doc -> many chunks)
    relevant_doc_ids: list[str]  # ground truth: which docs are relevant
    notes: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
