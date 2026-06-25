"""In-memory session store + retrieval orchestration.

Per-session architecture: each browsing session has its own small corpus
(the open tabs), its own index, and its own retriever instance. This mirrors
how the eval harness works and keeps sessions isolated from each other.

Production would swap this for pgvector + Redis, but the in-memory version is
correct, fast, and dependency-free — good enough to ship a demo and run evals.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from ..core.config import get_settings
from ..core.logging import get_logger
from ..models.schemas import (
    Chunk, QueryRequest, RetrievalResult, ScoredChunk, TabPayload,
)
from .chunking import chunk_document
from .reranker import CrossEncoderReRanker, IdentityReRanker, ReRanker
from .retrieval import (
    BM25Retriever, DenseRetriever, HybridRetriever, Retriever,
)

log = get_logger(__name__)


@dataclass
class Session:
    """One user's browsing session: its tabs, chunks, and live retriever."""
    session_id: str
    chunks: list[Chunk] = field(default_factory=list)
    chunk_index: dict[str, Chunk] = field(default_factory=dict)
    retriever: Retriever | None = None
    reranker: ReRanker | None = None
    # Track which tab ids are present so re-ingest can replace (not duplicate).
    tab_ids: set[str] = field(default_factory=set)
    last_active: float = field(default_factory=time.time)
    _dirty: bool = True  # index needs rebuild after a tab change

    def add_tabs(self, tabs: list[TabPayload]) -> int:
        # Remove old chunks for re-ingested tabs, then add fresh ones.
        for t in tabs:
            if t.tab_id in self.tab_ids:
                self.chunks = [c for c in self.chunks if c.tab_id != t.tab_id]
            self.tab_ids.add(t.tab_id)
            new_chunks = chunk_document(
                tab_id=t.tab_id, url=str(t.url), title=t.title, text=t.text,
            )
            self.chunks.extend(new_chunks)
        self.chunk_index = {c.chunk_id: c for c in self.chunks}
        self._dirty = True
        self.last_active = time.time()
        return len(self.chunks)

    def build_retriever(self) -> Retriever:
        settings = get_settings()
        # Build hybrid: BM25 always; dense if its dependency is available.
        retrievers: list[Retriever] = [BM25Retriever()]
        try:
            retrievers.append(DenseRetriever(settings.embedding_model))
        except Exception as e:  # pragma: no cover - env dependent
            log.warning("Dense retriever unavailable, BM25-only: %s", e)
        weights = [settings.bm25_weight, settings.dense_weight][: len(retrievers)]
        # renormalize if dense dropped out
        total = sum(weights) or 1.0
        weights = [w / total for w in weights]
        self.retriever = HybridRetriever(retrievers, weights=weights)
        self.retriever.index(self.chunks)
        self._dirty = False
        log.info("Session %s indexed: %d chunks", self.session_id, len(self.chunks))
        return self.retriever

    def ensure_ready(self) -> None:
        if self.retriever is None or self._dirty:
            self.build_retriever()

    def get_reranker(self) -> ReRanker:
        if self.reranker is None:
            settings = get_settings()
            try:
                self.reranker = CrossEncoderReRanker(settings.reranker_source)
            except Exception as e:  # pragma: no cover
                log.warning("Re-ranker unavailable, using identity: %s", e)
                self.reranker = IdentityReRanker()
        return self.reranker


class SessionStore:
    """Process-local registry of sessions. Thread-safe enough for a demo."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str | None = None) -> Session:
        sid = session_id or uuid.uuid4().hex[:12]
        if sid not in self._sessions:
            self._sessions[sid] = Session(session_id=sid)
            log.info("Created session %s", sid)
        return self._sessions[sid]

    def ingest(self, session_id: str, tabs: list[TabPayload]) -> tuple[int, int]:
        sess = self.get_or_create(session_id)
        n_chunks = sess.add_tabs(tabs)
        return n_chunks, len(sess.tab_ids)

    def retrieve(self, req: QueryRequest) -> RetrievalResult:
        sess = self.get_or_create(req.session_id)
        sess.ensure_ready()
        assert sess.retriever is not None
        settings = get_settings()

        # First-stage retrieval (over-fetch for the re-ranker).
        hits = sess.retriever.search(req.question, k=settings.top_k_retrieval)
        # Second-stage re-ranking.
        reranker = sess.get_reranker()
        top = reranker.rerank(req.question, hits, k=settings.top_k_rerank)

        trace = {
            "retriever": sess.retriever.name if hasattr(sess.retriever, "name") else "hybrid",
            "n_candidates": len(hits),
            "n_kept": len(top),
            "reranker": type(reranker).__name__,
        }
        log.info("Retrieve[%s] q=%r -> %d candidates, %d kept",
                 req.session_id, req.question[:50], len(hits), len(top))
        return RetrievalResult(query=req.question, top_chunks=top, trace=trace)


# Module-level singleton — imported by the API layer.
store = SessionStore()
