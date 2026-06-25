"""The retrieval engine — the heart of the system.

Three retrievers, each a separate class:
  - BM25Retriever  (lexical, sparse)
  - DenseRetriever (embedding, dense)
  - HybridRetriever (weighted fusion of the two)

Keeping them as composable units means the eval harness can A/B each one in
isolation AND the combined hybrid score. That isolation is exactly what lets us
report "BM25-only vs dense-only vs hybrid vs hybrid+rerank" in the resume.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, Protocol

from ..core.logging import get_logger
from ..models.schemas import Chunk, ScoredChunk

log = get_logger(__name__)

# Lightweight tokenizer shared by BM25 + query preprocessing.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_STOPWORDS = frozenset(
    "the a an and or but if then else of to in on for with at by from is are was "
    "were be been being this that these those it its as into how what why when".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase + alphanumeric tokenization + stopword removal."""
    return [t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text)) if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Retriever protocol — anything that scores chunks against a query implements this.
# ---------------------------------------------------------------------------

class Retriever(Protocol):
    def index(self, chunks: list[Chunk]) -> None: ...
    def search(self, query: str, k: int) -> list[ScoredChunk]: ...
    @property
    def name(self) -> str: ...


# ---------------------------------------------------------------------------
# BM25 — sparse lexical retrieval. Pure-Python so it has zero infra deps and
# runs identically in the eval harness and the live backend.
# ---------------------------------------------------------------------------

class BM25Retriever:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freqs: list[dict[str, int]] = []
        self._df: dict[str, int] = {}  # document frequency per term
        self._avg_dl: float = 0.0
        self._N: int = 0

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._doc_tokens = [tokenize(c.text) for c in chunks]
        self._doc_freqs = []
        self._df = {}
        total_len = 0
        for toks in self._doc_tokens:
            freqs: dict[str, int] = {}
            for t in toks:
                freqs[t] = freqs.get(t, 0) + 1
            self._doc_freqs.append(freqs)
            for term in freqs:
                self._df[term] = self._df.get(term, 0) + 1
            total_len += len(toks)
        self._N = len(chunks)
        self._avg_dl = (total_len / self._N) if self._N else 0.0
        log.info("BM25 indexed %d chunks (avg dl=%.1f)", self._N, self._avg_dl)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        q_terms = tokenize(query)
        scores = [0.0] * self._N
        idf_cache: dict[str, float] = {}
        for term in q_terms:
            if term not in self._df:
                continue
            if term not in idf_cache:
                # smoothed idf, never negative
                idf = (self._N - self._df[term] + 0.5) / (self._df[term] + 0.5) + 1.0
                idf_cache[term] = idf
            idf = idf_cache[term]
            for i, freqs in enumerate(self._doc_freqs):
                f = freqs.get(term, 0)
                if f == 0:
                    continue
                dl = len(self._doc_tokens[i])
                denom = f + self.k1 * (1 - self.b + self.b * (dl / self._avg_dl if self._avg_dl else 0))
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        ranked = sorted(
            (ScoredChunk(chunk=self._chunks[i], score=s) for i, s in enumerate(scores) if s > 0),
            key=lambda sc: sc.score,
            reverse=True,
        )
        return ranked[:k]

    def __len__(self) -> int:
        return self._N


# ---------------------------------------------------------------------------
# Dense — embeddings via sentence-transformers. Lazily imported so the module
# still loads (and BM25 still works) in environments without torch installed,
# e.g. lightweight CI for the eval harness baselines.
# ---------------------------------------------------------------------------

class DenseRetriever:
    name = "dense"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._chunks: list[Chunk] = []
        self._vectors = None  # np.ndarray

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "sentence-transformers is required for dense retrieval. "
                    "pip install sentence-transformers"
                ) from e
            self._model = SentenceTransformer(self.model_name)
            log.info("Dense model loaded: %s", self.model_name)

    def index(self, chunks: list[Chunk]) -> None:
        if not chunks:
            self._chunks = []
            self._vectors = None
            return
        self._ensure_model()
        import numpy as np

        self._chunks = chunks
        self._vectors = self._model.encode(
            [c.text for c in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        log.info("Dense indexed %d chunks (dim=%d)", len(chunks), self._vectors.shape[1])

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if self._vectors is None or not self._chunks:
            return []
        import numpy as np

        self._ensure_model()
        qv = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        # cosine sim (vectors are normalized) -> dot product
        sims = self._vectors @ qv
        top_idx = np.argsort(-sims)[:k]
        return [
            ScoredChunk(chunk=self._chunks[i], score=float(sims[i]))
            for i in top_idx
            if sims[i] > 0
        ]


# ---------------------------------------------------------------------------
# Hybrid — Reciprocal Rank Fusion (RRF) is more robust than raw score blending
# because BM25 and cosine scores live on different scales. RRF only uses ranks.
# We expose a weighted variant so the eval harness can sweep fusion weights.
# ---------------------------------------------------------------------------

class HybridRetriever:
    name = "hybrid"

    def __init__(self, retrievers: list[Retriever], weights: list[float] | None = None, rrf_k: int = 60):
        self.retrievers = retrievers
        if weights is None:
            weights = [1.0 / len(retrievers)] * len(retrievers)
        assert len(weights) == len(retrievers), "one weight per retriever"
        self.weights = weights
        self.rrf_k = rrf_k  # standard RRF constant; 60 is the literature default

    def index(self, chunks: list[Chunk]) -> None:
        for r in self.retrievers:
            r.index(chunks)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        fused: dict[str, float] = {}
        held: dict[str, Chunk] = {}
        for weight, retriever in zip(self.weights, self.retrievers):
            hits = retriever.search(query, k=k * 3)  # over-fetch then fuse
            for rank, scored in enumerate(hits):
                # RRF contribution: weight * 1/(rrf_k + rank)
                fused[scored.chunk.chunk_id] = fused.get(scored.chunk.chunk_id, 0.0) + weight * (
                    1.0 / (self.rrf_k + rank + 1)
                )
                held[scored.chunk.chunk_id] = scored.chunk
        ranked = sorted(
            (ScoredChunk(chunk=held[cid], score=s) for cid, s in fused.items()),
            key=lambda sc: sc.score,
            reverse=True,
        )
        return ranked[:k]

    @property
    def trace_label(self) -> str:
        names = [r.name for r in self.retrievers]
        return "+".join(names)
