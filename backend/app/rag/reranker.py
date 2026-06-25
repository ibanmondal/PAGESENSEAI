"""Cross-encoder re-ranker.

A cross-encoder scores (query, chunk) pairs jointly — far more accurate than
bi-encoder similarity for ranking, but too expensive to run over the whole
corpus. So the pipeline is: BM25+dense retrieve top-N -> re-rank to top-k.

This module wraps a HuggingFace cross-encoder and adds a sigmoid normalization
so scores land in [0, 1], which doubles as a transparent "relevance confidence"
in citations and the eval harness.

The fine-tuned weights (from eval/scripts/finetune_reranker.py) load via the
`reranker_finetuned_path` config setting — the SAME interface, drop-in.
"""
from __future__ import annotations

from typing import Protocol

from ..core.logging import get_logger
from ..models.schemas import Chunk, ScoredChunk

log = get_logger(__name__)


class ReRanker(Protocol):
    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]: ...


def _sigmoid(x: float) -> float:
    # Numerically stable sigmoid; maps raw logit-ish scores into (0, 1).
    if x >= 0:
        z = 2.718281828459045 ** (-x)
        return 1.0 / (1.0 + z)
    z = 2.718281828459045 ** x
    return z / (1.0 + z)


class CrossEncoderReRanker:
    """Wraps a HuggingFace cross-encoder (e.g. ms-marco-MiniLM-L-6-v2)."""

    def __init__(self, model_source: str, max_chunks: int = 50):
        self.model_source = model_source
        self.max_chunks = max_chunks
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "sentence-transformers required for re-ranking. "
                    "pip install sentence-transformers"
                ) from e
            # max_length is model-dependent; ms-marco MiniLM uses 512.
            self._model = CrossEncoder(self.model_source, max_length=512)
            log.info("Re-ranker loaded: %s", self.model_source)

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        self._ensure_model()
        cand = candidates[: self.max_chunks]
        pairs = [(query, c.chunk.text) for c in cand]
        raw_scores = self._model.predict(pairs, show_progress_bar=False)
        scored = [
            ScoredChunk(chunk=c.chunk, score=_sigmoid(float(s)))
            for c, s in zip(cand, raw_scores)
        ]
        scored.sort(key=lambda sc: sc.score, reverse=True)
        log.info(
            "Re-ranked %d -> top %d (best=%.3f)", len(cand), min(k, len(scored)),
            scored[0].score if scored else 0.0,
        )
        return scored[:k]


class IdentityReRanker:
    """No-op re-ranker. Keeps the top-K from retrieval as-is.

    Used as the *baseline* in the eval harness so we can measure the exact
    lift the re-ranker adds. Don't remove it — it's a real experimental control.
    """

    name = "identity"

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int) -> list[ScoredChunk]:
        # already sorted by retriever; just trim to k
        return sorted(candidates, key=lambda sc: sc.score, reverse=True)[:k]
