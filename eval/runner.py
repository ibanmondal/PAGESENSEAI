"""Eval harness runner — runs N configurations over the labeled dataset and
emits a comparison table. This is THE script that produces resume numbers.

Usage:
    python -m eval.runner --data eval/data/evalset.jsonl --config all

It compares these configurations (a real experimental design):
  1. bm25           — lexical only
  2. dense          — embeddings only
  3. hybrid         — BM25 + dense RRF fusion
  4. hybrid+rerank  — hybrid top-N, then cross-encoder re-rank
  (5. hybrid+finetuned — same, with YOUR fine-tuned re-ranker — the headline win)

Each config sees the IDENTICAL queries + corpus, so differences are attributable
only to the retrieval/re-ranking strategy. That's the scientific discipline that
makes the numbers defensible in an interview.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Support both `python -m eval.runner` and direct execution.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.core.logging import get_logger   # noqa: E402
from backend.app.models.schemas import Chunk, EvalRow  # noqa: E402
from backend.app.rag.chunking import chunk_document  # noqa: E402
from backend.app.rag.reranker import CrossEncoderReRanker, IdentityReRanker  # noqa: E402
from backend.app.rag.retrieval import (  # noqa: E402
    BM25Retriever, DenseRetriever, HybridRetriever,
)

from eval import metrics as M  # noqa: E402

log = get_logger("eval")


# ---------------------------------------------------------------------------
# Dataset I/O — JSONL with one EvalRow per line.
# ---------------------------------------------------------------------------

def load_evalset(path: Path) -> list[EvalRow]:
    rows: list[EvalRow] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(EvalRow(
                query=obj["query"],
                pool_doc_ids=obj["pool_doc_ids"],
                relevant_doc_ids=obj["relevant_doc_ids"],
                notes=obj.get("notes", ""),
            ))
    log.info("Loaded %d eval queries from %s", len(rows), path)
    return rows


# ---------------------------------------------------------------------------
# A "config" is a factory that builds the retriever (+ optional re-ranker)
# for each query. Rebuilt per query so there's zero cross-query leakage.
# ---------------------------------------------------------------------------

@dataclass
class Config:
    name: str
    build: Callable[[], "Pipeline"]


class Pipeline:
    """One retrieval setup. `run` returns the ranked doc-id list for grading."""
    def __init__(self, retriever, reranker=None, top_n: int = 20, top_k: int = 10):
        self.retriever = retriever
        self.reranker = reranker
        self.top_n = top_n
        self.top_k = top_k

    def index(self, chunks: list[Chunk]) -> None:
        self.retriever.index(chunks)

    def run(self, query: str) -> list[str]:
        hits = self.retriever.search(query, k=self.top_n)
        if self.reranker is not None:
            hits = self.reranker.rerank(query, hits, k=self.top_k)
        else:
            hits = hits[: self.top_k]
        # Deduplicate by doc id, preserving rank order.
        seen: set[str] = set()
        ordered: list[str] = []
        for h in hits:
            doc_id = h.chunk.tab_id  # tab_id == doc id in the eval pool
            if doc_id not in seen:
                seen.add(doc_id)
                ordered.append(doc_id)
        return ordered


def _configs(settings, include_dense: bool) -> list[Config]:
    """Build the experimental configs. Dense-based ones are optional so the
    harness can run baseline-only on machines without torch.

    IMPORTANT: heavy models (embedding + cross-encoder) are instantiated ONCE
    here and shared across every query's Pipeline. Only the per-query index
    rebuilds. This is what keeps the harness fast — re-loading the model per
    query would be ~100x slower and would time out on any non-trivial set.
    """
    # Shared heavy models — load each exactly once.
    shared_dense = None
    shared_reranker = None
    if include_dense:
        shared_dense = DenseRetriever(settings.embedding_model)
        shared_dense._ensure_model()  # eager load, done once
        shared_reranker = CrossEncoderReRanker(settings.reranker_source)
        shared_reranker._ensure_model()  # eager load, done once

    cfgs: list[Config] = []
    cfgs.append(Config("bm25", lambda: Pipeline(
        BM25Retriever(), top_n=20, top_k=10)))
    if include_dense:
        cfgs.append(Config("dense", lambda: Pipeline(
            _fresh_dense(shared_dense), top_n=20, top_k=10)))
        cfgs.append(Config("hybrid", lambda: Pipeline(
            HybridRetriever(
                [BM25Retriever(), _fresh_dense(shared_dense)],
                weights=[settings.bm25_weight, settings.dense_weight],
            ),
            top_n=20, top_k=10,
        )))
        cfgs.append(Config("hybrid+rerank", lambda: Pipeline(
            HybridRetriever(
                [BM25Retriever(), _fresh_dense(shared_dense)],
                weights=[settings.bm25_weight, settings.dense_weight],
            ),
            reranker=shared_reranker,
            top_n=20, top_k=10,
        )))
    return cfgs


def _fresh_dense(shared: "DenseRetriever") -> "DenseRetriever":
    """Return a DenseRetriever that REUSES the shared model but starts with an
    empty index, so per-query re-indexing doesn't leak between queries."""
    d = DenseRetriever(shared.model_name)
    d._model = shared._model  # reuse the already-loaded SentenceTransformer
    return d


# ---------------------------------------------------------------------------
# Per-query corpus construction: the doc POOL for that query is its own index.
# This mirrors how a per-session RAG system actually works (small per-query corpus).
# ---------------------------------------------------------------------------

def build_corpus_for_query(row: EvalRow, docs: dict[str, str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_id in row.pool_doc_ids:
        text = docs.get(doc_id, "")
        if not text:
            continue
        chunks.extend(chunk_document(
            tab_id=doc_id, url=f"eval://{doc_id}", title=doc_id, text=text,
        ))
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="PageSense retrieval eval harness")
    parser.add_argument("--data", type=Path, default=Path("eval/data/evalset.jsonl"))
    parser.add_argument("--docs", type=Path, default=Path("eval/data/docs.jsonl"),
                        help="doc_id -> text corpus (JSONL)")
    parser.add_argument("--config", default="all", help="all | bm25 | dense | hybrid")
    parser.add_argument("--no-dense", action="store_true",
                        help="skip dense/hybrid configs (no torch needed)")
    parser.add_argument("--out", type=Path, default=Path("eval/results.json"))
    args = parser.parse_args()

    settings = get_settings()
    rows = load_evalset(args.data)
    # docs corpus: id -> full text
    docs: dict[str, str] = {}
    if args.docs.exists():
        with args.docs.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                docs[obj["doc_id"]] = obj["text"]

    available = _configs(settings, include_dense=not args.no_dense)
    if args.config != "all":
        available = [c for c in available if c.name == args.config]
        if not available:
            log.error("Unknown config '%s'", args.config)
            sys.exit(1)

    results: dict[str, dict] = {}
    for cfg in available:
        per_query = []
        for row in rows:
            corpus = build_corpus_for_query(row, docs)
            if not corpus:
                log.warning("Empty corpus for query: %s", row.query[:60])
                continue
            pipe = cfg.build()
            pipe.index(corpus)
            ranked_doc_ids = pipe.run(row.query)
            per_query.append(
                M.score_query(ranked_doc_ids, row.relevant_doc_ids, k_values=(5, 10))
            )
        agg = M.aggregate(per_query)
        results[cfg.name] = {
            "n_queries": len(per_query),
            "metrics": agg,
        }
        log.info("[%s] %s", cfg.name, {k: round(v, 4) for k, v in agg.items()})

    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Wrote results -> %s", args.out)

    # Print a compact comparison table to stdout.
    print("\n=== Retrieval Quality Comparison ===")
    print(f"{'config':<22}{'mrr':>8}{'ndcg@10':>10}{'recall@10':>12}{'map':>8}")
    for name, res in results.items():
        m = res["metrics"]
        print(f"{name:<22}{m.get('mrr',0):>8.4f}{m.get('ndcg@10',0):>10.4f}"
              f"{m.get('recall@10',0):>12.4f}{m.get('map',0):>8.4f}")


if __name__ == "__main__":
    main()
