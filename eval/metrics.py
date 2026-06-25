"""Retrieval quality metrics — the scientific core of this project.

Every function here is a pure function over ranked lists. This is deliberate:
pure functions are trivially unit-testable and let the eval harness compare
retrievers/re-rankers on identical footing. If you can't measure it, you can't
claim it improves — these functions are how we put real numbers in the resume.

Grading is at the *document* level, not chunk level. A chunk is "relevant" iff
its parent document is in the ground-truth relevant set. This is standard for
RAG eval (the user cares whether the right source was surfaced, not the exact
passage — re-ranking handles passage precision).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

# A relevance grade: 0 = irrelevant, 1 = relevant. Binary relevance for now.
Grade = int
GRADES = (0, 1)


def grade_ranking(
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: Iterable[str],
) -> list[Grade]:
    """Project a ranked list onto binary relevance grades (0/1)."""
    relevant = set(relevant_doc_ids)
    return [1 if doc_id in relevant else 0 for doc_id in ranked_doc_ids]


def precision_at_k(grades: Sequence[Grade], k: int) -> float:
    """P@k = (# relevant in top k) / k."""
    if k <= 0:
        return 0.0
    top = grades[:k]
    return sum(top) / k


def recall_at_k(grades: Sequence[Grade], total_relevant: int, k: int) -> float:
    """Recall@k = (# relevant in top k) / (total relevant).

    total_relevant is the size of the ground-truth relevant set, not len(grades).
    """
    if total_relevant <= 0:
        return 0.0
    return sum(grades[:k]) / total_relevant


def average_precision(grades: Sequence[Grade]) -> float:
    """Average Precision for one query.

    AP = mean over relevant ranks r of (precision@r).
    Rewards relevant docs appearing EARLY. Averaged over queries -> MAP.
    """
    if not grades:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, g in enumerate(grades, start=1):
        if g:
            hits += 1
            sum_prec += hits / i
    return sum_prec / hits if hits else 0.0


def reciprocal_rank(grades: Sequence[Grade]) -> float:
    """RR = 1 / rank of first relevant doc. 0 if none.
    Averaged over queries -> MRR. The headline metric for this project.
    """
    for i, g in enumerate(grades, start=1):
        if g:
            return 1.0 / i
    return 0.0


def dcg_at_k(grades: Sequence[Grade], k: int) -> float:
    """Discounted Cumulative Gain.
    DCG = sum over i of (grade_i / log2(i+1)). Position-discounted.
    """
    total = 0.0
    for i, g in enumerate(grades[:k], start=1):
        total += g / math.log2(i + 1)
    return total


def ndcg_at_k(grades: Sequence[Grade], k: int) -> float:
    """Normalized DCG = DCG / IDCG (ideal = relevant docs ranked first).
    Range [0, 1]. The primary retrieval-quality metric for this project.
    """
    if k <= 0:
        return 0.0
    dcg = dcg_at_k(grades, k)
    # Ideal ranking = sort grades descending (all 1s first).
    ideal = sorted(grades, reverse=True)[:k]
    idcg = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Aggregation over a query set
# ---------------------------------------------------------------------------

def aggregate(metrics: Iterable[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across queries. Keys must match across dicts."""
    rows = list(metrics)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: sum(r[k] for r in rows) / len(rows) for k in keys}


def score_query(
    ranked_doc_ids: Sequence[str],
    relevant_doc_ids: Iterable[str],
    k_values: Sequence[int] = (5, 10),
) -> dict[str, float]:
    """Compute all metrics for one query at the given k cutoffs.
    Returns flat dict keys like 'ndcg@10', 'mrr', 'recall@5', 'precision@5'.
    """
    grades = grade_ranking(ranked_doc_ids, relevant_doc_ids)
    total_rel = len(set(relevant_doc_ids))
    out: dict[str, float] = {"mrr": reciprocal_rank(grades), "map": average_precision(grades)}
    for k in k_values:
        out[f"precision@{k}"] = precision_at_k(grades, k)
        out[f"recall@{k}"] = recall_at_k(grades, total_rel, k)
        out[f"ndcg@{k}"] = ndcg_at_k(grades, k)
    return out
