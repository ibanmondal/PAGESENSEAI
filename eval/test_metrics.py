"""Unit tests for the retrieval metrics.

These exist so an interviewer (or you) can verify the eval harness is correct
before trusting its numbers. If a metric is wrong, every resume claim built on
it is wrong. Run with: pytest eval/test_metrics.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import metrics as M


def test_grade_ranking_basic():
    assert M.grade_ranking(["a", "b", "c"], {"b"}) == [0, 1, 0]


def test_precision_at_k():
    # [1,0,1] in top2 -> 1 relevant of 2 -> 0.5
    assert M.precision_at_k([1, 0, 1], k=2) == 0.5
    assert M.precision_at_k([1, 1, 0], k=2) == 1.0
    assert M.precision_at_k([0, 0, 0], k=2) == 0.0


def test_recall_at_k():
    # 2 relevant total, 1 surfaced in top k=3 -> 0.5
    assert M.recall_at_k([1, 0, 0], total_relevant=2, k=3) == 0.5
    assert M.recall_at_k([1, 1, 0], total_relevant=2, k=3) == 1.0


def test_reciprocal_rank():
    assert M.reciprocal_rank([0, 0, 1, 0]) == 1 / 3
    assert M.reciprocal_rank([1, 0, 0]) == 1.0
    assert M.reciprocal_rank([0, 0, 0]) == 0.0


def test_average_precision():
    # relevant at positions 1 and 3 -> AP = (1/1 + 2/3)/2
    ap = M.average_precision([1, 0, 1])
    assert abs(ap - ((1.0 + 2/3) / 2)) < 1e-9


def test_ndcg_perfect_is_one():
    # all relevant first -> ndcg = 1.0
    assert abs(M.ndcg_at_k([1, 1, 0], k=3) - 1.0) < 1e-9


def test_ndcg_zero_when_no_relevant():
    assert M.ndcg_at_k([0, 0, 0], k=3) == 0.0


def test_ndcg_decreases_when_relevant_lower():
    perfect = M.ndcg_at_k([1, 1, 0, 0], k=4)
    worse = M.ndcg_at_k([0, 0, 1, 1], k=4)
    assert perfect > worse
    assert 0.0 < worse < perfect < 1.0 + 1e-9


def test_score_query_keys():
    out = M.score_query(["a", "b", "c"], {"b"}, k_values=(5, 10))
    for key in ("mrr", "map", "precision@5", "recall@5", "ndcg@5",
                "precision@10", "recall@10", "ndcg@10"):
        assert key in out


def test_aggregate_means_across_queries():
    a = M.score_query(["a", "b", "c"], {"a"})
    b = M.score_query(["x", "y", "z"], {"z"})
    agg = M.aggregate([a, b])
    assert set(agg.keys()) == set(a.keys())
    # mrr should be mean of 1.0 and 1/3
    assert abs(agg["mrr"] - ((1.0 + 1/3) / 2)) < 1e-9


def test_grade_ranking_unseen_relevant_kept():
    # a relevant doc not in the ranked list is simply never hit -> grade 0s
    assert M.grade_ranking(["a", "b"], {"c"}) == [0, 0]
