"""
tests/test_retrieval_metrics.py
Unit tests for retrieval evaluation metrics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.evaluation.retrieval_metrics import (
    ndcg_at_k, precision_at_k, recall_at_k,
    mean_reciprocal_rank, hit_rate_at_k,
    compute_retrieval_metrics, rank_distribution_analysis,
)


# ---------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------

RETRIEVED_PERFECT = ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]
RETRIEVED_MISS    = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]
RETRIEVED_PARTIAL = ["x1", "p1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9"]

QRELS_BINARY = {"p1": 1, "p2": 1}
QRELS_GRADED = {"p1": 2, "p2": 1}
QRELS_EMPTY  = {}


# ---------------------------------------------------------------
# nDCG tests
# ---------------------------------------------------------------

class TestNDCG:
    def test_perfect_retrieval(self):
        score = ndcg_at_k(RETRIEVED_PERFECT, QRELS_BINARY, k=10)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_zero_retrieval_miss(self):
        score = ndcg_at_k(RETRIEVED_MISS, QRELS_BINARY, k=10)
        assert score == pytest.approx(0.0, abs=0.001)

    def test_partial_retrieval(self):
        score = ndcg_at_k(RETRIEVED_PARTIAL, QRELS_BINARY, k=10)
        assert 0.0 < score < 1.0

    def test_empty_qrels(self):
        score = ndcg_at_k(RETRIEVED_PERFECT, QRELS_EMPTY, k=10)
        assert score == 0.0

    def test_graded_relevance(self):
        # Higher relevance at rank 1 should give higher nDCG
        ret_good = ["p1", "p2", "x1"]
        ret_bad  = ["p2", "p1", "x1"]
        s_good = ndcg_at_k(ret_good, QRELS_GRADED, k=3)
        s_bad  = ndcg_at_k(ret_bad, QRELS_GRADED, k=3)
        assert s_good >= s_bad

    def test_k_cutoff(self):
        # Relevant passage at rank 11 — shouldn't contribute to nDCG@10
        retrieved = ["x"] * 10 + ["p1"]
        score_10 = ndcg_at_k(retrieved, QRELS_BINARY, k=10)
        assert score_10 == pytest.approx(0.0, abs=0.001)


# ---------------------------------------------------------------
# Precision and Recall tests
# ---------------------------------------------------------------

class TestPrecisionRecall:
    def test_perfect_precision(self):
        # All top-2 are relevant
        ret = ["p1", "p2", "x1", "x2"]
        p = precision_at_k(ret, QRELS_BINARY, k=2)
        assert p == pytest.approx(1.0)

    def test_zero_precision(self):
        p = precision_at_k(RETRIEVED_MISS, QRELS_BINARY, k=5)
        assert p == pytest.approx(0.0)

    def test_perfect_recall(self):
        ret = ["p1", "p2", "x1", "x2", "x3"]
        r = recall_at_k(ret, QRELS_BINARY, k=5)
        assert r == pytest.approx(1.0)

    def test_partial_recall(self):
        ret = ["p1", "x1", "x2"]
        r = recall_at_k(ret, QRELS_BINARY, k=3)
        assert r == pytest.approx(0.5)  # found 1 of 2 relevant


# ---------------------------------------------------------------
# MRR tests
# ---------------------------------------------------------------

class TestMRR:
    def test_first_rank(self):
        mrr = mean_reciprocal_rank(["p1", "x1", "x2"], QRELS_BINARY)
        assert mrr == pytest.approx(1.0)

    def test_second_rank(self):
        mrr = mean_reciprocal_rank(["x1", "p1", "x2"], QRELS_BINARY)
        assert mrr == pytest.approx(0.5)

    def test_no_relevant(self):
        mrr = mean_reciprocal_rank(RETRIEVED_MISS, QRELS_BINARY)
        assert mrr == pytest.approx(0.0)

    def test_empty_retrieved(self):
        mrr = mean_reciprocal_rank([], QRELS_BINARY)
        assert mrr == pytest.approx(0.0)


# ---------------------------------------------------------------
# Hit Rate tests
# ---------------------------------------------------------------

class TestHitRate:
    def test_hit(self):
        assert hit_rate_at_k(["p1", "x1"], QRELS_BINARY, k=2) == 1.0

    def test_miss(self):
        assert hit_rate_at_k(RETRIEVED_MISS, QRELS_BINARY, k=10) == 0.0

    def test_hit_boundary(self):
        ret = ["x1", "x2", "p1"]
        assert hit_rate_at_k(ret, QRELS_BINARY, k=2) == 0.0
        assert hit_rate_at_k(ret, QRELS_BINARY, k=3) == 1.0


# ---------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------

class TestBatchEvaluation:
    def test_compute_retrieval_metrics_shape(self):
        results = [
            {
                "conv_id": "c1",
                "retrieved": RETRIEVED_PERFECT,
                "qrels": QRELS_BINARY,
                "metadata": {"question_type": "factoid", "domain": "clapnq"},
            },
            {
                "conv_id": "c2",
                "retrieved": RETRIEVED_MISS,
                "qrels": QRELS_BINARY,
                "metadata": {"question_type": "explanatory", "domain": "govt"},
            },
        ]
        metrics = compute_retrieval_metrics(results, k_values=[1, 5, 10])
        assert "overall" in metrics
        assert "ndcg@10" in metrics["overall"]
        assert "mrr" in metrics["overall"]

    def test_breakdown_by_field(self):
        results = [
            {
                "conv_id": "c1",
                "retrieved": RETRIEVED_PERFECT,
                "qrels": QRELS_BINARY,
                "metadata": {"question_type": "factoid"},
            },
            {
                "conv_id": "c2",
                "retrieved": RETRIEVED_MISS,
                "qrels": QRELS_BINARY,
                "metadata": {"question_type": "explanatory"},
            },
        ]
        metrics = compute_retrieval_metrics(results, k_values=[1, 5, 10], breakdown_field="question_type")
        assert "by_question_type" in metrics
        assert "factoid" in metrics["by_question_type"]
        assert "explanatory" in metrics["by_question_type"]
        # factoid had perfect retrieval, explanatory had miss
        assert (metrics["by_question_type"]["factoid"]["ndcg@10"] >
                metrics["by_question_type"]["explanatory"]["ndcg@10"])

    def test_rank_distribution(self):
        results = [
            {"retrieved": ["p1", "x1", "x2"], "qrels": {"p1": 1}},
            {"retrieved": ["x1", "p1", "x2"], "qrels": {"p1": 1}},
            {"retrieved": ["x1", "x2", "x3"], "qrels": {"p1": 1}},
        ]
        dist = rank_distribution_analysis(results)
        assert "not_found_rate" in dist
        assert dist["rank_1_rate"] == pytest.approx(1 / 3, abs=0.01)
        assert dist["not_found_rate"] == pytest.approx(1 / 3, abs=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
