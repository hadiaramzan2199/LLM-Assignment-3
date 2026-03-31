"""
tests/test_generation_metrics.py
Unit tests for generation evaluation metrics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.evaluation.generation_metrics import (
    rouge_l_score, batch_rouge_l, harmonic_mean,
    is_refusal, unanswerable_detection_metrics,
    response_statistics,
)


# ---------------------------------------------------------------
# ROUGE-L tests
# ---------------------------------------------------------------

class TestROUGEL:
    def test_identical_strings(self):
        score = rouge_l_score("the cat sat on the mat", "the cat sat on the mat")
        assert score["f"] == pytest.approx(1.0)

    def test_no_overlap(self):
        score = rouge_l_score("quick brown fox", "lazy sleeping dog")
        assert score["f"] == pytest.approx(0.0)

    def test_partial_overlap(self):
        score = rouge_l_score("the cat sat", "the cat ran away fast")
        assert 0.0 < score["f"] < 1.0

    def test_empty_hypothesis(self):
        score = rouge_l_score("", "reference text here")
        assert score["f"] == pytest.approx(0.0)

    def test_empty_reference(self):
        score = rouge_l_score("hypothesis text", "")
        assert score["f"] == pytest.approx(0.0)

    def test_returns_prf(self):
        score = rouge_l_score("cat sat", "cat sat on mat")
        assert "f" in score
        assert "p" in score
        assert "r" in score
        assert 0.0 <= score["f"] <= 1.0
        assert 0.0 <= score["p"] <= 1.0
        assert 0.0 <= score["r"] <= 1.0

    def test_precision_recall_f1_relationship(self):
        score = rouge_l_score("cat sat on the big beautiful mat", "cat sat")
        # High precision (short hypothesis mostly in reference), but low recall
        assert score["p"] > score["r"]

    def test_batch(self):
        hyps = ["cat sat", "dog ran", "bird flew"]
        refs = ["cat sat on mat", "dog ran fast", "bird flew away"]
        scores = batch_rouge_l(hyps, refs)
        assert len(scores) == 3
        assert all("f" in s for s in scores)


# ---------------------------------------------------------------
# Harmonic mean tests
# ---------------------------------------------------------------

class TestHarmonicMean:
    def test_equal_values(self):
        hm = harmonic_mean(0.5, 0.5, 0.5)
        assert hm == pytest.approx(0.5)

    def test_zero_value(self):
        hm = harmonic_mean(0.8, 0.0, 0.6)
        assert hm == pytest.approx(0.0)

    def test_all_zeros(self):
        hm = harmonic_mean(0.0, 0.0, 0.0)
        assert hm == pytest.approx(0.0)

    def test_all_ones(self):
        hm = harmonic_mean(1.0, 1.0, 1.0)
        assert hm == pytest.approx(1.0)

    def test_harmonic_lt_arithmetic(self):
        vals = [0.9, 0.3, 0.6]
        hm = harmonic_mean(*vals)
        am = sum(vals) / len(vals)
        assert hm < am  # harmonic mean < arithmetic mean for non-equal values

    def test_two_values(self):
        hm = harmonic_mean(0.6, 0.9)
        assert hm == pytest.approx(2 / (1/0.6 + 1/0.9), abs=0.0001)


# ---------------------------------------------------------------
# Refusal detection tests
# ---------------------------------------------------------------

class TestRefusalDetection:
    def test_clear_refusal(self):
        assert is_refusal("I cannot answer this question based on the provided passages.")
        assert is_refusal("The provided passages do not contain enough information.")
        assert is_refusal("There is no information about this in the context.")
        assert is_refusal("This question is unanswerable.")
        assert is_refusal("I don't know the answer to this question.")

    def test_non_refusal(self):
        assert not is_refusal("The answer is Paris, France.")
        assert not is_refusal("Based on the passages, the key fact is that X equals Y.")
        assert not is_refusal("The 2018 Farm Bill included provisions for organic farmers.")

    def test_partial_refusal(self):
        # Sentences that contain some refusal signals
        assert is_refusal("I cannot find a direct answer, but the passages suggest X.")

    def test_case_insensitive(self):
        assert is_refusal("THE PROVIDED PASSAGES DO NOT CONTAIN THIS INFORMATION.")
        assert is_refusal("i cannot answer this.")


# ---------------------------------------------------------------
# Unanswerable detection metrics
# ---------------------------------------------------------------

class TestUnanswerableDetection:
    def _make_results(self, pairs):
        """pairs: (generated_text, answerability)"""
        return [{"generated": g, "answerability": a} for g, a in pairs]

    def test_perfect_detection(self):
        results = self._make_results([
            ("I cannot answer this.", "unanswerable"),
            ("I cannot answer this.", "unanswerable"),
            ("The answer is Paris.", "answerable"),
            ("Based on the passages, X is Y.", "answerable"),
        ])
        metrics = unanswerable_detection_metrics(results)
        assert metrics["precision"] == pytest.approx(1.0)
        assert metrics["recall"] == pytest.approx(1.0)
        assert metrics["f1"] == pytest.approx(1.0)

    def test_all_hallucinate(self):
        results = self._make_results([
            ("The answer is definitely 42.", "unanswerable"),
            ("Based on the passages, it is X.", "unanswerable"),
        ])
        metrics = unanswerable_detection_metrics(results)
        assert metrics["recall"] == pytest.approx(0.0)
        assert metrics["true_positives"] == 0
        assert metrics["false_negatives"] == 2

    def test_false_positive(self):
        results = self._make_results([
            ("I cannot answer this.", "answerable"),  # Wrong refusal
        ])
        metrics = unanswerable_detection_metrics(results)
        assert metrics["false_positives"] == 1
        assert metrics["precision"] == pytest.approx(0.0)

    def test_empty(self):
        metrics = unanswerable_detection_metrics([])
        assert metrics["f1"] == 0.0


# ---------------------------------------------------------------
# Response statistics
# ---------------------------------------------------------------

class TestResponseStatistics:
    def test_basic_stats(self):
        responses = ["one two three", "a b c d e", "hello"]
        stats = response_statistics(responses)
        assert "mean_length" in stats
        assert "median_length" in stats
        assert stats["min_length"] == 1
        assert stats["max_length"] == 5

    def test_single_response(self):
        stats = response_statistics(["word1 word2"])
        assert stats["mean_length"] == pytest.approx(2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
