"""
tests/test_faithfulness.py
Unit tests for faithfulness scorer and error taxonomy.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.evaluation.faithfulness import (
    NLIFaithfulnessScorer, split_sentences,
    contains_pronoun, analyze_coreference_impact,
)
from src.diagnostic.error_taxonomy import (
    ErrorTaxonomy, classify_retrieval_failure,
    classify_generation_failure, FAILURE_CATEGORIES,
)


# ---------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------

class TestSentenceSplitter:
    def test_basic_split(self):
        text = "The cat sat on the mat. The dog ran away. Birds fly high."
        sents = split_sentences(text)
        assert len(sents) == 3

    def test_single_sentence(self):
        sents = split_sentences("This is one sentence.")
        assert len(sents) == 1

    def test_empty_string(self):
        sents = split_sentences("")
        assert len(sents) == 0

    def test_short_words_filtered(self):
        # Sentences with <= 2 words should be filtered out
        sents = split_sentences("OK. This is a longer sentence.")
        assert len(sents) == 1  # "OK" filtered, longer one kept


# ---------------------------------------------------------------
# Faithfulness scorer (keyword fallback mode)
# ---------------------------------------------------------------

class TestNLIFaithfulnessScorer:
    @pytest.fixture
    def scorer(self):
        # Use keyword fallback (no actual NLI model needed for unit tests)
        s = NLIFaithfulnessScorer.__new__(NLIFaithfulnessScorer)
        s.model = None
        s.mode = "keyword_fallback"
        return s

    def test_high_overlap(self, scorer):
        passage = "The Eiffel Tower is located in Paris France and was built in 1889."
        response = "The Eiffel Tower is in Paris France."
        result = scorer.score_response(response, [passage])
        assert result["faithfulness_score"] > 0.5

    def test_low_overlap_hallucination(self, scorer):
        passage = "The cat sat on the mat."
        response = "The quantum computer revolutionized the nuclear industry."
        result = scorer.score_response(response, [passage])
        assert result["faithfulness_score"] < 0.5

    def test_empty_response(self, scorer):
        result = scorer.score_response("", ["Some passage text."])
        assert result["faithfulness_score"] == pytest.approx(1.0)
        assert result["num_sentences"] == 0

    def test_returns_expected_keys(self, scorer):
        result = scorer.score_response(
            "This is a test sentence.",
            ["This is test passage content."]
        )
        assert "faithfulness_score" in result
        assert "num_sentences" in result
        assert "entailed_sentences" in result
        assert "sentence_scores" in result
        assert "hallucination_sentences" in result

    def test_aggregate(self, scorer):
        scores = [
            {"faithfulness_score": 1.0, "hallucination_sentences": [], "num_sentences": 2},
            {"faithfulness_score": 0.5, "hallucination_sentences": ["bad sent"], "num_sentences": 2},
            {"faithfulness_score": 0.0, "hallucination_sentences": ["x", "y"], "num_sentences": 2},
        ]
        agg = scorer.aggregate_faithfulness(scores)
        assert agg["mean_faithfulness"] == pytest.approx(0.5)
        assert "fully_faithful_rate" in agg
        assert "low_faithfulness_rate" in agg


# ---------------------------------------------------------------
# Coreference analysis
# ---------------------------------------------------------------

class TestCoreferenceAnalysis:
    def test_contains_pronoun(self):
        assert contains_pronoun("How did they affect the farmers?")
        assert contains_pronoun("What is its impact?")
        assert not contains_pronoun("What is the Farm Bill?")

    def test_impact_analysis(self):
        results = [
            {"query": "How did they affect this?", "ndcg@10": 0.4},
            {"query": "What is the Farm Bill?", "ndcg@10": 0.7},
            {"query": "What is its purpose?", "ndcg@10": 0.3},
            {"query": "Describe the provisions.", "ndcg@10": 0.8},
        ]
        analysis = analyze_coreference_impact(results, metric_key="ndcg@10")
        # Queries with pronouns should have lower scores
        assert analysis["with_pronoun"]["mean"] < analysis["without_pronoun"]["mean"]
        assert analysis["gap"] > 0.0
        assert analysis["with_pronoun"]["count"] == 2
        assert analysis["without_pronoun"]["count"] == 2


# ---------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------

class TestErrorTaxonomy:
    def test_retrieval_miss(self):
        failures = classify_retrieval_failure(
            retrieved=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"],
            qrels={"p1": 1},
            rank_threshold=5,
        )
        assert "retrieval_miss" in failures

    def test_retrieval_rank_error(self):
        # Relevant passage at rank 8 (beyond threshold=5)
        retrieved = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "p1", "x8", "x9"]
        failures = classify_retrieval_failure(retrieved, {"p1": 1}, rank_threshold=5)
        assert "retrieval_rank_error" in failures
        assert "retrieval_miss" not in failures

    def test_perfect_retrieval(self):
        retrieved = ["p1", "p2", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
        failures = classify_retrieval_failure(retrieved, {"p1": 1, "p2": 1}, rank_threshold=5)
        assert len(failures) == 0

    def test_unanswerable_hallucination(self):
        failures = classify_generation_failure(
            generated="The answer is definitely X and Y based on my knowledge.",
            reference="I cannot answer this.",
            passages=[],
            faithfulness_score=0.0,
            answerability="unanswerable",
            num_conv_turns=5,
            query="What is the secret?",
        )
        assert "unanswerable_hallucination" in failures

    def test_correct_refusal(self):
        failures = classify_generation_failure(
            generated="I cannot answer this based on the provided passages.",
            reference="I cannot answer this.",
            passages=[],
            faithfulness_score=1.0,
            answerability="unanswerable",
            num_conv_turns=3,
            query="What is unanswerable?",
        )
        assert "unanswerable_hallucination" not in failures

    def test_taxonomy_classify_dataset(self):
        results = [
            {
                "conv_id": "c1",
                "retrieved": ["p1", "x1"],
                "qrels": {"p1": 1},
                "generated": "The answer is X.",
                "reference": "The answer is X.",
                "passages": ["The answer is X as per passage."],
                "faithfulness_score": 0.9,
                "answerability": "answerable",
                "num_turns": 3,
                "query": "What is the answer?",
                "metadata": {"question_type": "factoid"},
            },
            {
                "conv_id": "c2",
                "retrieved": ["x1", "x2"],
                "qrels": {"p1": 1},
                "generated": "The answer is definitely secret code 999.",
                "reference": "",
                "passages": [],
                "faithfulness_score": 0.0,
                "answerability": "unanswerable",
                "num_turns": 6,
                "query": "What is the secret?",
                "metadata": {"question_type": "factoid"},
            },
        ]
        taxonomy = ErrorTaxonomy()
        taxonomy.classify_dataset(results)

        assert len(taxonomy.classified) == 2
        dist = taxonomy.get_failure_distribution()
        assert isinstance(dist, dict)

        # c2 should have failures
        c2 = next(c for c in taxonomy.classified if c["conv_id"] == "c2")
        assert len(c2["failures"]) > 0

    def test_failure_categories_defined(self):
        for cat in FAILURE_CATEGORIES:
            assert isinstance(cat, str)

    def test_summary_report(self):
        results = [{
            "conv_id": "c1",
            "retrieved": ["p1"],
            "qrels": {"p1": 1},
            "generated": "Answer.",
            "reference": "Answer.",
            "passages": ["Answer is in here."],
            "faithfulness_score": 1.0,
            "answerability": "answerable",
            "num_turns": 3,
            "query": "What is this?",
            "metadata": {},
        }]
        taxonomy = ErrorTaxonomy()
        taxonomy.classify_dataset(results)
        report = taxonomy.summary_report()
        assert "ERROR TAXONOMY" in report
        assert "Total conversations" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
