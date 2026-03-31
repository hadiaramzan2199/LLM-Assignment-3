"""
src/evaluation/faithfulness.py
NLI-based faithfulness scorer for Task B.

Uses a cross-encoder NLI model to measure entailment between
the generated response and the reference passages.

Score = fraction of response sentences entailed by at least one passage.
"""

import re
from typing import List, Optional, Tuple

import numpy as np


def split_sentences(text: str) -> List[str]:
    """Simple sentence splitter."""
    # Split on . ! ? followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip() and len(s.split()) > 2]


class NLIFaithfulnessScorer:
    """
    Scores faithfulness of generated responses using NLI.

    For each sentence in the generated response, we check whether
    any reference passage entails it. The faithfulness score is
    the fraction of entailed sentences.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, device=self.device)
            self.mode = "cross_encoder"
            print(f"[Faithfulness] Loaded NLI model: {self.model_name}")
        except Exception as e:
            print(f"[Faithfulness] WARNING: Could not load NLI model ({e}). Using keyword fallback.")
            self.model = None
            self.mode = "keyword_fallback"

    def _nli_entailment_score(self, premise: str, hypothesis: str) -> float:
        """Return entailment probability for (premise, hypothesis) pair."""
        if self.model is None:
            return self._keyword_fallback(premise, hypothesis)
        try:
            # Cross-encoder NLI: labels = [contradiction, entailment, neutral]
            scores = self.model.predict([(premise, hypothesis)], apply_softmax=True)
            # entailment is index 1 for deberta-v3-base
            return float(scores[0][1])
        except Exception:
            return self._keyword_fallback(premise, hypothesis)

    def _keyword_fallback(self, premise: str, hypothesis: str) -> float:
        """Fallback: token overlap as a proxy for entailment."""
        premise_tokens = set(premise.lower().split())
        hyp_tokens = set(hypothesis.lower().split())
        if not hyp_tokens:
            return 0.0
        overlap = len(premise_tokens & hyp_tokens) / len(hyp_tokens)
        # Rough threshold: >0.4 overlap = entailed
        return min(overlap * 2, 1.0)

    def score_response(
        self,
        response: str,
        passages: List[str],
        entailment_threshold: float = 0.5,
    ) -> dict:
        """
        Score a single response against reference passages.

        Returns:
          {
            "faithfulness_score": float,   # fraction of sentences entailed
            "num_sentences": int,
            "entailed_sentences": int,
            "sentence_scores": [float, ...],
            "hallucination_sentences": [str, ...]
          }
        """
        sentences = split_sentences(response)
        if not sentences:
            return {
                "faithfulness_score": 1.0,
                "num_sentences": 0,
                "entailed_sentences": 0,
                "sentence_scores": [],
                "hallucination_sentences": [],
            }

        sentence_scores = []
        hallucinations = []

        for sent in sentences:
            # Max entailment across all passages
            max_score = 0.0
            for passage in passages:
                score = self._nli_entailment_score(passage, sent)
                max_score = max(max_score, score)
            sentence_scores.append(max_score)
            if max_score < entailment_threshold:
                hallucinations.append(sent)

        entailed = sum(1 for s in sentence_scores if s >= entailment_threshold)
        faithfulness = entailed / len(sentences)

        return {
            "faithfulness_score": faithfulness,
            "num_sentences": len(sentences),
            "entailed_sentences": entailed,
            "sentence_scores": sentence_scores,
            "hallucination_sentences": hallucinations,
        }

    def batch_score(
        self,
        responses: List[str],
        passages_list: List[List[str]],
        entailment_threshold: float = 0.5,
    ) -> List[dict]:
        """Score a batch of responses."""
        assert len(responses) == len(passages_list)
        return [
            self.score_response(r, p, entailment_threshold)
            for r, p in zip(responses, passages_list)
        ]

    def aggregate_faithfulness(self, scores: List[dict]) -> dict:
        """Aggregate faithfulness scores across a dataset."""
        fs = [s["faithfulness_score"] for s in scores]
        hallucination_rates = [
            len(s["hallucination_sentences"]) / max(s["num_sentences"], 1)
            for s in scores
        ]
        return {
            "mean_faithfulness": float(np.mean(fs)),
            "median_faithfulness": float(np.median(fs)),
            "std_faithfulness": float(np.std(fs)),
            "mean_hallucination_rate": float(np.mean(hallucination_rates)),
            "fully_faithful_rate": float(np.mean([s == 1.0 for s in fs])),
            "low_faithfulness_rate": float(np.mean([s < 0.5 for s in fs])),
        }


# ---------------------------------------------------------------
# Coreference resolution quality analysis
# ---------------------------------------------------------------

PRONOUN_LIST = {"he", "she", "they", "it", "his", "her", "their", "its",
                "him", "them", "this", "that", "these", "those"}


def contains_pronoun(text: str) -> bool:
    """Check if a query contains pronouns suggesting coreference dependency."""
    tokens = set(text.lower().split())
    return bool(tokens & PRONOUN_LIST)


def analyze_coreference_impact(
    results: List[dict],
    metric_key: str = "ndcg@10",
) -> dict:
    """
    Compare metric scores for queries with vs. without pronouns.
    Indicates whether coreference resolution is impacting performance.
    """
    with_pronoun = []
    without_pronoun = []

    for r in results:
        query = r.get("query", "")
        score = r.get(metric_key, 0.0)
        if contains_pronoun(query):
            with_pronoun.append(score)
        else:
            without_pronoun.append(score)

    return {
        "with_pronoun": {
            "count": len(with_pronoun),
            "mean": float(np.mean(with_pronoun)) if with_pronoun else 0.0,
        },
        "without_pronoun": {
            "count": len(without_pronoun),
            "mean": float(np.mean(without_pronoun)) if without_pronoun else 0.0,
        },
        "gap": (
            float(np.mean(without_pronoun)) - float(np.mean(with_pronoun))
            if (with_pronoun and without_pronoun) else 0.0
        ),
    }
