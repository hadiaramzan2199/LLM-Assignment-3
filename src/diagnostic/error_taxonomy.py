"""
src/diagnostic/error_taxonomy.py
Error taxonomy for multi-turn RAG failures.

Classifies each result into one or more failure categories:
  1. retrieval_rank_error       — correct passage exists but ranked >5
  2. retrieval_miss             — correct passage not retrieved in top-10
  3. coreference_failure        — query has pronoun, retrieval failed
  4. synthesis_failure          — multi-passage evidence required, partial answer
  5. hallucination              — generation adds unsupported content
  6. unanswerable_hallucination — model generates for unanswerable Qs
  7. history_neglect            — long conv, history ignored
  8. partial_answer             — incomplete but not hallucinated
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------
# Failure category definitions
# ---------------------------------------------------------------

FAILURE_CATEGORIES = [
    "retrieval_rank_error",
    "retrieval_miss",
    "coreference_failure",
    "synthesis_failure",
    "hallucination",
    "unanswerable_hallucination",
    "history_neglect",
    "partial_answer",
    "correct",           # No failure
]

PRONOUN_TOKENS = {"he", "she", "they", "it", "his", "her", "their", "its",
                  "him", "them", "this", "that", "these", "those"}

REFUSAL_PATTERNS = re.compile(
    r"i (don't|do not|cannot|can't) (know|find|answer|provide)|"
    r"the (provided|given|available) (passages?|documents?|context) (do(es)? not|don't|doesn't)|"
    r"(there is|there's) no (information|answer|evidence)|"
    r"(not|cannot be) answered|"
    r"(insufficient|not enough) (information|evidence|context)|"
    r"unanswerable",
    re.IGNORECASE,
)


def classify_retrieval_failure(
    retrieved: List[str],
    qrels: Dict[str, int],
    rank_threshold: int = 5,
) -> Set[str]:
    """
    Classify retrieval-side failures.
    Returns a set of applicable failure categories.
    """
    failures = set()
    relevant_passages = {pid for pid, rel in qrels.items() if rel >= 1}

    if not relevant_passages:
        return failures  # unanswerable — no relevant passages expected

    retrieved_set = set(retrieved[:10])
    top_k_set = set(retrieved[:rank_threshold])

    # Check retrieval miss (not in top-10 at all)
    if not relevant_passages & retrieved_set:
        failures.add("retrieval_miss")
    # Check rank error (in top-10 but not top-k)
    elif not relevant_passages & top_k_set:
        failures.add("retrieval_rank_error")

    return failures


def classify_generation_failure(
    generated: str,
    reference: str,
    passages: List[str],
    faithfulness_score: float,
    answerability: str,
    num_conv_turns: int,
    query: str,
    faithfulness_threshold: float = 0.7,
    history_length_threshold: int = 5,
) -> Set[str]:
    """
    Classify generation-side failures.
    Returns a set of applicable failure categories.
    """
    failures = set()

    # 1. Unanswerable hallucination
    if answerability == "unanswerable" and not bool(REFUSAL_PATTERNS.search(generated)):
        failures.add("unanswerable_hallucination")

    # 2. Faithfulness / hallucination
    if faithfulness_score < faithfulness_threshold and answerability == "answerable":
        failures.add("hallucination")

    # 3. History neglect — long conversation but very short output
    gen_len = len(generated.split())
    if num_conv_turns >= history_length_threshold and gen_len < 20:
        failures.add("history_neglect")

    # 4. Coreference failure proxy — query has pronoun AND generation is short/refusal
    has_pronoun = bool(set(query.lower().split()) & PRONOUN_TOKENS)
    if has_pronoun and gen_len < 15:
        failures.add("coreference_failure")

    # 5. Synthesis failure — reference has multiple sentences but generation is short
    ref_sentences = [s.strip() for s in re.split(r'[.!?]', reference) if s.strip()]
    if len(ref_sentences) >= 3 and gen_len < 30:
        failures.add("synthesis_failure")

    # 6. Partial answer — moderate faithfulness but short generation
    if 0.4 <= faithfulness_score < faithfulness_threshold and gen_len < 50:
        failures.add("partial_answer")

    if not failures and answerability == "answerable":
        failures.add("correct")

    return failures


def classify_result(result: dict) -> Set[str]:
    """
    Full classification of a single evaluation result.

    Expected result dict:
      {
        "conv_id": str,
        "retrieved": [pid, ...],
        "qrels": {pid: int},
        "generated": str,
        "reference": str,
        "passages": [str, ...],
        "faithfulness_score": float,
        "answerability": str,
        "num_turns": int,
        "query": str,
        "metadata": {...}
      }
    """
    failures = set()

    # Retrieval failures
    if "retrieved" in result and "qrels" in result:
        failures |= classify_retrieval_failure(result["retrieved"], result["qrels"])

    # Generation failures
    if "generated" in result:
        failures |= classify_generation_failure(
            generated=result.get("generated", ""),
            reference=result.get("reference", ""),
            passages=result.get("passages", []),
            faithfulness_score=result.get("faithfulness_score", 1.0),
            answerability=result.get("answerability", "answerable"),
            num_conv_turns=result.get("num_turns", 1),
            query=result.get("query", ""),
        )

    return failures


# ---------------------------------------------------------------
# Taxonomy aggregation
# ---------------------------------------------------------------

class ErrorTaxonomy:
    """
    Builds and analyzes a complete error taxonomy over a dataset.
    """

    def __init__(self):
        self.results: List[dict] = []
        self.classified: List[dict] = []  # {conv_id, failures, metadata}

    def classify_dataset(self, results: List[dict]) -> "ErrorTaxonomy":
        """Classify all results and store."""
        self.results = results
        self.classified = []
        for r in results:
            failures = classify_result(r)
            self.classified.append({
                "conv_id": r.get("conv_id"),
                "failures": list(failures),
                "metadata": r.get("metadata", {}),
            })
        return self

    def get_failure_distribution(self) -> Dict[str, int]:
        """Count how often each failure category occurs."""
        counts = defaultdict(int)
        for c in self.classified:
            for f in c["failures"]:
                counts[f] += 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    def get_failure_rate_by_category(self) -> Dict[str, float]:
        """Fraction of conversations with each failure type."""
        total = len(self.classified)
        if total == 0:
            return {}
        dist = self.get_failure_distribution()
        return {k: v / total for k, v in dist.items()}

    def get_co_occurrence_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        Compute co-occurrence counts between failure categories.
        Useful for finding compound failures.
        """
        matrix = defaultdict(lambda: defaultdict(int))
        for c in self.classified:
            fs = c["failures"]
            for i, f1 in enumerate(fs):
                for f2 in fs[i:]:
                    matrix[f1][f2] += 1
                    if f1 != f2:
                        matrix[f2][f1] += 1
        return {k: dict(v) for k, v in matrix.items()}

    def breakdown_by_metadata(self, field: str) -> Dict[str, Dict[str, float]]:
        """
        Failure rate breakdown by a metadata field.
        e.g., field="question_type" → {factoid: {hallucination: 0.2, ...}, ...}
        """
        groups = defaultdict(list)
        for c in self.classified:
            cat = c["metadata"].get(field, "unknown")
            groups[cat].append(c["failures"])

        output = {}
        for cat, failure_lists in groups.items():
            total = len(failure_lists)
            counts = defaultdict(int)
            for fl in failure_lists:
                for f in fl:
                    counts[f] += 1
            output[cat] = {k: v / total for k, v in counts.items()}
        return output

    def get_examples(
        self, failure_type: str, n: int = 5
    ) -> List[dict]:
        """Retrieve N example conversations with a given failure type."""
        examples = [
            c for c in self.classified if failure_type in c["failures"]
        ]
        return examples[:n]

    def summary_report(self) -> str:
        """Generate a human-readable summary of the error taxonomy."""
        dist = self.get_failure_distribution()
        rate = self.get_failure_rate_by_category()
        total = len(self.classified)

        lines = [
            "=" * 60,
            "ERROR TAXONOMY SUMMARY",
            f"Total conversations analyzed: {total}",
            "=" * 60,
            "",
            "Failure Category              Count   Rate",
            "-" * 50,
        ]
        for cat in FAILURE_CATEGORIES:
            if cat in dist:
                lines.append(
                    f"{cat:<30} {dist[cat]:<8} {rate[cat]:.1%}"
                )
        lines.append("")
        return "\n".join(lines)
