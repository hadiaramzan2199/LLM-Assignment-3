"""
src/evaluation/retrieval_metrics.py
Comprehensive retrieval metrics for Task A.
Supports: nDCG@k, Precision@k, Recall@k, MRR, Hit Rate@k
Breakdowns by: question type, multiturn type, domain, answerability
"""

import math
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------

def dcg(relevances: List[int], k: int) -> float:
    """Compute DCG@k given a ranked list of relevance scores."""
    relevances = relevances[:k]
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def ndcg_at_k(retrieved: List[str], qrels: Dict[str, int], k: int) -> float:
    """
    Compute nDCG@k.
    retrieved: ranked list of passage IDs
    qrels: {passage_id: relevance_score}
    """
    if not qrels:
        return 0.0
    rel_scores = [qrels.get(pid, 0) for pid in retrieved[:k]]
    ideal_scores = sorted(qrels.values(), reverse=True)
    actual_dcg = dcg(rel_scores, k)
    ideal_dcg = dcg(ideal_scores, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def precision_at_k(retrieved: List[str], qrels: Dict[str, int], k: int) -> float:
    """Precision@k — fraction of top-k that are relevant (relevance >= 1)."""
    if not retrieved:
        return 0.0
    hits = sum(1 for pid in retrieved[:k] if qrels.get(pid, 0) >= 1)
    return hits / k


def recall_at_k(retrieved: List[str], qrels: Dict[str, int], k: int) -> float:
    """Recall@k — fraction of all relevant passages retrieved in top-k."""
    relevant_total = sum(1 for v in qrels.values() if v >= 1)
    if relevant_total == 0:
        return 0.0
    hits = sum(1 for pid in retrieved[:k] if qrels.get(pid, 0) >= 1)
    return hits / relevant_total


def mean_reciprocal_rank(retrieved: List[str], qrels: Dict[str, int]) -> float:
    """MRR — reciprocal rank of the first relevant passage."""
    for rank, pid in enumerate(retrieved, start=1):
        if qrels.get(pid, 0) >= 1:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(retrieved: List[str], qrels: Dict[str, int], k: int) -> float:
    """Hit Rate@k — 1 if any relevant passage is in top-k, else 0."""
    return float(any(qrels.get(pid, 0) >= 1 for pid in retrieved[:k]))


# ---------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------

def compute_retrieval_metrics(
    results: List[dict],
    k_values: List[int] = [1, 3, 5, 10],
    breakdown_field: Optional[str] = None,
) -> dict:
    """
    Compute all retrieval metrics over a list of results.

    Each result dict:
      {
        "conv_id": str,
        "retrieved": [pid1, pid2, ...],   # ranked list
        "qrels": {pid: relevance},
        "metadata": {
            "question_type": ...,
            "multiturn_type": ...,
            "domain": ...,
            "answerability": ...,
            "history_mode": ...
        }
      }

    Returns:
      {
        "overall": {ndcg@1: ..., ndcg@3: ..., p@5: ..., ...},
        "by_<breakdown_field>": {category: {metric: value}, ...}
      }
    """
    # Group by category if requested
    groups = defaultdict(list)
    all_results = []

    for r in results:
        retrieved = r["retrieved"]
        qrels = r["qrels"]
        meta = r.get("metadata", {})

        entry = {}
        for k in k_values:
            entry[f"ndcg@{k}"] = ndcg_at_k(retrieved, qrels, k)
            entry[f"p@{k}"] = precision_at_k(retrieved, qrels, k)
            entry[f"r@{k}"] = recall_at_k(retrieved, qrels, k)
            entry[f"hit@{k}"] = hit_rate_at_k(retrieved, qrels, k)
        entry["mrr"] = mean_reciprocal_rank(retrieved, qrels)
        entry["conv_id"] = r["conv_id"]
        entry["metadata"] = meta

        all_results.append(entry)

        if breakdown_field and breakdown_field in meta:
            groups[meta[breakdown_field]].append(entry)

    def _aggregate(entries):
        if not entries:
            return {}
        keys = [k for k in entries[0].keys()
                if k not in ("conv_id", "metadata")]
        return {k: float(np.mean([e[k] for e in entries])) for k in keys}

    output = {"overall": _aggregate(all_results), "per_query": all_results}

    if breakdown_field:
        output[f"by_{breakdown_field}"] = {
            cat: _aggregate(entries) for cat, entries in groups.items()
        }

    return output


# ---------------------------------------------------------------
# Multi-breakdown analysis
# ---------------------------------------------------------------

def full_breakdown_analysis(results: List[dict], k_values: List[int] = [1, 3, 5, 10]) -> dict:
    """
    Run breakdowns across all metadata dimensions simultaneously.
    Returns a dict of breakdown_field -> category -> metrics.
    """
    breakdown_fields = ["question_type", "multiturn_type", "domain",
                        "answerability", "history_mode"]
    output = {}
    for field in breakdown_fields:
        analysis = compute_retrieval_metrics(results, k_values, breakdown_field=field)
        if f"by_{field}" in analysis:
            output[field] = analysis[f"by_{field}"]
    output["overall"] = compute_retrieval_metrics(results, k_values)["overall"]
    return output


# ---------------------------------------------------------------
# Rank analysis helpers
# ---------------------------------------------------------------

def get_rank_of_first_relevant(retrieved: List[str], qrels: Dict[str, int]) -> Optional[int]:
    """Return the 1-indexed rank of the first relevant passage, or None."""
    for rank, pid in enumerate(retrieved, start=1):
        if qrels.get(pid, 0) >= 1:
            return rank
    return None


def rank_distribution_analysis(results: List[dict]) -> dict:
    """
    Analyze where relevant passages tend to appear in ranked lists.
    Returns distribution of first-relevant-rank across queries.
    """
    ranks = []
    not_found = 0
    for r in results:
        rank = get_rank_of_first_relevant(r["retrieved"], r["qrels"])
        if rank is not None:
            ranks.append(rank)
        else:
            not_found += 1

    if not ranks:
        return {"not_found_rate": 1.0}

    return {
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "rank_1_rate": float(np.mean([r == 1 for r in ranks])),
        "rank_3_rate": float(np.mean([r <= 3 for r in ranks])),
        "rank_5_rate": float(np.mean([r <= 5 for r in ranks])),
        "rank_10_rate": float(np.mean([r <= 10 for r in ranks])),
        "not_found_rate": not_found / (len(ranks) + not_found),
        "rank_distribution": {str(i): ranks.count(i) for i in range(1, 11)},
    }


def format_retrieval_table(metrics: dict, k_values: List[int] = [1, 3, 5, 10]) -> str:
    """Format retrieval metrics as a readable table."""
    try:
        from tabulate import tabulate
    except ImportError:
        return str(metrics)

    if "overall" in metrics:
        data = [["Overall"]]
        headers = ["System"]
        for k in k_values:
            headers += [f"nDCG@{k}", f"P@{k}", f"R@{k}"]
        headers.append("MRR")

        row = ["Overall"]
        m = metrics["overall"]
        for k in k_values:
            row += [f"{m.get(f'ndcg@{k}', 0):.4f}",
                    f"{m.get(f'p@{k}', 0):.4f}",
                    f"{m.get(f'r@{k}', 0):.4f}"]
        row.append(f"{m.get('mrr', 0):.4f}")
        return tabulate([row], headers=headers, tablefmt="grid")
    return str(metrics)
