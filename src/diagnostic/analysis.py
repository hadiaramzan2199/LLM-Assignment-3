"""
src/diagnostic/analysis.py
Diagnostic analysis module.

Provides:
  - Breakdown analysis by metadata dimensions
  - Retrieval-generation coupling analysis
  - History impact quantification
  - Domain-level performance analysis
  - Visualization helpers
"""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------
# Retrieval-Generation Coupling Analysis
# ---------------------------------------------------------------

def retrieval_generation_coupling(
    results: List[dict],
    retrieval_metric: str = "ndcg@10",
    generation_metric: str = "harmonic_mean",
) -> dict:
    """
    Analyze correlation between retrieval quality and generation faithfulness.

    Bins results by retrieval score and computes mean generation score per bin.
    Tests H2: non-linear relationship and diminishing returns above threshold.

    Each result must have:
      retrieval_metric (float) and generation_metric (float)
    """
    pairs = [
        (r.get(retrieval_metric, 0.0), r.get(generation_metric, 0.0))
        for r in results
        if retrieval_metric in r and generation_metric in r
    ]

    if not pairs:
        return {"error": "No paired retrieval/generation data found"}

    ret_scores, gen_scores = zip(*pairs)

    # Pearson correlation
    try:
        from scipy import stats
        corr, pval = stats.pearsonr(ret_scores, gen_scores)
    except ImportError:
        n = len(ret_scores)
        mean_r = np.mean(ret_scores)
        mean_g = np.mean(gen_scores)
        num = sum((r - mean_r) * (g - mean_g) for r, g in zip(ret_scores, gen_scores))
        den = (sum((r - mean_r) ** 2 for r in ret_scores) *
               sum((g - mean_g) ** 2 for g in gen_scores)) ** 0.5
        corr = num / den if den > 0 else 0.0
        pval = None

    # Bin analysis (quartiles)
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    n = len(sorted_pairs)
    quartiles = [sorted_pairs[i * n // 4: (i + 1) * n // 4] for i in range(4)]

    bin_analysis = {}
    for i, q in enumerate(quartiles):
        if q:
            ret_vals, gen_vals = zip(*q)
            bin_analysis[f"Q{i+1}"] = {
                "retrieval_range": (min(ret_vals), max(ret_vals)),
                "mean_retrieval": float(np.mean(ret_vals)),
                "mean_generation": float(np.mean(gen_vals)),
                "count": len(q),
            }

    # Diminishing returns analysis
    # Compare Q3->Q4 gain vs Q1->Q2 gain
    if len(bin_analysis) >= 4:
        low_gain = bin_analysis["Q2"]["mean_generation"] - bin_analysis["Q1"]["mean_generation"]
        high_gain = bin_analysis["Q4"]["mean_generation"] - bin_analysis["Q3"]["mean_generation"]
        diminishing_returns = high_gain < low_gain * 0.5  # less than half the gain

    return {
        "pearson_correlation": float(corr),
        "p_value": float(pval) if pval is not None else "N/A",
        "bin_analysis": bin_analysis,
        "diminishing_returns_observed": diminishing_returns if len(bin_analysis) >= 4 else False,
        "n_pairs": len(pairs),
    }


# ---------------------------------------------------------------
# History window impact
# ---------------------------------------------------------------

def history_impact_analysis(
    ablation_results: Dict[str, dict],
    metric_key: str = "ndcg@10",
) -> dict:
    """
    Analyze the impact of conversation history on retrieval performance.
    Uses results from history_window ablation.

    Returns:
      - Scores per history mode
      - Best history mode
      - Follow-up vs topic-shift differential
    """
    if not ablation_results:
        return {}

    scores = {}
    for mode, metrics in ablation_results.items():
        if isinstance(metrics, dict):
            score = (
                metrics.get(metric_key)
                or metrics.get("overall", {}).get(metric_key, 0.0)
            )
        else:
            score = float(metrics)
        scores[mode] = score

    best_mode = max(scores, key=scores.get) if scores else None
    no_history_score = scores.get("no_history", 0.0)
    full_history_score = scores.get("full", 0.0)

    return {
        "scores_by_mode": scores,
        "best_mode": best_mode,
        "best_score": scores.get(best_mode, 0.0) if best_mode else 0.0,
        "history_gain_over_none": full_history_score - no_history_score,
        "recommendation": (
            best_mode if best_mode else "full"
        ),
    }


# ---------------------------------------------------------------
# Question-type stratified analysis
# ---------------------------------------------------------------

def question_type_analysis(
    results: List[dict],
    metric_keys: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Break down metrics by question type.

    Results must have metadata.question_type.
    """
    groups = defaultdict(list)
    for r in results:
        qt = r.get("metadata", {}).get("question_type", "unknown")
        groups[qt].append(r)

    analysis = {}
    for qt, group_results in groups.items():
        analysis[qt] = {"count": len(group_results)}
        for key in metric_keys:
            vals = [r.get(key, 0.0) for r in group_results]
            if vals:
                analysis[qt][key] = float(np.mean(vals))

    return analysis


# ---------------------------------------------------------------
# Domain-level analysis
# ---------------------------------------------------------------

def domain_analysis(results: List[dict], metric_keys: List[str]) -> Dict[str, dict]:
    """Break down metrics by dataset domain (clapnq, govt, etc.)."""
    groups = defaultdict(list)
    for r in results:
        domain = r.get("metadata", {}).get("domain", "unknown")
        groups[domain].append(r)

    output = {}
    for domain, group in groups.items():
        output[domain] = {"count": len(group)}
        for key in metric_keys:
            vals = [r.get(key, 0.0) for r in group]
            if vals:
                output[domain][key] = float(np.mean(vals))
    return output


# ---------------------------------------------------------------
# Multi-passage synthesis analysis
# ---------------------------------------------------------------

def synthesis_requirement_analysis(results: List[dict]) -> dict:
    """
    Analyze performance on queries requiring multi-passage synthesis
    vs single-passage queries.

    Heuristic: if qrels has >=2 relevant passages → multi-passage query.
    """
    multi_passage = []
    single_passage = []

    for r in results:
        qrels = r.get("qrels", {})
        relevant = [pid for pid, rel in qrels.items() if rel >= 1]
        target = r.get("harmonic_mean", r.get("ndcg@10", 0.0))

        if len(relevant) >= 2:
            multi_passage.append(target)
        else:
            single_passage.append(target)

    return {
        "single_passage": {
            "count": len(single_passage),
            "mean_score": float(np.mean(single_passage)) if single_passage else 0.0,
        },
        "multi_passage": {
            "count": len(multi_passage),
            "mean_score": float(np.mean(multi_passage)) if multi_passage else 0.0,
        },
        "gap": (
            (float(np.mean(single_passage)) - float(np.mean(multi_passage)))
            if (single_passage and multi_passage) else 0.0
        ),
    }


# ---------------------------------------------------------------
# Conversation length impact
# ---------------------------------------------------------------

def conversation_length_analysis(results: List[dict], metric_key: str = "ndcg@10") -> dict:
    """
    Analyze whether conversation length (number of turns) affects performance.
    Groups: short (<4 turns), medium (4-7), long (>7).
    """
    groups = {"short": [], "medium": [], "long": []}
    for r in results:
        n_turns = r.get("num_turns", r.get("metadata", {}).get("num_turns", 5))
        score = r.get(metric_key, 0.0)
        if n_turns < 4:
            groups["short"].append(score)
        elif n_turns <= 7:
            groups["medium"].append(score)
        else:
            groups["long"].append(score)

    return {
        group: {
            "count": len(scores),
            "mean": float(np.mean(scores)) if scores else 0.0,
        }
        for group, scores in groups.items()
    }


# ---------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------

def save_retrieval_breakdown_plot(
    breakdown: Dict[str, Dict[str, float]],
    field: str,
    metric: str = "ndcg@10",
    output_path: str = "artifacts/results/figures/retrieval_breakdown.png",
):
    """Save a bar chart of retrieval metric by metadata field."""
    try:
        import matplotlib.pyplot as plt
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        categories = list(breakdown.keys())
        values = [breakdown[c].get(metric, 0.0) for c in categories]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(categories, values, color="#4472C4", alpha=0.85, edgecolor="white")
        ax.set_xlabel(field.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f"{metric.upper()} by {field.replace('_', ' ').title()}", fontsize=14)
        ax.set_ylim(0, 1.0)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved: {output_path}")
    except ImportError:
        print("[Plot] matplotlib not available, skipping plot.")


def save_ablation_plot(
    ablation_scores: Dict[str, float],
    ablation_name: str,
    metric: str,
    output_path: str,
):
    """Save a line/bar chart for ablation results."""
    try:
        import matplotlib.pyplot as plt
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        settings = list(ablation_scores.keys())
        values = [ablation_scores[s] for s in settings]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(settings, values, color="#70AD47", alpha=0.85, edgecolor="white")
        ax.set_xlabel(ablation_name, fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"Ablation: {ablation_name} vs {metric}", fontsize=13)
        ax.set_ylim(0, max(values) * 1.2 if values else 1.0)
        for i, (s, v) in enumerate(zip(settings, values)):
            ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved: {output_path}")
    except ImportError:
        print("[Plot] matplotlib not available.")


def save_error_taxonomy_plot(
    failure_dist: Dict[str, int],
    output_path: str = "artifacts/results/figures/error_taxonomy.png",
):
    """Save a horizontal bar chart for error taxonomy."""
    try:
        import matplotlib.pyplot as plt
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        categories = list(failure_dist.keys())
        counts = [failure_dist[c] for c in categories]

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = range(len(categories))
        bars = ax.barh(list(y_pos), counts, color="#ED7D31", alpha=0.85)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels([c.replace("_", " ").title() for c in categories])
        ax.set_xlabel("Count", fontsize=12)
        ax.set_title("Error Taxonomy — Failure Mode Distribution", fontsize=13)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    str(count), va="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Plot] Saved: {output_path}")
    except ImportError:
        print("[Plot] matplotlib not available.")
