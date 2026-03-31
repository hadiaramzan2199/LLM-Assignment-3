"""
src/evaluation/generation_metrics.py
Generation evaluation metrics for Task B.
Computes: ROUGE-L, BERTScore, harmonic mean (RL_F, RB_llm, RB_alg),
unanswerable detection F1, response length statistics.
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------
# ROUGE-L
# ---------------------------------------------------------------

def lcs_length(x: List[str], y: List[str]) -> int:
    """Compute length of LCS between two token lists."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l_score(hypothesis: str, reference: str) -> dict:
    """Compute ROUGE-L F1, Precision, Recall."""
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not hyp_tokens or not ref_tokens:
        return {"f": 0.0, "p": 0.0, "r": 0.0}
    lcs = lcs_length(hyp_tokens, ref_tokens)
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"f": f1, "p": precision, "r": recall}


def batch_rouge_l(hypotheses: List[str], references: List[str]) -> List[dict]:
    """Compute ROUGE-L for a batch of hypothesis/reference pairs."""
    assert len(hypotheses) == len(references)
    return [rouge_l_score(h, r) for h, r in zip(hypotheses, references)]


# ---------------------------------------------------------------
# BERTScore (via HuggingFace evaluate)
# ---------------------------------------------------------------

def batch_bertscore(
    hypotheses: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    device: Optional[str] = None,
) -> dict:
    """
    Compute BERTScore for a batch.
    Returns dict with lists of P, R, F1 per sample.
    """
    try:
        import evaluate
        bertscore = evaluate.load("bertscore")
        result = bertscore.compute(
            predictions=hypotheses,
            references=references,
            model_type=model_type,
            device=device,
        )
        return {
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "mean_f1": float(np.mean(result["f1"])),
        }
    except Exception as e:
        print(f"[BERTScore] WARNING: {e}. Falling back to ROUGE-L proxy.")
        rouge_scores = batch_rouge_l(hypotheses, references)
        f1s = [s["f"] for s in rouge_scores]
        return {
            "precision": f1s,
            "recall": f1s,
            "f1": f1s,
            "mean_f1": float(np.mean(f1s)),
            "note": "BERTScore unavailable, using ROUGE-L proxy",
        }


# ---------------------------------------------------------------
# Harmonic mean (official MTRAGEval metric)
# ---------------------------------------------------------------

def harmonic_mean(*values: float) -> float:
    """
    Compute harmonic mean of given values.
    Official MTRAGEval Task B metric: harmonic mean of RL_F, RB_llm, RB_alg.
    Returns 0.0 if any value is 0 (a zero component collapses the harmonic mean).
    """
    if any(v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1.0 / v for v in values)


def compute_harmonic_mean_score(rl_f: float, rb_llm: float, rb_alg: float) -> float:
    """
    Official harmonic mean of:
      RL_F   — ROUGE-L F1 (lexical faithfulness)
      RB_llm — LLM-based faithfulness judgment
      RB_alg — Algorithmic faithfulness (NLI-based)
    """
    return harmonic_mean(rl_f, rb_llm, rb_alg)


# ---------------------------------------------------------------
# Unanswerable detection
# ---------------------------------------------------------------

REFUSAL_PATTERNS = [
    r"i (don't|do not|cannot|can't) (know|find|answer|provide)",
    r"the (provided|given|available) (passages?|documents?|context) (do(es)? not|don't|doesn't)",
    r"(there is|there's) no (information|answer|evidence|mention)",
    r"(not|cannot be) answered",
    r"(insufficient|not enough) (information|evidence|context)",
    r"unanswerable",
    r"i'm not able to",
]

REFUSAL_REGEX = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


def is_refusal(response: str) -> bool:
    """Detect whether a generation is a refusal/unanswerable response."""
    return bool(REFUSAL_REGEX.search(response))


def unanswerable_detection_metrics(
    results: List[dict],
) -> dict:
    """
    Compute unanswerable detection precision, recall, F1.

    Each result:
      {"generated": str, "answerability": "answerable" | "unanswerable"}
    """
    tp = fp = fn = tn = 0
    for r in results:
        predicted_refusal = is_refusal(r["generated"])
        true_unanswerable = r["answerability"] == "unanswerable"

        if predicted_refusal and true_unanswerable:
            tp += 1
        elif predicted_refusal and not true_unanswerable:
            fp += 1
        elif not predicted_refusal and true_unanswerable:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "refusal_rate": (tp + fp) / len(results) if results else 0.0,
    }


# ---------------------------------------------------------------
# Response statistics
# ---------------------------------------------------------------

def response_statistics(responses: List[str]) -> dict:
    """Compute basic statistics on generated response lengths."""
    lengths = [len(r.split()) for r in responses]
    return {
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "std_length": float(np.std(lengths)),
        "min_length": int(min(lengths)),
        "max_length": int(max(lengths)),
    }


# ---------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------

def compute_generation_metrics(
    results: List[dict],
    bertscore_model: str = "microsoft/deberta-xlarge-mnli",
    breakdown_field: Optional[str] = None,
) -> dict:
    """
    Full generation evaluation over a list of results.

    Each result:
      {
        "conv_id": str,
        "generated": str,
        "reference": str,
        "passages": [str, ...],    # reference passages
        "answerability": str,
        "metadata": {...}
      }

    Returns:
      {
        "overall": {rouge_l, bertscore_f1, harmonic_mean, ...},
        "unanswerable": {...},
        "by_<field>": {...}
      }
    """
    from collections import defaultdict

    hypotheses = [r["generated"] for r in results]
    references = [r["reference"] for r in results]

    # ROUGE-L
    rouge_scores = batch_rouge_l(hypotheses, references)
    rouge_l_mean = float(np.mean([s["f"] for s in rouge_scores]))

    # BERTScore
    bs_result = batch_bertscore(hypotheses, references, model_type=bertscore_model)
    bs_mean = bs_result["mean_f1"]

    # Harmonic mean (using ROUGE-L as RL_F proxy and BERTScore as RB proxies)
    hm_scores = [
        harmonic_mean(rouge_scores[i]["f"], bs_result["f1"][i], bs_result["f1"][i])
        for i in range(len(results))
    ]
    hm_mean = float(np.mean(hm_scores))

    # Unanswerable detection
    unans_metrics = unanswerable_detection_metrics(results)

    # Response stats
    stats = response_statistics(hypotheses)

    overall = {
        "rouge_l_f1": rouge_l_mean,
        "bertscore_f1": bs_mean,
        "harmonic_mean": hm_mean,
        "unanswerable_f1": unans_metrics["f1"],
        **stats,
    }

    output = {
        "overall": overall,
        "unanswerable_detection": unans_metrics,
        "per_query": [
            {
                "conv_id": r["conv_id"],
                "rouge_l": rouge_scores[i],
                "bertscore_f1": bs_result["f1"][i],
                "harmonic_mean": hm_scores[i],
                "is_refusal": is_refusal(r["generated"]),
                "answerability": r.get("answerability"),
                "metadata": r.get("metadata", {}),
            }
            for i, r in enumerate(results)
        ],
    }

    # Breakdown
    if breakdown_field:
        groups = defaultdict(list)
        for i, r in enumerate(results):
            cat = r.get("metadata", {}).get(breakdown_field, "unknown")
            groups[cat].append({
                "rouge_l_f1": rouge_scores[i]["f"],
                "bertscore_f1": bs_result["f1"][i],
                "harmonic_mean": hm_scores[i],
            })
        output[f"by_{breakdown_field}"] = {
            cat: {
                "rouge_l_f1": float(np.mean([e["rouge_l_f1"] for e in entries])),
                "bertscore_f1": float(np.mean([e["bertscore_f1"] for e in entries])),
                "harmonic_mean": float(np.mean([e["harmonic_mean"] for e in entries])),
                "count": len(entries),
            }
            for cat, entries in groups.items()
        }

    return output
