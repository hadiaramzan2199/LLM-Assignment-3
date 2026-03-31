"""
src/diagnostic/ablation.py
Ablation study runner for A3.

Ablations:
  1. History window size (0, 1, 2, 3, 5, full)
  2. Retrieval count k (1, 3, 5, 10)
  3. Prompt variant (standard, faithfulness_constrained, chain_of_thought, unanswerable_aware)
  4. Retrieval method (bm25, dense, hybrid)

Each ablation holds everything else constant (seed=42).
"""

import json
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------
# Prompt variant templates
# ---------------------------------------------------------------

PROMPT_TEMPLATES = {
    "standard": (
        "You are a helpful assistant. Answer the question based on the provided passages.\n\n"
        "Conversation history:\n{history}\n\n"
        "Reference passages:\n{passages}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    "faithfulness_constrained": (
        "You are a faithful assistant. Answer ONLY using information from the provided passages. "
        "If the passages do not contain enough information, say 'I cannot answer based on the provided information.'\n\n"
        "Conversation history:\n{history}\n\n"
        "Reference passages:\n{passages}\n\n"
        "Question: {question}\n\n"
        "Answer (grounded in passages only):"
    ),
    "chain_of_thought": (
        "You are a careful assistant. Think step by step before answering.\n\n"
        "Conversation history:\n{history}\n\n"
        "Reference passages:\n{passages}\n\n"
        "Question: {question}\n\n"
        "Let me think step by step:\n"
        "1. What is the question asking?\n"
        "2. What relevant information do the passages provide?\n"
        "3. What is my final answer?\n\n"
        "Answer:"
    ),
    "unanswerable_aware": (
        "You are a careful assistant. If the provided passages do not contain sufficient information "
        "to answer the question, explicitly state that the question cannot be answered from the given context. "
        "Do not guess or hallucinate.\n\n"
        "Conversation history:\n{history}\n\n"
        "Reference passages:\n{passages}\n\n"
        "Question: {question}\n\n"
        "Answer (or state if unanswerable):"
    ),
}


def format_prompt(
    template_name: str,
    history: str,
    passages: List[str],
    question: str,
) -> str:
    """Format a generation prompt from a named template."""
    template = PROMPT_TEMPLATES.get(template_name, PROMPT_TEMPLATES["standard"])
    passages_str = "\n\n".join(
        f"[Passage {i+1}]: {p}" for i, p in enumerate(passages)
    )
    return template.format(
        history=history or "None",
        passages=passages_str,
        question=question,
    )


# ---------------------------------------------------------------
# History window builder
# ---------------------------------------------------------------

def build_history_string(turns: List[dict], window_size: int) -> str:
    """
    Build a conversation history string.
    window_size: number of prior Q&A turns to include. -1 = all. 0 = none.
    """
    prior_turns = turns[:-1]  # all but the final question

    if window_size == 0:
        return ""
    elif window_size > 0:
        prior_turns = prior_turns[-window_size:]
    # else: -1 = full history

    lines = []
    for t in prior_turns:
        lines.append(f"User: {t['question']}")
        if "answer" in t:
            lines.append(f"Assistant: {t['answer']}")
    return "\n".join(lines)


# ---------------------------------------------------------------
# Ablation experiment runner
# ---------------------------------------------------------------

class AblationRunner:
    """
    Runs controlled ablation experiments.

    Usage:
      runner = AblationRunner(config, retriever, generator, dataset)
      results = runner.run_history_window_ablation()
      results = runner.run_retrieval_k_ablation()
      results = runner.run_prompt_variant_ablation()
    """

    def __init__(
        self,
        config: dict,
        retriever=None,
        generator=None,
        dataset=None,
    ):
        self.config = config
        self.retriever = retriever
        self.generator = generator
        self.dataset = dataset
        self.seed = config.get("seed", 42)

    def run_history_window_ablation(
        self,
        evaluate_fn: Callable,
        window_sizes: Optional[List[int]] = None,
    ) -> Dict[str, dict]:
        """
        Ablate history window size while keeping retrieval method fixed (hybrid).

        window_sizes: list of window sizes. -1 = full. 0 = no history.
        """
        window_sizes = window_sizes or self.config["diagnostic"]["ablation"]["history_window_sizes"]
        results = {}

        for ws in window_sizes:
            label = "full" if ws == -1 else f"window_{ws}" if ws > 0 else "no_history"
            print(f"[Ablation] History window: {label}")
            # Caller provides evaluate_fn that takes history_mode and returns metrics
            metrics = evaluate_fn(history_mode=label)
            results[label] = metrics

        return results

    def run_retrieval_k_ablation(
        self,
        evaluate_fn: Callable,
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, dict]:
        """
        Ablate number of retrieved passages (k) for generation quality.
        """
        k_values = k_values or self.config["diagnostic"]["ablation"]["retrieval_k_values"]
        results = {}

        for k in k_values:
            print(f"[Ablation] Retrieval k={k}")
            metrics = evaluate_fn(top_k=k)
            results[f"k_{k}"] = metrics

        return results

    def run_prompt_variant_ablation(
        self,
        evaluate_fn: Callable,
        variants: Optional[List[str]] = None,
    ) -> Dict[str, dict]:
        """
        Ablate prompt template while keeping retrieval and model fixed.
        """
        variants = variants or self.config["diagnostic"]["ablation"]["prompt_variants"]
        results = {}

        for variant in variants:
            print(f"[Ablation] Prompt variant: {variant}")
            metrics = evaluate_fn(prompt_variant=variant)
            results[variant] = metrics

        return results

    def run_retrieval_method_ablation(
        self,
        evaluate_fn: Callable,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, dict]:
        """
        Compare retrieval methods: BM25, Dense, Hybrid.
        This reproduces A2 results systematically for A3 comparison.
        """
        methods = methods or ["bm25", "dense", "hybrid"]
        results = {}
        for method in methods:
            print(f"[Ablation] Retrieval method: {method}")
            metrics = evaluate_fn(retrieval_method=method)
            results[method] = metrics
        return results

    def run_all_ablations(self, evaluate_fn: Callable) -> dict:
        """Run all ablation studies and return combined results."""
        return {
            "history_window": self.run_history_window_ablation(evaluate_fn),
            "retrieval_k": self.run_retrieval_k_ablation(evaluate_fn),
            "prompt_variant": self.run_prompt_variant_ablation(evaluate_fn),
            "retrieval_method": self.run_retrieval_method_ablation(evaluate_fn),
        }


# ---------------------------------------------------------------
# Ablation result analysis
# ---------------------------------------------------------------

def analyze_ablation_results(ablation_results: dict, primary_metric: str = "ndcg@10") -> dict:
    """
    Analyze ablation results to find optimal settings and deltas.

    Returns:
      {
        ablation_type: {
          "best_setting": str,
          "best_score": float,
          "worst_setting": str,
          "worst_score": float,
          "delta": float,
          "scores": {setting: score}
        }
      }
    """
    analysis = {}

    for ablation_type, results in ablation_results.items():
        scores = {}
        for setting, metrics in results.items():
            # Support nested dicts: metrics["overall"][primary_metric]
            if isinstance(metrics, dict):
                score = (
                    metrics.get(primary_metric)
                    or metrics.get("overall", {}).get(primary_metric)
                    or metrics.get("harmonic_mean")
                    or 0.0
                )
            else:
                score = float(metrics)
            scores[setting] = score

        if not scores:
            continue

        best = max(scores, key=scores.get)
        worst = min(scores, key=scores.get)

        analysis[ablation_type] = {
            "best_setting": best,
            "best_score": scores[best],
            "worst_setting": worst,
            "worst_score": scores[worst],
            "delta": scores[best] - scores[worst],
            "scores": scores,
        }

    return analysis


def format_ablation_table(analysis: dict) -> str:
    """Format ablation analysis as a readable table."""
    lines = [
        "=" * 70,
        "ABLATION STUDY RESULTS",
        "=" * 70,
    ]
    for ablation_type, result in analysis.items():
        lines.append(f"\n--- {ablation_type.replace('_', ' ').title()} ---")
        lines.append(f"  Best:  {result['best_setting']} ({result['best_score']:.4f})")
        lines.append(f"  Worst: {result['worst_setting']} ({result['worst_score']:.4f})")
        lines.append(f"  Delta: {result['delta']:.4f}")
        lines.append("  All scores:")
        for setting, score in sorted(result["scores"].items(), key=lambda x: -x[1]):
            lines.append(f"    {setting:<25} {score:.4f}")
    return "\n".join(lines)
