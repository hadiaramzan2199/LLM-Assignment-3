#!/usr/bin/env python3
"""
scripts/run_ablations.py
Ablation study runner for Assignment 3.

Controlled experiments:
  1. History window size → impact on retrieval nDCG@10
  2. Retrieval k → impact on generation harmonic mean
  3. Prompt variant → impact on faithfulness and hallucination rate
  4. Retrieval method → BM25 vs Dense vs Hybrid (systematic comparison)

All comparisons hold other variables fixed. Seed=42 throughout.

Usage:
    python scripts/run_ablations.py --config configs/default.yaml
    python scripts/run_ablations.py --config configs/default.yaml --ablation history_window
    python scripts/run_ablations.py --config configs/default.yaml --mock
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import (
    set_seed, load_config, setup_logger, save_json,
    Timer, ensure_dirs, generate_mock_dataset,
)
from src.evaluation.retrieval_metrics import compute_retrieval_metrics
from src.evaluation.generation_metrics import compute_generation_metrics
from src.diagnostic.ablation import (
    AblationRunner, analyze_ablation_results, format_ablation_table,
    PROMPT_TEMPLATES, format_prompt, build_history_string,
)
from src.diagnostic.analysis import (
    history_impact_analysis, save_ablation_plot,
)


# ---------------------------------------------------------------
# Evaluation function factory
# ---------------------------------------------------------------

def make_retrieval_eval_fn(conversations, corpus, qrels, config, logger):
    """
    Returns a callable that evaluates retrieval given arbitrary settings.
    Used by AblationRunner.
    """
    from scripts.run_evaluation import (
        evaluate_task_a, run_bm25_retrieval, run_dense_retrieval, run_hybrid_retrieval,
        _build_query, RETRIEVAL_FNS,
    )

    def evaluate_fn(
        history_mode: str = "full_history",
        top_k: int = 10,
        retrieval_method: str = "hybrid",
    ) -> dict:
        result = evaluate_task_a(
            conversations, corpus, qrels, config, logger,
            history_mode=history_mode,
            retrieval_method=retrieval_method,
            top_k=top_k,
        )
        return result["metrics"]["overall"]

    return evaluate_fn


def make_generation_eval_fn(conversations, corpus, references, config, logger, model=None, tokenizer=None):
    """Returns a callable for generation ablation experiments."""
    from scripts.run_evaluation import evaluate_task_b
    from src.evaluation.faithfulness import NLIFaithfulnessScorer

    faithfulness_scorer = NLIFaithfulnessScorer(
        model_name=config["evaluation"]["task_b"]["faithfulness_model"]
    )

    def evaluate_fn(
        prompt_variant: str = "standard",
        top_k: int = 5,
        history_mode: str = "full_history",
        model_name: str = "Mock",
    ) -> dict:
        result = evaluate_task_b(
            conversations, corpus, references, config, logger,
            model=model, tokenizer=tokenizer,
            model_name=model_name,
            prompt_variant=prompt_variant,
            history_mode=history_mode,
            top_k_passages=top_k,
            faithfulness_scorer=faithfulness_scorer,
        )
        return result["metrics"]["overall"]

    return evaluate_fn


# ---------------------------------------------------------------
# Ablation 1: History window size
# ---------------------------------------------------------------

def run_history_window_ablation(conversations, corpus, qrels, config, logger) -> dict:
    """
    Ablate history window size.
    Fixed: hybrid retrieval, k=10.
    Varies: history_mode in [no_history, window_1, window_2, window_3, full_history]
    """
    logger.info("=== ABLATION 1: History Window Size ===")
    eval_fn = make_retrieval_eval_fn(conversations, corpus, qrels, config, logger)

    window_configs = {
        "no_history":     {"history_mode": "no_history",     "top_k": 10, "retrieval_method": "hybrid"},
        "window_1":       {"history_mode": "window_1",       "top_k": 10, "retrieval_method": "hybrid"},
        "window_2":       {"history_mode": "window_2",       "top_k": 10, "retrieval_method": "hybrid"},
        "window_3":       {"history_mode": "window_3",       "top_k": 10, "retrieval_method": "hybrid"},
        "full_history":   {"history_mode": "full_history",   "top_k": 10, "retrieval_method": "hybrid"},
    }

    results = {}
    for label, kwargs in window_configs.items():
        logger.info(f"  History mode: {label}")
        with Timer(label):
            metrics = eval_fn(**kwargs)
        results[label] = metrics
        logger.info(f"    nDCG@10 = {metrics.get('ndcg@10', 0):.4f}")

    return results


# ---------------------------------------------------------------
# Ablation 2: Retrieval method comparison
# ---------------------------------------------------------------

def run_retrieval_method_ablation(conversations, corpus, qrels, config, logger) -> dict:
    """
    Compare BM25 / Dense / Hybrid with fixed history=full_history, k=10.
    Also breaks down by question type for each method.
    """
    logger.info("=== ABLATION 2: Retrieval Method ===")
    eval_fn = make_retrieval_eval_fn(conversations, corpus, qrels, config, logger)

    methods = ["bm25", "dense", "hybrid"]
    results = {}

    for method in methods:
        logger.info(f"  Method: {method}")
        with Timer(method):
            metrics = eval_fn(history_mode="full_history", top_k=10, retrieval_method=method)
        results[method] = metrics
        logger.info(f"    nDCG@10 = {metrics.get('ndcg@10', 0):.4f}, MRR = {metrics.get('mrr', 0):.4f}")

    return results


# ---------------------------------------------------------------
# Ablation 3: Retrieval k for generation
# ---------------------------------------------------------------

def run_retrieval_k_ablation(conversations, corpus, references, config, logger) -> dict:
    """
    Ablate number of retrieved passages fed to the generator.
    Fixed: hybrid retrieval, standard prompt, full history.
    Varies: k in [1, 3, 5, 10]
    """
    logger.info("=== ABLATION 3: Retrieval k for Generation ===")
    eval_fn = make_generation_eval_fn(conversations, corpus, references, config, logger)

    k_values = config["diagnostic"]["ablation"]["retrieval_k_values"]
    results = {}

    for k in k_values:
        label = f"k_{k}"
        logger.info(f"  k = {k}")
        with Timer(label):
            metrics = eval_fn(prompt_variant="standard", top_k=k, history_mode="full_history")
        results[label] = metrics
        logger.info(f"    Harmonic mean = {metrics.get('harmonic_mean', 0):.4f}")

    return results


# ---------------------------------------------------------------
# Ablation 4: Prompt variant
# ---------------------------------------------------------------

def run_prompt_variant_ablation(conversations, corpus, references, config, logger) -> dict:
    """
    Ablate prompt template.
    Fixed: hybrid retrieval, k=5, full history.
    Varies: prompt_variant in [standard, faithfulness_constrained, chain_of_thought, unanswerable_aware]
    """
    logger.info("=== ABLATION 4: Prompt Variant ===")
    eval_fn = make_generation_eval_fn(conversations, corpus, references, config, logger)

    variants = config["diagnostic"]["ablation"]["prompt_variants"]
    results = {}

    for variant in variants:
        logger.info(f"  Prompt: {variant}")
        with Timer(variant):
            metrics = eval_fn(prompt_variant=variant, top_k=5, history_mode="full_history")
        results[variant] = metrics

        hm = metrics.get("harmonic_mean", 0)
        rl = metrics.get("rouge_l_f1", 0)
        logger.info(f"    HM={hm:.4f}  ROUGE-L={rl:.4f}")

    return results


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="A3: Ablation Studies")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ablation",
                        choices=["history_window", "retrieval_method",
                                 "retrieval_k", "prompt_variant", "all"],
                        default="all")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    ensure_dirs(config)
    logger = setup_logger("ablations", config["output"]["logs_dir"])

    output_dir = args.output or config["output"]["results_dir"]
    os.makedirs(output_dir, exist_ok=True)
    fig_dir = config["output"]["figures_dir"]
    os.makedirs(fig_dir, exist_ok=True)

    # ---- Load data ----
    if args.mock:
        logger.info("Using mock dataset")
        mock = generate_mock_dataset(n_conversations=50, seed=config["seed"])
        conversations = mock["conversations"]
        corpus = mock["corpus"]
        qrels = mock["qrels"]
        references = mock["references"]
    else:
        from src.data.loader import MTRAGDataLoader
        loader = MTRAGDataLoader(config, split="val")
        conversations = loader.conversations
        corpus = loader.corpus
        qrels = loader.qrels
        references = loader.references

    ablation_results = {}

    # Run selected ablations
    if args.ablation in ("history_window", "all"):
        res = run_history_window_ablation(conversations, corpus, qrels, config, logger)
        ablation_results["history_window"] = res
        save_ablation_plot(
            {k: v.get("ndcg@10", 0) for k, v in res.items()},
            "History Window Size", "nDCG@10",
            os.path.join(fig_dir, "ablation_history_window.png")
        )

    if args.ablation in ("retrieval_method", "all"):
        res = run_retrieval_method_ablation(conversations, corpus, qrels, config, logger)
        ablation_results["retrieval_method"] = res
        save_ablation_plot(
            {k: v.get("ndcg@10", 0) for k, v in res.items()},
            "Retrieval Method", "nDCG@10",
            os.path.join(fig_dir, "ablation_retrieval_method.png")
        )

    if args.ablation in ("retrieval_k", "all"):
        res = run_retrieval_k_ablation(conversations, corpus, references, config, logger)
        ablation_results["retrieval_k"] = res
        save_ablation_plot(
            {k: v.get("harmonic_mean", 0) for k, v in res.items()},
            "Retrieved Passages (k)", "Harmonic Mean",
            os.path.join(fig_dir, "ablation_retrieval_k.png")
        )

    if args.ablation in ("prompt_variant", "all"):
        res = run_prompt_variant_ablation(conversations, corpus, references, config, logger)
        ablation_results["prompt_variant"] = res
        save_ablation_plot(
            {k: v.get("harmonic_mean", 0) for k, v in res.items()},
            "Prompt Variant", "Harmonic Mean",
            os.path.join(fig_dir, "ablation_prompt_variant.png")
        )

    # Analysis
    analysis = analyze_ablation_results(ablation_results)
    logger.info("\n" + format_ablation_table(analysis))

    save_json({"ablation_results": ablation_results, "analysis": analysis},
              os.path.join(output_dir, "ablation_results.json"))
    logger.info(f"Ablation results saved to {output_dir}")


if __name__ == "__main__":
    main()
