#!/usr/bin/env python3
"""
scripts/run_faithfulness.py
Faithfulness & reasoning analysis for Assignment 3 (Track A requirement).

Performs:
  1. NLI-based faithfulness scoring per response sentence
  2. Hallucination sentence identification and classification
  3. Faithfulness breakdown by question type, multiturn type, answerability
  4. Comparison of faithfulness across prompt variants
  5. Model-level faithfulness comparison (Llama vs Qwen)
  6. Correlation between response length and faithfulness
  7. Unanswerable hallucination deep-dive

Usage:
    python scripts/run_faithfulness.py --config configs/default.yaml \
        --input artifacts/results/task_b_results.json

    python scripts/run_faithfulness.py --config configs/default.yaml --mock
"""

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.utils import (
    set_seed, load_config, setup_logger, save_json, load_json,
    Timer, ensure_dirs, generate_mock_dataset,
)
from src.evaluation.faithfulness import NLIFaithfulnessScorer, analyze_coreference_impact
from src.evaluation.generation_metrics import (
    is_refusal, unanswerable_detection_metrics, response_statistics,
)
from src.diagnostic.analysis import save_retrieval_breakdown_plot


# ---------------------------------------------------------------
# Faithfulness breakdown helpers
# ---------------------------------------------------------------

def faithfulness_by_field(scored_results: list, field: str) -> dict:
    """
    Group faithfulness scores by a metadata field.
    Returns mean faithfulness per category.
    """
    groups = defaultdict(list)
    for r in scored_results:
        cat = r.get("metadata", {}).get(field, "unknown")
        groups[cat].append(r["faithfulness_score"])

    return {
        cat: {
            "count": len(scores),
            "mean_faithfulness": float(np.mean(scores)),
            "std_faithfulness": float(np.std(scores)),
            "fully_faithful_rate": float(np.mean([s == 1.0 for s in scores])),
            "low_faithfulness_rate": float(np.mean([s < 0.5 for s in scores])),
        }
        for cat, scores in groups.items()
    }


def hallucination_analysis(scored_results: list) -> dict:
    """
    Analyze hallucination patterns across the dataset.
    """
    all_hallucinations = []
    hallucination_counts = []
    total_sentences = []

    for r in scored_results:
        hall_sents = r.get("hallucination_sentences", [])
        n_total = r.get("num_sentences", 0)
        all_hallucinations.extend(hall_sents)
        hallucination_counts.append(len(hall_sents))
        total_sentences.append(n_total)

    total = sum(total_sentences)
    total_hallucinated = sum(hallucination_counts)

    # Average hallucinated sentences per response
    per_response = [h / max(t, 1) for h, t in zip(hallucination_counts, total_sentences)]

    return {
        "total_sentences_analyzed": total,
        "total_hallucinated_sentences": total_hallucinated,
        "overall_hallucination_rate": total_hallucinated / total if total > 0 else 0.0,
        "mean_hallucination_rate_per_response": float(np.mean(per_response)),
        "responses_with_any_hallucination": sum(1 for c in hallucination_counts if c > 0),
        "responses_fully_faithful": sum(1 for c in hallucination_counts if c == 0),
        "example_hallucinations": all_hallucinations[:10],
    }


def length_faithfulness_correlation(scored_results: list) -> dict:
    """
    Analyze whether response length correlates with faithfulness.
    Longer responses may have more hallucination risk.
    """
    lengths = [len(r.get("generated", "").split()) for r in scored_results]
    faithfulness = [r.get("faithfulness_score", 1.0) for r in scored_results]

    if len(lengths) < 5:
        return {"error": "Not enough data"}

    try:
        from scipy import stats
        corr, pval = stats.pearsonr(lengths, faithfulness)
    except ImportError:
        n = len(lengths)
        ml, mf = np.mean(lengths), np.mean(faithfulness)
        num = sum((l - ml) * (f - mf) for l, f in zip(lengths, faithfulness))
        den = (sum((l - ml) ** 2 for l in lengths) *
               sum((f - mf) ** 2 for f in faithfulness)) ** 0.5
        corr = num / den if den > 0 else 0.0
        pval = None

    # Bin by response length
    bins = {"short (<30w)": [], "medium (30-80w)": [], "long (>80w)": []}
    for l, f in zip(lengths, faithfulness):
        if l < 30:
            bins["short (<30w)"].append(f)
        elif l <= 80:
            bins["medium (30-80w)"].append(f)
        else:
            bins["long (>80w)"].append(f)

    return {
        "pearson_correlation": float(corr),
        "p_value": float(pval) if pval is not None else "N/A",
        "by_length_bin": {
            k: {"count": len(v), "mean_faithfulness": float(np.mean(v)) if v else 0.0}
            for k, v in bins.items()
        },
    }


def prompt_variant_faithfulness(task_b_results: dict, scorer: NLIFaithfulnessScorer) -> dict:
    """
    Compare faithfulness across prompt variants.
    """
    variant_results = {}

    for key, result in task_b_results.items():
        prompt_variant = result.get("config", {}).get("prompt_variant", key)
        per_query = result.get("per_query", [])

        if not per_query:
            continue

        responses = [r["generated"] for r in per_query]
        passages_list = [r.get("passages", []) for r in per_query]

        with Timer(f"Faith scoring: {key}"):
            scores = scorer.batch_score(responses, passages_list)

        agg = scorer.aggregate_faithfulness(scores)
        variant_results[prompt_variant] = {
            "mean_faithfulness": agg["mean_faithfulness"],
            "mean_hallucination_rate": agg["mean_hallucination_rate"],
            "fully_faithful_rate": agg["fully_faithful_rate"],
            "count": len(per_query),
        }

    return variant_results


# ---------------------------------------------------------------
# Mock generation results for --mock mode
# ---------------------------------------------------------------

def generate_mock_generation_results(n: int = 50, seed: int = 42) -> dict:
    import random
    random.seed(seed)

    mock = generate_mock_dataset(n_conversations=n, seed=seed)
    conversations = mock["conversations"]
    corpus = mock["corpus"]
    references = mock["references"]

    def make_per_query(prompt_variant: str, model_name: str):
        results = []
        for conv in conversations:
            cid = conv["id"]
            ref = references.get(cid, {})
            ans = conv.get("answerability", "answerable")
            ref_passages = [corpus[pid]["text"] for pid in ref.get("passages", []) if pid in corpus]

            if ans == "unanswerable":
                generated = (
                    "I cannot answer this question from the provided passages."
                    if prompt_variant in ("faithfulness_constrained", "unanswerable_aware")
                    else "The answer is definitely 42 and relates to undisclosed secrets."
                )
            else:
                if prompt_variant == "chain_of_thought":
                    generated = (
                        "Let me think step by step:\n"
                        "1. The question asks about the key value.\n"
                        "2. The passages mention this value explicitly.\n"
                        f"3. Therefore, {ref.get('reference', 'the answer is in the passages.')}"
                    )
                else:
                    generated = ref.get("reference", "Based on the passages, the answer is provided.")

            results.append({
                "conv_id": cid,
                "generated": generated,
                "reference": ref.get("reference", ""),
                "passages": ref_passages,
                "faithfulness_score": random.uniform(0.7, 1.0) if ans == "answerable" else random.uniform(0.0, 0.4),
                "answerability": ans,
                "num_turns": len(conv.get("turns", [])),
                "metadata": {
                    "question_type": conv.get("question_type"),
                    "multiturn_type": conv.get("multiturn_type"),
                    "domain": conv.get("domain"),
                    "answerability": ans,
                    "model": model_name,
                    "prompt_variant": prompt_variant,
                },
            })
        return results

    return {
        "Llama-3-8B_standard": {
            "config": {"model": "Llama-3-8B", "prompt_variant": "standard"},
            "per_query": make_per_query("standard", "Llama-3-8B"),
        },
        "Llama-3-8B_faithfulness_constrained": {
            "config": {"model": "Llama-3-8B", "prompt_variant": "faithfulness_constrained"},
            "per_query": make_per_query("faithfulness_constrained", "Llama-3-8B"),
        },
        "Qwen-2.5-7B_standard": {
            "config": {"model": "Qwen-2.5-7B", "prompt_variant": "standard"},
            "per_query": make_per_query("standard", "Qwen-2.5-7B"),
        },
        "Qwen-2.5-7B_unanswerable_aware": {
            "config": {"model": "Qwen-2.5-7B", "prompt_variant": "unanswerable_aware"},
            "per_query": make_per_query("unanswerable_aware", "Qwen-2.5-7B"),
        },
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="A3: Faithfulness Analysis")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", default=None, help="Path to task_b_results.json")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    ensure_dirs(config)
    logger = setup_logger("faithfulness", config["output"]["logs_dir"])

    output_dir = args.output or config["output"]["results_dir"]
    fig_dir = config["output"]["figures_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # ---- Load generation results ----
    if args.mock or args.input is None:
        logger.info("Generating mock generation results")
        task_b_results = generate_mock_generation_results(n=50, seed=config["seed"])
    else:
        logger.info(f"Loading generation results from {args.input}")
        task_b_results = load_json(args.input)

    # ---- Set up faithfulness scorer ----
    scorer = NLIFaithfulnessScorer(
        model_name=config["evaluation"]["task_b"]["faithfulness_model"]
    )

    faithfulness_output = {}

    # ================================================================
    # 1. Per-result faithfulness scoring
    # ================================================================
    logger.info("--- NLI Faithfulness Scoring ---")

    # Score the primary (standard prompt) results first
    primary_key = next(
        (k for k in task_b_results if "standard" in k and "Llama" in k), 
        list(task_b_results.keys())[0]
    )
    primary_results = task_b_results[primary_key].get("per_query", [])

    logger.info(f"Scoring {len(primary_results)} responses (primary: {primary_key})")
    with Timer("NLI scoring"):
        scored = scorer.batch_score(
            [r["generated"] for r in primary_results],
            [r.get("passages", []) for r in primary_results],
        )

    # Enrich primary_results with scored fields
    enriched = []
    for r, s in zip(primary_results, scored):
        entry = {**r, **s}
        enriched.append(entry)

    agg = scorer.aggregate_faithfulness(scored)
    faithfulness_output["aggregate"] = agg
    logger.info(f"  Mean faithfulness: {agg['mean_faithfulness']:.4f}")
    logger.info(f"  Hallucination rate: {agg['mean_hallucination_rate']:.4f}")
    logger.info(f"  Fully faithful: {agg['fully_faithful_rate']:.1%}")

    # ================================================================
    # 2. Hallucination analysis
    # ================================================================
    logger.info("--- Hallucination Analysis ---")
    hallucination = hallucination_analysis(enriched)
    faithfulness_output["hallucination_analysis"] = hallucination

    logger.info(f"  Overall hallucination rate: {hallucination['overall_hallucination_rate']:.4f}")
    logger.info(f"  Responses with any hallucination: {hallucination['responses_with_any_hallucination']}")

    # ================================================================
    # 3. Faithfulness by metadata dimension
    # ================================================================
    logger.info("--- Faithfulness Breakdown by Metadata ---")

    for field in ["question_type", "multiturn_type", "domain", "answerability"]:
        breakdown = faithfulness_by_field(enriched, field)
        faithfulness_output[f"by_{field}"] = breakdown
        logger.info(f"  By {field}:")
        for cat, stats in breakdown.items():
            logger.info(f"    {cat}: mean={stats['mean_faithfulness']:.3f}, n={stats['count']}")

        # Plot
        save_retrieval_breakdown_plot(
            {cat: {"ndcg@10": stats["mean_faithfulness"]} for cat, stats in breakdown.items()},
            field=field,
            metric="ndcg@10",
            output_path=os.path.join(fig_dir, f"faithfulness_by_{field}.png"),
        )

    # ================================================================
    # 4. Length-faithfulness correlation
    # ================================================================
    logger.info("--- Length vs Faithfulness Correlation ---")
    length_corr = length_faithfulness_correlation(enriched)
    faithfulness_output["length_faithfulness_correlation"] = length_corr
    logger.info(f"  Pearson r = {length_corr.get('pearson_correlation', 0):.4f}")

    # ================================================================
    # 5. Prompt variant comparison
    # ================================================================
    logger.info("--- Prompt Variant Faithfulness Comparison ---")
    prompt_comparison = prompt_variant_faithfulness(task_b_results, scorer)
    faithfulness_output["prompt_variant_comparison"] = prompt_comparison

    logger.info("  Prompt variant faithfulness:")
    for variant, stats in prompt_comparison.items():
        logger.info(f"    {variant}: mean={stats['mean_faithfulness']:.3f}, "
                    f"hall_rate={stats['mean_hallucination_rate']:.3f}")

    # ================================================================
    # 6. Unanswerable-specific hallucination
    # ================================================================
    logger.info("--- Unanswerable Hallucination Deep-Dive ---")
    unanswerable_results = [r for r in enriched if r.get("answerability") == "unanswerable"]
    answerable_results = [r for r in enriched if r.get("answerability") == "answerable"]

    if unanswerable_results:
        unans_faith = scorer.aggregate_faithfulness(
            [{"faithfulness_score": r.get("faithfulness_score", 0)} for r in unanswerable_results]
        )
        unans_refusal_rate = np.mean([is_refusal(r["generated"]) for r in unanswerable_results])
        faithfulness_output["unanswerable_analysis"] = {
            "count": len(unanswerable_results),
            "mean_faithfulness": float(np.mean([r.get("faithfulness_score", 0) for r in unanswerable_results])),
            "correct_refusal_rate": float(unans_refusal_rate),
            "hallucination_rate": float(1.0 - unans_refusal_rate),
        }
        logger.info(f"  Unanswerable correct refusals: {unans_refusal_rate:.1%}")
        logger.info(f"  Unanswerable hallucination rate: {1 - unans_refusal_rate:.1%}")

    # ================================================================
    # Save
    # ================================================================
    save_json(faithfulness_output, os.path.join(output_dir, "faithfulness_results.json"))
    logger.info(f"\nFaithfulness results saved to {output_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("FAITHFULNESS ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Mean faithfulness score:     {agg['mean_faithfulness']:.4f}")
    print(f"Mean hallucination rate:     {agg['mean_hallucination_rate']:.4f}")
    print(f"Fully faithful responses:    {agg['fully_faithful_rate']:.1%}")
    print(f"Low faithfulness (<0.5):     {agg['low_faithfulness_rate']:.1%}")
    print(f"Length-faithfulness corr:    {length_corr.get('pearson_correlation', 0):.4f}")
    if unanswerable_results:
        print(f"Unanswerable correct refusal:{float(unans_refusal_rate):.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
