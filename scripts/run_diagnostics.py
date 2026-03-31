#!/usr/bin/env python3
"""
scripts/run_diagnostics.py
Error taxonomy + multi-dimensional diagnostic analysis for Assignment 3.

Reads evaluation results from run_evaluation.py and produces:
  1. Error taxonomy with per-category counts and rates
  2. Breakdown by question type, multiturn type, domain, answerability
  3. Retrieval-generation coupling analysis
  4. Synthesis requirement analysis
  5. Conversation length impact
  6. Coreference failure analysis
  7. Diagnostic visualizations

Usage:
    python scripts/run_diagnostics.py --config configs/default.yaml \
        --input artifacts/results/eval_results.json

    python scripts/run_diagnostics.py --config configs/default.yaml --mock
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import (
    set_seed, load_config, setup_logger, save_json, load_json,
    Timer, ensure_dirs, generate_mock_dataset,
)
from src.diagnostic.error_taxonomy import ErrorTaxonomy, FAILURE_CATEGORIES
from src.diagnostic.analysis import (
    retrieval_generation_coupling,
    question_type_analysis,
    domain_analysis,
    synthesis_requirement_analysis,
    conversation_length_analysis,
    save_retrieval_breakdown_plot,
    save_error_taxonomy_plot,
)
from src.evaluation.retrieval_metrics import (
    ndcg_at_k, full_breakdown_analysis, rank_distribution_analysis,
)
from src.evaluation.generation_metrics import (
    compute_generation_metrics, unanswerable_detection_metrics,
)
from src.evaluation.faithfulness import analyze_coreference_impact


# ---------------------------------------------------------------
# Build unified per-query result list from eval output
# ---------------------------------------------------------------

def flatten_eval_results(eval_results: dict) -> list:
    """
    Flatten nested evaluation results into a flat per-query list
    for diagnostic analysis.

    Handles both task_a and task_b results.
    """
    flat = []

    # Task A results
    task_a = eval_results.get("task_a", {})
    # Use best retrieval config (hybrid_full_history by default)
    best_a_key = "hybrid_full_history" if "hybrid_full_history" in task_a else (
        list(task_a.keys())[0] if task_a else None
    )
    if best_a_key:
        per_query_a = task_a[best_a_key].get("per_query", [])
        flat.extend(per_query_a)

    return flat


def build_combined_results(eval_results: dict) -> list:
    """
    Build a unified result list that has both retrieval and generation metrics
    per conversation (for coupling analysis).
    """
    task_a = eval_results.get("task_a", {})
    task_b = eval_results.get("task_b", {})

    # Index retrieval results by conv_id
    ret_by_id = {}
    for key, result in task_a.items():
        if "hybrid" in key and "full_history" in key:
            for pq in result.get("per_query", []):
                ret_by_id[pq["conv_id"]] = pq

    # Index generation results by conv_id
    gen_by_id = {}
    for key, result in task_b.items():
        if "standard" in key:
            for pq in result.get("per_query", []):
                gen_by_id[pq["conv_id"]] = pq

    # Merge
    combined = []
    from src.evaluation.retrieval_metrics import ndcg_at_k
    for conv_id, ret in ret_by_id.items():
        entry = {
            "conv_id": conv_id,
            "ndcg@10": ndcg_at_k(ret.get("retrieved", []), ret.get("qrels", {}), 10),
            "retrieved": ret.get("retrieved", []),
            "qrels": ret.get("qrels", {}),
            "num_turns": ret.get("num_turns", 1),
            "query": ret.get("query", ""),
            "metadata": ret.get("metadata", {}),
        }
        if conv_id in gen_by_id:
            gen = gen_by_id[conv_id]
            entry.update({
                "generated": gen.get("generated", ""),
                "reference": gen.get("reference", ""),
                "passages": gen.get("passages", []),
                "faithfulness_score": gen.get("faithfulness_score", 1.0),
                "answerability": gen.get("answerability", "answerable"),
                "harmonic_mean": gen.get("harmonic_mean", 0.0),
                "rouge_l_f1": gen.get("rouge_l", {}).get("f", 0.0),
            })
        combined.append(entry)

    return combined


# ---------------------------------------------------------------
# Mock diagnostic data (for --mock mode)
# ---------------------------------------------------------------

def generate_mock_eval_results(n: int = 50, seed: int = 42) -> dict:
    """Generate mock eval results for pipeline testing."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

    mock_data = generate_mock_dataset(n_conversations=n, seed=seed)
    conversations = mock_data["conversations"]
    corpus = mock_data["corpus"]
    qrels = mock_data["qrels"]
    references = mock_data["references"]

    # Simulate retrieval results
    passage_ids = list(corpus.keys())
    per_query_a = []
    for conv in conversations:
        conv_id = conv["id"]
        conv_qrels = qrels.get(conv_id, {})
        relevant_pids = list(conv_qrels.keys())
        # Simulate realistic retrieval (put relevant at position 1-6)
        retrieved = []
        if relevant_pids:
            insert_pos = random.randint(0, 5)
            retrieved = random.sample([p for p in passage_ids if p not in relevant_pids], 9)
            retrieved.insert(min(insert_pos, len(retrieved)), relevant_pids[0])
        else:
            retrieved = random.sample(passage_ids, 10)

        per_query_a.append({
            "conv_id": conv_id,
            "retrieved": retrieved[:10],
            "qrels": conv_qrels,
            "query": conv["turns"][-1]["question"] if conv.get("turns") else "",
            "num_turns": len(conv.get("turns", [])),
            "metadata": {
                "question_type": conv.get("question_type", "factoid"),
                "multiturn_type": conv.get("multiturn_type", "follow_up"),
                "domain": conv.get("domain", "clapnq"),
                "answerability": conv.get("answerability", "answerable"),
                "history_mode": "full_history",
                "retrieval_method": "hybrid",
            },
        })

    # Simulate generation results
    per_query_b = []
    for conv in conversations:
        conv_id = conv["id"]
        ref = references.get(conv_id, {})
        answerability = conv.get("answerability", "answerable")
        if answerability == "unanswerable":
            generated = random.choice([
                "I cannot answer this based on the provided passages.",
                "The passages contain information that relates to the topic but does not specifically address this query.",
            ])
        else:
            generated = ref.get("reference", "The answer based on the passages is the relevant information.")

        per_query_b.append({
            "conv_id": conv_id,
            "generated": generated,
            "reference": ref.get("reference", ""),
            "passages": [corpus[pid]["text"] for pid in ref.get("passages", []) if pid in corpus],
            "faithfulness_score": random.uniform(0.6, 1.0) if answerability == "answerable" else random.uniform(0.0, 0.5),
            "answerability": answerability,
            "num_turns": len(conv.get("turns", [])),
            "query": conv["turns"][-1]["question"] if conv.get("turns") else "",
            "metadata": {
                "question_type": conv.get("question_type"),
                "multiturn_type": conv.get("multiturn_type"),
                "domain": conv.get("domain"),
                "answerability": answerability,
            },
        })

    return {
        "task_a": {
            "hybrid_full_history": {
                "per_query": per_query_a,
                "metrics": {"overall": {"ndcg@10": 0.607, "mrr": 0.698}},
            }
        },
        "task_b": {
            "Mock_standard": {
                "per_query": per_query_b,
                "metrics": {"overall": {"harmonic_mean": 0.55, "rouge_l_f1": 0.32}},
            }
        },
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="A3: Diagnostic Analysis")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", default=None, help="Path to eval_results.json from run_evaluation.py")
    parser.add_argument("--mock", action="store_true", help="Generate and analyze mock results")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    ensure_dirs(config)
    logger = setup_logger("diagnostics", config["output"]["logs_dir"])

    output_dir = args.output or config["output"]["results_dir"]
    fig_dir = config["output"]["figures_dir"]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # ---- Load eval results ----
    if args.mock or args.input is None:
        logger.info("Generating mock evaluation results for diagnostic analysis")
        eval_results = generate_mock_eval_results(n=50, seed=config["seed"])
    else:
        logger.info(f"Loading evaluation results from {args.input}")
        eval_results = load_json(args.input)

    # ---- Build unified result lists ----
    per_query_retrieval = flatten_eval_results(eval_results)
    combined_results = build_combined_results(eval_results)

    logger.info(f"Analyzing {len(per_query_retrieval)} retrieval results, "
                f"{len(combined_results)} combined results")

    diagnostic_output = {}

    # ================================================================
    # 1. Error Taxonomy
    # ================================================================
    logger.info("--- Building Error Taxonomy ---")
    taxonomy = ErrorTaxonomy()

    # Enrich per_query with generation fields if available
    gen_by_id = {}
    for key, result in eval_results.get("task_b", {}).items():
        for pq in result.get("per_query", []):
            gen_by_id[pq["conv_id"]] = pq

    enriched_results = []
    for r in per_query_retrieval:
        entry = dict(r)
        conv_id = r.get("conv_id")
        if conv_id in gen_by_id:
            gen = gen_by_id[conv_id]
            entry.update({
                "generated": gen.get("generated", ""),
                "reference": gen.get("reference", ""),
                "passages": gen.get("passages", []),
                "faithfulness_score": gen.get("faithfulness_score", 1.0),
                "answerability": gen.get("answerability", r.get("metadata", {}).get("answerability", "answerable")),
            })
        else:
            entry.setdefault("answerability", r.get("metadata", {}).get("answerability", "answerable"))
            entry.setdefault("faithfulness_score", 1.0)
            entry.setdefault("generated", "")
            entry.setdefault("reference", "")
            entry.setdefault("passages", [])
        enriched_results.append(entry)

    taxonomy.classify_dataset(enriched_results)
    failure_dist = taxonomy.get_failure_distribution()
    failure_rate = taxonomy.get_failure_rate_by_category()
    co_occurrence = taxonomy.get_co_occurrence_matrix()
    qt_breakdown = taxonomy.breakdown_by_metadata("question_type")
    mt_breakdown = taxonomy.breakdown_by_metadata("multiturn_type")
    domain_breakdown_tax = taxonomy.breakdown_by_metadata("domain")

    logger.info("\n" + taxonomy.summary_report())

    diagnostic_output["error_taxonomy"] = {
        "failure_distribution": failure_dist,
        "failure_rates": failure_rate,
        "co_occurrence": co_occurrence,
        "by_question_type": qt_breakdown,
        "by_multiturn_type": mt_breakdown,
        "by_domain": domain_breakdown_tax,
    }

    save_error_taxonomy_plot(
        failure_dist,
        os.path.join(fig_dir, "error_taxonomy.png"),
    )

    # ================================================================
    # 2. Retrieval Metric Breakdowns
    # ================================================================
    logger.info("--- Retrieval Breakdown Analysis ---")
    breakdown = full_breakdown_analysis(per_query_retrieval, k_values=[1, 3, 5, 10])
    rank_dist = rank_distribution_analysis(per_query_retrieval)

    diagnostic_output["retrieval_breakdown"] = {
        "overall": breakdown.get("overall", {}),
        "by_question_type": breakdown.get("question_type", {}),
        "by_multiturn_type": breakdown.get("multiturn_type", {}),
        "by_domain": breakdown.get("domain", {}),
        "by_answerability": breakdown.get("answerability", {}),
        "rank_distribution": rank_dist,
    }

    for field in ["question_type", "multiturn_type", "domain", "answerability"]:
        if field in breakdown:
            save_retrieval_breakdown_plot(
                breakdown[field], field,
                metric="ndcg@10",
                output_path=os.path.join(fig_dir, f"retrieval_by_{field}.png"),
            )

    # ================================================================
    # 3. Question Type Analysis (Generation)
    # ================================================================
    logger.info("--- Question Type Analysis ---")
    qt_gen_analysis = question_type_analysis(
        combined_results,
        metric_keys=["ndcg@10", "harmonic_mean", "faithfulness_score", "rouge_l_f1"],
    )
    domain_gen_analysis = domain_analysis(
        combined_results,
        metric_keys=["ndcg@10", "harmonic_mean", "faithfulness_score"],
    )

    diagnostic_output["question_type_analysis"] = qt_gen_analysis
    diagnostic_output["domain_analysis"] = domain_gen_analysis

    logger.info("Question type breakdown:")
    for qt, metrics in qt_gen_analysis.items():
        logger.info(f"  {qt}: n={metrics['count']}, "
                    f"nDCG@10={metrics.get('ndcg@10', 0):.3f}, "
                    f"HM={metrics.get('harmonic_mean', 0):.3f}")

    # ================================================================
    # 4. Retrieval-Generation Coupling
    # ================================================================
    logger.info("--- Retrieval-Generation Coupling Analysis ---")
    coupling = retrieval_generation_coupling(
        combined_results,
        retrieval_metric="ndcg@10",
        generation_metric="harmonic_mean",
    )
    diagnostic_output["retrieval_generation_coupling"] = coupling

    logger.info(f"  Pearson correlation (nDCG@10 vs HM): {coupling.get('pearson_correlation', 0):.4f}")
    logger.info(f"  Diminishing returns observed: {coupling.get('diminishing_returns_observed', False)}")

    # ================================================================
    # 5. Synthesis Requirement Analysis
    # ================================================================
    logger.info("--- Multi-Passage Synthesis Analysis ---")
    synthesis = synthesis_requirement_analysis(combined_results)
    diagnostic_output["synthesis_analysis"] = synthesis

    logger.info(f"  Single-passage: {synthesis['single_passage']['count']} convs, "
                f"mean={synthesis['single_passage']['mean_score']:.3f}")
    logger.info(f"  Multi-passage:  {synthesis['multi_passage']['count']} convs, "
                f"mean={synthesis['multi_passage']['mean_score']:.3f}")
    logger.info(f"  Performance gap: {synthesis['gap']:.3f}")

    # ================================================================
    # 6. Conversation Length Impact
    # ================================================================
    logger.info("--- Conversation Length Impact ---")
    conv_length_ret = conversation_length_analysis(combined_results, metric_key="ndcg@10")
    conv_length_gen = conversation_length_analysis(combined_results, metric_key="harmonic_mean")

    diagnostic_output["conversation_length_analysis"] = {
        "retrieval_impact": conv_length_ret,
        "generation_impact": conv_length_gen,
    }

    logger.info("Retrieval nDCG@10 by conversation length:")
    for group, stats in conv_length_ret.items():
        logger.info(f"  {group}: n={stats['count']}, mean={stats['mean']:.3f}")

    # ================================================================
    # 7. Coreference Analysis
    # ================================================================
    logger.info("--- Coreference Failure Analysis ---")
    coref_ret = analyze_coreference_impact(
        [{**r, "ndcg@10": r.get("ndcg@10", 0)} for r in combined_results],
        metric_key="ndcg@10",
    )
    coref_gen = analyze_coreference_impact(
        [{**r, "ndcg@10": r.get("harmonic_mean", 0)} for r in combined_results],
        metric_key="ndcg@10",
    )

    diagnostic_output["coreference_analysis"] = {
        "retrieval": coref_ret,
        "generation": coref_gen,
    }

    logger.info(f"  Retrieval gap (no-pronoun vs pronoun): {coref_ret.get('gap', 0):.4f}")
    logger.info(f"  Generation gap (no-pronoun vs pronoun): {coref_gen.get('gap', 0):.4f}")

    # ================================================================
    # 8. Unanswerable Detection Analysis
    # ================================================================
    logger.info("--- Unanswerable Detection ---")
    gen_results_flat = []
    for key, result in eval_results.get("task_b", {}).items():
        for pq in result.get("per_query", []):
            if "standard" in key:
                gen_results_flat.append(pq)

    if gen_results_flat:
        unans_metrics = unanswerable_detection_metrics(gen_results_flat)
        diagnostic_output["unanswerable_detection"] = unans_metrics
        logger.info(f"  Unanswerable F1: {unans_metrics['f1']:.4f}")
        logger.info(f"  Precision: {unans_metrics['precision']:.4f}, Recall: {unans_metrics['recall']:.4f}")

    # ================================================================
    # Save all diagnostic results
    # ================================================================
    save_json(diagnostic_output, os.path.join(output_dir, "diagnostic_results.json"))
    logger.info(f"\nAll diagnostic results saved to {output_dir}")
    logger.info(f"Figures saved to {fig_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("DIAGNOSTIC ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Error taxonomy: {len(failure_dist)} failure types identified")
    print(f"Top failure: {list(failure_dist.keys())[0]} ({list(failure_rate.values())[0]:.1%})")
    print(f"Retrieval-generation correlation: {coupling.get('pearson_correlation', 0):.3f}")
    print(f"Synthesis gap (single vs multi): {synthesis['gap']:.3f}")
    print(f"Coreference retrieval gap: {coref_ret.get('gap', 0):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
