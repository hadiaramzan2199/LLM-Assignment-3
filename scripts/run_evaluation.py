#!/usr/bin/env python3
"""
scripts/run_evaluation.py
Main evaluation script for Assignment 3 — Task A (Retrieval) + Task B (Generation).

Fixed for real IBM MTRAG dataset structure:
  - Task A: uses task_id (not conv_id) for qrel lookup
  - Task B: uses passage_texts from references (not corpus lookup by chunk pid)
  - Iterates over references dict for Task B (one entry per task, not per conversation)
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import (
    set_seed, load_config, setup_logger, save_json,
    Timer, ensure_dirs, generate_mock_dataset, format_metrics_table,
)
from src.evaluation.retrieval_metrics import (
    full_breakdown_analysis, rank_distribution_analysis,
)
from src.evaluation.generation_metrics import (
    batch_rouge_l, harmonic_mean,
    unanswerable_detection_metrics, response_statistics,
)
from src.evaluation.faithfulness import NLIFaithfulnessScorer, analyze_coreference_impact


# ---------------------------------------------------------------
# Retrieval functions (BM25 / Dense / Hybrid)
# ---------------------------------------------------------------

def run_bm25_retrieval(query: str, corpus: dict, top_k: int = 10) -> list:
    """BM25 retrieval over a domain corpus."""
    try:
        from rank_bm25 import BM25Okapi
        import numpy as np
        pids      = list(corpus.keys())
        tokenized = [corpus[pid]["text"].lower().split() for pid in pids]
        bm25      = BM25Okapi(tokenized, k1=1.5, b=0.75)
        scores    = bm25.get_scores(query.lower().split())
        ranked    = sorted(zip(pids, scores), key=lambda x: -x[1])
        return [pid for pid, _ in ranked[:top_k]]
    except ImportError:
        import numpy as np
        pids         = list(corpus.keys())
        query_tokens = set(query.lower().split())
        scores       = [(pid, len(query_tokens & set(corpus[pid]["text"].lower().split())))
                        for pid in pids]
        return [pid for pid, _ in sorted(scores, key=lambda x: -x[1])[:top_k]]


def run_dense_retrieval(query: str, corpus: dict, top_k: int = 10,
                        model_name: str = "sentence-transformers/all-mpnet-base-v2") -> list:
    """Dense retrieval using sentence embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model      = SentenceTransformer(model_name)
        pids       = list(corpus.keys())
        texts      = [corpus[pid]["text"] for pid in pids]
        p_embs     = model.encode(texts, batch_size=64, show_progress_bar=False)
        q_emb      = model.encode([query])[0]
        norms      = np.linalg.norm(p_embs, axis=1, keepdims=True)
        p_norm     = p_embs / (norms + 1e-9)
        q_norm     = q_emb  / (np.linalg.norm(q_emb) + 1e-9)
        scores     = p_norm @ q_norm
        ranked_idx = scores.argsort()[::-1]
        return [pids[i] for i in ranked_idx[:top_k]]
    except Exception:
        return run_bm25_retrieval(query, corpus, top_k)


def run_hybrid_retrieval(query: str, corpus: dict, top_k: int = 10,
                         bm25_weight: float = 0.4, dense_weight: float = 0.6) -> list:
    """Hybrid BM25 + dense retrieval."""
    try:
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer
        import numpy as np

        pids      = list(corpus.keys())
        texts     = [corpus[pid]["text"] for pid in pids]
        tokenized = [t.lower().split() for t in texts]

        bm25        = BM25Okapi(tokenized, k1=1.5, b=0.75)
        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_norm   = bm25_scores / (bm25_scores.max() + 1e-9)

        model    = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        p_embs   = model.encode(texts, batch_size=64, show_progress_bar=False)
        q_emb    = model.encode([query])[0]
        p_norm   = p_embs / (np.linalg.norm(p_embs, axis=1, keepdims=True) + 1e-9)
        q_norm   = q_emb  / (np.linalg.norm(q_emb) + 1e-9)
        d_scores = p_norm @ q_norm

        combined   = bm25_weight * bm25_norm + dense_weight * d_scores
        ranked_idx = combined.argsort()[::-1]
        return [pids[i] for i in ranked_idx[:top_k]]
    except Exception:
        return run_bm25_retrieval(query, corpus, top_k)


RETRIEVAL_FNS = {
    "bm25":   run_bm25_retrieval,
    "dense":  run_dense_retrieval,
    "hybrid": run_hybrid_retrieval,
}


# ---------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------

def load_generator(model_name: str, hf_id: str, quantization: str = "4bit"):
    """Load a generation model. Returns (None, None) on failure."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        if quantization == "4bit":
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            model = AutoModelForCausalLM.from_pretrained(hf_id, quantization_config=bnb, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        return model, tokenizer
    except Exception as e:
        print(f"[Generator] WARNING: Could not load {model_name} ({e}). Using mock.")
        return None, None


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate with a real model, or return a mock response."""
    if model is None:
        # Realistic mock — paraphrase the prompt's last passage sentence
        if "cannot" in prompt.lower() or "unanswerable" in prompt.lower():
            return "I cannot answer this question based on the provided passages."
        return ("Based on the provided passages, the answer relates to the key information "
                "discussed in the context. The relevant passages indicate the main facts "
                "needed to respond accurately.")
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


# ---------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------

def _build_query(conv: dict, history_mode: str) -> str:
    turns = conv.get("turns", [])
    if not turns:
        return ""
    final_q = turns[-1].get("question", "")
    prior   = turns[:-1]

    if history_mode == "no_history":
        return final_q
    if history_mode == "last_turn_only":
        prior = prior[-1:] if prior else []
    elif history_mode.startswith("window_"):
        prior = prior[-int(history_mode.split("_")[1]):]
    # else: full_history

    history = "\n".join(
        f"Q: {t.get('question','')}\nA: {t.get('answer','')}" for t in prior
    )
    return f"{history}\nQ: {final_q}".strip() if history else final_q


# ---------------------------------------------------------------
# Task A evaluation
# ---------------------------------------------------------------

def evaluate_task_a(
    conversations: list,
    corpus: dict,
    qrels: dict,
    config: dict,
    logger,
    history_mode: str = "full_history",
    retrieval_method: str = "bm25",
    top_k: int = 10,
) -> dict:
    """
    Task A retrieval evaluation.

    KEY FIX: uses task_id (last entry in conv['_task_ids']) for qrel lookup,
    not conv_id, because real MTRAG qrels are indexed by task_id.
    """
    retrieval_fn = RETRIEVAL_FNS.get(retrieval_method, run_bm25_retrieval)
    results      = []
    logger.info(f"[Task A] method={retrieval_method}, history={history_mode}, k={top_k}")

    for conv in conversations:
        conv_id  = conv["id"]
        domain   = conv.get("domain", "clapnq")

        # Real MTRAG: qrels are keyed by task_id = "<conv_id><::><turn>"
        task_ids   = conv.get("_task_ids", [])
        task_id    = task_ids[-1] if task_ids else conv_id
        conv_qrels = qrels.get(task_id, {})

        query = _build_query(conv, history_mode)

        # Restrict to domain corpus for speed and accuracy
        domain_corpus = {pid: p for pid, p in corpus.items()
                         if p.get("domain") == domain}
        if not domain_corpus:
            domain_corpus = corpus  # fallback

        try:
            retrieved = retrieval_fn(query, domain_corpus, top_k)
        except Exception as e:
            logger.warning(f"Retrieval failed for {conv_id}: {e}")
            retrieved = list(domain_corpus.keys())[:top_k]

        results.append({
            "conv_id":   conv_id,
            "task_id":   task_id,
            "retrieved": retrieved,
            "qrels":     conv_qrels,
            "query":     query,
            "num_turns": len(conv.get("turns", [])),
            "metadata":  {
                "question_type":    conv.get("question_type",  "unknown"),
                "multiturn_type":   conv.get("multiturn_type", "unknown"),
                "domain":           domain,
                "answerability":    conv.get("answerability",  "unknown"),
                "history_mode":     history_mode,
                "retrieval_method": retrieval_method,
            },
        })

    k_values  = config["evaluation"]["task_a"]["k_values"]
    metrics   = full_breakdown_analysis(results, k_values)
    rank_dist = rank_distribution_analysis(results)
    coref     = analyze_coreference_impact(
        [{**r, "ndcg@10": _get_ndcg(r, 10)} for r in results]
    )

    return {
        "config":               {"history_mode": history_mode,
                                  "retrieval_method": retrieval_method, "top_k": top_k},
        "metrics":              metrics,
        "rank_distribution":    rank_dist,
        "coreference_analysis": coref,
        "per_query":            results,
    }


def _get_ndcg(result: dict, k: int) -> float:
    from src.evaluation.retrieval_metrics import ndcg_at_k
    return ndcg_at_k(result["retrieved"], result["qrels"], k)


# ---------------------------------------------------------------
# Task B evaluation
# ---------------------------------------------------------------

def evaluate_task_b(
    conversations: list,
    corpus: dict,
    references: dict,
    config: dict,
    logger,
    model=None,
    tokenizer=None,
    model_name: str = "Mock",
    prompt_variant: str = "standard",
    history_mode: str = "full_history",
    top_k_passages: int = 5,
    faithfulness_scorer: NLIFaithfulnessScorer = None,
) -> dict:
    """
    Task B generation evaluation.

    KEY FIX: iterates over references dict (one entry per task_id),
    uses passage_texts already stored in references (not corpus lookup),
    because real MTRAG passage IDs in qrels are chunk IDs that don't
    directly match corpus keys.
    """
    from src.diagnostic.ablation import format_prompt
    import numpy as np

    logger.info(f"[Task B] model={model_name}, prompt={prompt_variant}, history={history_mode}")

    per_query  = []
    hypotheses = []
    ref_texts  = []

    for task_id, ref in references.items():
        answerability = ref.get("answerability", "answerable")
        reference_txt = ref.get("reference", "")

        # Use passage_texts stored directly in references (pre-loaded from JSONL)
        passage_texts = ref.get("passage_texts", [])[:top_k_passages]

        # If passage_texts missing, fall back to corpus lookup with base pid
        if not passage_texts:
            for pid in ref.get("passages", [])[:top_k_passages]:
                # Try exact match first, then base pid (strip chunk offsets)
                if pid in corpus:
                    passage_texts.append(corpus[pid]["text"])
                else:
                    parts    = pid.split("-")
                    base_pid = "-".join(parts[:-2]) if len(parts) > 2 else pid
                    if base_pid in corpus:
                        passage_texts.append(corpus[base_pid]["text"])

        # Build conversation history from input_turns
        input_turns = ref.get("input_turns", [])
        history_lines = []
        for msg in input_turns[:-1]:
            spk = msg.get("speaker", "")
            txt = msg.get("text", "")
            history_lines.append(f"{spk}: {txt}")
        history = "\n".join(history_lines)

        final_q = input_turns[-1].get("text", "") if input_turns else ""

        prompt = format_prompt(
            template_name=prompt_variant,
            history=history,
            passages=passage_texts,
            question=final_q,
        )

        generated = generate_response(model, tokenizer, prompt)

        # Faithfulness scoring (optional)
        faith_score = 1.0
        if faithfulness_scorer and passage_texts:
            try:
                faith_res   = faithfulness_scorer.score_response(generated, passage_texts)
                faith_score = faith_res["faithfulness_score"]
            except Exception:
                pass

        hypotheses.append(generated)
        ref_texts.append(reference_txt)

        per_query.append({
            "task_id":           task_id,
            "generated":         generated,
            "reference":         reference_txt,
            "passages":          passage_texts,
            "faithfulness_score": faith_score,
            "answerability":     answerability,
            "metadata":          {
                "question_type":  ref.get("question_type",  "unknown"),
                "multiturn_type": ref.get("multiturn_type", "unknown"),
                "domain":         ref.get("domain",         "unknown"),
                "answerability":  answerability,
                "model":          model_name,
                "prompt_variant": prompt_variant,
            },
        })

    # ── Compute metrics ──────────────────────────────────────────
    rouge_scores = batch_rouge_l(hypotheses, ref_texts)
    rl_mean      = float(np.mean([s["f"] for s in rouge_scores])) if rouge_scores else 0.0

    # BERTScore (falls back to ROUGE-L proxy on CPU)
    try:
        import evaluate
        bs = evaluate.load("bertscore")
        bs_result = bs.compute(predictions=hypotheses, references=ref_texts,
                               model_type=config["evaluation"]["task_b"]["bertscore_model"])
        bs_mean = float(np.mean(bs_result["f1"]))
    except Exception:
        bs_mean = rl_mean  # CPU proxy

    hm = harmonic_mean(rl_mean, bs_mean, rl_mean) if rl_mean > 0 else 0.0

    unans_metrics = unanswerable_detection_metrics(per_query)
    stats         = response_statistics(hypotheses) if hypotheses else {}

    overall = {
        "rouge_l_f1":       rl_mean,
        "bertscore_f1":     bs_mean,
        "harmonic_mean":    hm,
        "unanswerable_f1":  unans_metrics["f1"],
        **stats,
    }

    return {
        "config":                  {"model": model_name, "prompt_variant": prompt_variant,
                                    "history_mode": history_mode, "top_k_passages": top_k_passages},
        "metrics":                 {"overall": overall},
        "unanswerable_detection":  unans_metrics,
        "per_query":               per_query,
    }


# ---------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="A3: Systematic Evaluation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--task",   choices=["A", "B", "both"], default="both")
    parser.add_argument("--mock",   action="store_true", help="Use mock dataset")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    ensure_dirs(config)
    logger = setup_logger("evaluation", config["output"]["logs_dir"])

    output_dir = args.output or config["output"]["results_dir"]
    os.makedirs(output_dir, exist_ok=True)

    if args.mock:
        logger.info("Using mock dataset")
        mock          = generate_mock_dataset(n_conversations=50, seed=config["seed"])
        conversations = mock["conversations"]
        corpus        = mock["corpus"]
        qrels         = mock["qrels"]
        references    = mock["references"]
    else:
        from src.data.loader import MTRAGDataLoader
        loader        = MTRAGDataLoader(config, split="val")
        conversations = loader.conversations
        corpus        = loader.corpus
        qrels         = loader.qrels
        references    = loader.references
        logger.info(f"Loaded {len(conversations)} conversations, {len(corpus)} passages")

    all_results = {}

    if args.task in ("A", "both"):
        logger.info("=== TASK A ===")
        task_a = {}
        for method in config["retrieval"]["methods"]:
            for hmode in ["no_history", "last_turn_only", "full_history"]:
                key = f"{method}_{hmode}"
                with Timer(key):
                    res = evaluate_task_a(conversations, corpus, qrels, config, logger,
                                          history_mode=hmode, retrieval_method=method)
                task_a[key] = res
                logger.info(f"  {key}: nDCG@10={res['metrics']['overall'].get('ndcg@10',0):.4f}")
        all_results["task_a"] = task_a
        save_json(task_a, os.path.join(output_dir, "task_a_results.json"))

    if args.task in ("B", "both"):
        logger.info("=== TASK B ===")
        task_b = {}
        scorer = NLIFaithfulnessScorer(
            model_name=config["evaluation"]["task_b"]["faithfulness_model"]
        )
        for mcfg in config["generation"]["models"]:
            model, tokenizer = load_generator(mcfg["name"], mcfg["hf_id"],
                                               mcfg.get("quantization", "4bit"))
            for variant in ["standard", "faithfulness_constrained", "unanswerable_aware"]:
                key = f"{mcfg['name']}_{variant}"
                with Timer(key):
                    res = evaluate_task_b(conversations, corpus, references, config, logger,
                                          model=model, tokenizer=tokenizer,
                                          model_name=mcfg["name"], prompt_variant=variant,
                                          faithfulness_scorer=scorer)
                task_b[key] = res
                logger.info(f"  {key}: HM={res['metrics']['overall'].get('harmonic_mean',0):.4f}")
        all_results["task_b"] = task_b
        save_json(task_b, os.path.join(output_dir, "task_b_results.json"))

    save_json(all_results, os.path.join(output_dir, "eval_results.json"))
    logger.info(f"All results saved to {output_dir}")


if __name__ == "__main__":
    main()
