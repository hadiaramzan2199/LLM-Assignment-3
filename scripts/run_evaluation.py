"""
scripts/run_evaluation.py
"""

import argparse
import os
import sys
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from src.utils import (
    set_seed, load_config, setup_logger, save_json,
    Timer, ensure_dirs, load_json,
)
from src.evaluation.retrieval_metrics import (
    full_breakdown_analysis, rank_distribution_analysis,
)
from src.evaluation.generation_metrics import (
    batch_rouge_l, harmonic_mean,
    unanswerable_detection_metrics,
)
from src.evaluation.faithfulness import (
    NLIFaithfulnessScorer, analyze_coreference_impact,
)


# ---------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------

def build_bm25_indexes(corpus: dict, domains: list) -> dict:
    """Build BM25 indexes per domain."""
    from rank_bm25 import BM25Okapi
    indexes = {}
    for domain in domains:
        dom  = {pid: p for pid, p in corpus.items() if p.get('domain') == domain}
        if not dom:
            continue
        pids  = list(dom.keys())
        texts = [dom[pid]['text'].lower().split() for pid in pids]
        indexes[domain] = (BM25Okapi(texts, k1=1.5, b=0.75), pids)
        print(f'  {domain}: {len(pids):,} passages indexed')
    return indexes


def build_dense_indexes(corpus: dict, domains: list, model, device: str) -> dict:
    """Build dense embedding indexes per domain."""
    indexes = {}
    for domain in domains:
        dom  = {pid: p for pid, p in corpus.items() if p.get('domain') == domain}
        if not dom:
            continue
        pids  = list(dom.keys())
        texts = [dom[pid]['text'] for pid in pids]
        print(f'  {domain}: encoding {len(pids):,} passages...')
        embs = model.encode(
            texts, batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        indexes[domain] = (embs, pids)
    return indexes


def bm25_retrieve(query: str, domain: str, indexes: dict, top_k: int = 10):
    if domain not in indexes:
        return []
    bm25, pids = indexes[domain]
    scores = bm25.get_scores(query.lower().split())
    return [pids[i] for i in np.argsort(-scores)[:top_k]]


def dense_retrieve(query: str, domain: str, indexes: dict, model, top_k: int = 10):
    if domain not in indexes:
        return []
    embs, pids = indexes[domain]
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    return [pids[i] for i in np.argsort(-(embs @ q_emb))[:top_k]]


def hybrid_retrieve(query: str, domain: str, bm25_idx: dict, dense_idx: dict,
                    model, top_k: int = 10, bw: float = 0.4, dw: float = 0.6):
    if domain not in bm25_idx or domain not in dense_idx:
        return []
    bm25, pids  = bm25_idx[domain]
    embs, _     = dense_idx[domain]
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_norm   = bm25_scores / (bm25_scores.max() + 1e-9)
    q_emb       = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    combined    = bw * bm25_norm + dw * (embs @ q_emb)
    return [pids[i] for i in np.argsort(-combined)[:top_k]]


def rerank(query: str, retrieved_pids: list, corpus: dict,
           reranker, domain: str, top_k: int = 5):
    """Cross-encoder reranking of retrieved passages."""
    if not retrieved_pids:
        return []
    passages = [(pid, corpus[pid]['text'])
                for pid in retrieved_pids if pid in corpus]
    if not passages:
        return retrieved_pids[:top_k]
    pairs  = [[query, text] for pid, text in passages]
    scores = reranker.predict(pairs)
    ranked = sorted(zip([pid for pid, _ in passages], scores),
                    key=lambda x: -x[1])
    return [pid for pid, _ in ranked[:top_k]]


def build_query(conv: dict, mode: str) -> str:
    """Build retrieval query from conversation history."""
    turns = conv.get('turns', [])
    if not turns:
        return ''
    q     = turns[-1].get('question', '')
    prior = turns[:-1]
    if mode == 'no_history':
        return q
    if mode == 'last_turn_only':
        prior = prior[-1:]
    elif mode.startswith('window_'):
        prior = prior[-int(mode.split('_')[1]):]
    hist = '\n'.join(
        f"Q: {t.get('question','')} A: {t.get('answer','')}" for t in prior
    )
    return f'{hist}\nQ: {q}'.strip() if hist else q


# ---------------------------------------------------------------
# Task A evaluation
# ---------------------------------------------------------------

def evaluate_task_a(
    conversations, corpus, qrels, config, logger,
    bm25_indexes=None, dense_indexes=None, dense_model=None,
    history_mode='no_history', retrieval_method='bm25',
    reranker=None, top_k=10,
):
    """
    Task A retrieval evaluation.
    Uses task_id (not conv_id) for qrel lookup — critical fix.
    """
    logger.info(f'[Task A] {retrieval_method} | history={history_mode} | k={top_k}')
    results = []

    for conv in conversations:
        conv_id  = conv['id']
        domain   = conv.get('domain', 'clapnq')
        task_ids = conv.get('_task_ids', [])
        # KEY FIX: use task_id, not conv_id
        task_id  = task_ids[-1] if task_ids else conv_id
        conv_qrels = qrels.get(task_id, {})
        query    = build_query(conv, history_mode)

        if retrieval_method == 'bm25' and bm25_indexes:
            retrieved = bm25_retrieve(query, domain, bm25_indexes, top_k)
        elif retrieval_method == 'dense' and dense_indexes and dense_model:
            retrieved = dense_retrieve(query, domain, dense_indexes, dense_model, top_k)
        elif retrieval_method == 'hybrid' and bm25_indexes and dense_indexes and dense_model:
            retrieved = hybrid_retrieve(query, domain, bm25_indexes, dense_indexes,
                                        dense_model, top_k)
        elif retrieval_method == 'hybrid_reranked' and reranker:
            retrieved_raw = hybrid_retrieve(query, domain, bm25_indexes, dense_indexes,
                                            dense_model, top_k=10)
            retrieved = rerank(query, retrieved_raw, corpus, reranker, domain, top_k=5)
        else:
            retrieved = bm25_retrieve(query, domain, bm25_indexes or {}, top_k)

        results.append({
            'conv_id':   conv_id,
            'task_id':   task_id,
            'retrieved': retrieved,
            'qrels':     conv_qrels,
            'query':     query,
            'num_turns': len(conv.get('turns', [])),
            'metadata':  {
                'question_type':    conv.get('question_type',  'unknown'),
                'multiturn_type':   conv.get('multiturn_type', 'unknown'),
                'domain':           domain,
                'answerability':    conv.get('answerability',  'unknown'),
                'history_mode':     history_mode,
                'retrieval_method': retrieval_method,
            },
        })

    k_values = config['evaluation']['task_a']['k_values']
    metrics  = full_breakdown_analysis(results, k_values)

    return {
        'config':  {'history_mode': history_mode, 'retrieval_method': retrieval_method,
                    'top_k': top_k},
        'metrics': metrics,
        'rank_distribution': rank_distribution_analysis(results),
        'per_query': results,
    }


# ---------------------------------------------------------------
# Task B evaluation
# ---------------------------------------------------------------

def load_model(hf_id: str, hf_token: str = None):
    """Load LLM in float16 (no quantization needed on A100)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f'Loading {hf_id}...')
    model = AutoModelForCausalLM.from_pretrained(
        hf_id,
        torch_dtype=torch.float16,
        device_map='auto',
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    if torch.cuda.is_available():
        print(f'Loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB')
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str,
                      max_new_tokens: int = 200) -> str:
    """Generate response with truncation to avoid OOM."""
    inputs = tokenizer(
        prompt, return_tensors='pt',
        truncation=True, max_length=1024,
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def evaluate_task_b(
    references, config, logger,
    model=None, tokenizer=None,
    model_name='unknown', prompt_variant='standard',
    faithfulness_scorer=None, top_k_passages=5,
):
    """
    Task B generation evaluation.
    Uses passage_texts from references directly — avoids corpus PID mismatch.
    """
    from src.diagnostic.ablation import format_prompt
    logger.info(f'[Task B] {model_name} | prompt={prompt_variant}')

    per_query, hyps, refs_list = [], [], []

    for task_id, ref in references.items():
        ans          = ref.get('answerability', 'answerable')
        ref_txt      = ref.get('reference', '')
        # KEY FIX: use pre-loaded passage_texts, not corpus lookup
        passage_texts = ref.get('passage_texts', [])[:top_k_passages]
        input_turns  = ref.get('input_turns', [])
        final_q      = input_turns[-1].get('text', '') if input_turns else ''
        hist         = '\n'.join(
            f"{m.get('speaker','')}: {m.get('text','')}"
            for m in input_turns[:-1]
        )

        prompt    = format_prompt(prompt_variant, hist, passage_texts, final_q)
        generated = generate_response(model, tokenizer, prompt) if model else ''

        # Optional faithfulness scoring
        faith_score = 1.0
        if faithfulness_scorer and passage_texts and generated:
            try:
                r = faithfulness_scorer.score_response(generated, passage_texts)
                faith_score = r.get('faithfulness_score', 1.0)
            except Exception:
                pass

        hyps.append(generated)
        refs_list.append(ref_txt)
        per_query.append({
            'task_id':           task_id,
            'generated':         generated,
            'reference':         ref_txt,
            'passages':          passage_texts,
            'faithfulness_score': faith_score,
            'answerability':     ans,
            'metadata': {
                'question_type':  ref.get('question_type',  'unknown'),
                'multiturn_type': ref.get('multiturn_type', 'unknown'),
                'domain':         ref.get('domain',         'unknown'),
                'answerability':  ans,
                'model':          model_name,
                'prompt_variant': prompt_variant,
            },
        })

    # Metrics
    rouge_scores = batch_rouge_l(hyps, refs_list)
    rl_mean      = float(np.mean([s['f'] for s in rouge_scores])) if rouge_scores else 0.0

    # BERTScore
    try:
        from bert_score import score as bert_score_fn
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _, _, F = bert_score_fn(
            hyps, refs_list,
            model_type='distilbert-base-uncased',
            device=device, verbose=False,
        )
        bs_mean = float(F.mean())
    except Exception:
        bs_mean = rl_mean  # fallback proxy

    unans_m = unanswerable_detection_metrics(per_query)
    hm = (3 / (1/rl_mean + 1/bs_mean + 1/unans_m['f1'])
          if rl_mean > 0 and bs_mean > 0 and unans_m['f1'] > 0
          else (2 / (1/rl_mean + 1/bs_mean) if rl_mean > 0 and bs_mean > 0 else 0.0))

    return {
        'config':  {'model': model_name, 'prompt_variant': prompt_variant,
                    'top_k_passages': top_k_passages},
        'metrics': {'overall': {
            'rouge_l_f1':      rl_mean,
            'bertscore_f1':    bs_mean,
            'harmonic_mean':   hm,
            'unanswerable_f1': unans_m['f1'],
        }},
        'unanswerable_detection': unans_m,
        'per_query': per_query,
    }


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--task',   choices=['A', 'B', 'both'], default='both')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config['seed'])
    ensure_dirs(config)
    logger = setup_logger('evaluation', config['output']['logs_dir'])

    output_dir = args.output or config['output']['results_dir']
    os.makedirs(output_dir, exist_ok=True)

    from src.data.loader import MTRAGDataLoader
    loader        = MTRAGDataLoader(config, split='val')
    conversations = loader.conversations
    corpus        = loader.corpus
    qrels         = loader.qrels
    references    = loader.references

    HF_TOKEN = os.environ.get('HF_TOKEN')
    DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_results = {}

    if args.task in ('A', 'both'):
        logger.info('=== TASK A ===')
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer, CrossEncoder

        print('Building BM25 indexes...')
        bm25_indexes = build_bm25_indexes(corpus, loader.DOMAINS)

        print('Building dense indexes...')
        dense_model  = SentenceTransformer(
            'sentence-transformers/all-mpnet-base-v2', device=DEVICE
        )
        dense_indexes = build_dense_indexes(corpus, loader.DOMAINS, dense_model, DEVICE)

        print('Loading reranker...')
        reranker = CrossEncoder(
            'cross-encoder/ms-marco-MiniLM-L-6-v2', device=DEVICE
        )

        task_a = {}
        configs = [
            ('bm25',           'no_history'),
            ('bm25',           'full_history'),
            ('dense',          'full_history'),
            ('hybrid',         'full_history'),
            ('hybrid_reranked','no_history'),
        ]
        for method, mode in configs:
            key = f'{method}_{mode}'
            with Timer(key):
                res = evaluate_task_a(
                    conversations, corpus, qrels, config, logger,
                    bm25_indexes=bm25_indexes,
                    dense_indexes=dense_indexes,
                    dense_model=dense_model,
                    reranker=reranker,
                    history_mode=mode,
                    retrieval_method=method,
                )
            task_a[key] = res
            m = res['metrics']['overall']
            logger.info(f'  {key}: nDCG@10={m.get("ndcg@10",0):.4f}  MRR={m.get("mrr",0):.4f}')

        all_results['task_a'] = task_a
        save_json(task_a, os.path.join(output_dir, 'task_a_results.json'))

    if args.task in ('B', 'both'):
        logger.info('=== TASK B ===')
        faith_scorer = NLIFaithfulnessScorer(
            model_name='cross-encoder/nli-deberta-v3-base',
            device=DEVICE,
        )

        MODELS = [
            ('Llama-3-8B-Instruct',  'meta-llama/Meta-Llama-3-8B-Instruct'),
            ('Qwen-2.5-7B-Instruct', 'Qwen/Qwen2.5-7B-Instruct'),
        ]
        VARIANTS = ['standard', 'faithfulness_constrained', 'unanswerable_aware']
        task_b = {}

        for model_name, hf_id in MODELS:
            model, tokenizer = load_model(hf_id, HF_TOKEN)
            for variant in VARIANTS:
                key = f'{model_name}_{variant}'
                with Timer(key):
                    res = evaluate_task_b(
                        references, config, logger,
                        model=model, tokenizer=tokenizer,
                        model_name=model_name, prompt_variant=variant,
                        faithfulness_scorer=faith_scorer,
                    )
                task_b[key] = res
                m = res['metrics']['overall']
                logger.info(
                    f'  {key}: ROUGE-L={m.get("rouge_l_f1",0):.4f}  '
                    f'BS={m.get("bertscore_f1",0):.4f}  '
                    f'HM={m.get("harmonic_mean",0):.4f}'
                )
                save_json(task_b, os.path.join(output_dir, 'task_b_results.json'))

            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        all_results['task_b'] = task_b
        save_json(task_b, os.path.join(output_dir, 'task_b_results.json'))

    save_json(all_results, os.path.join(output_dir, 'eval_results.json'))
    logger.info(f'All results saved to {output_dir}')


if __name__ == '__main__':
    main()
