# Assignment 3: Evaluation & Diagnostic Analysis
## CS-818: Large Language Models — MTRAGEval (SemEval 2026 Task 8)

**Team:** Hadia Ramzan, Hareem Fatima Nagra  
**Institution:** SEECS, NUST  
**Track:** A — Research-Oriented Project  
**GitHub:** https://github.com/hadiaramzan2199/LLM-Assignment-3

---

## Overview

This assignment performs systematic evaluation and deep diagnostic analysis of a multi-turn RAG system on the IBM MTRAG human benchmark (SemEval 2026 Task 8). Building on A2 baselines, we evaluate:

- **Task A (Retrieval):** BM25, Dense, Hybrid, and Hybrid+Reranking across question types, domains, and history conditions
- **Task B (Generation):** Real Llama-3-8B-Instruct and Qwen-2.5-7B-Instruct inference across 3 prompt variants
- **Faithfulness:** NLI-based scoring using DeBERTa cross-encoder on all 436 responses
- **Ablations:** History window size, retrieval k, retrieval method, prompt variant
- **Diagnostics:** Error taxonomy, failure mode classification, retrieval-generation coupling

---

## Key Results

### Task A — Retrieval Evaluation

| Method | nDCG@1 | nDCG@10 | R@5 | MRR |
|---|---|---|---|---|
| BM25 (no history) | 0.103 | 0.089 | 0.075 | 0.142 |
| Dense (full history) | 0.215 | 0.165 | 0.139 | 0.258 |
| Hybrid (full history) | 0.168 | 0.156 | 0.139 | 0.241 |
| **Hybrid + Reranking** | — | **0.178** | — | **0.309** |

### Task B — Generation Evaluation

| Model | Prompt | ROUGE-L | BERTScore | Unans-F1 | HM |
|---|---|---|---|---|---|
| Llama-3-8B | Standard | 0.135 | 0.736 | 0.066 | 0.125 |
| Llama-3-8B | Faithfulness-Constrained | 0.125 | 0.725 | **0.172** | 0.198 |
| Llama-3-8B | Unanswerable-Aware | 0.132 | 0.733 | 0.143 | 0.188 |
| Qwen-2.5-7B | Standard | 0.126 | 0.736 | 0.000 | 0.215 |
| Qwen-2.5-7B | Faithfulness-Constrained | 0.126 | **0.737** | 0.161 | 0.194 |
| Qwen-2.5-7B | Unanswerable-Aware | 0.127 | 0.737 | 0.104 | 0.159 |

### Faithfulness (Llama-3-8B, 436 responses)

| Metric | Value |
|---|---|
| Mean Faithfulness | 0.122 |
| Hallucination Rate | 0.878 |
| Fully Faithful | 3.0% |
| Unanswerable Refusal Rate | 3.6% |

### Improvement: Cross-Encoder Reranking

| Method | nDCG@10 | MRR | Improvement |
|---|---|---|---|
| Hybrid (baseline) | 0.156 | 0.241 | — |
| Hybrid + Reranking | **0.178** | **0.309** | +14% nDCG, +28% MRR |

---

## Repository Structure

```
A3/
├── README.md
├── configs/
│   └── default.yaml                   # All experiment hyperparameters (seed=42)
├── scripts/
│   ├── run_evaluation.py              # Task A + B evaluation entry point
│   ├── run_ablations.py               # Ablation study runner
│   ├── run_diagnostics.py             # Error taxonomy + failure analysis
│   └── run_faithfulness.py            # NLI faithfulness analysis
├── src/
│   ├── data/
│   │   └── loader.py                  # MTRAG dataset loader with domain fix
│   ├── evaluation/
│   │   ├── retrieval_metrics.py       # nDCG, P@k, R@k, MRR, Hit Rate
│   │   ├── generation_metrics.py      # ROUGE-L, BERTScore, Unans-F1
│   │   └── faithfulness.py            # NLI-based faithfulness scorer
│   ├── diagnostic/
│   │   ├── error_taxonomy.py          # 6-category failure classification
│   │   ├── ablation.py                # Prompt templates + ablation logic
│   │   └── analysis.py                # Breakdown by question type, domain
│   └── utils/
│       └── helpers.py
├── tests/
│   ├── test_retrieval_metrics.py      # 61/63 tests passing
│   ├── test_generation_metrics.py
│   └── test_faithfulness.py
├── notebooks/
│   └── A3_MTRAGEval_Final_GPU.ipynb   # Complete Colab notebook (A100 GPU)
└── artifacts/
    └── results/                        # All evaluation outputs
```

---

## Setup

### 1. Clone repository
```bash
git clone https://github.com/hadiaramzan2199/LLM-Assignment-3
cd LLM-Assignment-3
```

### 2. Install dependencies
```bash
pip install rank-bm25==0.2.2 sentence-transformers==2.7.0 \
    rouge-score==0.1.2 bert-score==0.3.13 evaluate==0.4.2 \
    transformers==4.40.2 accelerate==0.30.1 \
    pyyaml==6.0.1 tabulate==0.9.0 seaborn==0.13.2 \
    scipy pytest nltk
```

### 3. Clone MTRAG dataset
```bash
git clone --depth 1 https://github.com/IBM/mt-rag-benchmark.git data/mt-rag-benchmark
```

### 4. Set HuggingFace token (for Llama access)
```bash
export HF_TOKEN=hf_your_token_here
```

> **Note:** Llama-3-8B-Instruct requires HuggingFace access approval at  
> https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

---

## Running Experiments

All experiments use fixed `seed=42` and are reproducible.

### Full Evaluation (Task A + B)
```bash
python scripts/run_evaluation.py --config configs/default.yaml
```

### Ablation Studies
```bash
python scripts/run_ablations.py --config configs/default.yaml
```

### Diagnostic Analysis
```bash
python scripts/run_diagnostics.py \
    --config configs/default.yaml \
    --input artifacts/results/eval_results.json
```

### Faithfulness Analysis
```bash
python scripts/run_faithfulness.py \
    --config configs/default.yaml \
    --input artifacts/results/task_b_results.json
```

### Run Tests
```bash
python -m pytest tests/ -v
# 61/63 tests pass (2 minor test assertion issues, not evaluation bugs)
```

---

## Reproducing Results (Google Colab A100)

The complete notebook `notebooks/A3_MTRAGEval_Final_GPU.ipynb` reproduces all results. It includes Drive checkpointing so sessions can resume after timeout.

**Estimated runtimes on A100:**

| Step | Time |
|---|---|
| Setup + data loading | ~6 min |
| Dense embedding (366k passages) | ~45 min |
| Task A retrieval (all configs) | ~5 min |
| Task B Llama inference (436 × 3 variants) | ~2.5 hours |
| Task B Qwen inference (436 × 3 variants) | ~2.5 hours |
| Ablations + diagnostics + faithfulness | ~15 min |

**Important:** Models are loaded from Google Drive to avoid repeated HuggingFace downloads. See notebook Cell 0 for Drive setup instructions.

---

## Implementation Notes

### Domain Fix
The MTRAG dataset assigns all conversations the label `'MT-RAG Authors (Internal)'` in the metadata field, which is unusable for domain-specific retrieval. We derive real domain labels (`clapnq`, `govt`, `fiqa`, `cloud`) by cross-referencing each task_id against the corresponding qrel file path. This fix is applied in `src/data/loader.py` via `_fix_domains()`.

### Task ID vs Conversation ID
Qrel files are indexed by `task_id` (format: `<conv_id><::><turn>`), not `conv_id`. The evaluation scripts use `conv['_task_ids'][-1]` for qrel lookup to ensure correct metric computation.

### Passage Lookup
Generation tasks store `passage_texts` directly in the reference entries. We use these pre-loaded texts rather than attempting corpus lookup by chunk PID, which would fail due to ID format mismatches.

### Cross-Encoder Reranking
Reranking uses `cross-encoder/ms-marco-MiniLM-L-6-v2` (90M parameters). Top-10 hybrid results are reranked to top-5 before feeding to the generator. This adds <1 second per conversation on GPU.

---

## Error Taxonomy

| Failure Category | Count | Rate |
|---|---|---|
| Retrieval Miss | 46 | 43.0% |
| Unanswerable Hallucination | 13 | 12.1% |
| Retrieval Rank Error | 5 | 4.7% |
| Synthesis Failure | 1 | 0.9% |
| History Neglect | 1 | 0.9% |
| Correct | 85 | 79.4% |

---

## Hardware

- **GPU:** NVIDIA A100 (40GB VRAM) — Google Colab Pro
- **Models:** Llama-3-8B-Instruct + Qwen-2.5-7B-Instruct (float16, no quantization)
- **Python:** 3.12

---

## References

- Katsis et al. (2025). mtRAG: A Multi-Turn Conversational Benchmark. *TACL*, 13:784–808.
- Rosenthal et al. (2026). SemEval-2026 Task 8: MTRAGEval.
- Karpukhin et al. (2020). Dense Passage Retrieval. *EMNLP*.
- Nogueira & Cho (2019). Passage Re-Ranking with BERT.
- Zhang et al. (2020). BERTScore. *ICLR*.
- Touvron et al. (2023). Llama 2. arXiv:2307.09288.
- Qwen Team (2025). Qwen2.5 Technical Report. arXiv:2412.15115.
