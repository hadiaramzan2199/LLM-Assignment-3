# Assignment 3: Evaluation & Diagnostic Analysis
## CS-818: LLMs — MTRAGEval (SemEval 2026 Task 8)

**Team:** Hadia Ramzan, Hareem Fatima Nagra  
**Institution:** SEECS, NUST  
**Track:** A — Research-Oriented Project

---

## Overview

This assignment builds on A2 baselines to perform systematic evaluation and deep diagnostic analysis of our multi-turn RAG system. We analyze:

- **Task A (Retrieval):** BM25 / Dense / Hybrid across question types, domains, and history conditions  
- **Task B (Generation):** Faithfulness, hallucination patterns, unanswerable detection  
- **Ablations:** History window size, retrieval count, prompt variants  
- **Diagnostic experiments:** Error taxonomy, failure mode classification, retrieval-generation coupling

---

## Repository Structure

```
A3/
├── README.md
├── requirements.txt
├── environment.yml
├── configs/
│   └── default.yaml            # All experiment hyperparameters
├── scripts/
│   ├── run_evaluation.py       # Main evaluation entry point (Task A + B)
│   ├── run_ablations.py        # Ablation study runner
│   ├── run_diagnostics.py      # Error taxonomy + failure analysis
│   └── run_faithfulness.py     # Faithfulness & reasoning analysis
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py           # MTRAG dataset loader
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── retrieval_metrics.py   # nDCG, P@k, R@k, MRR, Hit Rate
│   │   ├── generation_metrics.py  # ROUGE-L, BERTScore, faithfulness
│   │   └── faithfulness.py        # NLI-based faithfulness scorer
│   ├── diagnostic/
│   │   ├── __init__.py
│   │   ├── error_taxonomy.py      # Failure classification
│   │   ├── ablation.py            # Ablation study logic
│   │   └── analysis.py            # Breakdown by question type, domain
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── test_retrieval_metrics.py
│   ├── test_generation_metrics.py
│   └── test_faithfulness.py
├── notebooks/
│   └── diagnostic_exploration.ipynb   # Exploration only (not graded)
├── artifacts/
│   ├── logs/
│   └── results/
└── reports/
    └── A3/
        └── A3_Report.pdf
```

---

## Setup

### 1. Clone and navigate
```bash
git clone https://github.com/hadiaramzan2199/LLM-Assignment-3
cd LLM-Assignment-3
```

### 2. Create conda environment
```bash
conda env create -f environment.yml
conda activate mtrag_a3
```

### 3. Install pip dependencies
```bash
pip install -r requirements.txt
```

### 4. Download MTRAG dataset
```bash
# Clone the IBM benchmark repo
git clone https://github.com/IBM/mt-rag-benchmark.git data/mt-rag-benchmark
```

### 5. Set environment variables (if needed)
```bash
cp .env.example .env
# Edit .env with your HuggingFace token for Llama/Qwen models
```

---

## Running Experiments

All experiments use fixed seed=42 and are fully reproducible.

### Full Evaluation (Task A + B)
```bash
python scripts/run_evaluation.py --config configs/default.yaml
```

### Ablation Studies
```bash
# History window ablation
python scripts/run_ablations.py --config configs/default.yaml --ablation history_window

# Retrieval count ablation
python scripts/run_ablations.py --config configs/default.yaml --ablation retrieval_k

# Prompt variant ablation
python scripts/run_ablations.py --config configs/default.yaml --ablation prompt_variant
```

### Diagnostic Analysis
```bash
python scripts/run_diagnostics.py --config configs/default.yaml --input artifacts/results/eval_results.json
```

### Faithfulness Analysis
```bash
python scripts/run_faithfulness.py --config configs/default.yaml --input artifacts/results/generation_results.json
```

---

## Reproducing Results

To reproduce all reported results from scratch:
```bash
# Step 1: Run full evaluation
python scripts/run_evaluation.py --config configs/default.yaml --output artifacts/results/eval_results.json

# Step 2: Run ablations
python scripts/run_ablations.py --config configs/default.yaml --output artifacts/results/ablation_results.json

# Step 3: Run diagnostics on evaluation output
python scripts/run_diagnostics.py --config configs/default.yaml --input artifacts/results/eval_results.json --output artifacts/results/diagnostic_results.json

# Step 4: Faithfulness analysis
python scripts/run_faithfulness.py --config configs/default.yaml --input artifacts/results/generation_results.json --output artifacts/results/faithfulness_results.json
```

Results will be written to `artifacts/results/`. Logs go to `artifacts/logs/`.

---

## Hardware

- **GPU:** NVIDIA A10G (24GB VRAM) or equivalent  
- **CUDA:** 11.8  
- **RAM:** 32GB+ recommended  
- **Python:** 3.10

---

## Key Results Summary

| System        | nDCG@10 | P@5   | R@5   | MRR   |
|---------------|---------|-------|-------|-------|
| BM25          | 0.523   | 0.482 | 0.451 | 0.612 |
| Dense         | 0.581   | 0.534 | 0.512 | 0.673 |
| Hybrid        | 0.607   | 0.561 | 0.538 | 0.698 |

| Model         | ROUGE-L | BERTScore | Faithfulness |
|---------------|---------|-----------|--------------|
| Llama-3-8B    | 0.312   | 0.867     | 0.850        |
| Qwen-2.5-7B   | 0.328   | 0.873     | 0.862        |

---

## References

- Katsis et al. (2025). MTRAG: A Multi-Turn Conversational Benchmark.
- Rosenthal et al. (2025). MTRAGEval at SemEval 2026.
