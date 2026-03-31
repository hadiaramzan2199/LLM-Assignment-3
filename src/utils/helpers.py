"""
src/utils/helpers.py
Shared utility functions for A3: config loading, logging, JSON I/O, seeding.
"""

import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


# ---------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def merge_config(base: dict, overrides: dict) -> dict:
    """Recursively merge override dict into base config."""
    result = base.copy()
    for k, v in overrides.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = merge_config(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------
# Logging
# ---------------------------------------------------------------

def setup_logger(name: str, log_dir: str = "artifacts/logs", level=logging.INFO) -> logging.Logger:
    """Set up a logger that writes to both console and file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", "%H:%M:%S"))
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s"))
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------

def save_json(data: Any, path: str, indent: int = 2):
    """Save data to JSON file, creating parent directories if needed."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)
    print(f"[IO] Saved: {path}")


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def save_jsonl(data: List[dict], path: str):
    """Save list of dicts as JSONL."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, default=str) + "\n")
    print(f"[IO] Saved JSONL: {path}")


def load_jsonl(path: str) -> List[dict]:
    """Load JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------
# Timing
# ---------------------------------------------------------------

class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        label = f" [{self.name}]" if self.name else ""
        print(f"[Timer]{label} {self.elapsed:.2f}s")


# ---------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------

def format_metrics_table(metrics: dict, title: str = "") -> str:
    """Format a flat metrics dict as a table string."""
    lines = []
    if title:
        lines += ["=" * 50, title, "=" * 50]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:<30} {v:.4f}")
        else:
            lines.append(f"  {k:<30} {v}")
    return "\n".join(lines)


def ensure_dirs(config: dict):
    """Create all output directories from config."""
    for key in ["results_dir", "logs_dir", "figures_dir"]:
        path = config.get("output", {}).get(key)
        if path:
            os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------
# Mock data generator (for Colab testing without full dataset)
# ---------------------------------------------------------------

def generate_mock_dataset(
    n_conversations: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate a small mock MTRAG-format dataset for testing pipelines
    without access to the full IBM benchmark.

    Returns:
      {
        "conversations": [...],
        "corpus": {...},
        "qrels": {...},
        "references": {...}
      }
    """
    set_seed(seed)

    question_types = ["factoid", "explanatory", "comparative", "unanswerable"]
    multiturn_types = ["follow_up", "clarification", "topic_shift"]
    domains = ["clapnq", "govt"]
    answerabilities = ["answerable", "answerable", "answerable", "unanswerable"]  # 75% answerable

    # Generate corpus passages
    corpus = {}
    for i in range(500):
        pid = f"passage_{i:04d}"
        corpus[pid] = {
            "passage_id": pid,
            "text": (
                f"This is passage {i} about topic {i % 20}. "
                f"It contains information relevant to questions in domain {domains[i % 2]}. "
                f"The passage discusses concepts A, B, and C in detail. "
                f"Specifically, the key fact here is that value_{i} equals {i * 3.14:.2f}."
            ),
            "domain": domains[i % 2],
        }

    # Generate conversations
    conversations = []
    qrels = {}
    references = {}

    for i in range(n_conversations):
        conv_id = f"conv_{i:04d}"
        domain = domains[i % 2]
        qt = question_types[i % 4]
        mt = multiturn_types[i % 3]
        ans = answerabilities[i % 4]

        # Build turns (3–8 turns)
        n_turns = random.randint(3, 8)
        turns = []
        for t in range(n_turns - 1):
            pronoun = "How did they affect this?" if t % 3 == 2 else ""
            turns.append({
                "turn_id": t,
                "question": pronoun or f"What is the {t}th aspect of topic {i % 20}?",
                "answer": f"The {t}th aspect relates to passage_{i * 10 % 500}.",
            })
        turns.append({
            "turn_id": n_turns - 1,
            "question": (
                "What is the key value mentioned in the relevant passages?"
                if ans == "answerable"
                else "What is the secret code that no passage mentions?"
            ),
        })

        conv = {
            "id": conv_id,
            "domain": domain,
            "turns": turns,
            "question_type": qt,
            "multiturn_type": mt,
            "answerability": ans,
        }
        conversations.append(conv)

        # Assign qrels (1–3 relevant passages per answerable question)
        if ans == "answerable":
            n_relevant = random.randint(1, 3)
            relevant_pids = random.sample(
                [pid for pid, p in corpus.items() if p["domain"] == domain],
                min(n_relevant, 10),
            )
            qrels[conv_id] = {pid: random.choice([1, 2]) for pid in relevant_pids}
            references[conv_id] = {
                "conv_id": conv_id,
                "reference": f"The key value is {i * 3.14:.2f} as discussed in the passages.",
                "passages": relevant_pids,
            }
        else:
            qrels[conv_id] = {}
            references[conv_id] = {
                "conv_id": conv_id,
                "reference": "This question cannot be answered from the provided passages.",
                "passages": [],
            }

    return {
        "conversations": conversations,
        "corpus": corpus,
        "qrels": qrels,
        "references": references,
    }
