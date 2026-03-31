"""
src/data/loader.py
MTRAG Dataset Loader — matches the real IBM mt-rag-benchmark structure.

Real paths (confirmed from dataset exploration):
  mtrag-human/conversations/conversations.json     → 110 raw conversations
  mtrag-human/generation_tasks/reference+RAG.jsonl → 436 tasks (Task B input)
  mtrag-human/retrieval_tasks/<domain>/qrels/dev.tsv → qrels (Task A)
  corpora/passage_level/<domain>.jsonl.zip          → corpus passages

Task ID format in qrels: "<conversation_id><::><turn_number>"
Passage ID in qrels has chunk offsets appended: "base_pid-offset1-offset2"
"""

import json
import os
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class MTRAGDataLoader:
    """
    Loads the real IBM mt-rag-benchmark dataset.

    After loading:
      self.conversations  — list of normalised conversation dicts
      self.corpus         — {passage_id: passage_dict}
      self.qrels          — {task_id: {passage_id: relevance}}
      self.references     — {task_id: reference_dict}
      self.tasks          — raw list of generation task dicts
    """

    DOMAINS = ["clapnq", "govt", "fiqa", "cloud"]

    def __init__(self, config: dict, split: str = "val"):
        self.config = config
        self.split = split
        self.seed = config.get("seed", 42)
        set_seed(self.seed)

        self.base       = Path(config["dataset"]["base_path"])
        self.human_dir  = self.base / "mtrag-human"
        self.corpus_dir = self.base / "corpora" / "passage_level"

        self.conversations: List[dict] = []
        self.corpus:        Dict[str, dict] = {}
        self.qrels:         Dict[str, Dict[str, int]] = {}
        self.references:    Dict[str, dict] = {}
        self.tasks:         List[dict] = []

        self._load()

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load(self):
        self._load_corpus()
        self._load_generation_tasks()   # builds conversations + references
        self._load_qrels()
        print(
            f"[DataLoader] {len(self.conversations)} conversations | "
            f"{len(self.tasks)} tasks | "
            f"{len(self.corpus)} passages | "
            f"{len(self.qrels)} qrel entries"
        )

    def _load_corpus(self):
        """Load passage_level/<domain>.jsonl.zip for all four domains."""
        for domain in self.DOMAINS:
            zip_path = self.corpus_dir / f"{domain}.jsonl.zip"
            if not zip_path.exists():
                print(f"[DataLoader] WARNING corpus missing: {zip_path}")
                continue
            try:
                with zipfile.ZipFile(zip_path) as z:
                    fname = z.namelist()[0]
                    with z.open(fname) as f:
                        for raw in f:
                            if not raw.strip():
                                continue
                            p   = json.loads(raw.decode("utf-8"))
                            pid = p.get("_id") or p.get("id")
                            if pid:
                                self.corpus[pid] = {
                                    "passage_id": pid,
                                    "text":  p.get("text",  ""),
                                    "title": p.get("title", ""),
                                    "domain": domain,
                                }
            except Exception as e:
                print(f"[DataLoader] ERROR loading {domain} corpus: {e}")

    def _parse_answerability(self, raw) -> str:
        s = str(raw).lower()
        if "unanswerable" in s:
            return "unanswerable"
        if "partial" in s:
            return "partial"
        return "answerable"

    def _parse_field(self, raw) -> str:
        """Unwrap list fields like ['factoid'] → 'factoid'."""
        if isinstance(raw, list):
            raw = raw[0] if raw else "unknown"
        return str(raw).lower().strip()

    def _load_generation_tasks(self):
        """
        Load reference+RAG.jsonl (436 tasks).

        Each task has:
          conversation_id, task_id, turn, dataset,
          contexts  → [{"document_id": pid, "text": "..."}],
          input     → [{"speaker": "user"|"agent", "text": "..."}],
          targets   → [{"text": "..."}],
          Question Type, Multi-Turn, Answerability
        """
        gen_file = self.human_dir / "generation_tasks" / "reference+RAG.jsonl"
        if not gen_file.exists():
            print(f"[DataLoader] WARNING: {gen_file} not found")
            return

        with open(gen_file) as f:
            for line in f:
                if not line.strip():
                    continue
                task = json.loads(line)
                self.tasks.append(task)

                task_id = task.get("task_id", "")
                domain  = task.get("dataset", "unknown")
                qt      = self._parse_field(task.get("Question Type",  "unknown"))
                mt      = self._parse_field(task.get("Multi-Turn",     "unknown"))
                ans     = self._parse_answerability(task.get("Answerability", "answerable"))

                contexts    = task.get("contexts", [])
                passage_ids = [c.get("document_id", "") for c in contexts]
                passage_txts= [c.get("text", "")        for c in contexts]
                targets     = task.get("targets", [])
                ref_text    = targets[0].get("text", "") if targets else ""

                self.references[task_id] = {
                    "task_id":       task_id,
                    "conv_id":       task.get("conversation_id", ""),
                    "reference":     ref_text,
                    "passages":      passage_ids,
                    "passage_texts": passage_txts,
                    "domain":        domain,
                    "question_type": qt,
                    "multiturn_type":mt,
                    "answerability": ans,
                    "input_turns":   task.get("input", []),
                }

        # Build conversations by grouping tasks per conversation_id
        tasks_by_conv: Dict[str, List[dict]] = defaultdict(list)
        for task in self.tasks:
            tasks_by_conv[task.get("conversation_id", "")].append(task)

        for conv_id, conv_tasks in tasks_by_conv.items():
            conv_tasks = sorted(conv_tasks, key=lambda t: t.get("turn", 0))
            last_task  = conv_tasks[-1]

            # Build turns from the input field of the last task
            input_msgs = last_task.get("input", [])
            turns = []
            i = 0
            while i < len(input_msgs):
                msg  = input_msgs[i]
                spk  = msg.get("speaker", "")
                txt  = msg.get("text", "")
                if spk == "user":
                    turn = {"turn_id": len(turns), "question": txt}
                    if (i + 1 < len(input_msgs) and
                            input_msgs[i+1].get("speaker") in ("agent","assistant")):
                        turn["answer"] = input_msgs[i+1].get("text", "")
                        i += 1
                    turns.append(turn)
                i += 1

            domain = last_task.get("dataset", "unknown")
            qt     = self._parse_field(last_task.get("Question Type",  "unknown"))
            mt     = self._parse_field(last_task.get("Multi-Turn",     "unknown"))
            ans    = self._parse_answerability(last_task.get("Answerability", "answerable"))

            self.conversations.append({
                "id":            conv_id,
                "domain":        domain,
                "turns":         turns,
                "question_type": qt,
                "multiturn_type":mt,
                "answerability": ans,
                "_task_ids":     [t["task_id"] for t in conv_tasks],
            })

    def _load_qrels(self):
        """
        Load retrieval_tasks/<domain>/qrels/dev.tsv.

        Line format:  <task_id>\t<chunk_passage_id>\t<relevance>
        task_id       = "<conv_id><::><turn>"
        chunk_pid     = "base_pid-offset1-offset2"  (last 2 parts are chunk offsets)
        We store both the full chunk_pid AND the base_pid so retrieval can match either.
        """
        for domain in self.DOMAINS:
            qrel_path = (self.human_dir / "retrieval_tasks" /
                         domain / "qrels" / "dev.tsv")
            if not qrel_path.exists():
                continue
            with open(qrel_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("query"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 3:
                        continue
                    task_id  = parts[0]
                    chunk_pid = parts[1]
                    rel       = int(parts[2]) if parts[2].isdigit() else 1

                    # Derive base passage id (drop last 2 offset parts)
                    pid_parts = chunk_pid.split("-")
                    base_pid  = ("-".join(pid_parts[:-2])
                                 if len(pid_parts) > 2 else chunk_pid)

                    if task_id not in self.qrels:
                        self.qrels[task_id] = {}
                    self.qrels[task_id][chunk_pid] = rel
                    self.qrels[task_id][base_pid]  = rel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_conversations(
        self,
        question_type: str = None,
        multiturn_type: str = None,
        answerability:  str = None,
        domain:         str = None,
    ) -> List[dict]:
        convs = self.conversations
        if question_type:
            convs = [c for c in convs if c.get("question_type") == question_type]
        if multiturn_type:
            convs = [c for c in convs if c.get("multiturn_type") == multiturn_type]
        if answerability:
            convs = [c for c in convs if c.get("answerability") == answerability]
        if domain:
            convs = [c for c in convs if c.get("domain") == domain]
        return convs

    def build_query(self, conv: dict, history_mode: str = "full_history") -> str:
        """Build a retrieval query string from conversation history."""
        turns   = conv.get("turns", [])
        if not turns:
            return ""
        final_q = turns[-1].get("question", "")
        prior   = turns[:-1]

        if history_mode == "no_history":
            return final_q
        if history_mode == "last_turn_only":
            prior = prior[-1:] if prior else []
        elif history_mode.startswith("window_"):
            n     = int(history_mode.split("_")[1])
            prior = prior[-n:]
        # else: full_history — use all prior turns

        history = "\n".join(
            f"Q: {t.get('question','')}\nA: {t.get('answer','')}"
            for t in prior
        )
        return f"{history}\nQ: {final_q}".strip() if history else final_q

    def get_corpus_for_domain(self, domain: str) -> Dict[str, dict]:
        return {pid: p for pid, p in self.corpus.items()
                if p.get("domain") == domain}

    def get_qrels(self, task_id: str) -> Dict[str, int]:
        return self.qrels.get(task_id, {})

    def get_reference(self, task_id: str) -> Optional[dict]:
        return self.references.get(task_id)

    def get_metadata_distribution(self) -> dict:
        from collections import Counter
        return {
            "question_type": dict(Counter(c.get("question_type") for c in self.conversations)),
            "multiturn_type":dict(Counter(c.get("multiturn_type") for c in self.conversations)),
            "answerability": dict(Counter(c.get("answerability")  for c in self.conversations)),
            "domain":        dict(Counter(c.get("domain")         for c in self.conversations)),
            "total":         len(self.conversations),
            "total_tasks":   len(self.tasks),
        }
