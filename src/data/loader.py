"""
src/data/loader.py
"""

import json
import os
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from src.utils import set_seed, load_config


class MTRAGDataLoader:
    """Loader for the IBM MTRAG human benchmark dataset."""

    DOMAINS = ['clapnq', 'govt', 'fiqa', 'cloud']

    def __init__(self, config: dict, split: str = 'val'):
        self.config    = config
        self.split     = split
        self.seed      = config.get('seed', 42)
        set_seed(self.seed)

        self.base       = Path(config['dataset']['base_path'])
        self.human_dir  = self.base / 'mtrag-human'
        self.corpus_dir = self.base / 'corpora' / 'passage_level'

        self.conversations: List[dict]            = []
        self.corpus:        Dict[str, dict]       = {}
        self.qrels:         Dict[str, Dict[str, int]] = {}
        self.references:    Dict[str, dict]       = {}
        self.tasks:         List[dict]            = []

        self._load()

    # ------------------------------------------------------------------
    # Main load sequence
    # ------------------------------------------------------------------

    def _load(self):
        self._load_corpus()
        self._load_generation_tasks()   # builds conversations + references
        self._load_qrels()
        self._fix_domains()             # fix domain from qrel file paths
        print(
            f'[DataLoader] {len(self.conversations)} conversations | '
            f'{len(self.tasks)} tasks | '
            f'{len(self.corpus):,} passages | '
            f'{len(self.qrels)} qrel entries'
        )

    # ------------------------------------------------------------------
    # Corpus
    # ------------------------------------------------------------------

    def _load_corpus(self):
        """Load passage_level/<domain>.jsonl.zip for all four domains."""
        for domain in self.DOMAINS:
            zip_path = self.corpus_dir / f'{domain}.jsonl.zip'
            if not zip_path.exists():
                print(f'[DataLoader] WARNING: corpus not found: {zip_path}')
                continue
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for name in zf.namelist():
                    if name.endswith('.jsonl'):
                        with zf.open(name) as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    passage = json.loads(line)
                                    pid = passage.get('id') or passage.get('pid') or passage.get('passage_id')
                                    if pid:
                                        passage['domain'] = domain
                                        self.corpus[str(pid)] = passage
                                except json.JSONDecodeError:
                                    continue

    # ------------------------------------------------------------------
    # Generation tasks → conversations + references
    # ------------------------------------------------------------------

    def _load_generation_tasks(self):
        """Load generation tasks JSONL. Builds conversations and references."""
        gen_path = self.human_dir / 'generation_tasks' / 'reference+RAG.jsonl'
        if not gen_path.exists():
            print(f'[DataLoader] WARNING: generation tasks not found: {gen_path}')
            return

        conv_map: Dict[str, dict] = {}

        with open(gen_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    task = json.loads(line)
                except json.JSONDecodeError:
                    continue

                self.tasks.append(task)
                conv_id = task.get('conversation_id', '')
                task_id = task.get('task_id', '')

                # ── Build reference entry ──────────────────────────
                # Store passage_texts directly so Task B doesn't need
                # corpus lookup (chunk PIDs don't match corpus keys)
                passage_texts = []
                for pid in task.get('passages', []):
                    pid_str = str(pid)
                    if pid_str in self.corpus:
                        passage_texts.append(self.corpus[pid_str]['text'])
                    else:
                        # Try base PID (strip chunk suffix)
                        parts = pid_str.split('-')
                        base  = '-'.join(parts[:-2]) if len(parts) > 2 else pid_str
                        if base in self.corpus:
                            passage_texts.append(self.corpus[base]['text'])

                self.references[task_id] = {
                    'task_id':       task_id,
                    'conversation_id': conv_id,
                    'reference':     task.get('answer', task.get('reference', '')),
                    'passages':      task.get('passages', []),
                    'passage_texts': passage_texts,
                    'answerability': task.get('answerability', 'answerable'),
                    'question_type': task.get('question_type', 'unknown'),
                    'multiturn_type': task.get('multiturn_type', 'unknown'),
                    'input_turns':   task.get('input_turns', []),
                    'domain':        task.get('dataset', 'unknown'),
                }

                # ── Build conversation entry ───────────────────────
                if conv_id not in conv_map:
                    conv_map[conv_id] = {
                        'id':            conv_id,
                        'turns':         [],
                        '_task_ids':     [],
                        'domain':        task.get('dataset', 'unknown'),
                        'question_type': task.get('question_type', 'unknown'),
                        'multiturn_type': task.get('multiturn_type', 'unknown'),
                        'answerability': task.get('answerability', 'answerable'),
                    }

                conv = conv_map[conv_id]
                if task_id not in conv['_task_ids']:
                    conv['_task_ids'].append(task_id)

                # Add turns from input_turns
                for msg in task.get('input_turns', []):
                    speaker = msg.get('speaker', '')
                    text    = msg.get('text', '')
                    if speaker.lower() in ('user', 'human', 'questioner'):
                        # Check if this turn already added
                        existing = [t.get('question', '') for t in conv['turns']]
                        if text not in existing:
                            conv['turns'].append({
                                'question': text,
                                'answer':   '',
                            })
                    elif speaker.lower() in ('agent', 'assistant', 'system'):
                        if conv['turns']:
                            conv['turns'][-1]['answer'] = text

        self.conversations = list(conv_map.values())

    # ------------------------------------------------------------------
    # Qrels
    # ------------------------------------------------------------------

    def _load_qrels(self):
        """Load qrel files from all four domains."""
        retrieval_dir = self.human_dir / 'retrieval_tasks'
        for domain in self.DOMAINS:
            qrel_path = retrieval_dir / domain / 'qrels' / 'dev.tsv'
            if not qrel_path.exists():
                continue
            with open(qrel_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('query_id'):
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        task_id   = parts[0]
                        passage_id = parts[1] if len(parts) == 3 else parts[2]
                        relevance  = int(parts[-1]) if parts[-1].isdigit() else 1
                        if task_id not in self.qrels:
                            self.qrels[task_id] = {}
                        self.qrels[task_id][str(passage_id)] = relevance

    # ------------------------------------------------------------------
    # Domain fix — THE KEY FIX
    # ------------------------------------------------------------------

    def _fix_domains(self):
        """
        Fix conversation and reference domains using qrel file paths.

        The 'dataset' field in generation tasks is 'MT-RAG Authors (Internal)'
        for all entries — completely useless. Real domain comes from which
        qrel file each task_id appears in (clapnq / govt / fiqa / cloud).
        """
        retrieval_dir = self.human_dir / 'retrieval_tasks'
        taskid_to_domain = {}

        for domain in self.DOMAINS:
            qrel_path = retrieval_dir / domain / 'qrels' / 'dev.tsv'
            if not qrel_path.exists():
                continue
            with open(qrel_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        task_id = line.split('\t')[0]
                        taskid_to_domain[task_id] = domain

        # Fix conversations
        fixed_convs = 0
        for conv in self.conversations:
            for tid in conv.get('_task_ids', []):
                if tid in taskid_to_domain:
                    conv['domain'] = taskid_to_domain[tid]
                    fixed_convs += 1
                    break

        # Fix references
        for task_id, ref in self.references.items():
            if task_id in taskid_to_domain:
                ref['domain'] = taskid_to_domain[task_id]

        print(f'[DataLoader] Domain fixed for {fixed_convs}/{len(self.conversations)} conversations')

    # ------------------------------------------------------------------
    # Metadata utilities
    # ------------------------------------------------------------------

    def get_metadata_distribution(self) -> dict:
        dist = {
            'question_type':  Counter(c.get('question_type', 'unknown')
                                      for c in self.conversations),
            'multiturn_type': Counter(c.get('multiturn_type', 'unknown')
                                      for c in self.conversations),
            'domain':         Counter(c.get('domain', 'unknown')
                                      for c in self.conversations),
            'answerability':  Counter(c.get('answerability', 'unknown')
                                      for c in self.conversations),
            'total_tasks':    len(self.tasks),
        }
        return dist
