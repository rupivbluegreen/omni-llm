"""Benchmark data loaders and evaluation metrics."""

from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    benchmark: str
    score: float
    details: dict = field(default_factory=dict)


_HUMANEVAL_STUB = [
    {
        "task_id": "HumanEval/0",
        "prompt": "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    \"\"\"Check if any two numbers are closer than threshold.\"\"\"\n",
        "canonical_solution": "    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n",
        "test": "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0], 0.3) == True\n",
        "entry_point": "has_close_elements",
    }
]


def load_humaneval(data_path: str | None = None) -> list[dict]:
    """Load HumanEval benchmark problems.

    Each problem has: task_id, prompt, canonical_solution, test, entry_point.
    If data_path is given and exists, load from there. Otherwise try the
    HuggingFace `openai/openai_humaneval` dataset. Falls back to a single-
    problem stub only if both fail (so tests can run offline).
    """
    if data_path and Path(data_path).exists():
        with open(data_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    try:
        from datasets import load_dataset
        ds = load_dataset("openai/openai_humaneval", split="test")
        return [dict(row) for row in ds]
    except Exception:
        return list(_HUMANEVAL_STUB)


_MT_BENCH_STUB = [
    {
        "question_id": 1,
        "category": "coding",
        "turns": [
            "Write a Python function to find the longest palindromic substring.",
            "Now optimize it to O(n) using Manacher's algorithm.",
        ],
    },
    {
        "question_id": 2,
        "category": "debugging",
        "turns": [
            "This code throws IndexError: `for i in range(len(arr)+1): print(arr[i])`. Why?",
            "How would you fix it and add proper error handling?",
        ],
    },
]


def load_mt_bench_questions(data_path: str | None = None) -> list[dict]:
    """Load MT-Bench multi-turn evaluation questions.

    Each question has: question_id, category, turns (list of prompts).
    If data_path exists, load JSONL from there. Otherwise try the
    HuggingFace `HuggingFaceH4/mt_bench_prompts` dataset. Falls back
    to a two-question stub only if both fail.
    """
    if data_path and Path(data_path).exists():
        with open(data_path) as f:
            text = f.read()
            # Accept either JSON array or JSONL
            if text.lstrip().startswith("["):
                return json.loads(text)
            return [json.loads(line) for line in text.splitlines() if line.strip()]

    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        out = []
        for row in ds:
            # Dataset exposes 'prompt' as a list of strings (the turns)
            turns = row.get("prompt") or row.get("turns") or []
            if isinstance(turns, str):
                turns = [turns]
            out.append({
                "question_id": row.get("prompt_id", row.get("question_id", len(out) + 1)),
                "category": row.get("category", "unknown"),
                "turns": list(turns),
            })
        return out
    except Exception:
        return list(_MT_BENCH_STUB)


class PassAtKMetric:
    """Compute pass@k metric for code generation."""

    @staticmethod
    def compute(n: int, c: int, k: int) -> float:
        """Compute pass@k.

        Args:
            n: total number of samples
            c: number of correct samples
            k: k value
        """
        if n - c < k:
            return 1.0
        result = 1.0
        for i in range(k):
            result *= (n - c - i) / (n - i)
        return 1.0 - result
