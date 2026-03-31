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


def load_humaneval(data_path: str | None = None) -> list[dict]:
    """Load HumanEval benchmark problems.

    Each problem has: task_id, prompt, canonical_solution, test, entry_point
    """
    if data_path and Path(data_path).exists():
        with open(data_path) as f:
            return [json.loads(line) for line in f if line.strip()]

    # Placeholder — in practice, download from HuggingFace
    return [
        {
            "task_id": "HumanEval/0",
            "prompt": "def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    \"\"\"Check if any two numbers are closer than threshold.\"\"\"\n",
            "canonical_solution": "    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False\n",
            "test": "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\nassert has_close_elements([1.0, 2.8, 3.0], 0.3) == True\n",
            "entry_point": "has_close_elements",
        }
    ]


def load_mt_bench_questions(data_path: str | None = None) -> list[dict]:
    """Load MT-Bench style multi-turn evaluation questions.

    Each question has: question_id, category, turns (list of prompts)
    """
    if data_path and Path(data_path).exists():
        with open(data_path) as f:
            return json.loads(f.read())

    return [
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
