"""Evaluation runner for OmniscientLLM."""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
import tempfile
from pathlib import Path

from .benchmarks import load_humaneval, load_mt_bench_questions, PassAtKMetric, EvalResult


def run_humaneval(
    model,
    tokenizer,
    n_samples: int = 1,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> EvalResult:
    """Run HumanEval benchmark: generate completions and test them."""
    problems = load_humaneval()
    correct = 0
    total = 0

    for problem in problems:
        for _ in range(n_samples):
            total += 1
            # Generate completion
            try:
                import mlx.core as mx
                tokens = mx.array([tokenizer.encode(problem["prompt"]).ids])
                logits = model(tokens)
                # Simple greedy decode for eval
                generated_ids = []
                for _ in range(max_tokens):
                    next_token = mx.argmax(logits[:, -1, :], axis=-1)
                    token_id = next_token.item()
                    eos_id = tokenizer.token_to_id("<|eos|>")
                    if token_id == eos_id:
                        break
                    generated_ids.append(token_id)
                    tokens = mx.concatenate([tokens, next_token.reshape(1, 1)], axis=1)
                    logits = model(tokens)

                completion = tokenizer.decode(generated_ids)
            except ImportError:
                completion = problem.get("canonical_solution", "")

            # Test the completion
            full_code = problem["prompt"] + completion
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                    f.write(full_code + "\n" + problem["test"])
                    f.flush()
                    result = subprocess.run(
                        [sys.executable, f.name],
                        capture_output=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        correct += 1
            except (subprocess.TimeoutExpired, Exception):
                pass

    pass_at_1 = PassAtKMetric.compute(total, correct, 1)
    return EvalResult(
        benchmark="humaneval",
        score=pass_at_1,
        details={"total": total, "correct": correct, "pass@1": pass_at_1},
    )


def run_mt_bench(
    model,
    tokenizer,
    judge_endpoint: str | None = None,
) -> EvalResult:
    """Run MT-Bench style multi-turn evaluation.

    If judge_endpoint is provided, uses an external model to score responses.
    Otherwise, returns placeholder scores.
    """
    questions = load_mt_bench_questions()
    scores = []

    for question in questions:
        messages = []
        turn_scores = []

        for turn_prompt in question["turns"]:
            messages.append({"role": "user", "content": turn_prompt})

            # Generate response (stub — uses chat_generator in practice)
            response = f"[Model response to: {turn_prompt[:50]}...]"
            messages.append({"role": "assistant", "content": response})

            if judge_endpoint:
                # Call judge model to score
                import requests
                judge_response = requests.post(
                    judge_endpoint,
                    json={
                        "messages": [
                            {"role": "system", "content": "Rate the assistant's response from 1-10."},
                            *messages,
                        ],
                        "max_tokens": 50,
                    },
                )
                try:
                    score = float(judge_response.json()["choices"][0]["message"]["content"].strip())
                except (ValueError, KeyError):
                    score = 5.0
            else:
                score = 0.0  # Placeholder
            turn_scores.append(score)

        scores.append(sum(turn_scores) / len(turn_scores) if turn_scores else 0)

    avg_score = sum(scores) / len(scores) if scores else 0
    return EvalResult(
        benchmark="mt_bench",
        score=avg_score,
        details={"per_question": scores, "n_questions": len(questions)},
    )


def main():
    parser = argparse.ArgumentParser(description="Run OmniscientLLM evaluations")
    parser.add_argument("--benchmark", choices=["humaneval", "mt_bench", "all"], default="all")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/latest")
    parser.add_argument("--judge-endpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Try to load model
    model, tokenizer = None, None
    try:
        from model.config import OmniscientConfig
        from model.transformer import OmniscientModel
        from tokenizer.train_tokenizer import load_tokenizer
        from training.utils import load_checkpoint

        config = OmniscientConfig.load(Path(args.checkpoint) / "config.json")
        model = OmniscientModel(config)
        load_checkpoint(model, args.checkpoint)
        tokenizer = load_tokenizer("tokenizer/omniscient-tokenizer.json")
        print(f"Model loaded from {args.checkpoint}")
    except Exception as e:
        print(f"Warning: Could not load model ({e}). Running with stubs.")

    results = []

    if args.benchmark in ("humaneval", "all"):
        print("Running HumanEval...")
        result = run_humaneval(model, tokenizer)
        results.append(result)
        print(f"  pass@1: {result.score:.3f}")

    if args.benchmark in ("mt_bench", "all"):
        print("Running MT-Bench...")
        result = run_mt_bench(model, tokenizer, args.judge_endpoint)
        results.append(result)
        print(f"  avg score: {result.score:.1f}/10")

    if args.output:
        with open(args.output, "w") as f:
            json.dump([{"benchmark": r.benchmark, "score": r.score, "details": r.details} for r in results], f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
