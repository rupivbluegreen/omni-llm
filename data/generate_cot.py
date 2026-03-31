"""Generate synthetic chain-of-thought data using a larger model API.

Sends coding problems to an LLM and asks it to produce step-by-step
reasoning wrapped in <|think_start|>...<|think_end|> tags.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

COT_CATEGORIES = [
    "debug_error",
    "explain_code",
    "fix_bug",
    "optimize",
    "write_function",
    "compare_approaches",
    "explain_concept",
    "review_code",
]

COT_GENERATION_PROMPT = """\
You are an expert programming assistant. Given a coding problem, produce a \
detailed chain-of-thought solution.

IMPORTANT: You MUST wrap your reasoning inside special tags:
<|think_start|>
... your step-by-step reasoning here ...
<|think_end|>

After the reasoning block, provide your final answer clearly.

Rules:
1. Break the problem into small, logical steps.
2. Consider edge cases and potential issues.
3. If the problem involves code, include code snippets in your reasoning.
4. Be thorough but concise — every step should add value.

Problem category: {category}

Problem:
{problem}

Respond with your chain-of-thought reasoning and final answer."""


def generate_cot_example(
    problem: dict,
    api_url: str,
    model: str,
) -> dict:
    """Send a problem to the LLM API and parse the CoT response.

    Args:
        problem: Dict with at least 'prompt' and optionally 'category' keys.
        api_url: URL of the OpenAI-compatible chat completions endpoint.
        model: Model name to use.

    Returns:
        Dict with 'messages' (conversation), 'category', and 'source_problem'.
    """
    category = problem.get("category", "write_function")
    prompt_text = problem.get("prompt", problem.get("text", ""))

    user_message = COT_GENERATION_PROMPT.format(
        category=category,
        problem=prompt_text,
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
    }

    response = requests.post(
        api_url,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=120,
    )
    response.raise_for_status()

    result = response.json()
    assistant_content = result["choices"][0]["message"]["content"]

    # Build the formatted SFT example
    formatted = {
        "messages": [
            {"role": "user", "content": prompt_text},
            {"role": "assistant", "content": assistant_content},
        ],
        "category": category,
        "source_problem": prompt_text,
    }

    return formatted


def process_batch(
    problems: list[dict],
    api_url: str,
    model: str,
    max_workers: int = 4,
) -> list[dict]:
    """Process a batch of problems in parallel using ThreadPoolExecutor.

    Args:
        problems: List of problem dicts.
        api_url: URL of the chat completions endpoint.
        model: Model name.
        max_workers: Number of parallel workers.

    Returns:
        List of generated CoT examples.
    """
    results = []
    total = len(problems)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, problem in enumerate(problems):
            # Rate limiting: stagger submissions slightly
            if idx > 0 and idx % max_workers == 0:
                time.sleep(1.0)

            future = executor.submit(
                generate_cot_example, problem, api_url, model
            )
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            completed += 1

            try:
                result = future.result()
                results.append(result)
                print(
                    f"  [{completed}/{total}] Generated CoT for problem {idx}",
                    file=sys.stderr,
                )
            except Exception as e:
                print(
                    f"  [{completed}/{total}] ERROR on problem {idx}: {e}",
                    file=sys.stderr,
                )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic chain-of-thought training data"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file of coding problems (each line: {\"prompt\": ..., \"category\": ...})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file for generated CoT examples",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:11434/v1/chat/completions",
        help="OpenAI-compatible chat completions API URL (default: http://localhost:11434/v1/chat/completions)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="Model name to use (default: gpt-4)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    args = parser.parse_args()

    # Load problems
    print(f"Loading problems from {args.input}...", file=sys.stderr)
    problems = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))

    print(
        f"Loaded {len(problems)} problems. Generating CoT with model '{args.model}'...",
        file=sys.stderr,
    )

    # Process in batches
    results = process_batch(
        problems,
        api_url=args.api_url,
        model=args.model,
        max_workers=args.max_workers,
    )

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for example in results:
            f.write(json.dumps(example) + "\n")

    print(
        f"\nDone! Generated {len(results)}/{len(problems)} examples -> {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
