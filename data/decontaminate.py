"""Strip eval-set leakage from pretrain shards.

For every JSONL shard under --input-dir this walks each document's text,
looks for any 13-gram (token-split) that also appears in any HumanEval,
MBPP, or MT-Bench prompt, and drops the whole document if so. Output goes
to --output-dir, one JSONL per input shard.

We use an n-gram overlap instead of exact substring because exact matches
miss reformatted problem statements, and anything smaller than 13-grams
catches too many natural collisions on short code snippets.

Usage:
    python -m data.decontaminate \\
        --input-dir data/pretrain_raw \\
        --output-dir data/pretrain_clean
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


NGRAM_SIZE = 13
_WORD_RE = re.compile(r"\w+")


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = _WORD_RE.findall(text.lower())
    if len(toks) < n:
        return set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def build_contamination_set(include_humaneval: bool, include_mbpp: bool, include_mtbench: bool) -> set[tuple[str, ...]]:
    """Collect n-grams from all eval sources we want to exclude from pretrain."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from evals.benchmarks import load_humaneval, load_mt_bench_questions

    bad: set[tuple[str, ...]] = set()

    if include_humaneval:
        for problem in load_humaneval():
            bad |= _ngrams(problem.get("prompt", ""), NGRAM_SIZE)
            # Canonical solutions are also memorization targets.
            if "canonical_solution" in problem:
                bad |= _ngrams(problem["canonical_solution"], NGRAM_SIZE)

    if include_mbpp:
        try:
            from evals.benchmarks import load_mbpp  # optional
            for problem in load_mbpp():
                bad |= _ngrams(problem.get("text", problem.get("prompt", "")), NGRAM_SIZE)
        except Exception:
            # MBPP loader not present — not fatal, just skip.
            pass

    if include_mtbench:
        for q in load_mt_bench_questions():
            bad |= _ngrams(q.get("turns", [""])[0] if isinstance(q.get("turns"), list) else q.get("prompt", ""), NGRAM_SIZE)

    return bad


def decontaminate_shard(shard_path: Path, out_path: Path, bad_ngrams: set[tuple[str, ...]]) -> tuple[int, int]:
    """Return (kept, removed) document counts."""
    kept = 0
    removed = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(shard_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                removed += 1
                continue
            text = doc.get("text", "")
            doc_ngrams = _ngrams(text, NGRAM_SIZE)
            if doc_ngrams & bad_ngrams:
                removed += 1
                continue
            fout.write(json.dumps(doc) + "\n")
            kept += 1
    return kept, removed


def main():
    parser = argparse.ArgumentParser(description="Decontaminate pretrain shards against eval sets")
    parser.add_argument("--input-dir", required=True, help="Directory of raw JSONL shards")
    parser.add_argument("--output-dir", required=True, help="Directory to write cleaned shards")
    parser.add_argument("--skip-humaneval", action="store_true")
    parser.add_argument("--skip-mbpp", action="store_true")
    parser.add_argument("--skip-mtbench", action="store_true")
    args = parser.parse_args()

    bad = build_contamination_set(
        include_humaneval=not args.skip_humaneval,
        include_mbpp=not args.skip_mbpp,
        include_mtbench=not args.skip_mtbench,
    )
    print(f"[decontaminate] built contamination set with {len(bad)} {NGRAM_SIZE}-grams")

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    shards = sorted(in_dir.glob("*.jsonl"))
    if not shards:
        raise SystemExit(f"No .jsonl files found in {in_dir}")

    total_kept = 0
    total_removed = 0
    for shard in shards:
        kept, removed = decontaminate_shard(shard, out_dir / shard.name, bad)
        total_kept += kept
        total_removed += removed
        print(f"[decontaminate] {shard.name}: kept={kept} removed={removed}")

    print(f"[decontaminate] TOTAL kept={total_kept} removed={total_removed}")
    if total_removed == 0:
        print("[decontaminate] WARNING: 0 documents removed — sanity check your input-dir and eval loaders")


if __name__ == "__main__":
    main()
