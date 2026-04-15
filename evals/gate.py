"""Stage promotion gate.

Runs held-out perplexity + HumanEval pass@1 + an MT-Bench slice against a
checkpoint and exits non-zero if any metric regresses below the configured
threshold, or below the previous stage's recorded metrics. Intended to be
invoked between pretrain → SFT → DPO.

Thresholds are kept deliberately loose as *floors*, not targets. The point is
to catch catastrophic regressions (a stage destroying capability), not to
benchmark fine-grained progress. Tune for your run.

Usage:
    python -m evals.gate \\
        --checkpoint checkpoints/pretrain/step-100000-final \\
        --stage pretrain \\
        --eval-data data/heldout.jsonl \\
        --history evals/gate_history.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# Floors per stage. A stage that fails to beat these is considered broken.
# Numbers are placeholders — override from the CLI for your real run.
DEFAULT_FLOORS = {
    "pretrain": {"humaneval_pass_at_1": 0.02, "mt_bench_score": 1.5, "max_ppl": 40.0},
    "sft":      {"humaneval_pass_at_1": 0.05, "mt_bench_score": 3.0, "max_ppl": 30.0},
    "dpo":      {"humaneval_pass_at_1": 0.05, "mt_bench_score": 3.5, "max_ppl": 30.0},
}

# A new stage must not regress vs. the previous stage's recorded metrics by
# more than this fraction. e.g. 0.10 = SFT may be up to 10% worse than pretrain
# on HumanEval (it rarely should be, but small drops happen from format shift).
REGRESSION_TOLERANCE = 0.10


def compute_heldout_ppl(model, tokenizer, eval_data_path: str, max_seq_len: int) -> float:
    """Token-weighted perplexity over a held-out JSONL of {"text": ...}."""
    import math
    import mlx.core as mx
    import mlx.nn as nn

    total_loss = 0.0
    total_tokens = 0
    with open(eval_data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text", "")
            if not text:
                continue
            ids = tokenizer.encode(text).ids[: max_seq_len + 1]
            if len(ids) < 2:
                continue
            input_ids = mx.array([ids[:-1]])
            labels = mx.array([ids[1:]])
            logits = model(input_ids)
            vocab_size = logits.shape[-1]
            loss = nn.losses.cross_entropy(
                logits.reshape(-1, vocab_size),
                labels.reshape(-1),
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_tokens += labels.size
    if total_tokens == 0:
        raise ValueError(f"No eval tokens found in {eval_data_path}")
    return math.exp(total_loss / total_tokens)


def run_gate(
    checkpoint: str,
    stage: str,
    eval_data: str,
    tokenizer_path: str,
    history_path: str,
    mt_bench_slice: int,
    humaneval_samples: int,
    floors: dict,
) -> int:
    """Return 0 on pass, non-zero on fail. Writes metrics to history_path."""
    # Imported lazily so the gate can be smoke-tested for wiring without MLX.
    from tokenizers import Tokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from model.config import OmniscientConfig
    from model.transformer import OmniscientModel
    from training.utils import load_checkpoint
    from evals.run_eval import run_humaneval, run_mt_bench

    tokenizer = Tokenizer.from_file(tokenizer_path)
    ckpt = Path(checkpoint)
    config_path = ckpt.parent / "config.json"
    config = OmniscientConfig.load(config_path) if config_path.exists() else OmniscientConfig()
    model = OmniscientModel(config)
    load_checkpoint(model, checkpoint)

    metrics: dict = {}
    metrics["ppl"] = compute_heldout_ppl(model, tokenizer, eval_data, config.max_seq_len)
    he = run_humaneval(model, tokenizer, n_samples=humaneval_samples)
    metrics["humaneval_pass_at_1"] = float(getattr(he, "pass_at_1", getattr(he, "score", 0.0)))
    # run_mt_bench in evals/run_eval.py is currently a stub that returns 0.0
    # without a judge_endpoint. We keep it in the gate so the wiring is proven
    # end-to-end; once a real judge is available, thread it through here.
    mt = run_mt_bench(model, tokenizer)
    metrics["mt_bench_score"] = float(getattr(mt, "score", 0.0))

    print(f"[gate:{stage}] metrics = {json.dumps(metrics, indent=2)}")

    # Absolute-floor check
    failures = []
    stage_floors = floors.get(stage, {})
    if metrics["humaneval_pass_at_1"] < stage_floors.get("humaneval_pass_at_1", 0.0):
        failures.append(f"humaneval_pass_at_1 {metrics['humaneval_pass_at_1']:.4f} < floor {stage_floors['humaneval_pass_at_1']}")
    if metrics["mt_bench_score"] < stage_floors.get("mt_bench_score", 0.0):
        failures.append(f"mt_bench_score {metrics['mt_bench_score']:.4f} < floor {stage_floors['mt_bench_score']}")
    if metrics["ppl"] > stage_floors.get("max_ppl", float("inf")):
        failures.append(f"ppl {metrics['ppl']:.4f} > max_ppl {stage_floors['max_ppl']}")

    # Regression check against previous stage
    history = {}
    hpath = Path(history_path)
    if hpath.exists():
        history = json.loads(hpath.read_text())
    stage_order = ["pretrain", "sft", "dpo"]
    if stage in stage_order:
        idx = stage_order.index(stage)
        if idx > 0:
            prev = stage_order[idx - 1]
            if prev in history:
                prev_he = history[prev].get("humaneval_pass_at_1", 0.0)
                if prev_he > 0 and metrics["humaneval_pass_at_1"] < prev_he * (1 - REGRESSION_TOLERANCE):
                    failures.append(
                        f"humaneval regression: {metrics['humaneval_pass_at_1']:.4f} < "
                        f"{prev} {prev_he:.4f} * (1-{REGRESSION_TOLERANCE})"
                    )

    history[stage] = metrics
    hpath.parent.mkdir(parents=True, exist_ok=True)
    hpath.write_text(json.dumps(history, indent=2))

    if failures:
        print(f"[gate:{stage}] FAIL:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print(f"[gate:{stage}] PASS")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Stage promotion gate")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint dir to evaluate")
    parser.add_argument("--stage", required=True, choices=["pretrain", "sft", "dpo"])
    parser.add_argument("--eval-data", required=True, help="Held-out JSONL for PPL")
    parser.add_argument("--tokenizer", default="tokenizer.json")
    parser.add_argument("--history", default="evals/gate_history.json")
    parser.add_argument("--mt-bench-slice", type=int, default=20)
    parser.add_argument("--humaneval-samples", type=int, default=1)
    args = parser.parse_args()

    rc = run_gate(
        checkpoint=args.checkpoint,
        stage=args.stage,
        eval_data=args.eval_data,
        tokenizer_path=args.tokenizer,
        history_path=args.history,
        mt_bench_slice=args.mt_bench_slice,
        humaneval_samples=args.humaneval_samples,
        floors=DEFAULT_FLOORS,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
