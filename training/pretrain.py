"""Pretraining loop for OmniscientLLM using MLX."""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import OmniscientConfig
from model.transformer import OmniscientModel
from training.utils import (
    cosine_lr_schedule,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    prune_old_checkpoints,
    TrainingLogger,
)
from training.fim import apply_fim_augmentation


def load_jsonl_shards(data_dir: str | Path):
    """Iterate over JSONL shard files in data_dir, yielding text strings."""
    data_dir = Path(data_dir)
    shard_files = sorted(data_dir.glob("*.jsonl"))
    if not shard_files:
        raise FileNotFoundError(f"No .jsonl files found in {data_dir}")
    for shard in shard_files:
        with open(shard) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                yield data.get("text", "")


def chunk_tokens(token_ids: list[int], max_seq_len: int):
    """Split a flat list of token IDs into fixed-length chunks."""
    for i in range(0, len(token_ids) - max_seq_len, max_seq_len):
        yield token_ids[i : i + max_seq_len + 1]  # +1 for labels shift


def build_batches(data_dir: str | Path, tokenizer, max_seq_len: int, batch_size: int, fim_rate: float = 0.0):
    """Yield batches of (input_ids, labels) from JSONL shards.

    Each batch is a dict with:
        - input_ids: mx.array of shape (batch_size, max_seq_len)
        - labels: mx.array of shape (batch_size, max_seq_len)
    """
    buffer = []
    batch_inputs = []
    batch_labels = []

    for text in load_jsonl_shards(data_dir):
        if not text:
            continue

        # Optionally apply FIM augmentation
        if fim_rate > 0:
            [text] = apply_fim_augmentation([text], fim_rate=fim_rate)

        token_ids = tokenizer.encode(text).ids

        buffer.extend(token_ids)

        # Chunk buffer into sequences
        while len(buffer) >= max_seq_len + 1:
            chunk = buffer[: max_seq_len + 1]
            buffer = buffer[max_seq_len:]

            input_ids = chunk[:-1]
            labels = chunk[1:]

            batch_inputs.append(input_ids)
            batch_labels.append(labels)

            if len(batch_inputs) == batch_size:
                yield {
                    "input_ids": mx.array(batch_inputs),
                    "labels": mx.array(batch_labels),
                }
                batch_inputs = []
                batch_labels = []

    # Yield remaining partial batch
    if batch_inputs:
        yield {
            "input_ids": mx.array(batch_inputs),
            "labels": mx.array(batch_labels),
        }


def train_step(model, batch, loss_fn):
    """Single forward + loss computation."""
    logits = model(batch["input_ids"])
    loss = loss_fn(logits, batch["labels"])
    return loss


def main():
    parser = argparse.ArgumentParser(description="Pretrain OmniscientLLM")
    parser.add_argument("--config", type=str, required=True, help="Model config JSON path")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory with JSONL shards")
    parser.add_argument("--output-dir", type=str, default="checkpoints/pretrain", help="Checkpoint output directory")
    parser.add_argument("--max-steps", type=int, default=100_000, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max-lr", type=float, default=3e-4, help="Maximum learning rate")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Minimum learning rate")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--checkpoint-every", type=int, default=5000, help="Checkpoint interval (steps)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--fim-rate", type=float, default=0.5, help="FIM augmentation rate (applied in-pretrain, StarCoder-style)")
    parser.add_argument("--keep-last-n", type=int, default=5, help="Rolling checkpoint window; older checkpoints are pruned")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Path to tokenizer.json")
    args = parser.parse_args()

    # Load config and build model
    config = OmniscientConfig.load(args.config)
    model = OmniscientModel(config)

    logger = TrainingLogger("pretrain", use_wandb=args.wandb)
    param_count = count_parameters(model)
    logger.log(0, {"params": param_count, "msg": "Model initialized"})

    # Optimizer: AdamW
    optimizer = optim.AdamW(
        learning_rate=args.max_lr,
        betas=[0.9, 0.95],
        weight_decay=0.1,
    )

    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        state = load_checkpoint(model, args.resume)
        start_step = state["step"]
        logger.log(start_step, {"msg": f"Resumed from {args.resume}"})

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # Loss function
    def loss_fn(logits, labels):
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        labels = labels.reshape(-1)
        return nn.losses.cross_entropy(logits, labels, reduction="mean")

    # Build loss_and_grad function
    loss_and_grad_fn = nn.value_and_grad(model, lambda m, batch: train_step(m, batch, loss_fn))

    # Training loop
    step = start_step
    accum_loss = 0.0
    accum_count = 0
    tokens_seen = 0
    last_log_time = time.time()
    last_log_tokens = 0

    # Accumulated gradients storage
    accumulated_grads = None

    while step < args.max_steps:
        data_iter = build_batches(
            args.data_dir,
            tokenizer,
            config.max_seq_len,
            args.batch_size,
            fim_rate=args.fim_rate,
        )

        for batch in data_iter:
            if step >= args.max_steps:
                break

            # Update LR
            lr = cosine_lr_schedule(step, args.max_lr, args.min_lr, args.warmup_steps, args.max_steps)
            optimizer.learning_rate = lr

            # Forward + backward
            loss, grads = loss_and_grad_fn(model, batch)
            mx.eval(loss)

            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = mx.tree_map(lambda a, b: a + b, accumulated_grads, grads)

            accum_loss += loss.item()
            accum_count += 1
            tokens_seen += args.batch_size * config.max_seq_len

            # Step optimizer every grad_accum micro-batches
            if accum_count % args.grad_accum == 0:
                # Average gradients
                scale = 1.0 / args.grad_accum
                averaged_grads = mx.tree_map(lambda g: g * scale, accumulated_grads)

                # Apply optimizer step
                optimizer.update(model, averaged_grads)
                mx.eval(model.parameters(), optimizer.state)

                accumulated_grads = None
                step += 1

                # Log every 10 steps
                if step % 10 == 0:
                    avg_loss = accum_loss / (args.grad_accum * 10)
                    now = time.time()
                    elapsed = now - last_log_time
                    toks_per_sec = (tokens_seen - last_log_tokens) / elapsed if elapsed > 0 else 0

                    logger.log(step, {
                        "loss": avg_loss,
                        "lr": lr,
                        "tokens_per_sec": toks_per_sec,
                    })

                    accum_loss = 0.0
                    last_log_time = now
                    last_log_tokens = tokens_seen

                # Checkpoint
                if step % args.checkpoint_every == 0:
                    ckpt_path = Path(args.output_dir) / f"step-{step}"
                    save_checkpoint(model, optimizer, step, avg_loss if step % 10 == 0 else 0.0, ckpt_path)
                    prune_old_checkpoints(Path(args.output_dir), args.keep_last_n)
                    logger.log(step, {"msg": f"Checkpoint saved to {ckpt_path}"})

                if step >= args.max_steps:
                    break

    # Final checkpoint
    ckpt_path = Path(args.output_dir) / f"step-{step}-final"
    save_checkpoint(model, optimizer, step, 0.0, ckpt_path)
    logger.log(step, {"msg": f"Training complete. Final checkpoint at {ckpt_path}"})


if __name__ == "__main__":
    main()
