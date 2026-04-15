"""Supervised Fine-Tuning (SFT) for OmniscientLLM using MLX."""

import argparse
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_map

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.config import OmniscientConfig
from model.transformer import OmniscientModel
from training.utils import (
    cosine_lr_schedule,
    masked_cross_entropy,
    save_checkpoint,
    load_checkpoint,
    count_parameters,
    TrainingLogger,
)
from data.sft_data import load_sft_dataset, create_sft_dataloader


def train_step(model, batch):
    """Single forward pass + masked cross-entropy loss."""
    logits = model(batch["input_ids"])
    loss = masked_cross_entropy(logits, batch["labels"])
    return loss


def main():
    parser = argparse.ArgumentParser(description="SFT training for OmniscientLLM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to pretrained checkpoint")
    parser.add_argument("--data", type=str, required=True, help="SFT JSONL dataset path")
    parser.add_argument("--output-dir", type=str, default="checkpoints/sft", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Maximum epochs (upper bound; --max-steps may stop earlier)")
    parser.add_argument("--max-steps", type=int, default=8000, help="Hard cap on optimizer steps")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--scheduler", type=str, default="linear-warmup-constant", choices=["linear-warmup-constant", "cosine"], help="LR schedule")
    parser.add_argument("--early-stop-patience", type=int, default=2, help="Stop if epoch loss fails to improve for N epochs (0 disables)")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Path to tokenizer.json")
    args = parser.parse_args()

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # Load config from checkpoint directory and build model
    ckpt_path = Path(args.checkpoint)
    config_path = ckpt_path.parent / "config.json"
    if config_path.exists():
        config = OmniscientConfig.load(config_path)
    else:
        # Fallback: use default config
        config = OmniscientConfig()

    model = OmniscientModel(config)

    # Load pretrained weights
    state = load_checkpoint(model, args.checkpoint)
    logger = TrainingLogger("sft", use_wandb=args.wandb)
    param_count = count_parameters(model)
    logger.log(0, {"params": param_count, "msg": f"Loaded pretrained model from {args.checkpoint}"})

    # Optimizer
    optimizer = optim.AdamW(
        learning_rate=args.lr,
        betas=[0.9, 0.95],
        weight_decay=0.1,
    )

    # Load SFT dataset
    conversations = load_sft_dataset(args.data)
    logger.log(0, {"msg": f"Loaded {len(conversations)} conversations from {args.data}"})

    # Total optimizer steps is min(epoch-derived, --max-steps). The LR schedule
    # uses this effective horizon so warmup + decay land at the real end of training.
    steps_per_epoch = max(1, len(conversations) // (args.batch_size * args.grad_accum))
    epoch_derived_steps = steps_per_epoch * args.epochs
    total_steps = min(epoch_derived_steps, args.max_steps)

    # Build loss_and_grad function
    loss_and_grad_fn = nn.value_and_grad(model, lambda m, batch: train_step(m, batch))

    # Training loop
    global_step = 0
    best_loss = float("inf")
    epochs_since_improve = 0
    accumulated_grads = None
    accum_count = 0
    accum_loss = 0.0
    stop_training = False

    for epoch in range(args.epochs):
        if stop_training:
            break
        epoch_loss = 0.0
        epoch_batches = 0

        dataloader = create_sft_dataloader(
            conversations,
            tokenizer,
            batch_size=args.batch_size,
            max_seq_len=config.max_seq_len,
            shuffle=True,
        )

        for batch in dataloader:
            if global_step >= args.max_steps:
                stop_training = True
                break

            # Convert to mx arrays
            batch = {
                "input_ids": mx.array(batch["input_ids"]),
                "labels": mx.array(batch["labels"]),
            }

            # Update LR
            if args.scheduler == "cosine":
                lr = cosine_lr_schedule(global_step, args.lr, args.lr * 0.1, args.warmup_steps, total_steps)
            else:  # linear-warmup-constant
                lr = args.lr * min(1.0, (global_step + 1) / max(1, args.warmup_steps))
            optimizer.learning_rate = lr

            # Forward + backward
            loss, grads = loss_and_grad_fn(model, batch)
            mx.eval(loss)

            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(lambda a, b: a + b, accumulated_grads, grads)

            accum_loss += loss.item()
            accum_count += 1
            epoch_loss += loss.item()
            epoch_batches += 1

            # Step optimizer every grad_accum micro-batches
            if accum_count % args.grad_accum == 0:
                # Average gradients
                scale = 1.0 / args.grad_accum
                averaged_grads = tree_map(lambda g: g * scale, accumulated_grads)

                # Apply optimizer step
                optimizer.update(model, averaged_grads)
                mx.eval(model.parameters(), optimizer.state)

                accumulated_grads = None
                global_step += 1

                avg_loss = accum_loss / args.grad_accum
                accum_loss = 0.0

                # Log every 10 steps
                if global_step % 10 == 0:
                    logger.log(global_step, {
                        "loss": avg_loss,
                        "lr": lr,
                        "epoch": epoch + 1,
                    })

        # End of epoch
        if epoch_batches > 0:
            epoch_avg_loss = epoch_loss / epoch_batches
        else:
            epoch_avg_loss = float("inf")

        logger.log(global_step, {
            "epoch_loss": epoch_avg_loss,
            "epoch": epoch + 1,
            "msg": f"Epoch {epoch + 1}/{args.epochs} complete",
        })

        # Save best model + early-stop bookkeeping
        if epoch_avg_loss < best_loss:
            best_loss = epoch_avg_loss
            epochs_since_improve = 0
            ckpt_path = Path(args.output_dir) / "best"
            save_checkpoint(model, optimizer, global_step, best_loss, ckpt_path)
            logger.log(global_step, {"msg": f"Best model saved to {ckpt_path} (loss={best_loss:.4f})"})
        else:
            epochs_since_improve += 1
            if args.early_stop_patience > 0 and epochs_since_improve >= args.early_stop_patience:
                logger.log(global_step, {"msg": f"Early stop: no improvement for {epochs_since_improve} epochs"})
                stop_training = True

        # Save epoch checkpoint
        ckpt_path = Path(args.output_dir) / f"epoch-{epoch + 1}"
        save_checkpoint(model, optimizer, global_step, epoch_avg_loss, ckpt_path)

    logger.log(global_step, {"msg": "SFT training complete"})


if __name__ == "__main__":
    main()
