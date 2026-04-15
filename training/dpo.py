"""Direct Preference Optimization (DPO) alignment training for OmniscientLLM using MLX."""

import argparse
import json
import copy
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
    TrainingLogger,
)


def get_sequence_logprobs(
    model: nn.Module,
    input_ids: mx.array,
    labels: mx.array,
) -> mx.array:
    """Compute per-sequence sum of log probabilities where labels != -100.

    Args:
        model: The language model.
        input_ids: Shape (batch_size, seq_len).
        labels: Shape (batch_size, seq_len). -100 for masked positions.

    Returns:
        Shape (batch_size,) — sum of log probs per sequence.
    """
    logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
    vocab_size = logits.shape[-1]

    # Shift: predict next token
    shift_logits = logits[:, :-1, :]  # (B, S-1, V)
    shift_labels = labels[:, 1:]      # (B, S-1)

    # Compute log softmax
    log_probs = shift_logits - mx.logsumexp(shift_logits, axis=-1, keepdims=True)  # (B, S-1, V)

    # Gather log probs for the target tokens
    mask = shift_labels != -100
    safe_labels = mx.where(mask, shift_labels, 0)  # (B, S-1)

    # Index into log_probs: gather along vocab dimension
    # For each position, get log_prob of the target token
    batch_size, seq_len = safe_labels.shape
    batch_idx = mx.broadcast_to(
        mx.arange(batch_size).reshape(-1, 1),
        safe_labels.shape,
    )
    seq_idx = mx.broadcast_to(
        mx.arange(seq_len).reshape(1, -1),
        safe_labels.shape,
    )
    token_log_probs = log_probs[batch_idx, seq_idx, safe_labels]  # (B, S-1)

    # Mask and sum
    token_log_probs = token_log_probs * mask
    return token_log_probs.sum(axis=-1)  # (B,)


def dpo_loss(
    policy_chosen_logps: mx.array,
    policy_rejected_logps: mx.array,
    ref_chosen_logps: mx.array,
    ref_rejected_logps: mx.array,
    beta: float = 0.1,
) -> mx.array:
    """Standard DPO loss.

    loss = -log(sigmoid(beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))))

    Args:
        policy_chosen_logps: Log probs of chosen under policy, shape (B,).
        policy_rejected_logps: Log probs of rejected under policy, shape (B,).
        ref_chosen_logps: Log probs of chosen under reference, shape (B,).
        ref_rejected_logps: Log probs of rejected under reference, shape (B,).
        beta: DPO temperature parameter.

    Returns:
        Scalar loss.
    """
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    logits = chosen_rewards - rejected_rewards
    loss = -nn.losses.log_sigmoid(logits)
    return loss.mean()


def load_preference_data(path: str | Path):
    """Load preference pairs from JSONL.

    Each line should have:
        {"prompt": str, "chosen": str, "rejected": str}

    Returns list of dicts.
    """
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def tokenize_preference_pair(example: dict, tokenizer, max_seq_len: int):
    """Tokenize a preference pair into input_ids and labels for chosen/rejected.

    Returns dict with chosen_ids, chosen_labels, rejected_ids, rejected_labels.
    """
    prompt = example["prompt"]
    chosen = example["chosen"]
    rejected = example["rejected"]

    prompt_ids = tokenizer.encode(prompt).ids
    chosen_ids = tokenizer.encode(chosen).ids
    rejected_ids = tokenizer.encode(rejected).ids

    # Concatenate prompt + response
    chosen_input_ids = (prompt_ids + chosen_ids)[:max_seq_len]
    rejected_input_ids = (prompt_ids + rejected_ids)[:max_seq_len]

    # Labels: mask prompt tokens with -100
    prompt_len = len(prompt_ids)
    chosen_labels = [-100] * min(prompt_len, len(chosen_input_ids)) + chosen_input_ids[prompt_len:]
    rejected_labels = [-100] * min(prompt_len, len(rejected_input_ids)) + rejected_input_ids[prompt_len:]

    return {
        "chosen_ids": chosen_input_ids,
        "chosen_labels": chosen_labels,
        "rejected_ids": rejected_input_ids,
        "rejected_labels": rejected_labels,
    }


def pad_and_batch(examples: list[dict], pad_id: int):
    """Pad tokenized preference pairs and create batched mx arrays.

    Returns dict with chosen_ids, chosen_labels, rejected_ids, rejected_labels as mx arrays.
    """
    # Find max lengths
    max_chosen_len = max(len(ex["chosen_ids"]) for ex in examples)
    max_rejected_len = max(len(ex["rejected_ids"]) for ex in examples)

    chosen_ids_batch = []
    chosen_labels_batch = []
    rejected_ids_batch = []
    rejected_labels_batch = []

    for ex in examples:
        # Pad chosen
        c_pad = max_chosen_len - len(ex["chosen_ids"])
        chosen_ids_batch.append(ex["chosen_ids"] + [pad_id] * c_pad)
        chosen_labels_batch.append(ex["chosen_labels"] + [-100] * c_pad)

        # Pad rejected
        r_pad = max_rejected_len - len(ex["rejected_ids"])
        rejected_ids_batch.append(ex["rejected_ids"] + [pad_id] * r_pad)
        rejected_labels_batch.append(ex["rejected_labels"] + [-100] * r_pad)

    return {
        "chosen_ids": mx.array(chosen_ids_batch),
        "chosen_labels": mx.array(chosen_labels_batch),
        "rejected_ids": mx.array(rejected_ids_batch),
        "rejected_labels": mx.array(rejected_labels_batch),
    }


def main():
    parser = argparse.ArgumentParser(description="DPO alignment training for OmniscientLLM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SFT checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Preference pairs JSONL path")
    parser.add_argument("--output-dir", type=str, default="checkpoints/dpo", help="Output directory")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta parameter")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro-batch size")
    parser.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max-steps", type=int, default=3000, help="Maximum training steps")
    parser.add_argument("--warmup-steps", type=int, default=150, help="Linear warmup steps")
    parser.add_argument("--scheduler", type=str, default="constant", choices=["constant", "cosine"], help="LR schedule")
    parser.add_argument("--min-effective-pairs", type=int, default=32, help="Fail-fast floor on batch_size * grad_accum")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Path to tokenizer.json")
    args = parser.parse_args()

    effective_pairs = args.batch_size * args.grad_accum
    if effective_pairs < args.min_effective_pairs:
        raise SystemExit(
            f"DPO effective batch = {effective_pairs} preference pairs "
            f"(batch_size={args.batch_size} * grad_accum={args.grad_accum}); "
            f"must be >= {args.min_effective_pairs}. Increase --grad-accum."
        )

    # Load tokenizer
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        pad_id = 0

    # Load config from checkpoint directory
    ckpt_path = Path(args.checkpoint)
    config_path = ckpt_path.parent / "config.json"
    if config_path.exists():
        config = OmniscientConfig.load(config_path)
    else:
        config = OmniscientConfig()

    # Build policy model (trainable)
    policy_model = OmniscientModel(config)
    load_checkpoint(policy_model, args.checkpoint)

    # Build reference model (frozen) — separate instance with same weights
    ref_model = OmniscientModel(config)
    load_checkpoint(ref_model, args.checkpoint)
    ref_model.freeze()

    logger = TrainingLogger("dpo", use_wandb=args.wandb)
    param_count = count_parameters(policy_model)
    logger.log(0, {"params": param_count, "msg": f"Loaded SFT model from {args.checkpoint}"})

    # Optimizer
    optimizer = optim.AdamW(
        learning_rate=args.lr,
        betas=[0.9, 0.95],
        weight_decay=0.01,
    )

    # Load preference data
    preference_data = load_preference_data(args.data)
    logger.log(0, {"msg": f"Loaded {len(preference_data)} preference pairs"})

    # Tokenize all examples
    tokenized_examples = [
        tokenize_preference_pair(ex, tokenizer, config.max_seq_len)
        for ex in preference_data
    ]

    # DPO forward pass for policy model (used for value_and_grad)
    def compute_dpo_loss(policy, batch):
        # Policy log probs
        policy_chosen_logps = get_sequence_logprobs(policy, batch["chosen_ids"], batch["chosen_labels"])
        policy_rejected_logps = get_sequence_logprobs(policy, batch["rejected_ids"], batch["rejected_labels"])

        # Reference log probs (no grad)
        ref_chosen_logps = mx.stop_gradient(
            get_sequence_logprobs(ref_model, batch["chosen_ids"], batch["chosen_labels"])
        )
        ref_rejected_logps = mx.stop_gradient(
            get_sequence_logprobs(ref_model, batch["rejected_ids"], batch["rejected_labels"])
        )

        loss = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=args.beta,
        )
        return loss

    loss_and_grad_fn = nn.value_and_grad(policy_model, compute_dpo_loss)

    # Training loop
    import random
    step = 0
    accumulated_grads = None
    accum_count = 0
    accum_loss = 0.0

    while step < args.max_steps:
        # Shuffle data each epoch
        random.shuffle(tokenized_examples)

        for i in range(0, len(tokenized_examples), args.batch_size):
            if step >= args.max_steps:
                break

            batch_examples = tokenized_examples[i : i + args.batch_size]
            if not batch_examples:
                continue

            batch = pad_and_batch(batch_examples, pad_id)

            # Update LR
            if args.scheduler == "cosine":
                lr = cosine_lr_schedule(step, args.lr, args.lr * 0.1, args.warmup_steps, args.max_steps)
            else:  # constant w/ linear warmup
                lr = args.lr * min(1.0, (step + 1) / max(1, args.warmup_steps))
            optimizer.learning_rate = lr

            # Forward + backward
            loss, grads = loss_and_grad_fn(policy_model, batch)
            mx.eval(loss)

            # Accumulate gradients
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = mx.tree_map(lambda a, b: a + b, accumulated_grads, grads)

            accum_loss += loss.item()
            accum_count += 1

            # Step optimizer every grad_accum micro-batches
            if accum_count % args.grad_accum == 0:
                scale = 1.0 / args.grad_accum
                averaged_grads = mx.tree_map(lambda g: g * scale, accumulated_grads)

                optimizer.update(policy_model, averaged_grads)
                mx.eval(policy_model.parameters(), optimizer.state)

                accumulated_grads = None
                step += 1

                avg_loss = accum_loss / args.grad_accum
                accum_loss = 0.0

                # Log every 10 steps
                if step % 10 == 0:
                    # Compute reward margins for logging
                    with mx.no_grad():
                        p_chosen = get_sequence_logprobs(policy_model, batch["chosen_ids"], batch["chosen_labels"])
                        p_rejected = get_sequence_logprobs(policy_model, batch["rejected_ids"], batch["rejected_labels"])
                        chosen_margin = (p_chosen - p_rejected).mean().item()

                    logger.log(step, {
                        "loss": avg_loss,
                        "chosen_reward_margin": chosen_margin,
                    })

                # Checkpoint every 500 steps
                if step % 500 == 0:
                    ckpt_path = Path(args.output_dir) / f"step-{step}"
                    save_checkpoint(policy_model, optimizer, step, avg_loss, ckpt_path)
                    logger.log(step, {"msg": f"Checkpoint saved to {ckpt_path}"})

                if step >= args.max_steps:
                    break

    # Final checkpoint
    ckpt_path = Path(args.output_dir) / f"step-{step}-final"
    save_checkpoint(policy_model, optimizer, step, 0.0, ckpt_path)
    logger.log(step, {"msg": f"DPO training complete. Final checkpoint at {ckpt_path}"})


if __name__ == "__main__":
    main()
