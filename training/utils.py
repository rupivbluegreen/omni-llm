"""Training utilities: LR schedules, checkpointing, loss functions."""

import math
import time
import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def cosine_lr_schedule(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def masked_cross_entropy(logits: mx.array, labels: mx.array) -> mx.array:
    """Cross-entropy loss ignoring positions where labels == -100."""
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size)
    labels = labels.reshape(-1)
    mask = labels != -100
    # Replace -100 with 0 for indexing (will be masked out)
    safe_labels = mx.where(mask, labels, 0)
    loss = nn.losses.cross_entropy(logits, safe_labels, reduction="none")
    return (loss * mask).sum() / mx.maximum(mask.sum(), 1)


def save_checkpoint(
    model: nn.Module,
    optimizer,
    step: int,
    loss: float,
    path: str | Path,
) -> None:
    """Save model weights and training state."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    # Save model weights
    mx.savez(str(path / "model.npz"), **dict(model.parameters()))
    # Save training state
    state = {"step": step, "loss": loss}
    (path / "state.json").write_text(json.dumps(state, indent=2))


def load_checkpoint(model: nn.Module, path: str | Path) -> dict:
    """Load model weights. Returns training state dict."""
    path = Path(path)
    weights = mx.load(str(path / "model.npz"))
    model.load_weights(list(weights.items()))
    state = json.loads((path / "state.json").read_text())
    return state


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.size for _, p in model.parameters())


class TrainingLogger:
    """Simple training logger with optional wandb."""

    def __init__(self, run_name: str, use_wandb: bool = False):
        self.logger = logging.getLogger(run_name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
            self.logger.addHandler(handler)
        self.use_wandb = use_wandb
        self._start_time = time.time()
        if use_wandb:
            import wandb
            wandb.init(project="omniscient-llm", name=run_name)

    def log(self, step: int, metrics: dict) -> None:
        elapsed = time.time() - self._start_time
        msg = f"step={step} | " + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
        msg += f" | elapsed={elapsed:.0f}s"
        self.logger.info(msg)
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)
