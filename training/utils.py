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
    """Save model weights, optimizer state, and training state for true resume.

    `model.parameters()` returns a nested dict tree; `mx.savez` only accepts
    flat array kwargs, so we flatten with `tree_flatten` before unpacking.
    """
    from mlx.utils import tree_flatten
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    flat_weights = dict(tree_flatten(model.parameters()))
    mx.savez(str(path / "model.npz"), **flat_weights)
    # Optimizer state must be persisted for resume — otherwise Adam moments reset
    # and the LR schedule jumps discontinuously on restart.
    try:
        flat_opt = dict(tree_flatten(optimizer.state))
        mx.savez(str(path / "optimizer.npz"), **flat_opt)
    except Exception as e:
        (path / "optimizer.MISSING").write_text(f"{type(e).__name__}: {e}\n")
    state = {"step": step, "loss": loss}
    (path / "state.json").write_text(json.dumps(state, indent=2))


def load_checkpoint(model: nn.Module, path: str | Path, optimizer=None) -> dict:
    """Load model weights (and optimizer state, if provided). Returns training state dict."""
    from mlx.utils import tree_unflatten
    path = Path(path)
    weights = mx.load(str(path / "model.npz"))
    # tree_unflatten reconstructs the nested dict expected by load_weights / update.
    model.update(tree_unflatten(list(weights.items())))
    if optimizer is not None and (path / "optimizer.npz").exists():
        try:
            opt_state = mx.load(str(path / "optimizer.npz"))
            optimizer.state.update(tree_unflatten(list(opt_state.items())))
        except Exception:
            pass
    state = json.loads((path / "state.json").read_text())
    return state


def prune_old_checkpoints(output_dir: Path, keep_last_n: int) -> None:
    """Keep only the N most-recent step-* checkpoint directories in output_dir.

    Preserves any directory whose name ends in '-final' or is named 'best'.
    """
    if keep_last_n <= 0:
        return
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return
    candidates = [
        p for p in output_dir.iterdir()
        if p.is_dir()
        and p.name.startswith("step-")
        and not p.name.endswith("-final")
        and p.name != "best"
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime)
    for old in candidates[:-keep_last_n]:
        import shutil
        shutil.rmtree(old, ignore_errors=True)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    from mlx.utils import tree_flatten
    return sum(p.size for _, p in tree_flatten(model.parameters()))


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
