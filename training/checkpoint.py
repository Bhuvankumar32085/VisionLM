"""Checkpoint saving and loading utilities."""

import os
from typing import Any, Dict, Optional
import torch

from utils.logging import get_logger

logger = get_logger("checkpoint")


def save_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.amp.GradScaler],
    epoch: int,
    global_step: int,
    config: Dict[str, Any],
    is_best: bool = False,
    tag: Optional[str] = None,
) -> str:
    """Saves complete training checkpoint containing model, projector, and optimizer states.

    Args:
        checkpoint_dir: Directory where checkpoints are stored.
        model: The VLM model.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        scaler: AMP GradScaler instance (if used).
        epoch: Current epoch index.
        global_step: Current global training step.
        config: Full configuration dictionary.
        is_best: Whether this checkpoint achieved best validation loss.
        tag: Optional custom name tag.

    Returns:
        Path to saved checkpoint file.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_data = {
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "model_state_dict": model.state_dict(),
        "projector_state_dict": model.projector.state_dict() if hasattr(model, "projector") else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
    }

    if tag:
        filename = f"checkpoint_{tag}.pt"
    else:
        filename = f"checkpoint_step_{global_step}.pt"

    save_path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint_data, save_path)
    logger.info(f"Saved checkpoint to: {save_path}")

    if is_best:
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        torch.save(checkpoint_data, best_path)
        logger.info(f"Updated best checkpoint at: {best_path}")

    return save_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, Any]:
    """Loads a training checkpoint and restores weights and states.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        model: VLM model to load weights into.
        optimizer: Optional optimizer to restore state into.
        scheduler: Optional scheduler to restore state into.
        scaler: Optional GradScaler to restore state into.

    Returns:
        Dictionary containing restored metadata ('epoch', 'global_step', 'config').
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    elif "projector_state_dict" in checkpoint and hasattr(model, "projector"):
        model.projector.load_state_dict(checkpoint["projector_state_dict"])

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # Load scaler state
    if scaler is not None and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"] is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "global_step": checkpoint.get("global_step", 0),
        "config": checkpoint.get("config", {}),
    }
