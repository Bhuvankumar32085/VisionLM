"""Optimizer and Learning Rate Scheduler setup."""

from typing import Any, Dict, Tuple
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: Dict[str, Any],
    num_training_steps: int,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Creates the AdamW optimizer and LR scheduler for trainable parameters.

    Args:
        model: VisionLanguageModel instance.
        config: Training configuration dictionary.
        num_training_steps: Total number of optimization steps across epochs.

    Returns:
        Tuple of (optimizer, scheduler).
    """
    train_cfg = config.get("training", {})
    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    warmup_steps = int(train_cfg.get("warmup_steps", 0))
    scheduler_type = train_cfg.get("scheduler_type", "cosine").lower()

    # Filter strictly for parameters that require gradients (e.g. Projector)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if not trainable_params:
        raise ValueError("No trainable parameters found in model! Check freezing configurations.")

    optimizer = AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    if scheduler_type == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    else:
        # Constant LR
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    return optimizer, scheduler
