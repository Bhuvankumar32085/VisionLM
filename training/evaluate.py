"""Validation and evaluation functions for VLM."""

import math
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader

from utils.logging import get_logger

logger = get_logger("evaluate")


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    amp_dtype: Optional = None,
) -> Dict[str, float]:
    """Evaluates the model over the validation dataset.

    Calculates average cross-entropy loss and text perplexity.
    (Note: loss/perplexity measures language modeling fit on the answers,
    and should not be conflated with full visual reasoning capability).

    Args:
        model: The VisionLanguageModel.
        dataloader: DataLoader for validation data.
        device: Device to run evaluation on.
        amp_dtype: Optional AMP precision dtype (e.g. torch.float16, torch.bfloat16).

    Returns:
        Dictionary with 'val_loss' and 'val_perplexity'.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    use_amp = (amp_dtype is not None and device.type == "cuda")

    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                    output = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
            else:
                output = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            loss = output.loss
            if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                num_batches += 1

    avg_loss = (total_loss / num_batches) if num_batches > 0 else 0.0
    try:
        perplexity = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    except OverflowError:
        perplexity = float("inf")

    return {
        "val_loss": avg_loss,
        "val_perplexity": perplexity,
    }
