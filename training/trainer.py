"""Custom PyTorch Training Loop for Multimodal VLM."""

import time
from typing import Any, Dict, Optional
import torch
from torch.utils.data import DataLoader

from training.checkpoint import load_checkpoint, save_checkpoint
from training.evaluate import evaluate
from training.optimizer import build_optimizer_and_scheduler
from utils.device import get_device, get_gpu_memory_mb
from utils.logging import get_logger

logger = get_logger("trainer")


class VLMTrainer:
    """Modular custom PyTorch Trainer for VisionLanguageModel."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initializes the trainer.

        Args:
            model: VisionLanguageModel instance.
            train_dataloader: DataLoader for training dataset.
            val_dataloader: Optional DataLoader for validation dataset.
            config: Full configuration dictionary.
            device: Optional torch.device to train on.
        """
        self.config = config or {}
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        train_cfg = self.config.get("training", {})
        self.epochs = int(train_cfg.get("epochs", 3))
        self.grad_accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
        self.max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
        self.logging_steps = int(train_cfg.get("logging_steps", 1))
        self.eval_steps = int(train_cfg.get("eval_steps", 10))

        # Precision & AMP setup
        precision_str = str(train_cfg.get("precision", "fp16")).lower()
        self.scaler: Optional[torch.amp.GradScaler] = None
        self.amp_dtype: Optional[torch.dtype] = None

        if self.device.type == "cuda":
            if precision_str == "bf16":
                if torch.cuda.is_bf16_supported():
                    self.amp_dtype = torch.bfloat16
                    logger.info("Using Mixed Precision: BF16 (No GradScaler needed)")
                else:
                    logger.warning("BF16 requested but not supported on this GPU. Falling back to FP16.")
                    self.amp_dtype = torch.float16
                    self.scaler = torch.amp.GradScaler("cuda")
            elif precision_str == "fp16":
                self.amp_dtype = torch.float16
                self.scaler = torch.amp.GradScaler("cuda")
                logger.info("Using Mixed Precision: FP16 with GradScaler")
            else:
                self.amp_dtype = None
                logger.info("Using Full Precision: FP32")
        else:
            self.amp_dtype = None
            logger.info("Running on CPU: Full Precision FP32")

        # Total training steps
        steps_per_epoch = len(self.train_dataloader)
        self.total_training_steps = (steps_per_epoch * self.epochs) // self.grad_accum_steps
        if self.total_training_steps == 0:
            self.total_training_steps = 1

        # Optimizer and Scheduler
        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            model=self.model,
            config=self.config,
            num_training_steps=self.total_training_steps,
        )

        # Tracking variables
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Checkpoint directory
        ckpt_cfg = self.config.get("checkpoint", {})
        self.checkpoint_dir = ckpt_cfg.get("directory", "checkpoints")
        self.save_every_steps = int(ckpt_cfg.get("save_every_steps", 50))
        self.save_best = bool(ckpt_cfg.get("save_best", True))

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """Restores model and optimizer states from a checkpoint.

        Args:
            checkpoint_path: Path to the .pt checkpoint file.
        """
        meta = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
        )
        self.start_epoch = meta.get("epoch", 0)
        self.global_step = meta.get("global_step", 0)
        logger.info(f"Resumed training from epoch {self.start_epoch}, global step {self.global_step}")

    def train(self) -> Dict[str, Any]:
        """Runs the complete training loop across all configured epochs.

        Returns:
            Dictionary with training history and metrics.
        """
        logger.info("=" * 60)
        logger.info("STARTING VLM TRAINING")
        logger.info(f"Epochs: {self.epochs} | Steps/epoch: {len(self.train_dataloader)} | Grad accum: {self.grad_accum_steps}")
        logger.info(f"Total optimization steps: {self.total_training_steps}")
        logger.info(f"Device: {self.device} | Mixed precision: {self.amp_dtype}")
        logger.info("=" * 60)

        accumulated_loss = 0.0
        accumulated_samples = 0
        self.optimizer.zero_grad()
        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            self.model.train()
            # Ensure frozen submodules remain in eval mode
            if hasattr(self.model, "vision_encoder"):
                self.model.vision_encoder.eval()
            if hasattr(self.model, "language_model"):
                self.model.language_model.eval()

            logger.info(f"--- Epoch {epoch + 1}/{self.epochs} ---")

            for batch_idx, batch in enumerate(self.train_dataloader):
                step_start_time = time.time()

                pixel_values = batch["pixel_values"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass with optional AMP
                if self.amp_dtype is not None and self.device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype):
                        output = self.model(
                            pixel_values=pixel_values,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                        loss = output.loss / self.grad_accum_steps
                else:
                    output = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = output.loss / self.grad_accum_steps

                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accumulated_loss += (output.loss.item())
                accumulated_samples += 1

                # Optimizer step on accumulation boundary
                if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_dataloader):
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.max_grad_norm,
                        )
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        # Only advance scheduler if optimizer was not skipped due to Inf/NaN gradients
                        if scale_after >= scale_before and self.scheduler is not None:
                            self.scheduler.step()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in self.model.parameters() if p.requires_grad],
                            self.max_grad_norm,
                        )
                        self.optimizer.step()
                        if self.scheduler is not None:
                            self.scheduler.step()

                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % self.logging_steps == 0:
                        current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.get("training", {}).get("learning_rate", 1e-4)
                        gpu_mem = get_gpu_memory_mb()
                        step_time = time.time() - step_start_time
                        avg_loss = (accumulated_loss / accumulated_samples) if accumulated_samples > 0 else 0.0

                        mem_str = f" | GPU Mem: {gpu_mem:.1f} MB" if gpu_mem is not None else ""
                        logger.info(
                            f"Epoch {epoch + 1}/{self.epochs} | Step {self.global_step}/{self.total_training_steps} | "
                            f"Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                            f"Time: {step_time:.3f}s{mem_str}"
                        )
                        accumulated_loss = 0.0
                        accumulated_samples = 0

                    # Evaluation
                    if self.val_dataloader is not None and (self.global_step % self.eval_steps == 0):
                        val_metrics = evaluate(
                            model=self.model,
                            dataloader=self.val_dataloader,
                            device=self.device,
                            amp_dtype=self.amp_dtype,
                        )
                        val_loss = val_metrics["val_loss"]
                        val_ppl = val_metrics["val_perplexity"]
                        logger.info(f"[Validation] Step {self.global_step} | Val Loss: {val_loss:.4f} | Val Perplexity: {val_ppl:.2f}")

                        is_best = val_loss < self.best_val_loss
                        if is_best:
                            self.best_val_loss = val_loss

                        if self.save_best and is_best:
                            save_checkpoint(
                                checkpoint_dir=self.checkpoint_dir,
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                scaler=self.scaler,
                                epoch=epoch,
                                global_step=self.global_step,
                                config=self.config,
                                is_best=True,
                            )
                        self.model.train()

                    # Periodic Checkpointing
                    if self.global_step % self.save_every_steps == 0:
                        save_checkpoint(
                            checkpoint_dir=self.checkpoint_dir,
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                            epoch=epoch,
                            global_step=self.global_step,
                            config=self.config,
                            is_best=False,
                        )

        # Final checkpoint save at end of training
        final_ckpt_path = save_checkpoint(
            checkpoint_dir=self.checkpoint_dir,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=self.epochs,
            global_step=self.global_step,
            config=self.config,
            is_best=False,
            tag="final",
        )

        total_elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"TRAINING COMPLETE in {total_elapsed:.2f} seconds.")
        logger.info(f"Final checkpoint saved to: {final_ckpt_path}")
        logger.info("=" * 60)

        return {
            "global_step": self.global_step,
            "final_checkpoint": final_ckpt_path,
            "best_val_loss": self.best_val_loss,
        }
