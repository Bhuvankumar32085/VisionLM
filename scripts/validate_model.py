"""Comprehensive Post-Training Model Validation & Evaluation Script.

Evaluates trained VLM checkpoints on the validation dataset:
1. Calculates quantitative metrics (Average Loss, Perplexity).
2. Generates qualitative answers on validation samples and compares with Ground Truth.
"""

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Optional
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.collator import VLMDataCollator
from data.dataset import VLMInstructionDataset
from data.preprocessing import load_and_preprocess_image
from inference.generate import generate_answer
from models.language_model import LanguageModel
from models.vision_encoder import VisionEncoder
from models.vlm import VisionLanguageModel
from training.checkpoint import load_checkpoint
from utils.config import load_config
from utils.device import get_device


def validate_checkpoint(
    checkpoint_path: str = "checkpoints/checkpoint_best.pt",
    val_file: Optional[str] = None,
    config_path: str = "configs/config.yaml",
    batch_size: int = 1,
    num_samples_to_generate: int = 5,
    max_new_tokens: int = 64,
) -> None:
    """Performs quantitative and qualitative evaluation of a trained VLM checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        val_file: Path to validation.jsonl (if None, loaded from config).
        config_path: Path to config.yaml.
        batch_size: Evaluation batch size.
        num_samples_to_generate: Number of sample images to generate answers for.
        max_new_tokens: Max tokens to generate per answer.
    """
    config = load_config(config_path)
    device = get_device()

    # Determine validation file
    if val_file is None:
        val_file = config.get("data", {}).get("validation_file", "data/processed/validation.jsonl")

    if not os.path.exists(val_file):
        # Fallback to subset if processed validation doesn't exist
        if os.path.exists("data/subsets/llava_subset_500.jsonl"):
            val_file = "data/subsets/llava_subset_500.jsonl"
        else:
            raise FileNotFoundError(f"Validation file not found: {val_file}")

    print("=" * 65)
    print("VLM MODEL POST-TRAINING VALIDATION")
    print("=" * 65)
    print(f"Device:          {device}")
    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"Validation File: {val_file}")
    print("=" * 65)

    vision_cfg = config.get("vision", {})
    lang_cfg = config.get("language", {})
    proj_cfg = config.get("projector", {})
    train_cfg = config.get("training", {})

    # 1. Load Processors & Tokenizer
    image_processor = VisionEncoder.get_image_processor(vision_cfg.get("model_name", "google/vit-base-patch16-224-in21k"))
    tokenizer = LanguageModel.get_tokenizer(lang_cfg.get("model_name", "HuggingFaceTB/SmolLM2-360M-Instruct"))

    # 2. Instantiate Model
    print("\n[1/3] Loading Model Architecture...")
    model = VisionLanguageModel(
        vision_model_name=vision_cfg.get("model_name", "google/vit-base-patch16-224-in21k"),
        language_model_name=lang_cfg.get("model_name", "HuggingFaceTB/SmolLM2-360M-Instruct"),
        use_cls_token=bool(vision_cfg.get("use_cls_token", False)),
        projector_hidden_dim=int(proj_cfg.get("hidden_dim", 1536)),
        projector_activation=str(proj_cfg.get("activation", "gelu")),
        projector_dropout=float(proj_cfg.get("dropout", 0.0)),
        freeze_vision=True,
        freeze_language=True,
    ).to(device)

    # 3. Load Checkpoint
    if os.path.exists(checkpoint_path):
        ckpt_info = load_checkpoint(checkpoint_path=checkpoint_path, model=model)
        epoch = ckpt_info.get("epoch", "N/A")
        step = ckpt_info.get("global_step", "N/A")
        print(f"[SUCCESS] Loaded weights from {checkpoint_path} (Epoch: {epoch}, Step: {step})")
    else:
        print(f"[WARNING] Checkpoint {checkpoint_path} not found! Evaluating model with initial weights.")

    model.eval()

    # 4. Quantitative Evaluation (Loss & Perplexity)
    print("\n[2/3] Computing Quantitative Validation Metrics...")
    val_dataset = VLMInstructionDataset(data_source=val_file)
    collator = VLMDataCollator(
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_question_length=int(config.get("data", {}).get("max_question_length", 128)),
        max_answer_length=int(config.get("data", {}).get("max_answer_length", 256)),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    precision_str = str(train_cfg.get("precision", "bf16")).lower()
    amp_dtype = torch.bfloat16 if (precision_str == "bf16" and torch.cuda.is_bf16_supported()) else (torch.float16 if precision_str == "fp16" else None)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating Batches", unit="batch"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if amp_dtype is not None and device.type == "cuda":
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

            if output.loss is not None and not torch.isnan(output.loss):
                total_loss += output.loss.item()
                num_batches += 1

    avg_loss = (total_loss / num_batches) if num_batches > 0 else 0.0
    try:
        perplexity = math.exp(avg_loss) if avg_loss < 50 else float("inf")
    except OverflowError:
        perplexity = float("inf")

    print("\n" + "-" * 40)
    print("QUANTITATIVE RESULTS")
    print("-" * 40)
    print(f"Total Validation Samples: {len(val_dataset)}")
    print(f"Validation Loss:          {avg_loss:.4f}")
    print(f"Validation Perplexity:    {perplexity:.2f}")
    print("-" * 40)

    # 5. Qualitative Inference Evaluation
    if num_samples_to_generate > 0:
        print(f"\n[3/3] Generating Qualitative Visual QA Responses (First {min(num_samples_to_generate, len(val_dataset))} samples)...")
        print("=" * 65)

        for i in range(min(num_samples_to_generate, len(val_dataset))):
            sample = val_dataset[i]
            img = sample["image"]
            question = sample["question"]
            gt_answer = sample["answer"]

            generated_answer = generate_answer(
                model=model,
                image=img,
                question=question,
                image_processor=image_processor,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

            print(f"\n[Sample #{i + 1}]")
            print(f"Question:         {question}")
            print(f"Generated Answer: {generated_answer}")
            print(f"Ground Truth:     {gt_answer}")
            print("-" * 65)

    print("\n" + "=" * 65)
    print("VALIDATION COMPLETE")
    print("=" * 65)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained VLM on validation dataset.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/checkpoint_best.pt",
        help="Path to trained checkpoint (.pt).",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default=None,
        help="Path to validation dataset (.jsonl). If None, reads from configs/config.yaml.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of qualitative sample answers to generate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max tokens for generated answers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_checkpoint(
        checkpoint_path=args.checkpoint,
        val_file=args.val_file,
        config_path=args.config,
        num_samples_to_generate=args.num_samples,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
