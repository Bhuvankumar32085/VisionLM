"""Inference validation script for Phase 1 trained VLM checkpoint."""

import json
from pathlib import Path
import torch
from PIL import Image

from data.preprocessing import load_and_preprocess_image
from inference.generate import generate_answer
from models.language_model import LanguageModel
from models.vision_encoder import VisionEncoder
from models.vlm import VisionLanguageModel
from training.checkpoint import load_checkpoint
from utils.config import load_config
from utils.device import get_device


def run_inference_eval(
    checkpoint_path: str = "checkpoints/projector_realdata_test/checkpoint_final.pt",
    subset_file: str = "data/subsets/llava_subset_500.jsonl",
    config_path: str = "configs/config.yaml",
    num_samples: int = 4,
):
    config = load_config(config_path)
    device = get_device()
    print(f"==================================================")
    print(f"RUNNING MULTIMODAL INFERENCE TEST")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"==================================================")

    vision_cfg = config.get("vision", {})
    lang_cfg = config.get("language", {})
    proj_cfg = config.get("projector", {})

    # Load components
    image_processor = VisionEncoder.get_image_processor(vision_cfg.get("model_name", "google/vit-base-patch16-224-in21k"))
    tokenizer = LanguageModel.get_tokenizer(lang_cfg.get("model_name", "HuggingFaceTB/SmolLM2-360M-Instruct"))

    model = VisionLanguageModel(
        vision_model_name=vision_cfg.get("model_name", "google/vit-base-patch16-224-in21k"),
        language_model_name=lang_cfg.get("model_name", "HuggingFaceTB/SmolLM2-360M-Instruct"),
        projector_hidden_dim=proj_cfg.get("hidden_dim", 1536),
        freeze_vision=True,
        freeze_language=True,
    ).to(device)

    # Load checkpoint
    if Path(checkpoint_path).exists():
        ckpt_info = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
        )
        print(f"Successfully loaded checkpoint from {checkpoint_path} (Epoch {ckpt_info.get('epoch', 0)}, Step {ckpt_info.get('global_step', 0)})")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Running with initial weights.")

    model.eval()

    # Load test samples
    samples = []
    with open(subset_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) >= num_samples:
                break

    print(f"\nEvaluating on {len(samples)} real samples from subset:\n")

    test_questions = [
        "What is in this image?",
        "Describe the image in detail.",
        "What objects can you see?",
        "What is shown in this picture?",
    ]

    for idx, sample in enumerate(samples):
        img_path = sample.get("image")
        convs = sample.get("conversations", [])
        gt_question = convs[0]["value"].replace("<image>\n", "").replace("<image>", "").strip() if len(convs) > 0 else ""
        gt_answer = convs[1]["value"].strip() if len(convs) > 1 else ""

        # Load image
        img = load_and_preprocess_image(img_path)

        # 1. Test with dataset prompt
        gen_answer_gt_q = generate_answer(
            model=model,
            image=img,
            question=gt_question,
            image_processor=image_processor,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=64,
            do_sample=False,
        )

        # 2. Test with generic prompt
        eval_q = test_questions[idx % len(test_questions)]
        gen_answer_eval_q = generate_answer(
            model=model,
            image=img,
            question=eval_q,
            image_processor=image_processor,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=64,
            do_sample=False,
        )

        print(f"--------------------------------------------------")
        print(f"Sample #{idx + 1}")
        print(f"Image Path: {img_path} ({img.size[0]}x{img.size[1]})")
        print(f"\n[Prompt 1 - Original Question]: {gt_question}")
        print(f"[Generated Answer]: {gen_answer_gt_q}")
        print(f"[Ground Truth Answer]: {gt_answer}")
        print(f"\n[Prompt 2 - General Question]: {eval_q}")
        print(f"[Generated Answer]: {gen_answer_eval_q}")
        print(f"--------------------------------------------------\n")


if __name__ == "__main__":
    run_inference_eval()
