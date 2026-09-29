import argparse
import os
from pathlib import Path
import sys
from typing import Optional

# Ensure project root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from data.collator import VLMDataCollator
from data.dataset import VLMInstructionDataset
from models.language_model import LanguageModel
from models.vision_encoder import VisionEncoder
from models.vlm import VisionLanguageModel
from utils.device import get_device


def test_real_dataset_pipeline(data_file: Optional[str] = None) -> bool:
    """Performs end-to-end validation on real LLaVA dataset samples:

    dataset loading, collator shapes, label masking, forward pass, gradient flow, and parameter updates.
    """
    if data_file is None:
        if os.path.exists("data/processed/train.jsonl"):
            data_file = "data/processed/train.jsonl"
        elif os.path.exists("data/subsets/llava_subset_500.jsonl"):
            data_file = "data/subsets/llava_subset_500.jsonl"
        elif os.path.exists("data/sample_dataset.jsonl"):
            data_file = "data/sample_dataset.jsonl"
        else:
            raise FileNotFoundError("No dataset file found. Run download_dataset.py and prepare_dataset.py first.")

    device = get_device()
    print("=" * 65)
    print("REAL LLaVA-INSTRUCT DATASET PIPELINE VALIDATION")
    print(f"Device: {device} | Data file: {data_file}")
    print("=" * 65)

    # 1. Dataset Loading
    dataset = VLMInstructionDataset(data_source=data_file)
    print(f"\n[1] Dataset Loaded Successfully: Total Samples = {len(dataset)}")

    # 2. Print 3-5 Samples
    print("\n[2] Inspecting 3 Real Samples from Dataset:")
    print("-" * 65)
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        img = sample["image"]
        print(f"Sample {i + 1}:")
        print(f"  Image dimensions: {img.size} (Width x Height), Mode: {img.mode}")
        print(f"  Question:         {sample['question']}")
        # Truncate answer preview if long
        ans_preview = sample['answer'][:120] + ("..." if len(sample['answer']) > 120 else "")
        print(f"  Answer Preview:   {ans_preview}")
        print("-" * 65)

    # 3. Processors and Collator Setup
    image_processor = VisionEncoder.get_image_processor("google/vit-base-patch16-224-in21k")
    tokenizer = LanguageModel.get_tokenizer("HuggingFaceTB/SmolLM2-360M-Instruct")

    collator = VLMDataCollator(
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_question_length=128,
        max_answer_length=256,
    )

    # Create small batch of 2 samples
    batch_samples = [dataset[0], dataset[1]]
    batch = collator(batch_samples)

    pixel_values = batch["pixel_values"].to(device)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    print("\n[3] Collator Output Tensor Shapes (Batch Size = 2):")
    print(f"  pixel_values:   {list(pixel_values.shape)}  (Expected [2, 3, 224, 224])")
    print(f"  input_ids:      {list(input_ids.shape)}  (Expected [2, text_length])")
    print(f"  attention_mask: {list(attention_mask.shape)}  (Expected [2, text_length])")
    print(f"  labels:         {list(labels.shape)}  (Expected [2, text_length])")

    assert pixel_values.shape == (2, 3, 224, 224), f"Unexpected pixel_values shape: {pixel_values.shape}"
    assert input_ids.shape == attention_mask.shape == labels.shape, "Shape mismatch among text tensors!"

    # 4. Label Verification & Decoding
    print("\n[4] Label Masking & Target Token Verification:")
    sample_ids = input_ids[0].tolist()
    sample_labels = labels[0].tolist()

    # Find prompt tokens (labeled -100) vs target tokens (labeled with token ID)
    prompt_tokens_count = sum(1 for l in sample_labels if l == -100)
    target_tokens_count = sum(1 for l in sample_labels if l != -100)
    print(f"  Sample 0 total tokens: {len(sample_ids)} (Masked prompt/pad tokens: {prompt_tokens_count}, Active target answer tokens: {target_tokens_count})")

    # Decoded Full Input
    decoded_input = tokenizer.decode(sample_ids, skip_special_tokens=False)
    # Decoded Active Target Labels
    active_target_ids = [token_id for token_id, lbl in zip(sample_ids, sample_labels) if lbl != -100]
    decoded_target = tokenizer.decode(active_target_ids, skip_special_tokens=False)

    print(f"\n  [Decoded Input IDs]:\n  '{decoded_input}'")
    print(f"\n  [Decoded Target Label Tokens (Loss Computed ON ONLY THIS)]:\n  '{decoded_target}'")

    # 5. Model Instantiation & Forward Pass
    print("\n[5] VLM Model Forward Pass on Real Batch:")
    model = VisionLanguageModel(
        vision_model_name="google/vit-base-patch16-224-in21k",
        language_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        use_cls_token=False,
        projector_hidden_dim=1536,
        freeze_vision=True,
        freeze_language=True,
    ).to(device)
    model.train()
    model.vision_encoder.eval()
    model.language_model.eval()

    output = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_debug_dict=True,
    )

    debug = output.debug_dict
    print(f"  Visual Features (ViT):    {list(debug['vision_features'].shape)}")
    print(f"  Projected Features (MLP): {list(debug['projected_features'].shape)}")
    print(f"  Text Embeddings:          {list(debug['text_embeddings'].shape)}")
    print(f"  Combined Embeddings:      {list(debug['combined_embeddings'].shape)}")
    print(f"  Logits:                   {list(output.logits.shape)}")
    print(f"  Computed Causal Loss:     {output.loss.item():.4f}")

    assert output.logits.shape == (2, 196 + input_ids.shape[1], 49152), f"Logits shape mismatch: {output.logits.shape}"
    assert output.loss is not None and torch.isfinite(output.loss), "Loss is not a finite scalar!"

    # 6. Gradient Flow Verification
    print("\n[6] Gradient Flow & Parameter Freezing Check:")
    model.zero_grad()
    loss = output.loss
    loss.backward()

    # Projector gradient check & norm reporting
    projector_grads_ok = True
    print("  Projector Layer Gradients:")
    for name, param in model.projector.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"    [FAIL] {name}: NO gradient!")
                projector_grads_ok = False
            else:
                norm = param.grad.norm().item()
                print(f"    [PASS] {name}: grad_norm = {norm:.6f}")

    # Vision & LLM Frozen Check
    vision_frozen = all(param.grad is None for param in model.vision_encoder.parameters())
    llm_frozen = all(param.grad is None for param in model.language_model.parameters())
    print(f"  Vision Encoder Frozen:   {'PASS (grad == None)' if vision_frozen else 'FAIL'}")
    print(f"  Language Model Frozen:   {'PASS (grad == None)' if llm_frozen else 'FAIL'}")

    assert projector_grads_ok and vision_frozen and llm_frozen, "Gradient isolation test failed!"

    # 7. Parameter Update Verification
    print("\n[7] Parameter Update Test (1 Optimizer Step):")
    trainable_params = [p for p in model.projector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    initial_proj_params = {name: p.clone().detach() for name, p in model.projector.named_parameters()}
    initial_vit_sample = next(model.vision_encoder.parameters()).clone().detach()
    initial_llm_sample = next(model.language_model.parameters()).clone().detach()

    optimizer.step()

    # Check projector weights changed
    proj_updated = True
    for name, param in model.projector.named_parameters():
        delta = (param - initial_proj_params[name]).abs().max().item()
        if delta == 0.0:
            print(f"    [FAIL] {name} did not update!")
            proj_updated = False
        else:
            print(f"    [PASS] {name} delta = {delta:.6e}")

    # Check ViT and LLM weights did NOT change
    vit_delta = (next(model.vision_encoder.parameters()) - initial_vit_sample).abs().max().item()
    llm_delta = (next(model.language_model.parameters()) - initial_llm_sample).abs().max().item()

    print(f"  ViT weight delta: {vit_delta:.1e} ({'PASS (unchanged)' if vit_delta == 0 else 'FAIL'})")
    print(f"  LLM weight delta: {llm_delta:.1e} ({'PASS (unchanged)' if llm_delta == 0 else 'FAIL'})")

    assert proj_updated and vit_delta == 0.0 and llm_delta == 0.0, "Parameter update test failed!"

    print("\n" + "=" * 65)
    print("ALL REAL-DATA VALIDATION & PIPELINE TESTS: PASS")
    print("=" * 65)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test real dataset pipeline.")
    parser.add_argument("--data-file", type=str, default=None, help="Path to jsonl dataset.")
    args = parser.parse_args()
    test_real_dataset_pipeline(args.data_file)

