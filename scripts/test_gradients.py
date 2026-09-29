import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image

from models.language_model import LanguageModel
from models.vision_encoder import VisionEncoder
from models.vlm import VisionLanguageModel
from utils.device import get_device


def test_gradients() -> bool:
    """Verifies that only projector parameters receive gradients during backward."""
    print("=" * 50)
    print("TESTING GRADIENTS AND PARAMETER FREEZING")
    print("=" * 50)

    device = get_device()
    print(f"Using device: {device}")

    # Load model with frozen vision & language, trainable projector
    vlm = VisionLanguageModel(
        vision_model_name="google/vit-base-patch16-224-in21k",
        language_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        use_cls_token=False,
        projector_hidden_dim=1536,
        freeze_vision=True,
        freeze_language=True,
    ).to(device)

    # Ensure training mode for projector
    vlm.train()
    vlm.vision_encoder.eval()
    vlm.language_model.eval()

    processor = VisionEncoder.get_image_processor()
    tokenizer = LanguageModel.get_tokenizer()

    # Create dummy inputs
    sample_img = Image.new("RGB", (224, 224), color=(50, 100, 150))
    pixel_values = processor(images=sample_img, return_tensors="pt")["pixel_values"].to(device)

    prompt = "Question: What is in this image? Answer:"
    target_answer = " A blue square."
    full_text = prompt + target_answer

    prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    full_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
    attention_mask = torch.ones_like(full_tokens)

    labels = full_tokens.clone()
    prompt_len = prompt_tokens.shape[1]
    labels[:, :prompt_len] = -100

    # Forward pass
    vlm.zero_grad()
    output = vlm(
        pixel_values=pixel_values,
        input_ids=full_tokens,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = output.loss
    assert loss is not None, "Loss is None in test_gradients!"

    # Backward pass
    loss.backward()

    # Check projector gradients (Must NOT be None)
    projector_grads_ok = True
    for name, param in vlm.projector.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"[FAIL] Projector parameter {name} has NO gradient!")
                projector_grads_ok = False
            elif torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"[FAIL] Projector parameter {name} has NaN/Inf gradient!")
                projector_grads_ok = False

    # Check Vision Encoder parameters (Must be None)
    vision_frozen_ok = True
    for name, param in vlm.vision_encoder.named_parameters():
        if param.grad is not None:
            print(f"[FAIL] Vision parameter {name} unexpectedly has gradient!")
            vision_frozen_ok = False

    # Check LLM parameters (Must be None)
    llm_frozen_ok = True
    for name, param in vlm.language_model.named_parameters():
        if param.grad is not None:
            print(f"[FAIL] LLM parameter {name} unexpectedly has gradient!")
            llm_frozen_ok = False

    print(f"Projector gradients: {'PASS' if projector_grads_ok else 'FAIL'}")
    print(f"Vision frozen:       {'PASS' if vision_frozen_ok else 'FAIL'}")
    print(f"LLM frozen:          {'PASS' if llm_frozen_ok else 'FAIL'}")
    print("=" * 50)

    success = projector_grads_ok and vision_frozen_ok and llm_frozen_ok
    return success


if __name__ == "__main__":
    success = test_gradients()
    sys.exit(0 if success else 1)
