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


def test_forward() -> bool:
    """Verifies that the forward pass runs without error, NaNs, or shape mismatches."""
    print("=" * 50)
    print("TESTING VLM FORWARD PASS")
    print("=" * 50)

    device = get_device()
    print(f"Using device: {device}")

    # Load VLM
    vlm = VisionLanguageModel(
        vision_model_name="google/vit-base-patch16-224-in21k",
        language_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        use_cls_token=False,
        projector_hidden_dim=1536,
        freeze_vision=True,
        freeze_language=True,
    ).to(device)
    vlm.eval()

    processor = VisionEncoder.get_image_processor()
    tokenizer = LanguageModel.get_tokenizer()

    # Create dummy image and input text
    sample_img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    pixel_values = processor(images=sample_img, return_tensors="pt")["pixel_values"].to(device)

    prompt = "Question: What is shown in the image? Answer:"
    target_answer = " A solid blue image."
    full_text = prompt + target_answer

    # Tokenize
    prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    full_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
    attention_mask = torch.ones_like(full_tokens)

    # Build labels: -100 for prompt tokens, real IDs for target answer tokens
    labels = full_tokens.clone()
    prompt_len = prompt_tokens.shape[1]
    labels[:, :prompt_len] = -100

    print(f"Input image shape: {list(pixel_values.shape)}")
    print(f"Text token count: {full_tokens.shape[1]} (Prompt: {prompt_len}, Answer: {full_tokens.shape[1] - prompt_len})")

    # Run forward pass
    output = vlm(
        pixel_values=pixel_values,
        input_ids=full_tokens,
        attention_mask=attention_mask,
        labels=labels,
        return_debug_dict=True,
    )

    logits = output.logits
    loss = output.loss

    # Validations
    assert logits is not None, "Logits tensor is None!"
    assert loss is not None, "Loss tensor is None!"

    # Shape validation: [B, 196 + text_len, vocab_size]
    expected_seq_len = 196 + full_tokens.shape[1]
    assert logits.shape[0] == 1, f"Expected batch size 1, got {logits.shape[0]}"
    assert logits.shape[1] == expected_seq_len, f"Expected seq_len {expected_seq_len}, got {logits.shape[1]}"
    assert logits.shape[2] == vlm.language_model.vocab_size, "Vocab size mismatch in logits"

    # NaN / Inf validation
    assert not torch.isnan(logits).any(), "NaN detected in logits!"
    assert not torch.isinf(logits).any(), "Inf detected in logits!"
    assert not torch.isnan(loss).any(), "NaN detected in loss!"
    assert not torch.isinf(loss).any(), "Inf detected in loss!"

    print(f"Logits shape: {list(logits.shape)}")
    print(f"Loss value:   {loss.item():.4f}")
    print("=" * 50)
    print("FORWARD PASS: PASS")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = test_forward()
    sys.exit(0 if success else 1)
