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


def test_parameter_update() -> bool:
    """Verifies that an optimizer step properly updates projector parameters."""
    print("=" * 50)
    print("TESTING PARAMETER UPDATES")
    print("=" * 50)

    device = get_device()
    print(f"Using device: {device}")

    # Load model
    vlm = VisionLanguageModel(
        vision_model_name="google/vit-base-patch16-224-in21k",
        language_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        use_cls_token=False,
        projector_hidden_dim=1536,
        freeze_vision=True,
        freeze_language=True,
    ).to(device)

    # Initialize optimizer with ONLY trainable projector parameters
    trainable_params = [p for p in vlm.projector.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)

    processor = VisionEncoder.get_image_processor()
    tokenizer = LanguageModel.get_tokenizer()

    sample_img = Image.new("RGB", (224, 224), color=(20, 80, 140))
    pixel_values = processor(images=sample_img, return_tensors="pt")["pixel_values"].to(device)

    prompt = "Question: What color is the image? Answer:"
    target_answer = " Blue."
    full_text = prompt + target_answer

    prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    full_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"].to(device)
    attention_mask = torch.ones_like(full_tokens)

    labels = full_tokens.clone()
    prompt_len = prompt_tokens.shape[1]
    labels[:, :prompt_len] = -100

    # Save copy of projector parameters before step
    initial_params = {name: param.clone().detach() for name, param in vlm.projector.named_parameters()}

    # Forward + Backward + Step
    optimizer.zero_grad()
    output = vlm(
        pixel_values=pixel_values,
        input_ids=full_tokens,
        attention_mask=attention_mask,
        labels=labels,
    )
    loss = output.loss
    assert loss is not None, "Loss is None"
    loss.backward()
    optimizer.step()

    # Verify changes
    all_updated = True
    for name, param in vlm.projector.named_parameters():
        init_tensor = initial_params[name]
        diff = (param - init_tensor).abs().max().item()
        if diff == 0.0:
            print(f"[FAIL] Projector parameter {name} did NOT update!")
            all_updated = False
        else:
            print(f"Projector {name}: max diff = {diff:.6e}")

    print(f"Projector update: {'PASS' if all_updated else 'FAIL'}")
    print("=" * 50)
    return all_updated


if __name__ == "__main__":
    success = test_parameter_update()
    sys.exit(0 if success else 1)
