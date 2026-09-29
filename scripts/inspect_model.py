import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image

from models.language_model import LanguageModel
from models.projector import MLPProjector
from models.vision_encoder import VisionEncoder
from models.vlm import VisionLanguageModel
from utils.device import get_device


def inspect_model() -> None:
    """Instantiates components and prints tensor flow dimensions and parameter breakdown."""
    device = get_device()
    print(f"Running model inspection on device: {device}")
    print("=" * 60)

    # 1. Instantiate individual components
    print("Loading ViT Vision Encoder...")
    vision_encoder = VisionEncoder(model_name="google/vit-base-patch16-224-in21k", use_cls_token=False, freeze=True).to(device)
    processor = VisionEncoder.get_image_processor("google/vit-base-patch16-224-in21k")

    print("Loading SmolLM2 Language Model...")
    language_model = LanguageModel(model_name="HuggingFaceTB/SmolLM2-360M-Instruct", freeze=True).to(device)
    tokenizer = LanguageModel.get_tokenizer("HuggingFaceTB/SmolLM2-360M-Instruct")

    print("Instantiating MLP Projector...")
    projector = MLPProjector(
        input_dim=vision_encoder.hidden_size,
        hidden_dim=1536,
        output_dim=language_model.hidden_size,
        activation="gelu",
    ).to(device)

    # 2. Sample inputs
    # Synthetic RGB image of 224x224
    sample_img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    image_inputs = processor(images=sample_img, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device)  # [1, 3, 224, 224]

    # Sample text prompt
    prompt = "What is in this image?"
    text_inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = text_inputs["input_ids"].to(device)  # [1, text_len]
    attention_mask = text_inputs["attention_mask"].to(device)

    print("\n" + "=" * 60)
    print("TENSOR SHAPE INSPECTION")
    print("=" * 60)
    print(f"Input image: {list(pixel_values.shape)}")

    # ViT full output (with CLS)
    with torch.no_grad():
        raw_vit_out = vision_encoder.vit(pixel_values=pixel_values).last_hidden_state
    print(f"ViT output (with CLS): {list(raw_vit_out.shape)}")

    # VisionEncoder output (without CLS)
    patch_features = vision_encoder(pixel_values)
    print(f"Patch features (without CLS): {list(patch_features.shape)}")

    # Projector output
    projected_visual_features = projector(patch_features)
    print(f"Projected visual features: {list(projected_visual_features.shape)}")

    # Text token IDs & Embeddings
    print(f"Text token IDs: {list(input_ids.shape)}")
    with torch.no_grad():
        text_embeddings = language_model.get_input_embeddings()(input_ids)
    print(f"Text embeddings: {list(text_embeddings.shape)}")

    # 3. Instantiate full VisionLanguageModel
    print("\nInstantiating integrated VisionLanguageModel...")
    vlm = VisionLanguageModel(
        vision_model_name="google/vit-base-patch16-224-in21k",
        language_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        use_cls_token=False,
        projector_hidden_dim=1536,
        freeze_vision=True,
        freeze_language=True,
    ).to(device)

    # Run forward with debug dict
    output = vlm(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_debug_dict=True,
    )

    debug = output.debug_dict
    print(f"Combined embeddings: {list(debug['combined_embeddings'].shape)}")
    print(f"Attention mask: {list(debug['attention_mask'].shape)}")
    print(f"Position IDs: {list(debug['position_ids'].shape)}")
    print(f"Logits: {list(output.logits.shape)}")

    # 4. Parameter breakdown
    params = vlm.get_parameter_counts()
    print("\n" + "=" * 60)
    print("PARAMETER SUMMARY")
    print("=" * 60)
    print(f"Total parameters:      {params['total_params']:,}")
    print(f"Trainable parameters:  {params['trainable_params']:,}")
    print(f"Frozen parameters:     {params['frozen_params']:,}")
    print(f"Vision parameters:     {params['vision_params']:,} (Trainable: {params['vision_trainable']:,})")
    print(f"Projector parameters:  {params['projector_params']:,} (Trainable: {params['projector_trainable']:,})")
    print(f"LLM parameters:        {params['llm_params']:,} (Trainable: {params['llm_trainable']:,})")
    print("=" * 60)


if __name__ == "__main__":
    inspect_model()
