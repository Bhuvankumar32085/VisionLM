import argparse
from pathlib import Path
import sys
from typing import Any, Optional
import torch
from PIL import Image

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.preprocessing import format_prompt, load_and_preprocess_image
from models.language_model import LanguageModel
from models.vision_encoder import VisionEncoder
from models.vlm import VisionLanguageModel
from training.checkpoint import load_checkpoint
from utils.config import load_config
from utils.device import get_device


def generate_answer(
    model: VisionLanguageModel,
    image: Image.Image,
    question: str,
    image_processor: Any,
    tokenizer: Any,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = False,
    prompt_template: Optional[str] = None,
) -> str:
    """Generates an answer autoregressively given an input image and a question prompt.

    Args:
        model: Trained VisionLanguageModel.
        image: Input PIL Image.
        question: User query string.
        image_processor: ViT AutoImageProcessor.
        tokenizer: SmolLM2 Tokenizer.
        device: Device to run generation on.
        max_new_tokens: Max number of tokens to generate.
        temperature: Softmax sampling temperature.
        top_p: Nucleus sampling cutoff.
        do_sample: Whether to sample or use greedy argmax decoding.
        prompt_template: Optional prompt formatting template.

    Returns:
        Generated text string.
    """
    model.eval()

    # 1. Process image
    image_inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(device)

    # 2. Extract and project visual features
    with torch.no_grad():
        projected_visual_features = model.encode_images(pixel_values)  # [1, 196, 960]

    # 3. Format & Tokenize Question Prompt
    prompt_text = format_prompt(question, prompt_template)
    text_inputs = tokenizer(prompt_text, return_tensors="pt")
    prompt_ids = text_inputs["input_ids"].to(device)  # [1, prompt_len]

    # 4. Generate initial combined sequence embeddings
    with torch.no_grad():
        prompt_embeddings = model.language_model.get_input_embeddings()(prompt_ids)  # [1, prompt_len, 960]
        # Combined sequence: [visual tokens (196)] + [prompt tokens (prompt_len)]
        current_embeddings = torch.cat([projected_visual_features, prompt_embeddings], dim=1)

    generated_token_ids = []
    eos_token_id = tokenizer.eos_token_id

    # 5. Autoregressive generation loop
    with torch.no_grad():
        for step in range(max_new_tokens):
            seq_len = current_embeddings.shape[1]
            attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)

            # Pass into LLM decoder
            logits, _ = model.language_model(
                inputs_embeds=current_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

            # Extract logits for the last token position
            next_token_logits = logits[:, -1, :]  # [1, vocab_size]

            if do_sample and temperature > 0.0:
                scaled_logits = next_token_logits / temperature
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    scaled_logits = scaled_logits.masked_fill(indices_to_remove, -float("inf"))

                probs = torch.softmax(scaled_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1).item()
            else:
                # Greedy argmax decoding
                next_token_id = torch.argmax(next_token_logits, dim=-1).item()

            generated_token_ids.append(next_token_id)

            # Check stop criteria
            if eos_token_id is not None and next_token_id == eos_token_id:
                break

            # Embed the new token and append to sequence for the next iteration
            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            next_token_embed = model.language_model.get_input_embeddings()(next_token_tensor)
            current_embeddings = torch.cat([current_embeddings, next_token_embed], dim=1)

    # 6. Decode generated tokens
    generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
    return generated_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal VLM Generation Inference.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file.")
    parser.add_argument("--question", type=str, required=True, help="Question / Prompt for the model.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint (.pt).")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml.")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling.")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling instead of greedy decoding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device()

    vision_cfg = config.get("vision", {})
    lang_cfg = config.get("language", {})
    proj_cfg = config.get("projector", {})

    print(f"Loading models on device: {device}...")
    image_processor = VisionEncoder.get_image_processor(vision_cfg.get("model_name", "google/vit-base-patch16-224-in21k"))
    tokenizer = LanguageModel.get_tokenizer(lang_cfg.get("model_name", "HuggingFaceTB/SmolLM2-360M-Instruct"))

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

    if args.checkpoint:
        load_checkpoint(args.checkpoint, model=model)
        print(f"Loaded weights from checkpoint: {args.checkpoint}")

    # Load and preprocess image
    image = load_and_preprocess_image(args.image)

    print("\n" + "=" * 50)
    print(f"Image:    {args.image}")
    print(f"Question: {args.question}")
    print("=" * 50)

    answer = generate_answer(
        model=model,
        image=image,
        question=args.question,
        image_processor=image_processor,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    print(f"Answer:   {answer}")
    print("=" * 50)


if __name__ == "__main__":
    main()
