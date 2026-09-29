"""CLI Entry Point for Vision-Language Model Fine-Tuning."""

import argparse
import sys
from torch.utils.data import DataLoader

from data.collator import VLMDataCollator
from data.dataset import VLMInstructionDataset
from models.language_model import LanguageModel
from models.vision_encoder import VisionEncoder
from models.vlm import VisionLanguageModel
from training.trainer import VLMTrainer
from utils.config import load_config
from utils.device import get_device
from utils.logging import get_logger
from utils.seed import set_seed

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune custom Vision-Language Model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt file to resume training from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # 1. Reproducibility
    seed = config.get("project", {}).get("seed", 42)
    set_seed(seed)
    device = get_device()
    logger.info(f"Initialized training on device: {device} with seed: {seed}")

    # 2. Vision and Tokenizer processors
    vision_cfg = config.get("vision", {})
    lang_cfg = config.get("language", {})

    image_processor = VisionEncoder.get_image_processor(vision_cfg.get("model_name", "google/vit-base-patch16-224-in21k"))
    tokenizer = LanguageModel.get_tokenizer(lang_cfg.get("model_name", "HuggingFaceTB/SmolLM2-360M-Instruct"))

    # 3. Data Loaders
    data_cfg = config.get("data", {})
    train_file = data_cfg.get("train_file", "data/sample_dataset.jsonl")
    val_file = data_cfg.get("validation_file", "data/sample_dataset.jsonl")
    max_q_len = int(data_cfg.get("max_question_length", 128))
    max_a_len = int(data_cfg.get("max_answer_length", 128))
    num_workers = int(data_cfg.get("num_workers", 0))

    train_dataset = VLMInstructionDataset(data_source=train_file)
    val_dataset = VLMInstructionDataset(data_source=val_file) if val_file else None

    collator = VLMDataCollator(
        image_processor=image_processor,
        tokenizer=tokenizer,
        max_question_length=max_q_len,
        max_answer_length=max_a_len,
    )

    train_cfg = config.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 1))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
    )

    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
        )
        if val_dataset
        else None
    )

    # 4. Model Instantiation
    proj_cfg = config.get("projector", {})
    model = VisionLanguageModel(
        vision_model_name=vision_cfg.get("model_name", "google/vit-base-patch16-224-in21k"),
        language_model_name=lang_cfg.get("model_name", "HuggingFaceTB/SmolLM2-360M-Instruct"),
        use_cls_token=bool(vision_cfg.get("use_cls_token", False)),
        projector_hidden_dim=int(proj_cfg.get("hidden_dim", 1536)),
        projector_activation=str(proj_cfg.get("activation", "gelu")),
        projector_dropout=float(proj_cfg.get("dropout", 0.0)),
        freeze_vision=bool(vision_cfg.get("freeze", True)),
        freeze_language=bool(lang_cfg.get("freeze", True)),
    )

    # 5. Trainer
    trainer = VLMTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        device=device,
    )

    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    # Run Training
    results = trainer.train()
    logger.info(f"Training completed successfully! Final checkpoint: {results['final_checkpoint']}")


if __name__ == "__main__":
    main()
