"""LLaVA Dataset Preparation & Split Pipeline.

Processes raw LLaVA-Instruct metadata, validates image files on disk,
normalizes questions/answers, strips <image> markers, creates deterministic
train/validation splits (seed=42), and outputs processed JSONL files.
"""

import argparse
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image
from tqdm import tqdm

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.config import load_config


def clean_image_marker(text: str) -> str:
    """Strips '<image>' tags and cleans whitespace from prompt strings."""
    cleaned = re.sub(r"<image>\s*", "", text)
    cleaned = re.sub(r"\s*<image>", "", cleaned)
    return cleaned.strip()


def normalize_image_filename(img_ref: str) -> str:
    """Extracts a standardized filename (e.g. '000000123456.jpg') from any LLaVA image reference."""
    clean = os.path.basename(img_ref).replace("COCO_train2014_", "").replace("COCO_val2014_", "")
    base, ext = os.path.splitext(clean)
    if base.isdigit():
        return f"{int(base):012d}.jpg"
    return clean if clean.endswith((".jpg", ".png", ".jpeg")) else f"{clean}.jpg"


def validate_image_file(image_path: Path) -> bool:
    """Validates that image exists, is non-empty, and can be opened with PIL."""
    if not image_path.exists() or image_path.stat().st_size < 100:
        return False
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def prepare_dataset(
    raw_metadata_dir: str = "data/raw/llava/metadata",
    raw_images_dir: str = "data/raw/llava/images",
    output_dir: str = "data/processed",
    train_ratio: float = 0.98,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[int, int]:
    """Processes raw metadata and images into train.jsonl and validation.jsonl.

    Args:
        raw_metadata_dir: Directory containing raw LLaVA JSON files.
        raw_images_dir: Directory containing downloaded image files.
        output_dir: Output directory for processed JSONL files.
        train_ratio: Fraction of samples for training split (e.g. 0.98).
        seed: Random seed for deterministic splitting.
        max_samples: Optional cap on total processed samples.

    Returns:
        Tuple of (num_train_samples, num_val_samples).
    """
    meta_path = Path(raw_metadata_dir)
    images_path = Path(raw_images_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("LLaVA DATASET PREPARATION & VALIDATION PIPELINE")
    print("=" * 65)
    print(f"Raw Metadata Dir:   {meta_path.resolve()}")
    print(f"Raw Images Dir:     {images_path.resolve()}")
    print(f"Output Dir:         {out_path.resolve()}")
    print(f"Train/Val Split:    {train_ratio*100:.1f}% / {(1 - train_ratio)*100:.1f}%")
    print(f"Random Seed:        {seed}")
    print("=" * 65)

    # 1. Discover all raw metadata JSON / JSONL files
    json_files = list(meta_path.glob("*.json")) + list(meta_path.glob("*.jsonl"))
    if not json_files:
        raise FileNotFoundError(f"No JSON or JSONL metadata files found in {meta_path}. Run download_dataset.py first.")

    raw_items: List[Dict[str, Any]] = []
    for jf in json_files:
        print(f"[INFO] Reading metadata file: {jf.name}...")
        if jf.suffix == ".jsonl":
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        raw_items.append(json.loads(line))
        else:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    raw_items.extend(data)

    total_raw_samples = len(raw_items)
    print(f"\n[INFO] Total raw metadata samples loaded: {total_raw_samples}")

    # 2. Process and validate samples
    valid_samples: List[Dict[str, Any]] = []
    missing_images = 0
    corrupted_images = 0
    invalid_convs = 0

    print("[INFO] Validating image files and conversations...")
    for idx, item in enumerate(tqdm(raw_items, desc="Processing Samples")):
        # Extract ID
        sample_id = str(item.get("id", f"sample_{idx:08d}"))

        # Resolve image file
        raw_img = item.get("image")
        if not raw_img:
            missing_images += 1
            continue

        clean_img_name = normalize_image_filename(raw_img)
        img_full_path = images_path / clean_img_name

        if not img_full_path.exists():
            missing_images += 1
            continue

        # Fast verify cached image
        if not validate_image_file(img_full_path):
            corrupted_images += 1
            continue

        # Extract first human -> assistant pair
        convs = item.get("conversations", [])
        if not isinstance(convs, list) or len(convs) < 2:
            invalid_convs += 1
            continue

        human_val = ""
        gpt_val = ""
        for turn in convs:
            sender = turn.get("from", "").lower()
            val = turn.get("value", "")
            if sender in ("human", "user") and not human_val:
                human_val = val
            elif sender in ("gpt", "assistant") and not gpt_val:
                gpt_val = val

        question = clean_image_marker(human_val)
        answer = gpt_val.strip()

        if not question or not answer:
            invalid_convs += 1
            continue

        # Relative path from repository root for clean portability
        try:
            rel_image_path = img_full_path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel_image_path = img_full_path.as_posix()

        processed_sample = {
            "id": sample_id,
            "image": rel_image_path,
            "question": question,
            "answer": answer,
            "raw_metadata": {
                "original_image": raw_img,
                "num_turns": len(convs),
            },
        }
        valid_samples.append(processed_sample)

        if max_samples and len(valid_samples) >= max_samples:
            break

    # 3. Print Dataset Statistics
    invalid_samples = missing_images + corrupted_images + invalid_convs
    print("\n" + "=" * 65)
    print("DATASET PREPARATION STATISTICS")
    print("=" * 65)
    print(f"Total Raw Samples:      {total_raw_samples}")
    print(f"Valid Processed Samples:{len(valid_samples)}")
    print(f"Invalid / Skipped:      {invalid_samples}")
    print(f"  - Missing Images:     {missing_images}")
    print(f"  - Corrupted Images:   {corrupted_images}")
    print(f"  - Invalid Dialogues:  {invalid_convs}")
    print("=" * 65)

    if len(valid_samples) == 0:
        raise RuntimeError("No valid samples could be prepared. Ensure images have been downloaded with download_dataset.py.")

    # 4. Deterministic Train/Validation Split
    rng = random.Random(seed)
    rng.shuffle(valid_samples)

    num_train = int(len(valid_samples) * train_ratio)
    train_samples = valid_samples[:num_train]
    val_samples = valid_samples[num_train:]

    # Fallback to at least 1 validation sample if train_ratio is very high on small subset
    if len(val_samples) == 0 and len(valid_samples) > 1:
        val_samples = [train_samples.pop()]

    # 5. Save output JSONL files
    train_file = out_path / "train.jsonl"
    val_file = out_path / "validation.jsonl"

    print(f"\n[INFO] Saving {len(train_samples)} samples to {train_file}...")
    with open(train_file, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[INFO] Saving {len(val_samples)} samples to {val_file}...")
    with open(val_file, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("=" * 65)
    print("DATASET PREPARATION COMPLETE")
    print(f"Train File:      {train_file.resolve()} ({len(train_samples)} samples)")
    print(f"Validation File: {val_file.resolve()} ({len(val_samples)} samples)")
    print("=" * 65)

    return len(train_samples), len(val_samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LLaVA dataset and create train/validation splits.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml.")
    parser.add_argument("--raw-metadata-dir", type=str, default="data/raw/llava/metadata", help="Metadata directory.")
    parser.add_argument("--raw-images-dir", type=str, default="data/raw/llava/images", help="Images directory.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Processed output directory.")
    parser.add_argument("--train-ratio", type=float, default=0.98, help="Train split ratio (default 0.98).")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional max samples to process.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config) if Path(args.config).exists() else {}
    seed = config.get("project", {}).get("seed", args.seed)

    prepare_dataset(
        raw_metadata_dir=args.raw_metadata_dir,
        raw_images_dir=args.raw_images_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=seed,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
