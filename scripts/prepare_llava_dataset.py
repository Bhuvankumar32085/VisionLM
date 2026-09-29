"""Script to download, subset, and prepare real LLaVA-Instruct dataset with images."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
import random
import urllib.request
from typing import Any, Dict, List, Optional
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm


def download_single_coco_image(image_id_str: str, dest_path: str, timeout: int = 5) -> bool:
    """Attempts to download a COCO image from official URLs.

    Args:
        image_id_str: Standard image filename or ID.
        dest_path: Local destination file path.
        timeout: HTTP timeout.

    Returns:
        True if valid image exists or is downloaded, False otherwise.
    """
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        try:
            with Image.open(dest_path) as img:
                img.verify()
            return True
        except Exception:
            if os.path.exists(dest_path):
                os.remove(dest_path)

    clean_id = os.path.splitext(os.path.basename(image_id_str))[0]
    if clean_id.isdigit() and len(clean_id) < 12:
        clean_id = clean_id.zfill(12)

    candidate_urls = [
        f"http://images.cocodataset.org/train2014/COCO_train2014_{clean_id}.jpg",
        f"http://images.cocodataset.org/val2014/COCO_val2014_{clean_id}.jpg",
        f"http://images.cocodataset.org/train2017/{clean_id}.jpg",
        f"http://images.cocodataset.org/val2017/{clean_id}.jpg",
    ]

    for url in candidate_urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    data = response.read()
                    img = Image.open(io.BytesIO(data))
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(dest_path, "JPEG", quality=95)
                    return True
        except Exception:
            continue

    return False


def prepare_llava_subset(
    repo_id: str = "liuhaotian/LLaVA-Instruct-150K",
    filename: str = "detail_23k.json",
    num_samples: int = 500,
    seed: int = 42,
    output_dir: str = "data/subsets",
    images_dir: str = "data/subsets/images",
    max_workers: int = 10,
) -> str:
    """Downloads LLaVA metadata, samples a deterministic subset, and downloads images."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print(f"Downloading {filename} from Hugging Face ({repo_id})...")
    json_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")

    with open(json_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    print(f"Loaded {len(full_data)} total entries from {filename}.")

    # Deterministic order
    random.seed(seed)
    indices = list(range(len(full_data)))
    random.shuffle(indices)

    print(f"Sampling {num_samples} items with seed={seed}...")

    successful_items: List[Dict[str, Any]] = []

    # Check already downloaded files first
    for idx in indices:
        if len(successful_items) >= num_samples:
            break
        item = full_data[idx]
        img_field = item.get("image")
        if not img_field:
            continue
        clean_filename = os.path.basename(img_field)
        if not clean_filename.endswith(".jpg") and not clean_filename.endswith(".png"):
            clean_filename = f"{clean_filename}.jpg"

        local_img_path = os.path.join(images_dir, clean_filename)
        rel_img_path = os.path.relpath(local_img_path, start=".").replace("\\", "/")

        if download_single_coco_image(img_field, local_img_path):
            successful_items.append({
                "id": item.get("id", ""),
                "image": rel_img_path,
                "conversations": item.get("conversations", []),
            })

    print(f"Collected {len(successful_items)} verified samples.")

    # Save to JSONL
    subset_jsonl_name = f"llava_subset_{len(successful_items)}.jsonl"
    subset_jsonl_path = os.path.join(output_dir, subset_jsonl_name).replace("\\", "/")

    with open(subset_jsonl_path, "w", encoding="utf-8") as f:
        for sample in successful_items:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved real LLaVA subset to: {subset_jsonl_path}")
    return subset_jsonl_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LLaVA real dataset subset.")
    parser.add_argument("--repo_id", type=str, default="liuhaotian/LLaVA-Instruct-150K")
    parser.add_argument("--filename", type=str, default="detail_23k.json")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="data/subsets")
    parser.add_argument("--images_dir", type=str, default="data/subsets/images")
    parser.add_argument("--max_workers", type=int, default=10)

    args = parser.parse_args()
    prepare_llava_subset(
        repo_id=args.repo_id,
        filename=args.filename,
        num_samples=args.num_samples,
        seed=args.seed,
        output_dir=args.output_dir,
        images_dir=args.images_dir,
        max_workers=args.max_workers,
    )
