"""LLaVA-Instruct-150K Dataset Downloader.

Downloads metadata and MS COCO images for the complete LLaVA-Instruct-150K dataset
with support for resume, multi-threaded fetching, integrity verification, and subsets.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from PIL import Image
import requests
from tqdm import tqdm

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Official Hugging Face metadata endpoints for LLaVA-Instruct-150K
HF_BASE_URL = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main"
METADATA_FILES = {
    "detail": "detail_23k.json",
    "complex": "complex_reasoning_77k.json",
    "conversation": "conversation_58k.json",
    "all": "llava_instruct_150k.json",
}

# COCO Image repositories
COCO_IMAGE_URLS = [
    "http://images.cocodataset.org/train2014/COCO_train2014_{id}.jpg",
    "http://images.cocodataset.org/val2014/COCO_val2014_{id}.jpg",
    "http://images.cocodataset.org/train2017/{id}.jpg",
    "http://images.cocodataset.org/val2017/{id}.jpg",
]


def download_file_stream(url: str, output_path: Path, desc: str = "Downloading") -> bool:
    """Streams a file download with progress bar and atomic write."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(temp_path, "wb") as f, tqdm(
            desc=desc,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=True,
        ) as bar:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        if temp_path.exists():
            if output_path.exists():
                output_path.unlink()
            temp_path.rename(output_path)
        return True
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        print(f"[ERROR] Failed to download {url}: {e}")
        return False


def normalize_image_filename(img_ref: str) -> str:
    """Extracts a standardized filename (e.g. '000000123456.jpg') from any LLaVA image reference."""
    clean = os.path.basename(img_ref).replace("COCO_train2014_", "").replace("COCO_val2014_", "")
    # If purely digits or filename
    base, ext = os.path.splitext(clean)
    if base.isdigit():
        return f"{int(base):012d}.jpg"
    return clean if clean.endswith((".jpg", ".png", ".jpeg")) else f"{clean}.jpg"


def fetch_single_image(img_ref: str, images_dir: Path, session: requests.Session) -> Tuple[str, bool, str]:
    """Fetches a single COCO image if not already present on disk."""
    clean_filename = normalize_image_filename(img_ref)
    target_path = images_dir / clean_filename

    # Skip if valid file already exists
    if target_path.exists() and target_path.stat().st_size > 1000:
        return clean_filename, True, "exists"

    # Extract 12-digit ID for URL lookup
    base, _ = os.path.splitext(clean_filename)
    if base.isdigit():
        formatted_id = f"{int(base):012d}"
    else:
        formatted_id = base

    for url_tmpl in COCO_IMAGE_URLS:
        url = url_tmpl.format(id=formatted_id)
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 1000:
                with open(target_path, "wb") as f:
                    f.write(resp.content)
                return clean_filename, True, "downloaded"
        except Exception:
            continue

    return clean_filename, False, "not_found"


def verify_image(img_path: Path) -> bool:
    """Verifies that an image file can be opened and decoded properly by PIL."""
    if not img_path.exists() or img_path.stat().st_size < 100:
        return False
    try:
        with Image.open(img_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def download_dataset(
    output_dir: str = "data/raw/llava",
    splits: List[str] = ["detail"],
    subset_size: Optional[int] = None,
    workers: int = 16,
    verify: bool = True,
    resume: bool = True,
) -> bool:
    """Downloads LLaVA-Instruct-150K metadata and image files.

    Args:
        output_dir: Base directory to store raw metadata and images.
        splits: Metadata splits to download ('detail', 'complex', 'conversation', 'all').
        subset_size: Optional limit on number of images/samples to download (e.g. 1000 or None for all).
        workers: Number of concurrent worker threads for image fetching.
        verify: Whether to verify downloaded images with PIL.
        resume: Whether to resume downloading and skip already existing valid images.

    Returns:
        True if download succeeded.
    """
    base_dir = Path(output_dir)
    metadata_dir = base_dir / "metadata"
    images_dir = base_dir / "images"

    metadata_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("LLaVA-INSTRUCT-150K DATASET DOWNLOADER")
    print("=" * 65)
    print(f"Output Directory:   {base_dir.resolve()}")
    print(f"Metadata Directory: {metadata_dir.resolve()}")
    print(f"Images Directory:   {images_dir.resolve()}")
    print(f"Requested Splits:   {splits}")
    print(f"Subset Limit:       {subset_size if subset_size else 'FULL DATASET'}")
    print(f"Concurrent Workers: {workers}")
    print("=" * 65)

    # 1. Download Metadata Files
    image_references: Set[str] = set()
    loaded_metadata: List[Dict[str, Any]] = []

    for split_key in splits:
        filename = METADATA_FILES.get(split_key, f"{split_key}.json")
        target_json = metadata_dir / filename
        url = f"{HF_BASE_URL}/{filename}"

        if target_json.exists() and resume:
            print(f"[INFO] Metadata file already exists: {target_json.name}")
        else:
            print(f"[INFO] Downloading metadata: {filename} from Hugging Face...")
            success = download_file_stream(url, target_json, desc=f"Metadata ({filename})")
            if not success:
                print(f"[ERROR] Failed downloading metadata for split: {split_key}")
                return False

        # Parse metadata to collect image references
        try:
            with open(target_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    loaded_metadata.extend(data)
                    for item in data:
                        img = item.get("image")
                        if img:
                            image_references.add(img)
        except Exception as e:
            print(f"[ERROR] Failed parsing metadata {target_json}: {e}")
            return False

    print(f"\n[INFO] Total metadata samples loaded: {len(loaded_metadata)}")
    print(f"[INFO] Unique images referenced:     {len(image_references)}")

    # Apply subset limit if requested
    img_list = sorted(list(image_references))
    if subset_size is not None and subset_size > 0:
        img_list = img_list[:subset_size]
        print(f"[INFO] Restricted image download list to subset of {len(img_list)} images.")

    # 2. Multi-threaded Image Downloading
    print(f"\n[INFO] Checking and downloading {len(img_list)} images into {images_dir}...")
    success_count = 0
    exists_count = 0
    downloaded_count = 0
    failed_images: List[str] = []

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers * 2, pool_maxsize=workers * 2, max_retries=3)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch_single_image, img_ref, images_dir, session): img_ref for img_ref in img_list}

        with tqdm(total=len(img_list), desc="Images Progress", unit="img") as pbar:
            for future in as_completed(futures):
                img_ref = futures[future]
                try:
                    filename, ok, status = future.result()
                    if ok:
                        success_count += 1
                        if status == "exists":
                            exists_count += 1
                        else:
                            downloaded_count += 1
                    else:
                        failed_images.append(img_ref)
                except Exception:
                    failed_images.append(img_ref)
                pbar.update(1)

    print("\n" + "=" * 65)
    print("DOWNLOAD SUMMARY")
    print("=" * 65)
    print(f"Total Target Images:  {len(img_list)}")
    print(f"Already Existed:      {exists_count}")
    print(f"Newly Downloaded:     {downloaded_count}")
    print(f"Successfully Ready:   {success_count}")
    print(f"Failed / Missing:     {len(failed_images)}")

    # 3. Verification step
    if verify and success_count > 0:
        print("\n[INFO] Verifying image integrity with PIL...")
        corrupted = 0
        all_downloaded = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        for img_p in tqdm(all_downloaded, desc="Verifying Images", unit="img"):
            if not verify_image(img_p):
                corrupted += 1
                try:
                    img_p.unlink()  # Remove corrupted file so it can be re-downloaded
                except Exception:
                    pass
        print(f"[INFO] Verified {len(all_downloaded)} image files. Corrupted / removed: {corrupted}")

    print("=" * 65)
    print("DATASET DOWNLOAD COMPLETED")
    print("=" * 65)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download LLaVA-Instruct-150K dataset and COCO images.")
    parser.add_argument("--output-dir", type=str, default="data/raw/llava", help="Target raw dataset directory.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["detail"],
        choices=["detail", "complex", "conversation", "all"],
        help="LLaVA split(s) to download.",
    )
    parser.add_argument("--subset-size", type=int, default=None, help="Optional maximum number of images to download.")
    parser.add_argument("--workers", type=int, default=16, help="Number of download threads.")
    parser.add_argument("--no-verify", action="store_true", help="Skip image integrity verification.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip existing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    success = download_dataset(
        output_dir=args.output_dir,
        splits=args.splits,
        subset_size=args.subset_size,
        workers=args.workers,
        verify=not args.no_verify,
        resume=not args.no_resume,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
