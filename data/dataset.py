"""Instruction Dataset for Vision-Language Fine-Tuning."""

import json
import os
import re
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from torch.utils.data import Dataset

from data.preprocessing import load_and_preprocess_image


def clean_image_tokens(text: str) -> str:
    """Strips '<image>' tags and extraneous whitespace/newlines from questions.

    Args:
        text: Raw question/instruction string.

    Returns:
        Cleaned text prompt.
    """
    # Remove variants like '<image>\n', '\n<image>', '<image>'
    cleaned = re.sub(r"<image>\s*", "", text)
    cleaned = re.sub(r"\s*<image>", "", cleaned)
    return cleaned.strip()


class VLMInstructionDataset(Dataset):
    """Dataset class for instruction-following image-text pairs.

    Supports both LLaVA-style conversation format and flat question-answer format.
    """

    def __init__(
        self,
        data_source: Union[str, List[Dict[str, Any]]],
        image_base_dir: Optional[str] = None,
    ) -> None:
        """Initializes the dataset from a JSONL file, JSON file, or list of dictionaries.

        Args:
            data_source: Path to .jsonl / .json file or direct list of item dicts.
            image_base_dir: Base directory path to resolve relative image paths.
        """
        self.image_base_dir = image_base_dir
        self.samples: List[Dict[str, Any]] = []

        if isinstance(data_source, str):
            if not os.path.exists(data_source):
                raise FileNotFoundError(f"Data file not found: {data_source}")

            if data_source.endswith(".jsonl"):
                with open(data_source, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.samples.append(json.loads(line))
            elif data_source.endswith(".json"):
                with open(data_source, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.samples = data
                    else:
                        raise ValueError("JSON file root must contain a list of objects.")
            else:
                raise ValueError(f"Unsupported file format: {data_source}. Must be .json or .jsonl")
        elif isinstance(data_source, list):
            self.samples = data_source
        else:
            raise TypeError(f"Invalid data_source type: {type(data_source)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieves and prepares a single sample.

        Returns:
            Dictionary with:
                - 'image': PIL.Image (RGB)
                - 'question': str (cleaned question text)
                - 'answer': str (target response text)
        """
        raw_item = self.samples[idx]

        # 1. Extract question and answer
        if "conversations" in raw_item and isinstance(raw_item["conversations"], list):
            convs = raw_item["conversations"]
            human_val = ""
            gpt_val = ""
            for turn in convs:
                sender = turn.get("from", "").lower()
                val = turn.get("value", "")
                if sender in ("human", "user") and not human_val:
                    human_val = val
                elif sender in ("gpt", "assistant") and not gpt_val:
                    gpt_val = val

            question = clean_image_tokens(human_val)
            answer = gpt_val.strip()
        else:
            # Flat format
            raw_q = raw_item.get("question", "")
            question = clean_image_tokens(raw_q)
            answer = str(raw_item.get("answer", "")).strip()

        # 2. Resolve image
        image_val = raw_item.get("image")
        if isinstance(image_val, str):
            if self.image_base_dir and not os.path.isabs(image_val):
                img_path = os.path.join(self.image_base_dir, image_val)
            else:
                img_path = image_val
            image = load_and_preprocess_image(img_path)
        elif isinstance(image_val, Image.Image):
            image = load_and_preprocess_image(image_val)
        else:
            raise ValueError(f"Item {idx} missing valid 'image' key: {image_val}")

        return {
            "image": image,
            "question": question,
            "answer": answer,
        }
