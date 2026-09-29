"""Multimodal batch collator for VLM instruction fine-tuning."""

from typing import Any, Dict, List, Optional
import torch
from transformers import AutoImageProcessor, PreTrainedTokenizerBase

from data.preprocessing import format_prompt


class VLMDataCollator:
    """Multimodal collator that processes images, formats instruction prompts,

    tokenizes sequences, constructs causal loss label masks, and pads batches.
    """

    def __init__(
        self,
        image_processor: AutoImageProcessor,
        tokenizer: PreTrainedTokenizerBase,
        max_question_length: int = 128,
        max_answer_length: int = 128,
        prompt_template: Optional[str] = None,
    ) -> None:
        """Initializes the collator.

        Args:
            image_processor: Hugging Face AutoImageProcessor for ViT.
            tokenizer: Hugging Face PreTrainedTokenizer for SmolLM2.
            max_question_length: Maximum allowed question token length.
            max_answer_length: Maximum allowed answer token length.
            prompt_template: Optional prompt formatting template.
        """
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length
        self.prompt_template = prompt_template
        self.max_total_length = max_question_length + max_answer_length

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collates a list of dataset samples into a padded model batch.

        Args:
            batch: List of dicts with 'image', 'question', 'answer'.

        Returns:
            Dictionary containing:
                - pixel_values: [B, 3, 224, 224]
                - input_ids: [B, max_batch_seq_len]
                - attention_mask: [B, max_batch_seq_len]
                - labels: [B, max_batch_seq_len]
        """
        images = [item["image"] for item in batch]
        # Preprocess images using official ViT processor
        image_batch = self.image_processor(images=images, return_tensors="pt")
        pixel_values = image_batch["pixel_values"]  # [B, 3, 224, 224]

        batch_input_ids: List[List[int]] = []
        batch_attention_mask: List[List[int]] = []
        batch_labels: List[List[int]] = []

        eos_token = self.tokenizer.eos_token or ""

        for item in batch:
            question = item["question"]
            answer = item["answer"]

            prompt_text = format_prompt(question, self.prompt_template)
            # Ensure proper spacing and EOS token
            full_text = f"{prompt_text} {answer.strip()}{eos_token}"

            # Tokenize prompt to identify boundary
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
            full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)

            # Truncate if exceeds max length
            if len(full_ids) > self.max_total_length:
                full_ids = full_ids[: self.max_total_length]

            prompt_len = min(len(prompt_ids), len(full_ids))

            # Construct labels: -100 for question/instruction tokens, target IDs for answer tokens
            labels = list(full_ids)
            for i in range(prompt_len):
                labels[i] = -100

            attention_mask = [1] * len(full_ids)

            batch_input_ids.append(full_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        # Pad all sequences to the max length in this batch
        max_len = max(len(ids) for ids in batch_input_ids)
        pad_id = self.tokenizer.pad_token_id

        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []

        for ids, mask, lbls in zip(batch_input_ids, batch_attention_mask, batch_labels):
            pad_length = max_len - len(ids)
            padded_input_ids.append(ids + [pad_id] * pad_length)
            padded_attention_mask.append(mask + [0] * pad_length)
            padded_labels.append(lbls + [-100] * pad_length)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
