"""Preprocessing utilities for images and instruction text."""

from typing import Optional, Union
from PIL import Image


def load_and_preprocess_image(image_input: Union[str, Image.Image]) -> Image.Image:
    """Loads an image from a file path or validates an existing PIL Image.

    Converts non-RGB images to RGB.

    Args:
        image_input: File path string or PIL.Image instance.

    Returns:
        RGB PIL.Image instance.
    """
    if isinstance(image_input, str):
        image = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        image = image_input
    else:
        raise TypeError(f"Expected image path string or PIL Image, got {type(image_input)}")

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def format_prompt(question: str, prompt_template: Optional[str] = None) -> str:
    """Formats a user question into the standard instruction prompt template.

    Args:
        question: User query text.
        prompt_template: Optional custom string template containing '{question}'.

    Returns:
        Formatted prompt string.
    """
    if prompt_template is not None:
        return prompt_template.format(question=question)
    return f"Question: {question.strip()}\nAnswer:"
