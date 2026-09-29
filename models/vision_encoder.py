"""Vision Encoder Module using Pretrained ViT."""

from typing import Optional
import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel


class VisionEncoder(nn.Module):
    """Vision Encoder wrapper around pretrained Vision Transformer (ViT).

    Extracts visual patch representations from images without classification heads.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224-in21k",
        use_cls_token: bool = False,
        freeze: bool = True,
    ) -> None:
        """Initializes the VisionEncoder.

        Args:
            model_name: Hugging Face model identifier for the ViT backbone.
            use_cls_token: Whether to keep the CLS token in output sequence.
            freeze: Whether to freeze ViT parameters (disable gradients).
        """
        super().__init__()
        self.model_name = model_name
        self.use_cls_token = use_cls_token

        # Load pretrained ViT backbone without classification head
        self.vit = AutoModel.from_pretrained(model_name)
        self.hidden_size: int = self.vit.config.hidden_size
        self.image_size: int = getattr(self.vit.config, "image_size", 224)
        self.patch_size: int = getattr(self.vit.config, "patch_size", 16)
        self.num_patches: int = (self.image_size // self.patch_size) ** 2

        if freeze:
            self.freeze_parameters()

    def freeze_parameters(self) -> None:
        """Freezes all parameters of the ViT backbone."""
        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()

    def unfreeze_parameters(self) -> None:
        """Unfreezes all parameters of the ViT backbone."""
        for param in self.vit.parameters():
            param.requires_grad = True
        self.vit.train()

    @classmethod
    def get_image_processor(cls, model_name: str = "google/vit-base-patch16-224-in21k") -> AutoImageProcessor:
        """Loads and returns the official AutoImageProcessor for the vision model."""
        return AutoImageProcessor.from_pretrained(model_name)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass through the vision encoder.

        Args:
            pixel_values: Preprocessed image tensor of shape [B, 3, H, W] (typically [B, 3, 224, 224]).

        Returns:
            visual_features: Tensor of shape [B, N_patches, hidden_dim] (e.g., [B, 196, 768]),
                             or [B, N_patches + 1, hidden_dim] if use_cls_token=True.
        """
        # If parameters are frozen, run without building computation graph for ViT
        is_frozen = not any(p.requires_grad for p in self.vit.parameters())
        if is_frozen:
            with torch.no_grad():
                outputs = self.vit(pixel_values=pixel_values)
                last_hidden_state = outputs.last_hidden_state  # [B, 197, 768]
        else:
            outputs = self.vit(pixel_values=pixel_values)
            last_hidden_state = outputs.last_hidden_state

        if not self.use_cls_token:
            # Remove CLS token (index 0) to keep only the 196 patch tokens
            visual_features = last_hidden_state[:, 1:, :]  # [B, 196, 768]
        else:
            visual_features = last_hidden_state

        return visual_features
