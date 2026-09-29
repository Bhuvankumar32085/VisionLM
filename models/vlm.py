"""Vision-Language Model (VLM) connecting ViT-B/16, MLP Projector, and SmolLM2."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from models.language_model import LanguageModel
from models.projector import MLPProjector
from models.vision_encoder import VisionEncoder


@dataclass
class VLMOutput:
    """Output dataclass for VisionLanguageModel forward pass."""

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    debug_dict: Optional[Dict[str, Any]] = None


class VisionLanguageModel(nn.Module):
    """Custom Vision-Language Model constructed from modular components.

    Architecture:
    1. Pretrained ViT-B/16 Vision Encoder (extracts [B, 196, 768] patch tokens)
    2. Custom MLP Projector (projects [B, 196, 768] -> [B, 196, 960])
    3. Pretrained SmolLM2-360M-Instruct Causal LM (decodes multimodal sequence [B, 196 + text_len, 960])
    """

    def __init__(
        self,
        vision_model_name: str = "google/vit-base-patch16-224-in21k",
        language_model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        use_cls_token: bool = False,
        projector_hidden_dim: int = 1536,
        projector_activation: str = "gelu",
        projector_dropout: float = 0.0,
        freeze_vision: bool = True,
        freeze_language: bool = True,
    ) -> None:
        """Initializes the Vision-Language Model.

        Args:
            vision_model_name: ViT model identifier.
            language_model_name: SmolLM2 model identifier.
            use_cls_token: Whether to preserve the CLS token in visual sequence.
            projector_hidden_dim: Hidden dimension for intermediate MLP projection.
            projector_activation: Activation function for MLP ('gelu', etc.).
            projector_dropout: Dropout probability for MLP projector.
            freeze_vision: If True, freeze ViT parameters.
            freeze_language: If True, freeze SmolLM2 parameters.
        """
        super().__init__()

        # 1. Vision Encoder
        self.vision_encoder = VisionEncoder(
            model_name=vision_model_name,
            use_cls_token=use_cls_token,
            freeze=freeze_vision,
        )

        # 2. Language Model (load first to dynamically inspect hidden size)
        self.language_model = LanguageModel(
            model_name=language_model_name,
            freeze=freeze_language,
        )

        # Dynamically determine dimensions
        vision_hidden_dim = self.vision_encoder.hidden_size  # 768
        language_hidden_dim = self.language_model.hidden_size  # 960

        # 3. Trainable MLP Projector
        self.projector = MLPProjector(
            input_dim=vision_hidden_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=language_hidden_dim,
            activation=projector_activation,
            dropout=projector_dropout,
        )

    def get_parameter_counts(self) -> Dict[str, int]:
        """Calculates total, trainable, and frozen parameter counts by component."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        vision_trainable = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)

        projector_params = sum(p.numel() for p in self.projector.parameters())
        projector_trainable = sum(p.numel() for p in self.projector.parameters() if p.requires_grad)

        llm_params = sum(p.numel() for p in self.language_model.parameters())
        llm_trainable = sum(p.numel() for p in self.language_model.parameters() if p.requires_grad)

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "vision_params": vision_params,
            "vision_trainable": vision_trainable,
            "projector_params": projector_params,
            "projector_trainable": projector_trainable,
            "llm_params": llm_params,
            "llm_trainable": llm_trainable,
        }

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extracts and projects visual features from images.

        Args:
            pixel_values: Image tensor [B, 3, 224, 224].

        Returns:
            projected_features: Tensor of shape [B, N_patches, language_hidden_dim].
        """
        visual_features = self.vision_encoder(pixel_values)  # [B, 196, 768]
        projected_features = self.projector(visual_features)  # [B, 196, 960]
        return projected_features

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_debug_dict: bool = False,
    ) -> Union[VLMOutput, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Forward pass of the Vision-Language Model.

        Args:
            pixel_values: Image tensor [B, 3, 224, 224] (Optional, allows text-only forward).
            input_ids: Text token IDs [B, text_len].
            attention_mask: Text attention mask [B, text_len].
            position_ids: Explicit position IDs [B, total_seq_len] (Optional).
            labels: Text labels [B, text_len] with -100 on prompt tokens.
            return_debug_dict: If True, attaches detailed intermediate tensor flow dict.

        Returns:
            VLMOutput containing logits, loss, and debug_dict.
        """
        debug_dict = {} if return_debug_dict else None

        # 1. Process visual features if images are provided
        if pixel_values is not None:
            batch_size = pixel_values.shape[0]
            visual_features = self.vision_encoder(pixel_values)  # [B, 196, 768]
            projected_features = self.projector(visual_features)  # [B, 196, 960]
            num_visual_tokens = projected_features.shape[1]

            if return_debug_dict:
                debug_dict["vision_features"] = visual_features
                debug_dict["projected_features"] = projected_features
        else:
            batch_size = input_ids.shape[0] if input_ids is not None else 1
            projected_features = None
            num_visual_tokens = 0

        # 2. Process text embeddings
        if input_ids is not None:
            embedding_layer = self.language_model.get_input_embeddings()
            text_embeddings = embedding_layer(input_ids)  # [B, text_len, 960]
            text_length = input_ids.shape[1]

            if return_debug_dict:
                debug_dict["text_embeddings"] = text_embeddings
        else:
            text_embeddings = None
            text_length = 0

        # 3. Concatenate visual embeddings + text embeddings
        if projected_features is not None and text_embeddings is not None:
            combined_embeddings = torch.cat([projected_features, text_embeddings], dim=1)  # [B, 196 + text_len, 960]
        elif projected_features is not None:
            combined_embeddings = projected_features
        elif text_embeddings is not None:
            combined_embeddings = text_embeddings
        else:
            raise ValueError("At least one of pixel_values or input_ids must be provided.")

        total_seq_len = combined_embeddings.shape[1]

        # 4. Build combined attention mask
        if attention_mask is not None and projected_features is not None:
            vision_mask = torch.ones(
                (batch_size, num_visual_tokens),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            combined_attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        elif attention_mask is not None:
            combined_attention_mask = attention_mask
        else:
            combined_attention_mask = torch.ones(
                (batch_size, total_seq_len),
                dtype=torch.long,
                device=combined_embeddings.device,
            )

        # 5. Build explicit continuous position IDs
        if position_ids is None:
            position_ids = combined_attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(combined_attention_mask == 0, 1)

        # 6. Build combined labels for causal language modeling
        if labels is not None and projected_features is not None:
            # Mask all visual tokens with -100 so no loss is computed on visual tokens
            vision_labels = torch.full(
                (batch_size, num_visual_tokens),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            combined_labels = torch.cat([vision_labels, labels], dim=1)
        elif labels is not None:
            combined_labels = labels
        else:
            combined_labels = None

        if return_debug_dict:
            debug_dict["combined_embeddings"] = combined_embeddings
            debug_dict["attention_mask"] = combined_attention_mask
            debug_dict["position_ids"] = position_ids
            debug_dict["labels"] = combined_labels

        # Ensure embedding tensor matches LLM parameter dtype (e.g. when in AMP or eval)
        llm_dtype = getattr(self.language_model.llm, "dtype", combined_embeddings.dtype)
        if combined_embeddings.dtype != llm_dtype:
            combined_embeddings = combined_embeddings.to(dtype=llm_dtype)

        # 7. Pass into SmolLM2 Decoder
        logits, loss = self.language_model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask,
            position_ids=position_ids,
            labels=combined_labels,
        )

        if return_debug_dict:
            debug_dict["logits"] = logits
            debug_dict["loss"] = loss

        return VLMOutput(loss=loss, logits=logits, debug_dict=debug_dict)
