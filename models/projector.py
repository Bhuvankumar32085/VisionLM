"""Multimodal MLP Projector Module."""

from typing import Union
import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """Trainable Multi-Layer Perceptron (MLP) Projector.

    Projects visual patch representations from the vision encoder's embedding space
    (e.g., 768-d) into the language model's embedding space (e.g., 960-d).
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 1536,
        output_dim: int = 960,
        activation: Union[str, nn.Module] = "gelu",
        dropout: float = 0.0,
    ) -> None:
        """Initializes the MLP Projector.

        Args:
            input_dim: Dimensionality of input visual features (default: 768).
            hidden_dim: Intermediate projection dimension (default: 1536).
            output_dim: Target language model hidden dimension (default: 960).
            activation: Activation function ('gelu', 'relu', 'silu', etc.).
            dropout: Dropout probability (default: 0.0).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Resolve activation function
        if isinstance(activation, str):
            act_lower = activation.lower()
            if act_lower == "gelu":
                act_layer = nn.GELU()
            elif act_lower == "relu":
                act_layer = nn.ReLU()
            elif act_lower == "silu":
                act_layer = nn.SiLU()
            else:
                raise ValueError(f"Unsupported activation: {activation}. Choose from 'gelu', 'relu', 'silu'.")
        else:
            act_layer = activation

        layers = [
            nn.Linear(input_dim, hidden_dim, bias=True),
            act_layer,
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim, bias=True))

        self.projector = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize projector weights with Gaussian distribution and zero biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """Projects visual features to LLM embedding space.

        Args:
            visual_features: Tensor of shape [B, N_patches, input_dim] (e.g. [B, 196, 768]).

        Returns:
            projected_features: Tensor of shape [B, N_patches, output_dim] (e.g. [B, 196, 960]).
        """
        return self.projector(visual_features)
