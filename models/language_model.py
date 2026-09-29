"""Language Model Module wrapping AutoModelForCausalLM."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


class LanguageModel(nn.Module):
    """Language Model wrapper around SmolLM2-360M-Instruct (or compatible causal LLMs)."""

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        freeze: bool = True,
    ) -> None:
        """Initializes the LanguageModel wrapper.

        Args:
            model_name: Hugging Face model identifier for the Causal LM.
            freeze: Whether to freeze LLM parameters (disable gradients).
        """
        super().__init__()
        self.model_name = model_name

        # Load causal language model in float32 for uniform precision across components
        self.llm: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
        )

        # Retrieve dynamic hidden dimension from configuration
        self.hidden_size: int = self.llm.config.hidden_size
        self.vocab_size: int = self.llm.config.vocab_size

        if freeze:
            self.freeze_parameters()

    def freeze_parameters(self) -> None:
        """Freezes all parameters of the language model backbone."""
        for param in self.llm.parameters():
            param.requires_grad = False
        self.llm.eval()

    def unfreeze_parameters(self) -> None:
        """Unfreezes all parameters of the language model backbone."""
        for param in self.llm.parameters():
            param.requires_grad = True
        self.llm.train()

    def apply_lora(
        self,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[list] = None,
    ) -> None:
        """Applies LoRA adapters to the language model for Phase 2 fine-tuning.

        Args:
            r: LoRA attention dimension (rank).
            lora_alpha: LoRA alpha scaling factor.
            lora_dropout: LoRA dropout probability.
            target_modules: List of module names to apply LoRA to (defaults to q_proj, v_proj).
        """
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError("PEFT package is required for LoRA. Run 'uv sync' to install.")

        if target_modules is None:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()

    @classmethod
    def get_tokenizer(cls, model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct") -> PreTrainedTokenizerBase:
        """Loads and configures the AutoTokenizer for the model."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure padding token is defined
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        return tokenizer

    def get_input_embeddings(self) -> nn.Module:
        """Returns the token embedding layer of the LLM."""
        return self.llm.get_input_embeddings()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through the causal language model using inputs_embeds.

        Args:
            inputs_embeds: Combined embeddings tensor of shape [B, Seq_Len, Hidden_Dim].
            attention_mask: Attention mask of shape [B, Seq_Len].
            position_ids: Position indices of shape [B, Seq_Len].
            labels: Target token IDs of shape [B, Seq_Len] (with -100 for ignored tokens).

        Returns:
            Tuple of (logits, loss):
                - logits: Output logits of shape [B, Seq_Len, vocab_size]
                - loss: Cross-entropy loss scalar if labels are provided, else None
        """
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            return_dict=True,
        )
        return outputs.logits, outputs.loss
