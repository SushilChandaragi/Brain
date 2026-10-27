"""Base model adapter abstraction."""
from typing import Any, Dict, List, Optional, Protocol
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model configuration metadata."""
    model_name: str
    model_type: str  # "gemma", "llama", etc.
    revision: str
    device: str
    dtype: torch.dtype = torch.float32


@dataclass
class InferenceResult:
    """Result from a forward pass."""
    token_ids: torch.Tensor  # Shape: (batch_size, seq_len)
    logits: torch.Tensor  # Shape: (batch_size, seq_len, vocab_size)
    probabilities: torch.Tensor  # Shape: (batch_size, seq_len, vocab_size)
    top_k_tokens: List[List[tuple]]  # [(token_id, prob), ...] for each position
    top_k_logits: List[List[tuple]]  # [(token_id, logit), ...] for each position
    metadata: Dict[str, Any]
    hidden_states: Optional[tuple] = None
    attentions: Optional[tuple] = None


class ModelAdapter(Protocol):
    """Protocol for model adapters.
    
    Allows the rest of the system to work with different Transformer architectures
    without coupling to specific implementations.
    """
    
    def load(self) -> None:
        """Load the model and tokenizer."""
        ...
    
    def get_config(self) -> ModelConfig:
        """Get model configuration."""
        ...
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize input text.
        
        Returns:
            dict with 'input_ids', 'attention_mask', etc.
        """
        ...
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        ...
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> InferenceResult:
        """Run forward pass and return results.
        
        Args:
            input_ids: Shape (batch_size, seq_len)
            attention_mask: Optional, shape (batch_size, seq_len)
            
        Returns:
            InferenceResult with logits and probabilities
        """
        ...
    
    def apply_intervention(self, intervention: Dict[str, Any]) -> None:
        """Apply an intervention (e.g., ablate attention head).
        
        Args:
            intervention: Dict with intervention type, target, and parameters
        """
        ...
    
    def remove_intervention(self) -> None:
        """Remove/reset any applied interventions."""
        ...
    
    def get_module(self, module_path: str) -> Optional[nn.Module]:
        """Get a specific module by path (e.g., 'model.layers.5.self_attn')."""
        ...
