"""Gemma model adapter implementation."""
import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.models.adapter import ModelAdapter, ModelConfig, InferenceResult


class GemmaAdapter(ModelAdapter):
    """Adapter for Gemma models."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """Initialize Gemma adapter.
        
        Args:
            model_name: HuggingFace model ID (e.g., 'google/gemma-1.1-2b-it')
            device: Device to run on ('cuda', 'cpu', 'auto')
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.model = None
        self._active_hooks = []
        self._active_intervention = None
    
    def load(self) -> None:
        """Load model and tokenizer from HuggingFace."""
        from app.config import settings
        token = settings.HF_TOKEN if getattr(settings, 'HF_TOKEN', None) else None
        print(f"Loading {self.model_name} on device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            token=token,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate dtype
        dtype = torch.float32
        if self.device == "cuda":
            dtype = torch.float16  # Use float16 on GPU for memory efficiency
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            token=token,
        )

        if hasattr(self.model, "set_attn_implementation"):
            self.model.set_attn_implementation("eager")
        
        # Move model to device after loading
        self.model = self.model.to(self.device)
        
        self.model.eval()  # Evaluation mode
        print(f"Model loaded successfully. Device: {self.device}")
        print(f"Model size: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B parameters")
    
    def get_config(self) -> ModelConfig:
        """Get model configuration."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        return ModelConfig(
            model_name=self.model_name,
            model_type="gemma",
            revision=getattr(self.model.config, "model_type", "unknown"),
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize input text."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        # Move to device
        return {k: v.to(self.device) for k, v in encoding.items()}
    
    def decode(self, token_ids: Any) -> str:
        """Decode token IDs back to text."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Call load() first.")
        
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        elif hasattr(token_ids, "dim") and token_ids.dim() > 1:
            token_ids = token_ids[0]
        
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        capture_internals: bool = False,
    ) -> InferenceResult:
        """Run forward pass through Gemma.
        
        Args:
            input_ids: Shape (batch_size, seq_len)
            attention_mask: Optional attention mask
            
        Returns:
            InferenceResult with logits and probabilities
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Ensure inputs are on correct device
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=capture_internals,
            output_attentions=capture_internals,
        )
        
        logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)
        
        # Convert logits to probabilities
        probabilities = F.softmax(logits, dim=-1)
        
        # Get top-k predictions for each position
        top_k = 5
        top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=-1)
        
        # Convert to lists of tuples
        batch_size, seq_len = input_ids.shape
        top_k_tokens = []
        top_k_logits_list = []
        
        for b in range(batch_size):
            seq_tokens = []
            seq_logits = []
            for s in range(seq_len):
                tokens = [
                    (top_k_indices[b, s, k].item(), top_k_probs[b, s, k].item())
                    for k in range(top_k)
                ]
                seq_tokens.append(tokens)
                
                # Also get top-k logits
                top_logit_vals = torch.topk(logits[b, s], top_k)[0]
                logits_tuples = [
                    (idx.item(), val.item())
                    for idx, val in zip(top_k_indices[b, s], top_logit_vals)
                ]
                seq_logits.append(logits_tuples)
            
            top_k_tokens.append(seq_tokens)
            top_k_logits_list.append(seq_logits)
        
        return InferenceResult(
            token_ids=input_ids,
            logits=logits,
            probabilities=probabilities,
            top_k_tokens=top_k_tokens,
            top_k_logits=top_k_logits_list,
            metadata={
                "model": self.model_name,
                "device": self.device,
                "vocab_size": logits.shape[-1],
                "sequence_length": seq_len,
                "batch_size": batch_size,
                "num_attention_heads": getattr(self.model.config, "num_attention_heads", None),
                "head_dim": getattr(self.model.config, "head_dim", None),
            },
            hidden_states=outputs.hidden_states if capture_internals else None,
            attentions=outputs.attentions if capture_internals else None,
        )
    
    def apply_intervention(self, intervention: Dict[str, Any]) -> None:
        """Apply an intervention (placeholder for Phase 4)."""
        # This will be implemented in Phase 4 (Ablation Interventions)
        self._active_intervention = intervention
    
    def remove_intervention(self) -> None:
        """Remove active intervention."""
        self._active_intervention = None
        # Remove any registered hooks
        for hook_handle in self._active_hooks:
            hook_handle.remove()
        self._active_hooks.clear()
    
    def get_module(self, module_path: str) -> Optional[torch.nn.Module]:
        """Get a module by path."""
        if self.model is None:
            return None
        
        module = self.model
        for part in module_path.split("."):
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        
        return module
