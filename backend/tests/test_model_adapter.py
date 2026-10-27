"""Tests for model adapter."""
import pytest
import torch
from app.models.gemma import GemmaAdapter


@pytest.fixture
def adapter():
    """Create a Gemma adapter instance."""
    return GemmaAdapter(model_name="google/gemma-1.1-2b-it", device="cpu")


def test_adapter_initialization(adapter):
    """Test adapter can be initialized."""
    assert adapter is not None
    assert adapter.model_name == "google/gemma-1.1-2b-it"


def test_tokenization(adapter):
    """Test tokenization works."""
    adapter.load()
    
    tokens = adapter.tokenize("Hello world")
    assert "input_ids" in tokens
    assert tokens["input_ids"].shape[0] == 1  # batch size
    assert tokens["input_ids"].shape[1] > 0  # sequence length


def test_forward_pass(adapter):
    """Test forward pass works."""
    adapter.load()
    
    tokens = adapter.tokenize("The capital of France is")
    result = adapter.forward(tokens["input_ids"])
    
    assert result.logits is not None
    assert result.probabilities is not None
    assert len(result.top_k_tokens) > 0


def test_decode(adapter):
    """Test decoding works."""
    adapter.load()
    
    # Test with known token ID
    token_id = torch.tensor([1])
    decoded = adapter.decode(token_id)
    assert isinstance(decoded, str)
    assert len(decoded) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
