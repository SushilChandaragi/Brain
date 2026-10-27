"""Pydantic schemas for API requests/responses."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class TokenInfo(BaseModel):
    """Information about a single token."""
    token_id: int
    token_str: str
    position: int


class TopKPrediction(BaseModel):
    """Top-K prediction for a token position."""
    position: int
    token_id: int
    token_str: str
    probability: float
    logit: float
    rank: int


class InferenceRequest(BaseModel):
    """Request for model inference."""
    prompt: str
    max_new_tokens: int = 20
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    capture_layers: Optional[List[int]] = None
    capture_components: Optional[List[str]] = None


class InferenceResponse(BaseModel):
    """Response from model inference."""
    input_prompt: str
    input_tokens: List[TokenInfo]
    generated_text: str
    generated_tokens: List[TokenInfo]
    predictions: List[List[TopKPrediction]]  # [position][top_k]
    metadata: Dict[str, Any]


class HeadAblationRequest(BaseModel):
    prompt: str
    layer: int
    head: int
    target_token_id: Optional[int] = None


class ExperimentResponse(BaseModel):
    experiment_id: str
    prompt: str
    model: str
    layer: int
    head: int
    intervention: Dict[str, Any]
    baseline: Dict[str, Any]
    intervention_result: Dict[str, Any]
    metrics: Dict[str, Any]


class ModelInfoResponse(BaseModel):
    """Information about the loaded model."""
    model_name: str
    model_type: str
    device: str
    vocab_size: int
    num_parameters: Optional[int]
    dtype: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: Optional[str]
    message: str
