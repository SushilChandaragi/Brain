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


class ModelSwitchRequest(BaseModel):
    """Request to switch active model."""
    model_name: str


class InferenceRequest(BaseModel):
    """Request for model inference."""
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    capture_layers: Optional[List[int]] = None
    capture_components: Optional[List[str]] = None
    use_chat_template: Optional[bool] = None


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


class MlpAblationRequest(BaseModel):
    prompt: str
    layer: int
    target_token_id: Optional[int] = None


class HeadSweepRequest(BaseModel):
    prompt: str
    target_token_id: Optional[int] = None
    layers: Optional[List[int]] = None  # None = sweep all layers


class HeadScore(BaseModel):
    layer: int
    head: int
    probability_drop: float
    target_logit_delta: float


class HeadSweepResponse(BaseModel):
    experiment_id: str
    prompt: str
    target_token: str
    target_token_id: int
    baseline_probability: float
    baseline_logit: float
    layers_evaluated: List[int]
    num_heads: int
    # 2D matrix [layer_idx][head_idx] of probability drops
    sensitivity_matrix: List[List[float]]
    top_causal_heads: List[HeadScore]
    created_at: str


class ActivationPatchingRequest(BaseModel):
    clean_prompt: str
    corrupted_prompt: str
    target_layer: int
    component: str = "hidden"  # "hidden" (residual), "mlp", or "attention"
    target_token_id: Optional[int] = None


class ActivationPatchingResponse(BaseModel):
    experiment_id: str
    clean_prompt: str
    corrupted_prompt: str
    target_layer: int
    component: str
    clean_target_token: str
    clean_target_id: int
    corrupted_prediction: str
    patched_prediction: str
    clean_logit: float
    corrupted_logit: float
    patched_logit: float
    clean_prob: float
    corrupted_prob: float
    patched_prob: float
    causal_recovery: float  # (patched_logit - corrupted_logit) / (clean_logit - corrupted_logit)
    created_at: str


class ExperimentResponse(BaseModel):
    experiment_id: str
    prompt: str
    model: str
    layer: Optional[int] = None
    head: Optional[int] = None
    intervention: Dict[str, Any]
    baseline: Dict[str, Any]
    intervention_result: Dict[str, Any]
    metrics: Dict[str, Any]
    created_at: Optional[str] = None


class ExperimentListItem(BaseModel):
    id: str
    experiment_type: str
    prompt: str
    model: str
    layer: Optional[int] = None
    head: Optional[int] = None
    target_token: Optional[str] = None
    metrics_summary: Optional[str] = None
    created_at: str


class NeuralGraphRequest(BaseModel):
    prompt: str
    num_neurons_per_column: int = 10
    selected_layers: Optional[List[int]] = None


class SynapseLink(BaseModel):
    source_id: str
    target_id: str
    weight: float
    activity: float


class NeuronNode(BaseModel):
    id: str
    layer_idx: int
    layer_name: str
    neuron_idx: int
    channel_idx: int
    label: str
    activation: float
    normalized: float
    role: str
    details: Dict[str, Any]


class ModelThinkingStep(BaseModel):
    step_index: int
    stage_name: str
    layer_num: int
    top_hypothesized_token: str
    confidence: float
    description: str


class NeuralGraphResponse(BaseModel):
    prompt: str
    model: str
    columns: List[str]
    nodes: List[NeuronNode]
    links: List[SynapseLink]
    winner_node_id: str
    thinking_steps: Optional[List[ModelThinkingStep]] = None
    autoregressive_continuation: Optional[str] = None
    subword_explanation: Optional[str] = None


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


# ============================================================================
# AI AUDIT LAB & PROMPT INTELLIGENCE SCHEMAS
# ============================================================================

class AuditPipelineRequest(BaseModel):
    """Request for running the comprehensive AI Audit & Prompt Intelligence pipeline."""
    prompt: str
    objective: str = "factual_recall"  # factual_recall, hallucination_risk, ambiguity_detection, contradictory_instructions, context_distraction, bias_fairness, instruction_following, prompt_robustness, output_consistency, custom_hypothesis
    expected_answer: Optional[str] = None
    hypothesis: Optional[str] = None
    run_robustness_variants: bool = True
    max_new_tokens: int = 24
    temperature: float = 0.0


class PromptAnalysis(BaseModel):
    """Stage 1: Linguistic and structural prompt breakdown."""
    intended_task: str
    explicit_requirements: List[str]
    ambiguous_terms: List[str]
    missing_context: List[str]
    contradictions: List[str]
    misleading_assumptions: List[str]
    expected_format: str
    reliability_concerns: List[str]
    summary: str


class AuditEvidenceItem(BaseModel):
    """Stage 3: Evidence item with rigorous scientific provenance."""
    evidence_id: str
    category: str  # directly_measured, derived_metric, hypothesis_interpretation, externally_verified, unavailable
    observation: str
    source: str
    method: str
    confidence_or_uncertainty: Optional[str] = None
    limitations: str
    raw_data: Optional[Dict[str, Any]] = None


class RobustnessVariantResult(BaseModel):
    """Stage 4: Evaluation of a controlled prompt variation."""
    variant_id: str
    variant_type: str
    prompt: str
    rationale: str
    top_token: str
    top_probability: float
    generated_text: str
    consistent_with_baseline: bool
    matches_expected: Optional[bool] = None
    probability_shift: float


class PromptQualityWeakness(BaseModel):
    """Diagnosed prompt defect with causal evidence and suggested remediation."""
    id: str
    weakness_type: str
    problematic_text: str
    explanation: str
    evidence: str
    proposed_correction: str


class MechanisticCandidate(BaseModel):
    """Candidate transformer component for mechanistic deep-dive."""
    component_type: str  # layer, attention_head, mlp
    layer: int
    head: Optional[int] = None
    observation: str
    hypothesis: str
    suggested_action: str


class AuditSynthesis(BaseModel):
    """Stage 6: Multi-perspective synthesis answering key audit questions."""
    original_behavior_summary: str
    evidence_collected_count: int
    directly_observed_issues: List[str]
    empirically_supported_conclusions: List[str]
    unverified_hypotheses: List[str]
    discovered_weaknesses: List[str]
    recommended_improvements: List[str]
    recommended_followup_experiments: List[str]


class AuditPipelineResponse(BaseModel):
    """Comprehensive response from the AI Audit & Prompt Intelligence pipeline."""
    run_id: str
    timestamp: str
    prompt: str
    objective: str
    expected_answer: Optional[str] = None
    hypothesis: Optional[str] = None
    model_info: Dict[str, Any]
    analysis: PromptAnalysis
    baseline_inference: Dict[str, Any]
    evidence: List[AuditEvidenceItem]
    robustness_variants: List[RobustnessVariantResult]
    weaknesses: List[PromptQualityWeakness]
    mechanistic_candidates: List[MechanisticCandidate]
    synthesis: AuditSynthesis
    improved_prompt: str
    improved_prompt_explanation: str
    execution_timeline: List[Dict[str, Any]]

