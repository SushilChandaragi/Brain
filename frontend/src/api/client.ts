import axios, { AxiosInstance } from 'axios';

export interface TokenInfo {
  token_id: number;
  token_str: string;
  position: number;
}

export interface TopKPrediction {
  position: number;
  token_id: number;
  token_str: string;
  probability: number;
  logit: number;
  rank: number;
}

export interface InferenceResponse {
  input_prompt: string;
  input_tokens: TokenInfo[];
  generated_text: string;
  generated_tokens: TokenInfo[];
  predictions: TopKPrediction[][];
  metadata: Record<string, any>;
}

export interface ModelInfo {
  model_name: string;
  model_type: string;
  device: string;
  vocab_size: number;
  num_parameters?: number;
  dtype: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  device?: string;
  message: string;
}

export interface TraceCapture {
  path: string;
  layer: number;
  component: string;
  shape: number[];
  dtype: string;
  mean: number;
  std: number;
  min: number;
  max: number;
  abs_max: number;
  values?: number[][][];
  token_means?: number[];
  mean_entropy?: number;
  strongest_key_by_query?: number[];
}

export interface ExperimentTrace {
  trace_id: string;
  created_at: string;
  input_shape: number[];
  layer_count: number;
  selected_layers: number[];
  selected_components: string[];
  architecture: Array<{ path: string; layer: string; component: string; type: string }>;
  captures: TraceCapture[];
  capture_log: string[];
}

export interface LayerCatalog {
  layer_count: number;
  num_attention_heads: number;
  head_dim: number;
  layers: Array<{ layer: number; paths: Array<{ path: string; layer: string; component: string; type: string }> }>;
}

export interface HeadAblationExperiment {
  experiment_id: string;
  prompt: string;
  model: string;
  layer: number;
  head: number;
  intervention: Record<string, any>;
  baseline: Record<string, any>;
  intervention_result: Record<string, any>;
  metrics: Record<string, any>;
  created_at?: string;
}

export interface HeadScore {
  layer: number;
  head: number;
  probability_drop: number;
  target_logit_delta: number;
}

export interface HeadSweepResponse {
  experiment_id: string;
  prompt: string;
  target_token: string;
  target_token_id: number;
  baseline_probability: number;
  baseline_logit: number;
  layers_evaluated: number[];
  num_heads: number;
  sensitivity_matrix: number[][];
  top_causal_heads: HeadScore[];
  created_at: string;
}

export interface ActivationPatchingResponse {
  experiment_id: string;
  clean_prompt: string;
  corrupted_prompt: string;
  target_layer: number;
  component: string;
  clean_target_token: string;
  clean_target_id: number;
  corrupted_prediction: string;
  patched_prediction: string;
  clean_logit: number;
  corrupted_logit: number;
  patched_logit: number;
  clean_prob: number;
  corrupted_prob: number;
  patched_prob: number;
  causal_recovery: number;
  created_at: string;
}

export interface SynapseLink {
  source_id: string;
  target_id: string;
  weight: number;
  activity: number;
}

export interface NeuronNode {
  id: string;
  layer_idx: number;
  layer_name: string;
  neuron_idx: number;
  channel_idx: number;
  label: string;
  activation: number;
  normalized: number;
  role: string;
  details: Record<string, any>;
}

export interface ModelThinkingStep {
  step_index: number;
  stage_name: string;
  layer_num: number;
  top_hypothesized_token: string;
  confidence: number;
  description: string;
}

export interface NeuralGraphResponse {
  prompt: string;
  model: string;
  columns: string[];
  nodes: NeuronNode[];
  links: SynapseLink[];
  winner_node_id: string;
  thinking_steps?: ModelThinkingStep[];
  autoregressive_continuation?: string;
  subword_explanation?: string;
}

export interface ExperimentListItem {
  id: string;
  experiment_type: string;
  prompt: string;
  model: string;
  layer?: number;
  head?: number;
  target_token?: string;
  metrics_summary?: string;
  created_at: string;
}

class APIClient {
  private client: AxiosInstance;

  constructor(baseURL: string = 'http://localhost:8000') {
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  async health(): Promise<HealthResponse> {
    const response = await this.client.get<HealthResponse>('/health');
    return response.data;
  }

  async getModelInfo(): Promise<ModelInfo> {
    const response = await this.client.get<ModelInfo>('/model/info');
    return response.data;
  }

  async runInference(
    prompt: string,
    options: { maxNewTokens: number; temperature: number; captureLayers?: number[]; captureComponents?: string[] },
  ): Promise<InferenceResponse> {
    const response = await this.client.post<InferenceResponse>('/infer', {
      prompt,
      max_new_tokens: options.maxNewTokens,
      temperature: options.temperature,
      capture_layers: options.captureLayers,
      capture_components: options.captureComponents,
    });
    return response.data;
  }

  async testInference(): Promise<any> {
    const response = await this.client.post('/test-inference');
    return response.data;
  }

  async getTrace(traceId: string): Promise<ExperimentTrace> {
    const response = await this.client.get<ExperimentTrace>(`/traces/${traceId}`);
    return response.data;
  }

  async listLayers(): Promise<LayerCatalog> {
    const response = await this.client.get<LayerCatalog>('/model/layers');
    return response.data;
  }

  async runHeadAblation(prompt: string, layer: number, head: number): Promise<HeadAblationExperiment> {
    const response = await this.client.post<HeadAblationExperiment>('/experiments/head-ablation', { prompt, layer, head });
    return response.data;
  }

  async runMlpAblation(prompt: string, layer: number): Promise<HeadAblationExperiment> {
    const response = await this.client.post<HeadAblationExperiment>('/experiments/mlp-ablation', { prompt, layer });
    return response.data;
  }

  async runHeadSweep(prompt: string, layers?: number[]): Promise<HeadSweepResponse> {
    const response = await this.client.post<HeadSweepResponse>('/experiments/head-sweep', { prompt, layers });
    return response.data;
  }

  async runActivationPatching(
    clean_prompt: string,
    corrupted_prompt: string,
    target_layer: number,
    component: string = 'hidden',
  ): Promise<ActivationPatchingResponse> {
    const response = await this.client.post<ActivationPatchingResponse>('/experiments/activation-patching', {
      clean_prompt,
      corrupted_prompt,
      target_layer,
      component,
    });
    return response.data;
  }

  async listExperiments(limit: number = 50, expType?: string): Promise<ExperimentListItem[]> {
    const response = await this.client.get<ExperimentListItem[]>('/experiments', {
      params: { limit, experiment_type: expType },
    });
    return response.data;
  }

  async deleteExperiment(experimentId: string): Promise<void> {
    await this.client.delete(`/experiments/${experimentId}`);
  }

  async getNeuralGraph(prompt: string, numNeuronsPerColumn: number = 10): Promise<NeuralGraphResponse> {
    const response = await this.client.post<NeuralGraphResponse>('/model/neural-graph', {
      prompt,
      num_neurons_per_column: numNeuronsPerColumn,
    });
    return response.data;
  }

  async runAuditPipeline(request: AuditPipelineRequest): Promise<AuditPipelineResponse> {
    const response = await this.client.post<AuditPipelineResponse>('/audit/pipeline', request);
    return response.data;
  }
}

export const apiClient = new APIClient();

// ============================================================================
// AI AUDIT LAB & PROMPT INTELLIGENCE TYPES
// ============================================================================

export interface AuditPipelineRequest {
  prompt: string;
  objective?: string;
  expected_answer?: string;
  hypothesis?: string;
  run_robustness_variants?: boolean;
  max_new_tokens?: number;
  temperature?: number;
}

export interface PromptAnalysis {
  intended_task: string;
  explicit_requirements: string[];
  ambiguous_terms: string[];
  missing_context: string[];
  contradictions: string[];
  misleading_assumptions: string[];
  expected_format: string;
  reliability_concerns: string[];
  summary: string;
}

export interface AuditEvidenceItem {
  evidence_id: string;
  category: 'directly_measured' | 'derived_metric' | 'hypothesis_interpretation' | 'externally_verified' | 'unavailable';
  observation: string;
  source: string;
  method: string;
  confidence_or_uncertainty?: string;
  limitations: string;
  raw_data?: Record<string, any>;
}

export interface RobustnessVariantResult {
  variant_id: string;
  variant_type: string;
  prompt: string;
  rationale: string;
  top_token: string;
  top_probability: number;
  generated_text: string;
  consistent_with_baseline: boolean;
  matches_expected?: boolean | null;
  probability_shift: number;
}

export interface PromptQualityWeakness {
  id: string;
  weakness_type: string;
  problematic_text: string;
  explanation: string;
  evidence: string;
  proposed_correction: string;
}

export interface MechanisticCandidate {
  component_type: string;
  layer: number;
  head?: number;
  observation: string;
  hypothesis: string;
  suggested_action: string;
}

export interface AuditSynthesis {
  original_behavior_summary: string;
  evidence_collected_count: number;
  directly_observed_issues: string[];
  empirically_supported_conclusions: string[];
  unverified_hypotheses: string[];
  discovered_weaknesses: string[];
  recommended_improvements: string[];
  recommended_followup_experiments: string[];
}

export interface AuditTimelineStage {
  stage_index: number;
  stage_name: string;
  status: string;
  timestamp: string;
  details: string;
}

export interface AuditPipelineResponse {
  run_id: string;
  timestamp: string;
  prompt: string;
  objective: string;
  expected_answer?: string;
  hypothesis?: string;
  model_info: {
    model_name: string;
    device: string;
    total_layers: number;
    vocab_size: number;
  };
  analysis: PromptAnalysis;
  baseline_inference: {
    model_identifier: string;
    sequence_length: number;
    input_tokens: TokenInfo[];
    top_prediction: {
      rank: number;
      token_id: number;
      token_str: string;
      probability: number;
      logit: number;
    };
    top_5_predictions: Array<{
      rank: number;
      token_id: number;
      token_str: string;
      probability: number;
      logit: number;
    }>;
    predictive_entropy: number;
    logit_margin_over_runner_up: number;
    generated_text: string;
    matches_expected?: boolean | null;
    expected_token_stats?: Record<string, any>;
    temperature: number;
  };
  evidence: AuditEvidenceItem[];
  robustness_variants: RobustnessVariantResult[];
  weaknesses: PromptQualityWeakness[];
  mechanistic_candidates: MechanisticCandidate[];
  synthesis: AuditSynthesis;
  improved_prompt: string;
  improved_prompt_explanation: string;
  execution_timeline: AuditTimelineStage[];
}
