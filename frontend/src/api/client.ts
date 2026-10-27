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
}

export const apiClient = new APIClient();
