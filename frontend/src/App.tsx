import { useEffect, useMemo, useState } from 'react';
import {
  apiClient,
  ExperimentTrace,
  HeadAblationExperiment,
  InferenceResponse,
  LayerCatalog,
  ModelInfo,
  HealthResponse,
  TopKPrediction,
} from './api/client';
import './App.css';

function ProbabilityBars({ predictions }: { predictions: TopKPrediction[] }) {
  const maxProbability = predictions[0]?.probability || 1;
  return <div className="probability-chart" aria-label="Top token probabilities">
    {predictions.slice(0, 8).map((prediction) => <div className="probability-row" key={`${prediction.position}-${prediction.rank}`}>
      <span className="probability-rank">{String(prediction.rank).padStart(2, '0')}</span>
      <span className="probability-token">{prediction.token_str || 'space'}</span>
      <div className="probability-track"><span style={{ width: `${(prediction.probability / maxProbability) * 100}%` }} /></div>
      <strong>{(prediction.probability * 100).toFixed(1)}%</strong>
    </div>)}
  </div>;
}

function AttentionHeatmap({ values }: { values: number[][] }) {
  const flat = values.flat();
  const max = Math.max(...flat.map((value) => Math.abs(value)), 1e-6);
  return <div className="attention-heatmap" aria-label="Attention weight heatmap">
    {values.slice(0, 16).map((row, rowIndex) => row.slice(0, 16).map((value, columnIndex) => <i key={`${rowIndex}-${columnIndex}`} style={{ opacity: Math.min(1, Math.abs(value) / max + 0.12) }} />))}
  </div>;
}

function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [layerCatalog, setLayerCatalog] = useState<LayerCatalog | null>(null);
  const [prompt, setPrompt] = useState('The capital of Karnataka is');
  const [maxNewTokens, setMaxNewTokens] = useState(24);
  const [temperature, setTemperature] = useState(0.0);
  const [captureComponents, setCaptureComponents] = useState(['hidden', 'attention', 'mlp', 'intermediate']);
  const [captureLayers, setCaptureLayers] = useState('');
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  const [loading, setLoading] = useState(false);
  const [ablating, setAblating] = useState(false);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [trace, setTrace] = useState<ExperimentTrace | null>(null);
  const [experiment, setExperiment] = useState<HeadAblationExperiment | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const currentHealth = await apiClient.health();
        setHealth(currentHealth);
        if (currentHealth.model_loaded) {
          setModelInfo(await apiClient.getModelInfo());
          setLayerCatalog(await apiClient.listLayers());
        }
      } catch (err) { setError(`Backend unavailable: ${err}`); }
    };
    void checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  const lastPredictions = useMemo(() => result?.predictions[result.predictions.length - 1] ?? [], [result]);
  const topPrediction = lastPredictions[0];
  const firstGeneratedToken = result?.generated_tokens[0]?.token_str;
  const isTopCandidateSampled = topPrediction && firstGeneratedToken ? (topPrediction.token_str.trim().toLowerCase() === firstGeneratedToken.trim().toLowerCase() || firstGeneratedToken.includes(topPrediction.token_str.trim())) : false;
  const selectedCaptures = trace?.captures.filter((capture) => capture.layer === selectedLayer && capture.component !== 'logits') ?? [];
  const hasSelectedLayer = selectedCaptures.length > 0;

  const runInference = async (promptToUse?: string, layersOverride?: string) => {
    const activePrompt = promptToUse ?? prompt;
    if (!activePrompt.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const parsedLayers = (layersOverride ?? captureLayers).split(',').map((value) => Number(value.trim())).filter((value) => Number.isInteger(value) && value >= 0);
      const response = await apiClient.runInference(activePrompt, { maxNewTokens, temperature, captureLayers: parsedLayers.length ? parsedLayers : undefined, captureComponents });
      setResult(response);
      const traceId = response.metadata.trace_id;
      if (typeof traceId === 'string') setTrace(await apiClient.getTrace(traceId));
    } catch (err) { setError(`Inference failed: ${err}`); }
    finally { setLoading(false); }
  };

  const handleInference = () => { void runInference(); };

  const selectPresetPrompt = (preset: string) => {
    setPrompt(preset);
    void runInference(preset);
  };

  const retraceSelectedLayer = () => {
    setCaptureLayers(String(selectedLayer));
    void runInference(prompt, String(selectedLayer));
  };

  const handleAblation = async () => {
    setAblating(true);
    setError(null);
    try {
      const exp = await apiClient.runHeadAblation(prompt, selectedLayer, selectedHead);
      setExperiment(exp);
    }
    catch (err) { setError(`Ablation failed: ${err}`); }
    finally { setAblating(false); }
  };

  return <div className="app-shell">
    <header className="topbar"><div className="brand-lockup"><span className="brand-mark">NB</span><div><p className="eyebrow">NEURAL BEHAVIOR LAB / 01</p><h1>AI Brain Scanner</h1></div></div><div className={`connection-pill ${health?.model_loaded ? 'online' : ''}`}><span /> {health?.model_loaded ? 'MODEL ONLINE' : 'CONNECTING'}</div></header>
    <main className="workspace">
      <section className="intro-row"><div><p className="eyebrow accent">BASELINE / INFERENCE TRACE</p><h2>Read the next thought.</h2><p className="intro-copy">Probe a language model one token at a time, then inspect the distribution behind its answer.</p></div><div className="phase-stamp">PHASE 02<strong>DISSECTION</strong></div></section>
      <section className="control-board">
        <div className="board-heading"><div><span className="section-number">01</span><h3>Prompt chamber</h3></div><span className="muted-label">{prompt.length} CHARACTERS (Enter to Scan)</span></div>
        <textarea
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={(event) => {
            if ((event.key === 'Enter' && event.ctrlKey) || (event.key === 'Enter' && !event.shiftKey)) {
              event.preventDefault();
              handleInference();
            }
          }}
          placeholder="Enter a prompt to inspect (press Enter to run)..."
          disabled={loading}
        />
        <div className="preset-prompts">
          <span className="muted-label">PRESETS:</span>
          <button type="button" className="preset-chip" onClick={() => selectPresetPrompt('The capital of Karnataka is')}>Karnataka</button>
          <button type="button" className="preset-chip" onClick={() => selectPresetPrompt('The capital of France is')}>France</button>
          <button type="button" className="preset-chip" onClick={() => selectPresetPrompt('The sun rises in the')}>Sun</button>
          <button type="button" className="preset-chip" onClick={() => selectPresetPrompt('Python is a programming')}>Python</button>
        </div>
        <div className="control-row">
          <label>NEW TOKENS <input type="number" min="1" max="64" value={maxNewTokens} onChange={(event) => setMaxNewTokens(Number(event.target.value))} /></label>
          <label className="range-control" title={temperature === 0 ? 'Greedy / Deterministic (Top candidate always chosen)' : 'Stochastic Sampling (May pick lower-ranked tokens)'}>
            TEMPERATURE <input type="range" min="0" max="1.5" step="0.1" value={temperature} onChange={(event) => setTemperature(Number(event.target.value))} />
            <b>{temperature.toFixed(1)} {temperature === 0 ? '(Greedy)' : '(Sampled)'}</b>
          </label>
          <label>LAYERS <input className="layer-input" value={captureLayers} onChange={(event) => setCaptureLayers(event.target.value)} placeholder="auto" /></label>
          <button onClick={handleInference} disabled={loading || !health?.model_loaded || !prompt.trim()}>{loading ? 'SCANNING...' : 'RUN TRACE  →'}</button>
        </div>
        <div className="capture-options">
          <span className="muted-label">CAPTURE</span>
          {['hidden', 'attention', 'mlp', 'intermediate'].map((component) => <label key={component}><input type="checkbox" checked={captureComponents.includes(component)} onChange={() => setCaptureComponents((current) => current.includes(component) ? current.filter((item) => item !== component) : [...current, component])} /> {component}</label>)}
          <span className="capture-note">blank layers = first + last</span>
        </div>
        {error && <div className="error-message">{error}</div>}
      </section>
      <section className="signal-strip"><div><span>ENGINE</span><strong>{modelInfo?.model_name?.split('/').pop() ?? '—'}</strong></div><div><span>DEVICE</span><strong>{modelInfo?.device ?? '—'}</strong></div><div><span>PARAMETERS</span><strong>{modelInfo?.num_parameters ? `${(modelInfo.num_parameters / 1e9).toFixed(2)}B` : '—'}</strong></div><div><span>SEQUENCE</span><strong>{result?.metadata.sequence_length ?? '—'} TOKENS</strong></div></section>
      {result ? <div className="analysis-grid">
        <section className="panel token-panel">
          <div className="panel-heading"><div><span className="section-number">02</span><h3>Token stream</h3></div><span className="muted-label">{result.input_tokens.length} INPUT / {result.generated_tokens.length} OUTPUT</span></div>
          <div className="token-stream">
            {result.input_tokens.map((token) => <span className="stream-token input-token" key={`in-${token.position}`} title={`Token ID ${token.token_id}`}>{token.token_str}</span>)}
            <span className="stream-arrow">→</span>
            {result.generated_tokens.map((token) => <span className="stream-token output-token" key={`out-${token.position}`} title={`Token ID ${token.token_id}`}>{token.token_str}</span>)}
          </div>
          <div className="completion-box">
            <span className="muted-label">COMPLETION</span>
            <p>{result.input_prompt}{result.generated_text.startsWith(' ') || result.input_prompt.endsWith(' ') ? '' : ' '}<em>{result.generated_text}</em></p>
          </div>
        </section>
        <section className="panel metric-panel">
          <div className="panel-heading"><div><span className="section-number">03</span><h3>Next-Token Prediction</h3></div><span className="muted-label">TOP CANDIDATE</span></div>
          <div className="hero-metric">
            <strong>{topPrediction?.token_str || '—'}</strong>
            <span>{topPrediction ? `${(topPrediction.probability * 100).toFixed(1)}% confidence (Rank #${topPrediction.rank})` : 'No signal'}</span>
          </div>
          <div className="metric-line"><span>LOGIT</span><b>{topPrediction?.logit.toFixed(3) ?? '—'}</b></div>
          <div className="metric-line"><span>FIRST GENERATED TOKEN</span><b>"{firstGeneratedToken || '—'}"</b></div>
          <div className="sampling-status">
            {isTopCandidateSampled ? (
              <span className="status-badge match">✓ Generated token matches top candidate</span>
            ) : temperature > 0 ? (
              <span className="status-badge sampled">⚡ Differs from generated token due to temperature sampling ({temperature.toFixed(1)}). Set temp=0 for exact greedy match.</span>
            ) : null}
          </div>
        </section>
        <section className="panel distribution-panel"><div className="panel-heading"><div><span className="section-number">04</span><h3>Prediction field</h3></div><span className="muted-label">LAST INPUT POSITION</span></div><ProbabilityBars predictions={lastPredictions} /></section>
        <section className="panel metadata-panel"><div className="panel-heading"><div><span className="section-number">05</span><h3>Trace metadata</h3></div></div><div className="metadata-grid">{Object.entries(result.metadata).map(([key, value]) => <div key={key}><span>{key.replace(/_/g, ' ')}</span><b>{String(value)}</b></div>)}</div></section>
        {trace && <section className="panel trace-panel"><div className="panel-heading"><div><span className="section-number">06</span><h3>Layer explorer</h3></div><span className="muted-label">TRACE {trace.trace_id.slice(0, 8)}</span></div><div className="layer-explorer"><div className="layer-list">{layerCatalog?.layers.map((item) => <button className={selectedLayer === item.layer ? 'layer-button selected' : 'layer-button'} key={item.layer} onClick={() => setSelectedLayer(item.layer)}>L{String(item.layer).padStart(2, '0')}</button>)}</div><div className="layer-detail"><div className="layer-toolbar"><strong>LAYER {selectedLayer}</strong><label>HEAD <select value={selectedHead} onChange={(event) => setSelectedHead(Number(event.target.value))}>{Array.from({ length: layerCatalog?.num_attention_heads ?? 0 }, (_, index) => <option key={index} value={index}>{index}</option>)}</select></label><button onClick={retraceSelectedLayer} disabled={loading}>RETRACE LAYER</button><button onClick={handleAblation} disabled={loading || ablating || !hasSelectedLayer}>{ablating ? 'ABLATING...' : `ABLATE HEAD ${selectedHead}`}</button></div>{!hasSelectedLayer && <p className="layer-empty">Layer {selectedLayer} is not in this trace. Retrace it to inspect the real tensors.</p>}<div className="trace-captures">{selectedCaptures.map((capture) => <div className="capture-card" key={`${capture.path}-${capture.component}`}><div><strong>{capture.component}</strong><span>{capture.path}</span></div><b>{capture.shape.join(' × ')}</b><div className="stat-line"><span>MEAN {capture.mean.toFixed(3)}</span><span>STD {capture.std.toFixed(3)}</span><span>MAX {capture.abs_max.toFixed(3)}</span></div>{capture.component === 'attention_weights' && capture.values && <><AttentionHeatmap values={capture.values[selectedHead] || capture.values[0]} /><div className="attention-facts"><span>HEAD {selectedHead}</span><span>ENTROPY {capture.mean_entropy?.toFixed(3) ?? '—'}</span></div></>}{capture.token_means && <div className="activation-strip">{capture.token_means.slice(0, 32).map((value, index) => <i key={index} style={{ height: `${Math.max(8, Math.min(100, value * 35))}%` }} />)}</div>}</div>)}</div></div></div><div className="capture-log"><span className="muted-label">CAPTURE LOG</span>{trace.capture_log.map((entry) => <code key={entry}>{entry}</code>)}</div></section>}
        {experiment && <section className="panel experiment-panel"><div className="panel-heading"><div><span className="section-number">07</span><h3>Head ablation comparison</h3></div><span className="muted-label">L{experiment.layer} / HEAD {experiment.head}</span></div><div className="comparison-grid"><div><span>BASELINE PREDICTION</span><strong>{experiment.baseline.predicted_token}</strong><b>{(experiment.baseline.target_probability * 100).toFixed(2)}% / RANK {experiment.baseline.target_rank}</b></div><div><span>ABLATED PREDICTION</span><strong>{experiment.intervention_result.predicted_token}</strong><b>{(experiment.intervention_result.target_probability * 100).toFixed(2)}% / RANK {experiment.intervention_result.target_rank}</b></div><div><span>TARGET LOGIT DELTA</span><strong>{experiment.metrics.target_logit_delta.toFixed(4)}</strong><b>PROBABILITY {experiment.metrics.target_probability_delta.toFixed(4)}</b></div></div></section>}
      </div> : <section className="empty-state"><span className="empty-glyph">＋</span><h3>Awaiting a trace</h3><p>Run the prompt to reveal token flow and the model's probability field.</p></section>}
    </main>
    <footer>AI BRAIN SCANNER <span>LOCAL INFERENCE / {health?.device?.toUpperCase() ?? '—'}</span></footer>
  </div>;
}

export default App;
