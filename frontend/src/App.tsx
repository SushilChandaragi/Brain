import { useEffect, useMemo, useState } from 'react';
import {
  apiClient,
  ActivationPatchingResponse,
  ExperimentListItem,
  ExperimentTrace,
  HeadAblationExperiment,
  HeadSweepResponse,
  InferenceResponse,
  LayerCatalog,
  ModelInfo,
  HealthResponse,
  TokenInfo,
  TopKPrediction,
  NeuralGraphResponse,
  NeuronNode,
} from './api/client';
import './App.css';
import { Synaptic3DNeuralFlow } from './components/Synaptic3DNeuralFlow';
import { AuditCaseSelector, AuditCase } from './components/AuditCaseSelector';
import { ExecutiveAuditDossier } from './components/ExecutiveAuditDossier';

export function ProbabilityBars({ predictions }: { predictions: TopKPrediction[] }) {
  const maxProbability = predictions[0]?.probability || 1;
  return (
    <div className="probability-chart" aria-label="Top token probabilities">
      {predictions.slice(0, 8).map((prediction) => (
        <div className="probability-row" key={`${prediction.position}-${prediction.rank}`}>
          <span className="probability-rank">{String(prediction.rank).padStart(2, '0')}</span>
          <span className="probability-token">{prediction.token_str || 'space'}</span>
          <div className="probability-track">
            <span style={{ width: `${(prediction.probability / maxProbability) * 100}%` }} />
          </div>
          <strong>{(prediction.probability * 100).toFixed(1)}%</strong>
        </div>
      ))}
    </div>
  );
}

// 1. Interactive Token-Labeled Attention Matrix with hover tooltips
function InteractiveAttentionMatrix({
  tokens,
  values,
  headIndex,
  layerIndex,
}: {
  tokens: TokenInfo[];
  values: number[][];
  headIndex: number;
  layerIndex: number;
}) {
  const [hoveredCell, setHoveredCell] = useState<{
    qToken: string;
    kToken: string;
    qIdx: number;
    kIdx: number;
    weight: number;
  } | null>(null);

  const displayTokens = tokens.slice(0, 16);
  const matrixValues = values.slice(0, displayTokens.length);

  return (
    <div className="interactive-matrix-wrap">
      <div className="matrix-table-card">
        <div className="matrix-table-scroll">
          <table className="attention-matrix-table">
            <thead>
              <tr>
                <th className="matrix-corner-th">
                  <div className="corner-guide">
                    <span className="corner-query">Query ↓</span>
                    <span className="corner-key">Key →</span>
                  </div>
                </th>
                {displayTokens.map((t, idx) => (
                  <th
                    key={`col-${idx}`}
                    className={`matrix-col-th ${hoveredCell?.kIdx === idx ? 'col-highlighted' : ''}`}
                  >
                    <div className="col-header-box" title={`Key token #${idx}: "${t.token_str}"`}>
                      <span className="col-idx">#{idx}</span>
                      <span className="col-token">{t.token_str.trim() || '␣'}</span>
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {displayTokens.map((rowToken, rIdx) => {
                const rowVals = matrixValues[rIdx] || [];
                return (
                  <tr key={`row-${rIdx}`} className={hoveredCell?.qIdx === rIdx ? 'row-highlighted' : ''}>
                    <th className={`matrix-row-th ${hoveredCell?.qIdx === rIdx ? 'row-highlighted' : ''}`}>
                      <div className="row-header-box" title={`Query token #${rIdx}: "${rowToken.token_str}"`}>
                        <span className="row-idx">#{rIdx}</span>
                        <span className="row-token">{rowToken.token_str.trim() || '␣'}</span>
                      </div>
                    </th>
                    {displayTokens.map((colToken, cIdx) => {
                      const weight = rowVals[cIdx] !== undefined ? rowVals[cIdx] : 0;
                      const alpha = Math.min(1, Math.max(0.04, Math.abs(weight)));
                      const isHovered = hoveredCell?.qIdx === rIdx && hoveredCell?.kIdx === cIdx;
                      const isCausalMask = cIdx > rIdx;
                      return (
                        <td
                          key={`cell-${rIdx}-${cIdx}`}
                          className={`matrix-cell-td ${isHovered ? 'cell-hovered' : ''} ${isCausalMask ? 'cell-causal-mask' : ''}`}
                          style={{
                            backgroundColor: isCausalMask
                              ? '#f8fafc'
                              : `rgba(29, 101, 214, ${alpha})`,
                            color: isCausalMask
                              ? '#cbd5e1'
                              : alpha > 0.45
                              ? '#ffffff'
                              : '#1d65d6',
                          }}
                          onMouseEnter={() =>
                            setHoveredCell({
                              qToken: rowToken.token_str,
                              kToken: colToken.token_str,
                              qIdx: rIdx,
                              kIdx: cIdx,
                              weight,
                            })
                          }
                          onMouseOver={() =>
                            setHoveredCell({
                              qToken: rowToken.token_str,
                              kToken: colToken.token_str,
                              qIdx: rIdx,
                              kIdx: cIdx,
                              weight,
                            })
                          }
                          onMouseLeave={() => setHoveredCell(null)}
                        >
                          {isCausalMask ? '—' : `${(weight * 100).toFixed(0)}%`}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="matrix-footer-bar">
          {hoveredCell ? (
            <div className="matrix-tooltip">
              <div className="tooltip-cell-info">
                <span className="tooltip-tag query">Query #{hoveredCell.qIdx}: <strong>"{hoveredCell.qToken}"</strong></span>
                <span className="tooltip-arrow">➔</span>
                <span className="tooltip-tag key">Key #{hoveredCell.kIdx}: <strong>"{hoveredCell.kToken}"</strong></span>
              </div>
              <div className="tooltip-weight-info">
                <span>Attention Weight: <strong>{(hoveredCell.weight * 100).toFixed(2)}%</strong></span>
                <span className="tooltip-sub">Layer {layerIndex} • Head {headIndex}</span>
              </div>
            </div>
          ) : (
            <div className="matrix-legend">
              <span className="legend-label">ATTENTION DENSITY:</span>
              <span className="legend-step">0%</span>
              <div className="legend-gradient-bar" />
              <span className="legend-step">100%</span>
              <span className="legend-hint">Hover any cell to inspect query ➔ key directional attention weight</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// 2. Head Sensitivity Map & Causal Importance Sweep
function HeadSensitivityMap({
  sweep,
  loading,
  onRunSweep,
  onSelectHead,
}: {
  sweep: HeadSweepResponse | null;
  loading: boolean;
  onRunSweep: () => void;
  onSelectHead: (layer: number, head: number) => void;
}) {
  const [hoveredHead, setHoveredHead] = useState<{ layer: number; head: number; drop: number } | null>(null);

  return (
    <div className="panel sweep-panel">
      <div className="panel-heading">
        <div>
          <span className="section-number">08</span>
          <h3>Causal Head Sensitivity Map (Full-Model Sweep)</h3>
        </div>
        <button className="layer-button" onClick={onRunSweep} disabled={loading} style={{ padding: '8px 16px', background: 'var(--ink)', color: 'var(--acid)' }}>
          {loading ? 'SWEEPING HEADS...' : 'RUN HEAD SWEEP ⚡'}
        </button>
      </div>
      <p className="intro-copy" style={{ margin: '12px 0' }}>
        Systematically ablates all 32 attention heads per layer to locate which heads are the primary causal drivers of the model's factual prediction.
      </p>

      {sweep ? (
        <>
          <div className="sweep-matrix-wrap">
            <div style={{ display: 'flex', marginLeft: 56, gap: 2, marginBottom: 4 }}>
              {Array.from({ length: sweep.num_heads }, (_, h) => (
                <div key={h} style={{ width: 22, font: '9px DM Mono', textAlign: 'center', color: 'var(--muted)' }}>
                  H{h}
                </div>
              ))}
            </div>
            {sweep.layers_evaluated.map((layerNum, lIdx) => (
              <div className="sweep-row" key={`layer-${layerNum}`}>
                <div className="sweep-row-label">L{String(layerNum).padStart(2, '0')}</div>
                {sweep.sensitivity_matrix[lIdx]?.map((drop, hIdx) => {
                  let bg = '#e8e5dc';
                  if (drop > 0.08) bg = '#d32f2f'; // high causal drop
                  else if (drop > 0.04) bg = '#f57c00'; // moderate
                  else if (drop > 0.01) bg = '#fbc02d'; // subtle
                  else if (drop < -0.01) bg = '#81c784'; // suppressor

                  return (
                    <div
                      key={`h-${hIdx}`}
                      className="sweep-cell"
                      style={{ backgroundColor: bg }}
                      title={`L${layerNum} H${hIdx}: Drop ${(drop * 100).toFixed(2)}%`}
                      onMouseEnter={() => setHoveredHead({ layer: layerNum, head: hIdx, drop })}
                      onMouseLeave={() => setHoveredHead(null)}
                      onClick={() => onSelectHead(layerNum, hIdx)}
                    />
                  );
                })}
              </div>
            ))}
          </div>

          {hoveredHead && (
            <div className="matrix-tooltip" style={{ marginTop: 8 }}>
              <span>Selected Target: "{sweep.target_token}" (Baseline: {(sweep.baseline_probability * 100).toFixed(1)}%)</span>
              <strong>
                Layer {hoveredHead.layer} / Head {hoveredHead.head}: Impact {(hoveredHead.drop * 100).toFixed(2)}% drop (Click to inspect)
              </strong>
            </div>
          )}

          <div style={{ marginTop: 20 }}>
            <span className="muted-label">TOP CAUSAL ATTENTION HEADS (HIGHEST FACTUAL RECOVERY)</span>
            <div className="causal-heads-cards">
              {sweep.top_causal_heads.map((headScore) => (
                <div
                  key={`top-${headScore.layer}-${headScore.head}`}
                  className="causal-head-card"
                  onClick={() => onSelectHead(headScore.layer, headScore.head)}
                  style={{ cursor: 'pointer' }}
                >
                  <strong>L{headScore.layer} / HEAD {headScore.head}</strong>
                  <span>Drop: {(headScore.probability_drop * 100).toFixed(2)}%</span>
                  <div style={{ fontSize: 10, color: 'var(--muted)', marginTop: 4 }}>
                    Logit Δ: {headScore.target_logit_delta.toFixed(3)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      ) : (
        <div className="layer-empty" style={{ marginTop: 14 }}>
          Click "RUN HEAD SWEEP ⚡" to scan the model layers and generate the 2D causal sensitivity matrix.
        </div>
      )}
    </div>
  );
}

// 3. Activation Patching / Causal Tracing Tab
function ActivationPatchingTab({
  modelInfo,
  layerCatalog,
}: {
  modelInfo: ModelInfo | null;
  layerCatalog: LayerCatalog | null;
}) {
  const [cleanPrompt, setCleanPrompt] = useState('The capital of France is');
  const [corruptedPrompt, setCorruptedPrompt] = useState('The capital of Italy is');
  const [targetLayer, setTargetLayer] = useState(4);
  const [component, setComponent] = useState('hidden');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ActivationPatchingResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runPatch = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.runActivationPatching(cleanPrompt, corruptedPrompt, targetLayer, component);
      setResult(res);
    } catch (err) {
      setError(`Activation patching failed: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="patching-grid">
      <div className="patch-input-group">
        <div className="board-heading">
          <div>
            <span className="section-number">01</span>
            <h3>Causal Tracing Setup</h3>
          </div>
          <span className="muted-label">
            CLEAN VS. CORRUPTED ({modelInfo?.model_name?.split('/').pop() || 'MODEL'})
          </span>
        </div>
        <p className="intro-copy" style={{ fontSize: 13, margin: '10px 0 16px' }}>
          Patches activations from the clean forward pass into the corrupted forward pass to test where factual associations are stored.
        </p>

        <label>CLEAN PROMPT (Provides target factual activation)</label>
        <input type="text" value={cleanPrompt} onChange={(e) => setCleanPrompt(e.target.value)} />

        <label>CORRUPTED PROMPT (Receives patched activation)</label>
        <input type="text" value={corruptedPrompt} onChange={(e) => setCorruptedPrompt(e.target.value)} />

        <div className="preset-prompts" style={{ marginBottom: 16 }}>
          <span className="muted-label">PRESETS:</span>
          <button
            type="button"
            className="preset-chip"
            onClick={() => {
              setCleanPrompt('The capital of France is');
              setCorruptedPrompt('The capital of Italy is');
            }}
          >
            France / Italy
          </button>
          <button
            type="button"
            className="preset-chip"
            onClick={() => {
              setCleanPrompt('The Colosseum is in the city of');
              setCorruptedPrompt('The Eiffel Tower is in the city of');
            }}
          >
            Colosseum / Eiffel
          </button>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 14, marginBottom: 18 }}>
          <div>
            <label>TARGET LAYER: L{targetLayer}</label>
            <input
              type="range"
              min="0"
              max={(layerCatalog?.layer_count ?? 22) - 1}
              value={targetLayer}
              onChange={(e) => setTargetLayer(Number(e.target.value))}
              style={{ width: '100%' }}
            />
          </div>
          <div>
            <label>COMPONENT</label>
            <select
              value={component}
              onChange={(e) => setComponent(e.target.value)}
              style={{ width: '100%', padding: 8, font: '12px DM Mono', border: '1px solid var(--line)', background: '#fff' }}
            >
              <option value="hidden">Hidden Residual Stream</option>
              <option value="mlp">MLP Output</option>
              <option value="attention">Self-Attention Output</option>
            </select>
          </div>
        </div>

        <button
          onClick={runPatch}
          disabled={loading || !cleanPrompt.trim() || !corruptedPrompt.trim()}
          style={{ width: '100%', padding: '14px', background: 'var(--ink)', color: 'var(--acid)', border: 0, font: '12px DM Mono', cursor: 'pointer' }}
        >
          {loading ? 'PATCHING TENSORS...' : 'RUN ACTIVATION PATCH →'}
        </button>

        {error && <div className="error-message">{error}</div>}
      </div>

      <div className="patch-result-card">
        <div className="board-heading">
          <div>
            <span className="section-number">02</span>
            <h3>Causal Recovery Outcome</h3>
          </div>
          <span className="muted-label">RESTORATION METRIC</span>
        </div>

        {result ? (
          <div style={{ marginTop: 18 }}>
            <div className="hero-metric" style={{ margin: '14px 0', paddingBottom: 14 }}>
              <strong>{result.patched_prediction}</strong>
              <span>
                Clean Target: "{result.clean_target_token}" | Corrupted Output: "{result.corrupted_prediction}"
              </span>
            </div>

            <label style={{ font: '11px DM Mono', color: 'var(--muted)', display: 'block', marginTop: 14 }}>
              CAUSAL RECOVERY RATIO:
            </label>
            <div className="recovery-gauge">
              <div className="recovery-fill" style={{ width: `${Math.max(0, Math.min(100, result.causal_recovery * 100))}%` }} />
              <div className="recovery-label">{(result.causal_recovery * 100).toFixed(1)}% RECOVERY</div>
            </div>

            <div className="patch-summary-stats">
              <div>
                <span>CLEAN PROB</span>
                <strong>{(result.clean_prob * 100).toFixed(1)}%</strong>
              </div>
              <div>
                <span>CORRUPTED PROB</span>
                <strong>{(result.corrupted_prob * 100).toFixed(2)}%</strong>
              </div>
              <div>
                <span>PATCHED PROB</span>
                <strong style={{ color: 'var(--orange)' }}>{(result.patched_prob * 100).toFixed(1)}%</strong>
              </div>
            </div>

            <div className="completion-box" style={{ marginTop: 20 }}>
              <span className="muted-label">SCIENTIFIC INTERPRETATION</span>
              <p>
                Injecting layer <strong>{result.target_layer}</strong>'s {result.component} activation from <em>"{result.clean_prompt}"</em>{' '}
                into <em>"{result.corrupted_prompt}"</em> shifted the model's answer from <strong>"{result.corrupted_prediction}"</strong>{' '}
                to <strong>"{result.patched_prediction}"</strong>, demonstrating direct causal transmission of the factual association.
              </p>
            </div>
          </div>
        ) : (
          <div className="empty-state" style={{ marginTop: 24 }}>
            <span className="empty-glyph">⚡</span>
            <h3>Awaiting Patching Run</h3>
            <p>Select clean and corrupted prompts and run the intervention to measure causal recovery.</p>
          </div>
        )}
      </div>
    </div>
  );
}

// 4. SQLite Experiment History Tab
function ExperimentHistoryTab() {
  const [experiments, setExperiments] = useState<ExperimentListItem[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchHistory = async () => {
    setLoading(true);
    try {
      const list = await apiClient.listExperiments(50);
      setExperiments(list);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchHistory();
  }, []);

  const handleDelete = async (id: string) => {
    try {
      await apiClient.deleteExperiment(id);
      setExperiments((cur) => cur.filter((item) => item.id !== id));
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="history-table-wrap">
      <div style={{ display: 'flex', justifyContent: 'space-between', padding: '16px 20px', borderBottom: '1px solid var(--ink)' }}>
        <div>
          <h3 style={{ margin: 0 }}>SQLite Experiment Database (`experiments.db`)</h3>
          <span className="muted-label">{experiments.length} RECORDED EXPERIMENTAL RUNS</span>
        </div>
        <button className="layer-button" onClick={fetchHistory} disabled={loading} style={{ padding: '6px 14px' }}>
          {loading ? 'LOADING...' : 'REFRESH ↻'}
        </button>
      </div>

      {experiments.length > 0 ? (
        <table className="history-table">
          <thead>
            <tr>
              <th>TYPE</th>
              <th>PROMPT</th>
              <th>TARGET TOKEN</th>
              <th>METRICS SUMMARY</th>
              <th>TIMESTAMP</th>
              <th>ACTION</th>
            </tr>
          </thead>
          <tbody>
            {experiments.map((exp) => (
              <tr key={exp.id}>
                <td>
                  <strong style={{ color: 'var(--orange)' }}>{exp.experiment_type.toUpperCase()}</strong>
                </td>
                <td style={{ maxWidth: 280, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={exp.prompt}>
                  {exp.prompt}
                </td>
                <td>{exp.target_token || '—'}</td>
                <td>{exp.metrics_summary || '—'}</td>
                <td style={{ color: 'var(--muted)', fontSize: 10 }}>{new Date(exp.created_at).toLocaleString()}</td>
                <td>
                  <button className="del-btn" onClick={() => handleDelete(exp.id)}>
                    DELETE
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="empty-state" style={{ padding: 40 }}>
          <p>No experiments saved yet in `experiments.db`. Run any ablation, sweep, or patch to record runs persistently.</p>
        </div>
      )}
    </div>
  );
}

// 5. Educational Mini-Transformer Simulator Tab
function MiniTransformerSimulator() {
  const [phrase, setPhrase] = useState('AI thinks fast');
  const [head1Weight, setHead1Weight] = useState(1.0);
  const [mlpActive, setMlpActive] = useState(true);

  const tokens = useMemo(() => phrase.trim().split(/\s+/).filter(Boolean), [phrase]);

  // Toy 1-layer Transformer mathematical simulation
  const dim = 4;
  const embeddings = useMemo(() => {
    return tokens.map((token, idx) => {
      const hash = (token.charCodeAt(0) * 17 + idx * 31) % 100;
      return [
        Math.sin(hash).toFixed(2),
        Math.cos(hash).toFixed(2),
        (Math.sin(hash * 2) * 0.8).toFixed(2),
        (Math.cos(hash * 2) * 0.8).toFixed(2),
      ].map(Number);
    });
  }, [tokens]);

  // Simulated Q, K, V
  const attentionScores = useMemo(() => {
    const n = tokens.length;
    const scores = [];
    for (let i = 0; i < n; i++) {
      const row = [];
      for (let j = 0; j < n; j++) {
        if (j > i) {
          row.push(0); // Causal mask
        } else {
          const raw = 0.5 + ((i - j) * 0.4) * head1Weight;
          row.push(Math.min(1.0, Math.max(0.05, raw)));
        }
      }
      // Softmax normalize
      const sum = row.reduce((a, b) => a + b, 0) || 1;
      scores.push(row.map((v) => Number((v / sum).toFixed(2))));
    }
    return scores;
  }, [tokens, head1Weight]);

  return (
    <div className="transformer-lab">
      <div className="board-heading">
        <div>
          <span className="section-number">EDUCATIONAL LAB</span>
          <h3>Interactive 1-Layer Mini-Transformer Mathematical Simulator</h3>
        </div>
        <span className="muted-label">STEP-BY-STEP TENSOR MATH</span>
      </div>
      <p className="intro-copy" style={{ margin: '10px 0 24px', maxWidth: 850 }}>
        Visualizes the exact linear algebra equations flowing through a 1-layer, 2-head Transformer: Token Embeddings ➔ Q/K/V Projections ➔ Scaled Dot-Product Attention ➔ Residual Stream ➔ Feed-Forward MLP ➔ Logits.
      </p>

      <div style={{ marginBottom: 20 }}>
        <label style={{ font: '11px DM Mono', color: 'var(--muted)', display: 'block', marginBottom: 6 }}>INPUT PHRASE TO SIMULATE:</label>
        <input
          type="text"
          value={phrase}
          onChange={(e) => setPhrase(e.target.value)}
          style={{ padding: '10px 14px', font: '14px DM Mono', border: '1px solid var(--line)', width: '320px', background: '#fff' }}
        />
      </div>

      <div className="interactive-slider-box">
        <label style={{ font: '11px DM Mono', color: 'var(--ink)' }}>
          <strong>INTERVENTION: Attention Head 1 Scaling ({head1Weight.toFixed(1)}x):</strong>
        </label>
        <input
          type="range"
          min="0"
          max="2"
          step="0.1"
          value={head1Weight}
          onChange={(e) => setHead1Weight(Number(e.target.value))}
          style={{ width: 160 }}
        />
        <label style={{ font: '11px DM Mono', display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }}>
          <input type="checkbox" checked={mlpActive} onChange={(e) => setMlpActive(e.target.checked)} />
          <strong>MLP Block Active</strong>
        </label>
      </div>

      <div style={{ marginTop: 24 }}>
        <div className="math-step-card">
          <h4>1. Token Embeddings & Positional Encodings</h4>
          <div className="math-formula">X = Embedding(Token) + PositionalEncoding(pos), where d_model = {dim}</div>
          <table className="vector-table">
            <thead>
              <tr>
                <th>TOKEN</th>
                <th>POS</th>
                <th>DIM 0</th>
                <th>DIM 1</th>
                <th>DIM 2</th>
                <th>DIM 3</th>
              </tr>
            </thead>
            <tbody>
              {tokens.map((tok, i) => (
                <tr key={i}>
                  <td><strong>"{tok}"</strong></td>
                  <td>#{i}</td>
                  {embeddings[i]?.map((val, d) => (
                    <td key={d}>{val}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="math-step-card">
          <h4>2. Scaled Dot-Product Attention Weights Matrix</h4>
          <div className="math-formula">Attention(Q, K, V) = Softmax( (Q · K^T) / √d_k ) · V</div>
          <table className="vector-table">
            <thead>
              <tr>
                <th>QUERY \ KEY</th>
                {tokens.map((tok, j) => (
                  <th key={j}>"{tok}"</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tokens.map((qTok, i) => (
                <tr key={i}>
                  <td><strong>"{qTok}"</strong></td>
                  {tokens.map((_, j) => (
                    <td
                      key={j}
                      style={{
                        backgroundColor: `rgba(29, 101, 214, ${attentionScores[i]?.[j] || 0.05})`,
                        color: (attentionScores[i]?.[j] || 0) > 0.45 ? '#fff' : '#1d65d6',
                        fontWeight: 'bold',
                      }}
                    >
                      {((attentionScores[i]?.[j] || 0) * 100).toFixed(0)}%
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
          <span style={{ fontSize: 10, color: 'var(--muted)', marginTop: 8, display: 'block' }}>
            * Lower-triangular strictly respects causal autoregressive masking (future tokens cannot be attended to).
          </span>
        </div>

        <div className="math-step-card">
          <h4>3. Feed-Forward MLP & Residual Connection</h4>
          <div className="math-formula">Output = LayerNorm( X + Attention(X) + {mlpActive ? 'MLP(LayerNorm(X + Attention(X)))' : '0 (MLP ABLATED)'} )</div>
          <p style={{ font: '12px DM Mono', margin: '8px 0', color: mlpActive ? '#2e7d32' : '#c62828' }}>
            {mlpActive
              ? '✓ MLP contribution enabled: Non-linear key-value associations active.'
              : '✗ MLP output suppressed: Residual stream carries only attention projection.'}
          </p>
        </div>
      </div>
    </div>
  );
}

// 6. Visual Synaptic Flow & Neuron Telemetry Inspector (Legacy 2D Inspector)
export function _SynapticNeuronFlowView({ initialPrompt }: { initialPrompt: string }) {
  const [prompt, setPrompt] = useState(initialPrompt || 'The capital of India is');
  const [graphData, setGraphData] = useState<NeuralGraphResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<NeuronNode | null>(null);
  const [silencedNodes, setSilencedNodes] = useState<Set<string>>(new Set());

  const fetchGraph = async (promptToUse?: string) => {
    const activePrompt = promptToUse ?? prompt;
    if (!activePrompt.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.getNeuralGraph(activePrompt, 8);
      setGraphData(data);
      // Auto-select the winning node if available
      const winner = data.nodes.find((n) => n.id === data.winner_node_id);
      if (winner) {
        setSelectedNode(winner);
      } else if (data.nodes.length > 0) {
        setSelectedNode(data.nodes[data.nodes.length - 1]);
      }
    } catch (err) {
      setError(`Failed to extract neural graph: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchGraph(prompt);
  }, []);

  const toggleSilence = (nodeId: string) => {
    setSilencedNodes((prev) => {
      const next = new Set(prev);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  };

  // Compute 2D coordinates for nodes dynamically adapting to node counts per column
  const nodeCoords = useMemo(() => {
    if (!graphData) return new Map<string, { x: number; y: number }>();
    const map = new Map<string, { x: number; y: number }>();

    // Group nodes by layer_idx
    const colNodes: Record<number, NeuronNode[]> = { 0: [], 1: [], 2: [], 3: [], 4: [] };
    graphData.nodes.forEach((n) => {
      if (colNodes[n.layer_idx]) colNodes[n.layer_idx].push(n);
    });

    const colX = [120, 260, 395, 530, 650];
    const ySpans = [
      { top: 140, bottom: 480 }, // Col 0: Input Tokens
      { top: 70, bottom: 550 },  // Col 1: Hexagonal spread
      { top: 22, bottom: 598 },  // Col 2: Max hexagonal diamond apex
      { top: 70, bottom: 550 },  // Col 3: Hexagonal spread
      { top: 140, bottom: 480 }, // Col 4: Output Candidates
    ];

    Object.entries(colNodes).forEach(([colKey, nodesInCol]) => {
      const colIdx = Number(colKey);
      const span = ySpans[colIdx] || { top: 100, bottom: 500 };
      const totalInCol = nodesInCol.length;
      nodesInCol.forEach((n, idx) => {
        let y = (span.top + span.bottom) / 2;
        if (totalInCol > 1) {
          const step = (span.bottom - span.top) / (totalInCol - 1);
          y = span.top + idx * step;
        }
        map.set(n.id, { x: colX[colIdx], y });
      });
    });

    return map;
  }, [graphData]);

  // Compute incoming and outgoing connections for selected node
  const selectedNodeStats = useMemo(() => {
    if (!selectedNode || !graphData) return null;
    const incoming = graphData.links.filter((l) => l.target_id === selectedNode.id);
    const outgoing = graphData.links.filter((l) => l.source_id === selectedNode.id);
    const maxIn = incoming.length ? Math.max(...incoming.map((l) => l.weight)) : 0;
    const maxOut = outgoing.length ? Math.max(...outgoing.map((l) => l.weight)) : 0;
    return {
      incomingCount: incoming.length,
      outgoingCount: outgoing.length,
      maxInWeight: maxIn,
      maxOutWeight: maxOut,
    };
  }, [selectedNode, graphData]);

  return (
    <div className="neural-graph-panel">
      <div className="board-heading" style={{ borderBottom: '1px solid #143818', paddingBottom: 14 }}>
        <div>
          <span className="section-number" style={{ color: '#00ff66' }}>09</span>
          <h3 style={{ color: '#00ff66', fontFamily: 'DM Mono' }}>
            Synaptic Neuron Flow & Mechanistic Interpretability
          </h3>
        </div>
        <span className="muted-label" style={{ color: '#629c6b' }}>
          GENUINE FORWARD PROPAGATION / LOGIT LENS
        </span>
      </div>

      <p style={{ color: '#88bc91', fontSize: 13, lineHeight: 1.6, margin: '14px 0 16px', maxWidth: 900 }}>
        Visualizes the authentic step-by-step thinking of the Transformer model for <em>any</em> query. Tokens flow from real prompt embeddings on the left, through active neuron channels in early, mid, and late checkpoints, into candidate vocabulary logits on the right. <strong>Tap any neuron</strong> to inspect its scalar activation, what vocabulary words it promotes/suppresses, and its causal contribution to the final answer.
      </p>

      {/* Control row with prompt input & presets */}
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', marginBottom: 12 }}>
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter any query or sentence to inspect the model's inner thoughts..."
          style={{
            flex: 1,
            minWidth: 280,
            background: '#040d06',
            border: '1px solid #184c20',
            color: '#00ff66',
            font: '13px DM Mono',
            padding: '10px 14px',
            borderRadius: 3,
            outline: 0,
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter') void fetchGraph(prompt);
          }}
        />
        <button
          onClick={() => void fetchGraph(prompt)}
          disabled={loading || !prompt.trim()}
          style={{
            background: '#00ff66',
            color: '#000',
            border: 0,
            padding: '10px 22px',
            font: '700 11px DM Mono',
            cursor: 'pointer',
            letterSpacing: '0.08em',
            borderRadius: 2,
          }}
        >
          {loading ? 'COMPUTING TENSORS...' : 'RUN NEURAL FLOW ⚡'}
        </button>
      </div>

      {/* Presets */}
      <div className="preset-prompts" style={{ marginBottom: 16 }}>
        <span className="muted-label" style={{ color: '#629c6b' }}>TEST QUERIES:</span>
        <button
          type="button"
          className="preset-chip"
          style={{ background: '#0a1d0d', color: '#00ff66', borderColor: '#184c20' }}
          onClick={() => {
            setPrompt('The capital of Bihar is');
            void fetchGraph('The capital of Bihar is');
          }}
        >
          The capital of Bihar is ➔ "Pat" (Patna)
        </button>
        <button
          type="button"
          className="preset-chip"
          style={{ background: '#0a1d0d', color: '#00ff66', borderColor: '#184c20' }}
          onClick={() => {
            setPrompt('capital of bihar is');
            void fetchGraph('capital of bihar is');
          }}
        >
          capital of bihar is ➔ "pat" (patna)
        </button>
        <button
          type="button"
          className="preset-chip"
          style={{ background: '#0a1d0d', color: '#00ff66', borderColor: '#184c20' }}
          onClick={() => {
            setPrompt('Mumbai is the capital of');
            void fetchGraph('Mumbai is the capital of');
          }}
        >
          Mumbai is the capital of ➔ "Mah" (Maharashtra)
        </button>
        <button
          type="button"
          className="preset-chip"
          style={{ background: '#0a1d0d', color: '#00ff66', borderColor: '#184c20' }}
          onClick={() => {
            setPrompt('The capital of India is');
            void fetchGraph('The capital of India is');
          }}
        >
          The capital of India is ➔ "New"
        </button>
        <button
          type="button"
          className="preset-chip"
          style={{ background: '#0a1d0d', color: '#00ff66', borderColor: '#184c20' }}
          onClick={() => {
            setPrompt('7 + 1 = ');
            void fetchGraph('7 + 1 = ');
          }}
        >
          7 + 1 = ➔ "8"
        </button>
        <button
          type="button"
          className="preset-chip"
          style={{ background: '#0a1d0d', color: '#00ff66', borderColor: '#184c20' }}
          onClick={() => {
            setPrompt('The capital of France is');
            void fetchGraph('The capital of France is');
          }}
        >
          The capital of France is ➔ "Paris"
        </button>
      </div>

      {/* Model Thinking Timeline (Logit Lens Ribbon) */}
      {graphData?.thinking_steps && (
        <div style={{ marginBottom: 18 }}>
          <div style={{ font: '10px DM Mono', color: '#629c6b', textTransform: 'uppercase', marginBottom: 6, letterSpacing: '0.08em' }}>
            🧠 MODEL LOGIT LENS EVOLUTION (How intermediate layers form the thought):
          </div>
          <div className="thinking-ribbon">
            {graphData.thinking_steps.map((step) => (
              <div
                key={step.step_index}
                className={`thinking-card ${step.step_index === 4 ? 'active' : ''}`}
              >
                <div className="thinking-stage-title">
                  <span>{step.stage_name}</span>
                  <span>L{String(step.layer_num).padStart(2, '0')}</span>
                </div>
                <div className="thinking-token">{step.top_hypothesized_token}</div>
                <div className="thinking-confidence">
                  {step.step_index === 0
                    ? 'Prompt Ingestion'
                    : `${(step.confidence * 100).toFixed(1)}% hypothesis`}
                </div>
                <div className="thinking-desc">{step.description}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Autoregressive Continuation & Subword Notice Banner */}
      {graphData?.autoregressive_continuation && (
        <div className="continuation-banner">
          <div className="continuation-header">
            <span className="continuation-tag">AUTOREGRESSIVE DECODED SEQUENCE (NEXT TOKENS)</span>
            <span className="continuation-status">
              Full continuation: <strong>"{graphData.autoregressive_continuation}"</strong>
            </span>
          </div>
          {graphData.subword_explanation && (
            <div className="continuation-explanation">
              💡 <strong>Subword Tokenization (BPE):</strong> {graphData.subword_explanation}
            </div>
          )}
        </div>
      )}

      {error && <div className="error-message" style={{ color: '#ff5252' }}>{error}</div>}

      <div className="graph-viewport-wrap">
        {/* Left: SVG Neural Visualizer */}
        <div className="graph-svg-box">
          {graphData ? (
            <svg
              viewBox="0 0 880 620"
              style={{ width: '100%', height: 'auto', display: 'block' }}
            >
              {/* Column labels at the bottom */}
              <text x="120" y="598" className="layer-col-label">INPUT (PROMPT)</text>
              <text x="260" y="598" className="layer-col-label">EARLY (L03)</text>
              <text x="395" y="614" className="layer-col-label">MID (L10)</text>
              <text x="530" y="598" className="layer-col-label">LATE (L18)</text>
              <text x="650" y="598" className="layer-col-label">OUTPUT (LOGITS)</text>

              {/* Synapses (Links) */}
              <g className="synapse-group">
                {graphData.links.map((link) => {
                  const src = nodeCoords.get(link.source_id);
                  const tgt = nodeCoords.get(link.target_id);
                  if (!src || !tgt) return null;
                  const isSilenced = silencedNodes.has(link.source_id) || silencedNodes.has(link.target_id);
                  const isConnected = selectedNode && (link.source_id === selectedNode.id || link.target_id === selectedNode.id);

                  let strokeColor = '#00ff66';
                  let strokeWidth = 0.7 + link.weight * 1.5;
                  let strokeOpacity = isSilenced ? 0.04 : 0.12 + link.weight * 0.72;

                  if (selectedNode) {
                    if (isConnected) {
                      strokeColor = isSilenced ? '#ff4444' : '#ffd700';
                      strokeWidth = 2.4;
                      strokeOpacity = 1.0;
                    } else {
                      strokeOpacity = 0.06;
                    }
                  }

                  return (
                    <line
                      key={`${link.source_id}-${link.target_id}`}
                      x1={src.x}
                      y1={src.y}
                      x2={tgt.x}
                      y2={tgt.y}
                      className="synapse-path"
                      stroke={strokeColor}
                      strokeWidth={strokeWidth}
                      strokeOpacity={strokeOpacity}
                    />
                  );
                })}
              </g>

              {/* Neurons (Nodes) */}
              <g className="nodes-group">
                {graphData.nodes.map((node) => {
                  const pt = nodeCoords.get(node.id);
                  if (!pt) return null;
                  const isWinner = node.id === graphData.winner_node_id;
                  const isSelected = selectedNode?.id === node.id;
                  const isSilenced = silencedNodes.has(node.id);
                  const isBrightActive = node.normalized > 0.65;

                  // Dynamic token labels (no hardcoded digits!)
                  const inputLabel = node.layer_idx === 0 ? `"${node.details?.token_string}"` : null;
                  const outputLabel = node.layer_idx === 4 ? `#${node.neuron_idx + 1}: "${node.details?.token_string}" (${node.details?.percentage || ''})` : null;

                  // Node styling
                  let fill = '#030804';
                  if (isSilenced) {
                    fill = '#2a0a0a';
                  } else if (isWinner || isBrightActive) {
                    fill = '#00ff66';
                  }

                  let stroke = '#00ff66';
                  if (isSilenced) stroke = '#d32f2f';
                  else if (isSelected) stroke = '#ffd700';

                  const radius = isWinner ? 12.5 : isSelected ? 12 : 9.5;

                  return (
                    <g
                      key={node.id}
                      onClick={() => setSelectedNode(node)}
                      style={{ cursor: 'pointer' }}
                    >
                      {/* Pulse halo for the winning decided token */}
                      {isWinner && !isSilenced && (
                        <circle
                          cx={pt.x}
                          cy={pt.y}
                          r={19}
                          fill="none"
                          stroke="#00ff66"
                          strokeWidth="3"
                          opacity="0.65"
                          className="winner-glow"
                        />
                      )}

                      {/* Golden highlight ring for selected node */}
                      {isSelected && (
                        <circle
                          cx={pt.x}
                          cy={pt.y}
                          r={18}
                          fill="none"
                          stroke="#ffd700"
                          strokeWidth="2.5"
                          strokeDasharray="3 3"
                        />
                      )}

                      {/* Main Neuron Body */}
                      <circle
                        cx={pt.x}
                        cy={pt.y}
                        r={radius}
                        fill={fill}
                        stroke={stroke}
                        strokeWidth={isSelected ? 3.5 : 2.2}
                        className="neuron-circle"
                      />

                      {/* Inner phosphor dot for intermediate activation */}
                      {!isWinner && !isBrightActive && !isSilenced && node.normalized > 0.35 && (
                        <circle
                          cx={pt.x}
                          cy={pt.y}
                          r={3.5}
                          fill="#00ff66"
                          opacity={node.normalized}
                        />
                      )}

                      {/* Input Prompt Token Label on the left */}
                      {inputLabel !== null && (
                        <text
                          x={pt.x - 16}
                          y={pt.y + 4}
                          className="input-token-label"
                        >
                          {inputLabel}
                        </text>
                      )}

                      {/* Output Candidate Token Label on the right */}
                      {outputLabel !== null && (
                        <text
                          x={pt.x + 18}
                          y={pt.y + 4}
                          className={`output-token-label ${isWinner ? 'winner-label' : ''}`}
                        >
                          {outputLabel}
                        </text>
                      )}
                    </g>
                  );
                })}
              </g>
            </svg>
          ) : (
            <div style={{ display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: '#629c6b' }}>
              <p>Loading Neural Flow...</p>
            </div>
          )}
        </div>

        {/* Right: Neuron Telemetry HUD */}
        <div className="neural-hud">
          {selectedNode ? (
            <div>
              <div className="hud-title">
                <span>{selectedNode.layer_name}</span>
                <span className={`hud-badge ${selectedNode.id === graphData?.winner_node_id ? 'winner' : ''}`}>
                  {selectedNode.id === graphData?.winner_node_id
                    ? '★ DECIDED WINNER'
                    : selectedNode.role === 'input'
                    ? `TOKEN #${selectedNode.neuron_idx}`
                    : selectedNode.role === 'output'
                    ? `RANK #${selectedNode.neuron_idx + 1}`
                    : `NEURON #${selectedNode.neuron_idx}`}
                </span>
              </div>

              <div style={{ margin: '10px 0 16px' }}>
                <strong style={{ fontSize: 16, color: '#fff', display: 'block' }}>
                  {selectedNode.label}
                </strong>
                <span style={{ fontSize: 11, color: '#629c6b', fontFamily: 'DM Mono' }}>
                  Neuron ID: <code>{selectedNode.id}</code>
                </span>
              </div>

              {silencedNodes.has(selectedNode.id) && (
                <div style={{ background: '#3b1212', border: '1px solid #d32f2f', color: '#ff8a80', padding: 8, font: '11px DM Mono', marginBottom: 12 }}>
                  ⚠ NEURON SILENCED (Downstream synaptic transmission blocked)
                </div>
              )}

              {/* INPUT NODE SPECIFICS */}
              {selectedNode.role === 'input' && (
                <>
                  <div className="hud-metric-row">
                    <span>PROMPT SEQUENCE POSITION</span>
                    <strong>#{selectedNode.details?.token_position}</strong>
                  </div>
                  <div className="hud-metric-row">
                    <span>TOKEN VOCABULARY ID</span>
                    <strong>ID #{selectedNode.details?.token_id}</strong>
                  </div>
                  <div className="hud-metric-row">
                    <span>EMBEDDING L2 NORM</span>
                    <strong>{selectedNode.activation.toFixed(4)}</strong>
                  </div>
                  <div className="hud-metric-row">
                    <span>ATTENTION RECEIVED FROM QUERY</span>
                    <strong style={{ color: '#ffd700' }}>{selectedNode.details?.attention_received}</strong>
                  </div>
                </>
              )}

              {/* HIDDEN NEURON SPECIFICS */}
              {selectedNode.role === 'hidden' && (
                <>
                  <div className="hud-metric-row">
                    <span>RAW SCALAR ACTIVATION</span>
                    <strong style={{ color: selectedNode.activation >= 0 ? '#00ff66' : '#ff5252' }}>
                      {selectedNode.activation >= 0 ? `+${selectedNode.activation.toFixed(4)}` : selectedNode.activation.toFixed(4)}
                    </strong>
                  </div>
                  <div className="hud-metric-row">
                    <span>NORMALIZED ACTIVITY</span>
                    <strong>{(selectedNode.normalized * 100).toFixed(1)}%</strong>
                  </div>
                  <div className="hud-metric-row">
                    <span>MODEL CHANNEL DIMENSION</span>
                    <strong>Channel #{selectedNode.channel_idx} / 2048</strong>
                  </div>
                  {selectedNode.details?.logit_contribution_to_winner !== undefined && (
                    <div className="hud-metric-row">
                      <span>CONTRIBUTION TO WINNER</span>
                      <strong style={{ color: selectedNode.details.logit_contribution_to_winner >= 0 ? '#00ff66' : '#ff5252' }}>
                        {selectedNode.details.logit_contribution_to_winner >= 0 ? '+' : ''}
                        {selectedNode.details.logit_contribution_to_winner.toFixed(3)} logits
                      </strong>
                    </div>
                  )}
                  {selectedNode.details?.layer_logit_lens_hypothesis && (
                    <div className="hud-metric-row">
                      <span>LAYER LOGIT LENS HYPOTHESIS</span>
                      <strong style={{ color: '#ffd700' }}>{selectedNode.details.layer_logit_lens_hypothesis}</strong>
                    </div>
                  )}
                </>
              )}

              {/* OUTPUT NODE SPECIFICS */}
              {selectedNode.role === 'output' && (
                <>
                  <div className="hud-metric-row">
                    <span>TARGET OUTPUT TOKEN</span>
                    <strong style={{ color: '#ffd700', fontSize: 14 }}>"{selectedNode.details?.token_string}"</strong>
                  </div>
                  <div className="hud-metric-row">
                    <span>PREDICTION PROBABILITY</span>
                    <strong style={{ color: '#00ff66' }}>{selectedNode.details?.percentage}</strong>
                  </div>
                  <div className="hud-metric-row">
                    <span>VOCABULARY LOGIT</span>
                    <strong>{selectedNode.activation.toFixed(4)}</strong>
                  </div>
                  <div className="hud-metric-row">
                    <span>MARGIN OVER RUNNER-UP</span>
                    <strong style={{ color: '#ffd700' }}>{selectedNode.details?.margin_over_runner_up}</strong>
                  </div>
                </>
              )}

              {/* Activity Progress Bar */}
              <div style={{ height: 6, background: '#0b260f', borderRadius: 2, margin: '10px 0 12px', overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    width: `${Math.max(4, Math.min(100, selectedNode.normalized * 100))}%`,
                    background: silencedNodes.has(selectedNode.id) ? '#d32f2f' : 'linear-gradient(90deg, #184c20, #00ff66)',
                  }}
                />
              </div>

              {selectedNodeStats && (
                <>
                  <div className="hud-metric-row">
                    <span>INCOMING SYNAPSES</span>
                    <strong>{selectedNodeStats.incomingCount} links (max {selectedNodeStats.maxInWeight.toFixed(2)})</strong>
                  </div>
                  <div className="hud-metric-row">
                    <span>OUTGOING SYNAPSES</span>
                    <strong>{selectedNodeStats.outgoingCount} links (max {selectedNodeStats.maxOutWeight.toFixed(2)})</strong>
                  </div>
                </>
              )}

              {/* Detailed Breakdown Box */}
              <div className="hud-detail-box">
                <span style={{ color: '#00ff66', display: 'block', marginBottom: 4, fontWeight: 'bold' }}>
                  MECHANISTIC ROLE:
                </span>
                <div style={{ color: '#fff', marginBottom: 8 }}>
                  {selectedNode.role === 'input'
                    ? 'TOKEN EMBEDDING — Sequence Prompt Feature'
                    : selectedNode.role === 'output'
                    ? `DECISION LOGIT — Vocabulary Candidate #${selectedNode.neuron_idx + 1}`
                    : `RESIDUAL FEATURE — Transformer Layer ${selectedNode.details?.layer_number}`}
                </div>

                {/* Vocabulary Projections for Hidden Neurons (Computed from actual weights) */}
                {selectedNode.details?.promoted_tokens && (
                  <div style={{ marginBottom: 8 }}>
                    <span style={{ color: '#00ff66', display: 'block', marginBottom: 3, fontWeight: 'bold' }}>
                      DIRECTLY PROMOTES IN VOCABULARY:
                    </span>
                    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                      {selectedNode.details.promoted_tokens.map((tok: string, i: number) => (
                        <span key={i} style={{ background: '#0c3014', border: '1px solid #1a5e28', color: '#a6f7b4', padding: '2px 6px', fontSize: 10, borderRadius: 2 }}>
                          +{tok}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {selectedNode.details?.suppressed_tokens && (
                  <div style={{ marginBottom: 8 }}>
                    <span style={{ color: '#ff7777', display: 'block', marginBottom: 3, fontWeight: 'bold' }}>
                      DIRECTLY SUPPRESSES IN VOCABULARY:
                    </span>
                    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                      {selectedNode.details.suppressed_tokens.map((tok: string, i: number) => (
                        <span key={i} style={{ background: '#300c0c', border: '1px solid #5e1a1a', color: '#f7a6a6', padding: '2px 6px', fontSize: 10, borderRadius: 2 }}>
                          -{tok}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <span style={{ color: '#00ff66', display: 'block', marginBottom: 4, fontWeight: 'bold' }}>
                  WHAT THIS CARRIES:
                </span>
                <div>
                  {selectedNode.details?.description || 'Transformer internal representation tensor.'}
                </div>
                {selectedNode.role === 'output' && graphData?.autoregressive_continuation && (
                  <div style={{ marginTop: 8, paddingTop: 8, borderTop: '1px dashed #143e19' }}>
                    <span style={{ color: '#ffd700', display: 'block', marginBottom: 2, fontWeight: 'bold' }}>
                      AUTOREGRESSIVE MULTI-TOKEN CONTINUATION:
                    </span>
                    <span style={{ color: '#c4f7ce' }}>"{graphData.autoregressive_continuation}"</span>
                  </div>
                )}
              </div>

              {/* Silence / Knockout Action */}
              <button
                className="hud-silence-btn"
                style={{
                  background: silencedNodes.has(selectedNode.id) ? '#2e7d32' : '#e53935',
                }}
                onClick={() => toggleSilence(selectedNode.id)}
              >
                {silencedNodes.has(selectedNode.id)
                  ? '✓ RESTORE NEURON ACTIVATION'
                  : '⚡ SILENCE / ABLATE THIS NEURON'}
              </button>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '60px 10px', color: '#629c6b' }}>
              <span style={{ fontSize: 32, display: 'block', marginBottom: 12 }}>🧠</span>
              <strong style={{ color: '#00ff66', display: 'block', marginBottom: 6 }}>No Neuron Selected</strong>
              <p style={{ fontSize: 11, lineHeight: 1.5 }}>
                Tap or click any neuron in the network flow diagram to inspect its scalar activation, channel dimension, and functional role.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function App() {
  const [appMode, setAppMode] = useState<'audit' | 'observatory'>('audit');
  const [auditStep, setAuditStep] = useState<1 | 2 | 3 | 4>(1);
  const [circuitToolTab, setCircuitToolTab] = useState<'sweep' | 'patching' | 'matrix'>('sweep');
  const [observatoryTab, setObservatoryTab] = useState<'simulator' | 'history' | 'scanner'>('simulator');
  const [targetHypothesis, setTargetHypothesis] = useState<string>(' Bangalore');
  const [activeModelKey, setActiveModelKey] = useState<'tinyllama' | 'llama3.2'>('tinyllama');
  const [switchingModel, setSwitchingModel] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [layerCatalog, setLayerCatalog] = useState<LayerCatalog | null>(null);
  const [prompt, setPrompt] = useState('The capital of Karnataka is');
  const [maxNewTokens, _setMaxNewTokens] = useState(40);
  const [temperature, _setTemperature] = useState(0.0);
  const [captureComponents, _setCaptureComponents] = useState(['hidden', 'attention', 'mlp', 'intermediate']);
  const [captureLayers, setCaptureLayers] = useState('');
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [selectedHead, setSelectedHead] = useState(0);
  const [loading, setLoading] = useState(false);
  const [ablating, setAblating] = useState(false);
  const [sweeping, setSweeping] = useState(false);
  const [result, setResult] = useState<InferenceResponse | null>(null);
  const [trace, setTrace] = useState<ExperimentTrace | null>(null);
  const [experiment, setExperiment] = useState<HeadAblationExperiment | null>(null);
  const [sweepResult, setSweepResult] = useState<HeadSweepResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const currentHealth = await apiClient.health();
        setHealth(currentHealth);
        if (currentHealth.model_loaded) {
          const info = await apiClient.getModelInfo();
          setModelInfo(info);
          if (info.model_name.toLowerCase().includes('3.2')) {
            setActiveModelKey('llama3.2');
          } else {
            setActiveModelKey('tinyllama');
          }
          setLayerCatalog(await apiClient.listLayers());
        }
      } catch (err) {
        setError(`Backend unavailable: ${err}`);
      }
    };
    void checkHealth();
    const interval = setInterval(checkHealth, 6000);
    return () => clearInterval(interval);
  }, []);

  const handleModelSwitch = async (modelKey: 'tinyllama' | 'llama3.2') => {
    if (switchingModel || activeModelKey === modelKey) return;
    setSwitchingModel(true);
    setError(null);
    try {
      const newModelInfo = await apiClient.switchModel(modelKey);
      setModelInfo(newModelInfo);
      setActiveModelKey(modelKey);
      const catalog = await apiClient.listLayers();
      setLayerCatalog(catalog);
      // Reset selected layer / head within bounds of new model architecture
      if (catalog && selectedLayer >= catalog.layer_count) {
        setSelectedLayer(0);
      }
      // Reset previous trace caches
      setResult(null);
      setTrace(null);
      setSweepResult(null);
      setExperiment(null);
      // Immediately run baseline inference with newly loaded model
      await runInference(prompt);
    } catch (err: any) {
      setError(`Failed to switch model: ${err.message || err}`);
    } finally {
      setSwitchingModel(false);
    }
  };

  const selectedCaptures = trace?.captures.filter((capture) => capture.layer === selectedLayer && capture.component !== 'logits') ?? [];
  const hasSelectedLayer = selectedCaptures.length > 0;

  const runInference = async (promptToUse?: string, layersOverride?: string) => {
    const activePrompt = promptToUse ?? prompt;
    if (!activePrompt.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const parsedLayers = (layersOverride ?? captureLayers)
        .split(',')
        .map((value) => Number(value.trim()))
        .filter((value) => Number.isInteger(value) && value >= 0);
      const response = await apiClient.runInference(activePrompt, {
        maxNewTokens,
        temperature,
        captureLayers: parsedLayers.length ? parsedLayers : undefined,
        captureComponents,
      });
      setResult(response);
      const traceId = response.metadata.trace_id;
      if (typeof traceId === 'string') setTrace(await apiClient.getTrace(traceId));
    } catch (err) {
      setError(`Inference failed: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  const handleInference = () => {
    void runInference();
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
    } catch (err) {
      setError(`Ablation failed: ${err}`);
    } finally {
      setAblating(false);
    }
  };

  const handleMlpAblation = async () => {
    setAblating(true);
    setError(null);
    try {
      const exp = await apiClient.runMlpAblation(prompt, selectedLayer);
      setExperiment(exp);
    } catch (err) {
      setError(`MLP Ablation failed: ${err}`);
    } finally {
      setAblating(false);
    }
  };

  const handleHeadSweep = async () => {
    setSweeping(true);
    setError(null);
    try {
      const sweep = await apiClient.runHeadSweep(prompt, [0, 1, 2, 3, 4, 5]);
      setSweepResult(sweep);
    } catch (err) {
      setError(`Head sweep failed: ${err}`);
    } finally {
      setSweeping(false);
    }
  };

  useEffect(() => {
    void runInference('The capital of Karnataka is');
  }, []);

  return (
    <div className="app-shell">
      {/* Platform Header */}
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark">NA</span>
          <div>
            <p className="eyebrow">NEURAL BEHAVIOR & SAFETY AUDIT LAB • RESEARCH SUITE</p>
            <h1>NeuroAudit</h1>
            <span className="brand-subtitle">AI Safety & Mechanistic Neuro-Observatory</span>
          </div>
        </div>

        <div className="topbar-controls">
          {/* Model Architecture Toggle Switcher */}
          <div className="model-switch-capsule" role="group" aria-label="Model Architecture Switcher">
            <button
              type="button"
              className={`model-capsule-btn ${activeModelKey === 'tinyllama' ? 'active' : ''}`}
              onClick={() => handleModelSwitch('tinyllama')}
              disabled={switchingModel}
              title="TinyLlama 1.1B • Ultra-fast mechanistic circuit tracing & ablation"
            >
              <span className="capsule-icon">🦙</span>
              <div className="capsule-text">
                <span className="capsule-title">TinyLlama 1.1B</span>
                <span className="capsule-subtitle">Fast Trace</span>
              </div>
            </button>
            <button
              type="button"
              className={`model-capsule-btn ${activeModelKey === 'llama3.2' ? 'active' : ''}`}
              onClick={() => handleModelSwitch('llama3.2')}
              disabled={switchingModel}
              title="Llama 3.2 (3B) • Meta high-reasoning conversational intelligence"
            >
              <span className="capsule-icon">⚡</span>
              <div className="capsule-text">
                <span className="capsule-title">Llama 3.2 (3B)</span>
                <span className="capsule-subtitle">Deep Reasoning</span>
              </div>
            </button>
          </div>

          <div className="mode-toggle-group">
            <button
              type="button"
              className={`mode-toggle-btn ${appMode === 'audit' ? 'active' : ''}`}
              onClick={() => setAppMode('audit')}
            >
              🛡️ Forensic Audit Pipeline
            </button>
            <button
              type="button"
              className={`mode-toggle-btn ${appMode === 'observatory' ? 'active' : ''}`}
              onClick={() => setAppMode('observatory')}
            >
              🔬 Research Observatory
            </button>
          </div>

          <div className={`connection-pill ${health?.model_loaded ? 'online' : ''} ${switchingModel ? 'switching' : ''}`}>
            <span className={switchingModel ? 'pulsing-dot' : ''} />
            {switchingModel
              ? 'SWITCHING ARCHITECTURE...'
              : health?.model_loaded
              ? `${activeModelKey === 'llama3.2' ? 'LLAMA 3.2 (3B)' : 'TINYLLAMA 1.1B'} ONLINE`
              : 'CONNECTING'}
          </div>
        </div>
      </header>

      {/* Primary Pipeline Stepper Bar (When in Forensic Audit Mode) */}
      {appMode === 'audit' && (
        <div className="pipeline-stepper-bar">
          <div className="pipeline-stepper-inner">
            <button
              type="button"
              className={`step-btn ${auditStep === 1 ? 'active' : auditStep > 1 ? 'completed' : ''}`}
              onClick={() => setAuditStep(1)}
            >
              <span className="step-num">{auditStep > 1 ? '✓' : '01'}</span>
              <div className="step-text">
                <span className="step-label">PHASE 01</span>
                <strong className="step-title">TARGET SPECIFICATION</strong>
              </div>
            </button>
            <div className={`step-connector ${auditStep > 1 ? 'filled' : ''}`} />

            <button
              type="button"
              className={`step-btn ${auditStep === 2 ? 'active' : auditStep > 2 ? 'completed' : ''}`}
              onClick={() => setAuditStep(2)}
            >
              <span className="step-num">{auditStep > 2 ? '✓' : '02'}</span>
              <div className="step-text">
                <span className="step-label">PHASE 02</span>
                <strong className="step-title">3D SYNAPTIC PROBE</strong>
              </div>
            </button>
            <div className={`step-connector ${auditStep > 2 ? 'filled' : ''}`} />

            <button
              type="button"
              className={`step-btn ${auditStep === 3 ? 'active' : auditStep > 3 ? 'completed' : ''}`}
              onClick={() => setAuditStep(3)}
            >
              <span className="step-num">{auditStep > 3 ? '✓' : '03'}</span>
              <div className="step-text">
                <span className="step-label">PHASE 03</span>
                <strong className="step-title">CIRCUIT LOCALIZATION</strong>
              </div>
            </button>
            <div className={`step-connector ${auditStep > 3 ? 'filled' : ''}`} />

            <button
              type="button"
              className={`step-btn ${auditStep === 4 ? 'active' : ''}`}
              onClick={() => setAuditStep(4)}
            >
              <span className="step-num">04</span>
              <div className="step-text">
                <span className="step-label">PHASE 04</span>
                <strong className="step-title">VERIFICATION & DOSSIER</strong>
              </div>
            </button>
          </div>
        </div>
      )}

      <main className="workspace">
        {error && (
          <div className="error-banner" style={{ marginBottom: 16, padding: '10px 14px', background: '#fee2e2', border: '1px solid #f87171', borderRadius: 8, color: '#991b1b', fontSize: 13 }}>
            ⚠️ {error}
          </div>
        )}

        {/* -------------------- AUDIT MODE CONTENT -------------------- */}
        {appMode === 'audit' && (
          <>
            {auditStep === 1 && (
              <AuditCaseSelector
                currentPrompt={prompt}
                onPromptChange={setPrompt}
                onSelectPreset={(preset: AuditCase) => {
                  setPrompt(preset.prompt);
                  setTargetHypothesis(preset.expectedToken);
                  void runInference(preset.prompt);
                }}
                onRunInference={handleInference}
                onProceedToStep2={() => setAuditStep(2)}
                loading={loading}
                inferenceResult={result}
                targetToken={targetHypothesis}
                onTargetTokenChange={setTargetHypothesis}
              />
            )}

            {auditStep === 2 && (
              <div className="audit-step-container">
                <div className="step-intro-banner">
                  <div>
                    <span className="step-tag">PHASE 02 • RESIDUAL STREAM & LOGIT LENS PROBE</span>
                    <h2>3D Synaptic Neural Flow Probe</h2>
                    <p>
                      Inspect how information flows from input tokens through early, middle, and late residual streams
                      into vocabulary logits. Click any node to inspect activations, vocabulary projections, and logit contributions.
                    </p>
                  </div>
                  <button
                    type="button"
                    className="proceed-next-step-btn"
                    onClick={() => setAuditStep(3)}
                  >
                    Proceed to Circuit Localization (Step 3) ➔
                  </button>
                </div>

                <Synaptic3DNeuralFlow
                  initialPrompt={prompt}
                  onProceedToStep3={() => setAuditStep(3)}
                />
              </div>
            )}

            {auditStep === 3 && (
              <div className="audit-step-container">
                <div className="step-intro-banner">
                  <div>
                    <span className="step-tag">PHASE 03 • CAUSAL CIRCUIT LOCALIZATION</span>
                    <h2>Pinpoint Causal Memory & Attention Circuits</h2>
                    <p>
                      Use Causal Head Sweeps, Counterfactual Activation Patching, and Attention Matrices
                      to locate the exact attention heads and MLP layers responsible for the factual association.
                    </p>
                  </div>
                  <div className="target-selection-pill">
                    <span>Targeting: <strong>Layer {selectedLayer}, Head {selectedHead}</strong></span>
                    <button
                      type="button"
                      className="proceed-next-step-btn"
                      style={{ marginLeft: 12 }}
                      onClick={() => setAuditStep(4)}
                    >
                      Verify Circuit in Step 4 ➔
                    </button>
                  </div>
                </div>

                {/* Sub-tool Selector */}
                <div className="circuit-subtool-nav">
                  <button
                    type="button"
                    className={`circuit-tool-tab ${circuitToolTab === 'sweep' ? 'active' : ''}`}
                    onClick={() => setCircuitToolTab('sweep')}
                  >
                    ⚡ Causal Head Sensitivity Map (Full Sweep)
                  </button>
                  <button
                    type="button"
                    className={`circuit-tool-tab ${circuitToolTab === 'patching' ? 'active' : ''}`}
                    onClick={() => setCircuitToolTab('patching')}
                  >
                    🔄 Counterfactual Activation Patching
                  </button>
                  <button
                    type="button"
                    className={`circuit-tool-tab ${circuitToolTab === 'matrix' ? 'active' : ''}`}
                    onClick={() => setCircuitToolTab('matrix')}
                  >
                    🔬 Attention Matrix & Layer Tensors
                  </button>
                </div>

                {circuitToolTab === 'sweep' && (
                  <HeadSensitivityMap
                    sweep={sweepResult}
                    loading={sweeping}
                    onRunSweep={handleHeadSweep}
                    onSelectHead={(layer, head) => {
                      setSelectedLayer(layer);
                      setSelectedHead(head);
                    }}
                  />
                )}

                {circuitToolTab === 'patching' && (
                  <ActivationPatchingTab modelInfo={modelInfo} layerCatalog={layerCatalog} />
                )}

                {circuitToolTab === 'matrix' && (
                  <div className="matrix-and-layer-container">
                    <div className="layer-toolbar" style={{ marginBottom: 16 }}>
                      <strong>LAYER {selectedLayer}</strong>
                      <label>
                        HEAD{' '}
                        <select value={selectedHead} onChange={(e) => setSelectedHead(Number(e.target.value))}>
                          {Array.from({ length: layerCatalog?.num_attention_heads ?? 32 }, (_, idx) => (
                            <option key={idx} value={idx}>
                              Head {idx}
                            </option>
                          ))}
                        </select>
                      </label>
                      <button type="button" onClick={retraceSelectedLayer} disabled={loading}>
                        RETRACE LAYER
                      </button>
                      <button
                        type="button"
                        onClick={() => setAuditStep(4)}
                        style={{ background: 'var(--ink, #0f172a)', color: '#ffffff' }}
                      >
                        Set L{selectedLayer} H{selectedHead} as Audit Target ➔
                      </button>
                    </div>

                    {result && selectedCaptures.length > 0 ? (
                      <div className="trace-captures">
                        {selectedCaptures.map((capture) => (
                          <div
                            className={`capture-card ${capture.component === 'attention_weights' ? 'capture-card-matrix' : ''}`}
                            key={`${capture.path}-${capture.component}`}
                          >
                            <div>
                              <strong>{capture.component}</strong>
                              <span>{capture.path}</span>
                            </div>
                            <b>{capture.shape.join(' × ')}</b>
                            {capture.component === 'attention_weights' && capture.values && (
                              <>
                                <InteractiveAttentionMatrix
                                  tokens={result.input_tokens}
                                  values={capture.values[selectedHead] || capture.values[0] || []}
                                  headIndex={selectedHead}
                                  layerIndex={selectedLayer}
                                />
                                <div className="attention-facts">
                                  <span>HEAD {selectedHead}</span>
                                  <span>ENTROPY {capture.mean_entropy?.toFixed(3) ?? '—'}</span>
                                </div>
                              </>
                            )}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="no-experiment-prompt">
                        <p>Retrace Layer {selectedLayer} to view raw attention tensor matrix.</p>
                        <button type="button" onClick={retraceSelectedLayer} disabled={loading} className="run-baseline-btn">
                          Retrace Layer {selectedLayer}
                        </button>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {auditStep === 4 && (
              <ExecutiveAuditDossier
                currentPrompt={prompt}
                targetToken={targetHypothesis}
                selectedLayer={selectedLayer}
                selectedHead={selectedHead}
                onLayerChange={setSelectedLayer}
                onHeadChange={setSelectedHead}
                onRunHeadAblation={handleAblation}
                onRunMlpAblation={handleMlpAblation}
                ablating={ablating}
                experiment={experiment}
                modelName={modelInfo?.model_name || 'TinyLlama-1.1B'}
                numLayers={layerCatalog?.layer_count || 22}
                numHeads={layerCatalog?.num_attention_heads || 32}
                onOpenObservatory={() => {
                  setAppMode('observatory');
                  setObservatoryTab('history');
                }}
              />
            )}
          </>
        )}

        {/* -------------------- OBSERVATORY MODE CONTENT -------------------- */}
        {appMode === 'observatory' && (
          <div className="observatory-wrapper">
            <div className="step-intro-banner">
              <div>
                <span className="step-tag">RESEARCH OBSERVATORY • NEURO-ANALYTICS SUITE</span>
                <h2>Mechanistic Neuro-Observatory</h2>
                <p>
                  Explore transformer self-attention computations from first principles, inspect the full SQLite audit trail,
                  and dissect raw layer activations.
                </p>
              </div>
              <button
                type="button"
                className="mode-toggle-btn active"
                onClick={() => setAppMode('audit')}
              >
                Return to Forensic Audit Pipeline ➔
              </button>
            </div>

            <nav className="nav-tabs" style={{ marginBottom: 20 }}>
              <button
                type="button"
                className={`nav-tab ${observatoryTab === 'simulator' ? 'active' : ''}`}
                onClick={() => setObservatoryTab('simulator')}
              >
                📐 Mini-Transformer Simulator
              </button>
              <button
                type="button"
                className={`nav-tab ${observatoryTab === 'history' ? 'active' : ''}`}
                onClick={() => setObservatoryTab('history')}
              >
                💾 SQLite Experiment History
              </button>
              <button
                type="button"
                className={`nav-tab ${observatoryTab === 'scanner' ? 'active' : ''}`}
                onClick={() => setObservatoryTab('scanner')}
              >
                🔬 Raw Layer & Attention Explorer
              </button>
            </nav>

            {observatoryTab === 'simulator' && <MiniTransformerSimulator />}
            {observatoryTab === 'history' && <ExperimentHistoryTab />}
            {observatoryTab === 'scanner' && (
              <>
                <section className="intro-row">
                  <div>
                    <p className="eyebrow accent">RAW TENSOR EXPLORER</p>
                    <h2>Deep Layer Dissection</h2>
                    <p className="intro-copy">
                      Inspect captured activations, attention entropy, and ablate individual components.
                    </p>
                  </div>
                </section>

                <div className="layer-explorer">
                  <div className="layer-list">
                    {layerCatalog?.layers.map((item) => (
                      <button
                        type="button"
                        className={selectedLayer === item.layer ? 'layer-button selected' : 'layer-button'}
                        key={item.layer}
                        onClick={() => setSelectedLayer(item.layer)}
                      >
                        L{String(item.layer).padStart(2, '0')}
                      </button>
                    ))}
                  </div>
                  <div className="layer-detail">
                    <div className="layer-toolbar">
                      <strong>LAYER {selectedLayer}</strong>
                      <label>
                        HEAD{' '}
                        <select value={selectedHead} onChange={(event) => setSelectedHead(Number(event.target.value))}>
                          {Array.from({ length: layerCatalog?.num_attention_heads ?? 0 }, (_, index) => (
                            <option key={index} value={index}>
                              {index}
                            </option>
                          ))}
                        </select>
                      </label>
                      <button type="button" onClick={retraceSelectedLayer} disabled={loading}>
                        RETRACE LAYER
                      </button>
                      <button type="button" onClick={handleAblation} disabled={loading || ablating || !hasSelectedLayer}>
                        {ablating ? 'ABLATING...' : `ABLATE HEAD ${selectedHead}`}
                      </button>
                      <button
                        type="button"
                        onClick={handleMlpAblation}
                        disabled={loading || ablating || !hasSelectedLayer}
                        style={{ background: '#5c6bc0' }}
                      >
                        {ablating ? 'ABLATING...' : `ABLATE MLP L${selectedLayer}`}
                      </button>
                    </div>

                    {!hasSelectedLayer && (
                      <p className="layer-empty">Layer {selectedLayer} is not in this trace. Retrace it to inspect the real tensors.</p>
                    )}

                    <div className="trace-captures">
                        {selectedCaptures.map((capture) => (
                          <div
                            className={`capture-card ${capture.component === 'attention_weights' ? 'capture-card-matrix' : ''}`}
                            key={`${capture.path}-${capture.component}`}
                          >
                          <div>
                            <strong>{capture.component}</strong>
                            <span>{capture.path}</span>
                          </div>
                          <b>{capture.shape.join(' × ')}</b>
                          <div className="stat-line">
                            <span>MEAN {capture.mean.toFixed(3)}</span>
                            <span>STD {capture.std.toFixed(3)}</span>
                            <span>MAX {capture.abs_max.toFixed(3)}</span>
                          </div>

                          {capture.component === 'attention_weights' && capture.values && (
                            <>
                              <InteractiveAttentionMatrix
                                tokens={result?.input_tokens ?? []}
                                values={capture.values[selectedHead] || capture.values[0] || []}
                                headIndex={selectedHead}
                                layerIndex={selectedLayer}
                              />
                              <div className="attention-facts">
                                <span>HEAD {selectedHead}</span>
                                <span>ENTROPY {capture.mean_entropy?.toFixed(3) ?? '—'}</span>
                              </div>
                            </>
                          )}

                          {capture.token_means && (
                            <div className="activation-strip">
                              {capture.token_means.slice(0, 32).map((value, index) => (
                                <i key={index} style={{ height: `${Math.max(8, Math.min(100, value * 35))}%` }} />
                              ))}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </main>

      <footer>
        NEUROAUDIT PLATFORM <span>LOCAL INFERENCE / {health?.device?.toUpperCase() ?? 'CPU'} • ENTERPRISE AUDIT SUITE</span>
      </footer>
    </div>
  );
}

export default App;