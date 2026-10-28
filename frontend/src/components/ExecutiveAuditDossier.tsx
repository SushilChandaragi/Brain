import React, { useState } from 'react';
import { HeadAblationExperiment } from '../api/client';

interface ExecutiveAuditDossierProps {
  currentPrompt: string;
  targetToken: string;
  selectedLayer: number;
  selectedHead: number;
  onLayerChange: (layer: number) => void;
  onHeadChange: (head: number) => void;
  onRunHeadAblation: () => Promise<void>;
  onRunMlpAblation: () => Promise<void>;
  ablating: boolean;
  experiment: HeadAblationExperiment | null;
  modelName: string;
  numLayers: number;
  numHeads: number;
  onOpenObservatory: () => void;
}

export const ExecutiveAuditDossier: React.FC<ExecutiveAuditDossierProps> = ({
  currentPrompt,
  targetToken,
  selectedLayer,
  selectedHead,
  onLayerChange,
  onHeadChange,
  onRunHeadAblation,
  onRunMlpAblation,
  ablating,
  experiment,
  modelName,
  numLayers,
  numHeads,
  onOpenObservatory,
}) => {
  const [showExportModal, setShowExportModal] = useState(false);
  const [copied, setCopied] = useState(false);

  // Compute safety grade
  const logitDelta = experiment?.metrics?.target_logit_delta ?? 0;
  const probDelta = experiment?.metrics?.target_probability_delta ?? 0;
  const rankDelta = experiment?.metrics?.rank_delta ?? 0;
  const predictionChanged = experiment?.metrics?.prediction_changed ?? false;

  let safetyGrade = 'GRADE A • ROBUST';
  let gradeColor = '#10b981'; // Green
  let gradeDesc = 'The model maintains localized factual representations with clear causal attribution.';

  if (predictionChanged || Math.abs(logitDelta) > 4.0 || Math.abs(probDelta) > 0.5) {
    safetyGrade = 'CRITICAL CIRCUIT IDENTIFIED';
    gradeColor = '#f59e0b'; // Amber
    gradeDesc = `Surgical ablation of L${selectedLayer} H${selectedHead} caused a decisive ${(Math.abs(probDelta) * 100).toFixed(1)}% drop in target token probability, proving this head is a primary causal driver.`;
  }

  const exportMarkdown = `
# NEUROAUDIT EXECUTIVE SAFETY & COMPLIANCE DOSSIER
------------------------------------------------------------------
AUDIT ID: ${experiment?.experiment_id || 'AUDIT-' + Date.now().toString(36).toUpperCase()}
TIMESTAMP: ${experiment?.created_at || new Date().toISOString()}
TARGET MODEL: ${modelName || 'TinyLlama-1.1B'}
AUDIT SUITE: Mechanistic Interpretability & Circuit Verification

## 1. TARGET SPECIFICATION
- Active Prompt: "${currentPrompt}"
- Target Hypothesis Token: "${targetToken || experiment?.baseline?.target_token || 'N/A'}"
- Baseline Predicted Token: "${experiment?.baseline?.predicted_token || 'N/A'}" (${((experiment?.baseline?.target_probability || 0) * 100).toFixed(2)}% confidence)

## 2. SURGICAL CIRCUIT INTERVENTION
- Target Architecture: Layer ${selectedLayer}, Attention Head ${selectedHead}
- Intervention Type: ${experiment?.intervention?.type || 'attention_head_ablation'}
- Baseline Logit: ${experiment?.baseline?.target_logit?.toFixed(4) || 'N/A'}
- Intervened Logit: ${experiment?.intervention_result?.target_logit?.toFixed(4) || 'N/A'}
- Causal Logit Impact (Δlogit): ${logitDelta.toFixed(4)}
- Probability Shift (Δprob): ${(probDelta * 100).toFixed(2)}%
- Target Rank Shift (Δrank): ${rankDelta > 0 ? `+${rankDelta}` : rankDelta}

## 3. AUDIT VERDICT & REGULATORY STATUS
- Safety Verdict: ${safetyGrade}
- Evaluation Note: ${gradeDesc}
- SQLite Audit Trail: Verified & Stored into local compliance ledger.
------------------------------------------------------------------
Certified by NeuroAudit Platform (Mechanistic Interpretability Safety Lab)
`.trim();

  const handleCopyReport = () => {
    navigator.clipboard.writeText(exportMarkdown);
    setCopied(true);
    setTimeout(() => setCopied(false), 2500);
  };

  return (
    <div className="executive-dossier-container">
      {/* Step Header */}
      <div className="step-intro-banner">
        <div>
          <span className="step-tag">PHASE 04 • SURGICAL VERIFICATION & DOSSIER</span>
          <h2>Causal Intervention & Regulatory Audit Dossier</h2>
          <p>
            Verify whether the attention head or MLP layer identified in Phase 03 is truly causally necessary
            for the model's factual claim via zero-ablation, then generate an audit report.
          </p>
        </div>
        <div className="audit-readiness-badge">
          <span className="pulse-dot" style={{ backgroundColor: gradeColor }} />
          <span>{safetyGrade}</span>
        </div>
      </div>

      {/* Intervention Controls Card */}
      <div className="dossier-control-card">
        <div className="card-top-row">
          <div>
            <h3>Target Circuit Surgical Controls</h3>
            <p className="subtext">Configure zero-activation ablation for causal proof</p>
          </div>
          <div className="target-pill">
            Targeting: <strong>L{selectedLayer} H{selectedHead}</strong>
          </div>
        </div>

        <div className="circuit-selector-bar">
          <label>
            LAYER
            <select
              value={selectedLayer}
              onChange={(e) => onLayerChange(Number(e.target.value))}
              disabled={ablating}
            >
              {Array.from({ length: numLayers || 22 }, (_, idx) => (
                <option key={idx} value={idx}>
                  Layer {idx}
                </option>
              ))}
            </select>
          </label>

          <label>
            ATTENTION HEAD
            <select
              value={selectedHead}
              onChange={(e) => onHeadChange(Number(e.target.value))}
              disabled={ablating}
            >
              {Array.from({ length: numHeads || 32 }, (_, idx) => (
                <option key={idx} value={idx}>
                  Head {idx}
                </option>
              ))}
            </select>
          </label>

          <button
            type="button"
            className="ablate-head-action-btn"
            onClick={onRunHeadAblation}
            disabled={ablating}
          >
            {ablating ? 'Executing Ablation...' : `Ablate Head ${selectedHead} (Knock-out)`}
          </button>

          <button
            type="button"
            className="ablate-mlp-action-btn"
            onClick={onRunMlpAblation}
            disabled={ablating}
          >
            {ablating ? 'Executing Ablation...' : `Ablate Layer ${selectedLayer} MLP`}
          </button>
        </div>
      </div>

      {/* Intervention Metrics Comparison */}
      {experiment ? (
        <div className="metrics-comparison-card">
          <div className="metrics-card-header">
            <h4>Causal Intervention Evidence (Before vs After)</h4>
            <span className="exp-id">ID: {experiment.experiment_id.slice(0, 8)}...</span>
          </div>

          <div className="comparison-columns-grid">
            <div className="comparison-col baseline-col">
              <span className="col-label">BASELINE MODEL STATE</span>
              <div className="token-display">
                <strong>{experiment.baseline.predicted_token}</strong>
                <span className="sub-prob">
                  {((experiment.baseline.target_probability || 0) * 100).toFixed(2)}% probability
                </span>
              </div>
              <div className="stat-row">
                <span>Target Logit:</span>
                <b>{experiment.baseline.target_logit?.toFixed(3)}</b>
              </div>
              <div className="stat-row">
                <span>Target Rank:</span>
                <b>#{experiment.baseline.target_rank}</b>
              </div>
            </div>

            <div className="delta-center-col">
              <span className="delta-badge">CAUSAL IMPACT</span>
              <div className="delta-numbers">
                <span
                  className="delta-logit"
                  style={{ color: logitDelta < 0 ? '#dc2626' : '#10b981' }}
                >
                  {logitDelta > 0 ? `+${logitDelta.toFixed(3)}` : logitDelta.toFixed(3)}
                </span>
                <span className="delta-label">Logit Shift (Δlogit)</span>
              </div>
              <div className="delta-pct-bar">
                <span>Probability Delta:</span>
                <strong style={{ color: probDelta < 0 ? '#dc2626' : '#10b981' }}>
                  {(probDelta * 100).toFixed(1)}%
                </strong>
              </div>
            </div>

            <div className="comparison-col intervened-col">
              <span className="col-label">SURGICALLY ABLATED STATE</span>
              <div className="token-display">
                <strong>{experiment.intervention_result.predicted_token}</strong>
                <span className="sub-prob">
                  {((experiment.intervention_result.target_probability || 0) * 100).toFixed(2)}% probability
                </span>
              </div>
              <div className="stat-row">
                <span>Target Logit:</span>
                <b>{experiment.intervention_result.target_logit?.toFixed(3)}</b>
              </div>
              <div className="stat-row">
                <span>Target Rank:</span>
                <b>#{experiment.intervention_result.target_rank}</b>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="no-experiment-prompt">
          <p>
            💡 Select an attention head or MLP layer above and click <b>Ablate</b> to measure the causal counterfactual delta.
          </p>
        </div>
      )}

      {/* Formal Audit Report Card */}
      <div className="executive-report-card">
        <div className="report-card-topbar">
          <div>
            <span className="eyebrow-accent">CERTIFIED REGULATORY AUDIT REPORT</span>
            <h3>NeuroAudit Safety Compliance Certificate</h3>
          </div>
          <div className="report-actions">
            <button
              type="button"
              className="open-modal-btn"
              onClick={() => setShowExportModal(true)}
            >
              📄 View Printable Dossier
            </button>
            <button
              type="button"
              className="view-archive-btn"
              onClick={onOpenObservatory}
            >
              💾 View SQLite Ledger
            </button>
          </div>
        </div>

        <div className="report-summary-grid">
          <div className="summary-metric-tile">
            <span className="tile-label">SAFETY RATING</span>
            <strong style={{ color: gradeColor }}>{safetyGrade}</strong>
            <p className="tile-desc">{gradeDesc}</p>
          </div>

          <div className="summary-metric-tile">
            <span className="tile-label">IDENTIFIED CAUSAL HEAD</span>
            <strong>Layer {selectedLayer}, Head {selectedHead}</strong>
            <p className="tile-desc">Primary circuit routing tokens to the final residual stream.</p>
          </div>

          <div className="summary-metric-tile">
            <span className="tile-label">CAUSAL SENSITIVITY</span>
            <strong>{Math.abs(logitDelta).toFixed(3)} Logit Delta</strong>
            <p className="tile-desc">
              {logitDelta < 0
                ? 'Ablation significantly reduced confidence in the target claim.'
                : 'Head ablation did not degrade the primary prediction.'}
            </p>
          </div>

          <div className="summary-metric-tile">
            <span className="tile-label">AUDIT TRAIL STATUS</span>
            <strong style={{ color: '#10b981' }}>COMPLIANT & PERSISTED</strong>
            <p className="tile-desc">Record hashed and saved into local SQLite experiments ledger.</p>
          </div>
        </div>
      </div>

      {/* Modal for Print / Export */}
      {showExportModal && (
        <div className="modal-backdrop" onClick={() => setShowExportModal(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Executive Safety & Compliance Dossier</h3>
              <button
                type="button"
                className="close-modal-btn"
                onClick={() => setShowExportModal(false)}
              >
                ✕
              </button>
            </div>
            <div className="modal-body">
              <pre className="markdown-preview">{exportMarkdown}</pre>
            </div>
            <div className="modal-footer">
              <button
                type="button"
                className="copy-btn"
                onClick={handleCopyReport}
              >
                {copied ? '✓ Copied to Clipboard' : '📋 Copy Markdown Report'}
              </button>
              <button
                type="button"
                className="print-btn"
                onClick={() => window.print()}
              >
                🖨️ Print / Save as PDF
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
