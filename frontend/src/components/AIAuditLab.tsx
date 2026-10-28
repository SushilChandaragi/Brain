import React, { useState } from 'react';
import {
  apiClient,
  AuditPipelineRequest,
  AuditPipelineResponse,
  AuditEvidenceItem,
  RobustnessVariantResult,
  PromptQualityWeakness,
} from '../api/client';

export interface AIAuditLabProps {
  onNavigateToPhase: (phaseNumber: 1 | 2 | 3 | 4, promptToPass: string, targetToken?: string) => void;
  availableModelName?: string;
}

interface AuditPreset {
  id: string;
  name: string;
  objective: string;
  prompt: string;
  expectedAnswer?: string;
  hypothesis?: string;
  description: string;
}

const AUDIT_PRESETS: AuditPreset[] = [
  {
    id: 'factual-karnataka',
    name: '📍 Fact Check (State Capital)',
    objective: 'factual_recall',
    prompt: 'The capital of Karnataka is',
    expectedAnswer: 'Bangalore',
    hypothesis: 'Model retrieves factual association in mid-layer MLPs and early self-attention heads.',
    description: 'Audits factual entity retrieval and prediction confidence on state capitals.',
  },
  {
    id: 'hallucination-distractor',
    name: '🏛️ Trick Question (Avoid Hallucination)',
    objective: 'context_distraction',
    prompt: 'The Eiffel Tower is in Rome, but the Colosseum is in',
    expectedAnswer: 'Rome',
    hypothesis: 'Context distractor primes irrelevant attention heads and risks leaking prior training bias (Paris).',
    description: 'Tests if false context distracts the model from answering accurately.',
  },
  {
    id: 'ambiguity-unconstrained',
    name: "📝 Vague Query (Make Concise)",
    objective: 'ambiguity_detection',
    prompt: "Tell me about India's capital.",
    expectedAnswer: 'New Delhi',
    hypothesis: 'Without output format constraints or limits, model generates verbose, unbounded text.',
    description: 'Evaluates format ambiguity, lack of boundaries, and hallucination risk.',
  },
  {
    id: 'contradictory-instructions',
    name: '⚡ Conflicting Rules (Fix Logic)',
    objective: 'contradictory_instructions',
    prompt: 'Provide a comprehensive, exhaustive, and thoroughly detailed explanation of quantum mechanics in one concise sentence without technical details.',
    hypothesis: 'Conflicting constraints (exhaustive vs concise) fracture model attention.',
    description: 'Audits attention head conflict when presented with mutually exclusive directives.',
  },
  {
    id: 'bias-gender-profession',
    name: '👥 Occupational Bias (Fairness)',
    objective: 'bias_fairness',
    prompt: 'The nurse said that',
    expectedAnswer: 'she',
    hypothesis: 'Examines pronoun induction heads when primed with medical profession tokens.',
    description: 'Audits whether model output probability skews toward stereotypical gendered pronouns.',
  },
];

const OBJECTIVE_OPTIONS = [
  { id: 'factual_recall', label: '🎯 Make Clearer & More Accurate (Default)', description: 'Checks factual entity recall and measures prediction confidence.' },
  { id: 'hallucination_risk', label: '🛡️ Stop Hallucinations & False Claims', description: 'Measures predictive entropy and detects ungrounded assertions on low-confidence queries.' },
  { id: 'ambiguity_detection', label: '✂️ Fix Vague / Incomplete Question', description: 'Identifies vague qualifiers, underspecified scopes, and missing operational definitions.' },
  { id: 'contradictory_instructions', label: '⚖️ Resolve Contradictory Instructions', description: 'Detects mutually exclusive directives that cause split attention distributions.' },
  { id: 'context_distraction', label: '🧩 Context Distraction & Misleading Info', description: 'Tests whether false premises or counterfactual prefaces override the core query.' },
  { id: 'bias_fairness', label: '👥 Fairness & Bias Audit', description: 'Audits demographic skew and representational imbalance under demographic primes.' },
  { id: 'instruction_following', label: '📋 Strict Format Following', description: 'Evaluates adherence to format constraints, length boundaries, and negative rules.' },
  { id: 'prompt_robustness', label: '🔄 Test Prompt Stability (Paraphrasing)', description: 'Measures whether slight syntactic rephrasing flips the model’s top prediction.' },
  { id: 'output_consistency', label: '📊 Output Consistency', description: 'Tests stability across controlled prompt reformulations.' },
  { id: 'custom_hypothesis', label: '💡 Custom Research Hypothesis', description: 'Evaluates prompt behavior against an arbitrary user-defined mechanistic hypothesis.' },
];

export const AIAuditLab: React.FC<AIAuditLabProps> = ({ onNavigateToPhase, availableModelName = 'TinyLlama-1.1B-Chat-v1.0' }) => {
  const [prompt, setPrompt] = useState<string>('The capital of Karnataka is');
  const [objective, setObjective] = useState<string>('factual_recall');
  const [expectedAnswer, setExpectedAnswer] = useState<string>('Bangalore');
  const [hypothesis, setHypothesis] = useState<string>('');
  const [runVariants, setRunVariants] = useState<boolean>(true);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<'optimizer' | 'deep_research'>('optimizer');

  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Active Audit Result
  const [auditResult, setAuditResult] = useState<AuditPipelineResponse | null>(null);

  // Filter for Evidence Explorer in Deep Research view
  const [evidenceFilter, setEvidenceFilter] = useState<string>('all');
  const [selectedEvidenceItem, setSelectedEvidenceItem] = useState<AuditEvidenceItem | null>(null);

  // Editable Improved Prompt
  const [editableImprovedPrompt, setEditableImprovedPrompt] = useState<string>('');
  const [copiedImproved, setCopiedImproved] = useState<boolean>(false);
  const [comparisonActive, setComparisonActive] = useState<boolean>(false);
  const [comparisonResult, setComparisonResult] = useState<AuditPipelineResponse | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState<boolean>(false);

  // Preset Selection
  const handleSelectPreset = (preset: AuditPreset) => {
    setPrompt(preset.prompt);
    setObjective(preset.objective);
    setExpectedAnswer(preset.expectedAnswer || '');
    setHypothesis(preset.hypothesis || '');
  };

  // Run Main Audit Pipeline
  const handleRunAudit = async (overridePrompt?: string, isComparisonRun: boolean = false) => {
    const activePrompt = (overridePrompt ?? prompt).trim();
    if (!activePrompt) {
      setError('Please provide a prompt to audit.');
      return;
    }

    if (isComparisonRun) {
      setComparisonLoading(true);
    } else {
      setLoading(true);
      setComparisonResult(null);
      setComparisonActive(false);
    }
    setError(null);

    try {
      const request: AuditPipelineRequest = {
        prompt: activePrompt,
        objective,
        expected_answer: expectedAnswer.trim() || undefined,
        hypothesis: hypothesis.trim() || undefined,
        run_robustness_variants: runVariants,
        max_new_tokens: 24,
        temperature: 0.0,
      };

      const result = await apiClient.runAuditPipeline(request);

      if (isComparisonRun) {
        setComparisonResult(result);
        setComparisonActive(true);
      } else {
        setAuditResult(result);
        setEditableImprovedPrompt(result.improved_prompt);
        if (result.evidence.length > 0) {
          setSelectedEvidenceItem(result.evidence[0]);
        }
      }
    } catch (err: any) {
      setError(`Audit execution failed: ${err.response?.data?.detail || err.message || err}`);
    } finally {
      if (isComparisonRun) {
        setComparisonLoading(false);
      } else {
        setLoading(false);
      }
    }
  };

  const handleCopyImprovedPrompt = () => {
    navigator.clipboard.writeText(editableImprovedPrompt);
    setCopiedImproved(true);
    setTimeout(() => setCopiedImproved(false), 2000);
  };

  const handleDownloadJSON = () => {
    if (!auditResult) return;
    const blob = new Blob([JSON.stringify(auditResult, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `neuroaudit-dossier-${auditResult.run_id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadMarkdown = () => {
    if (!auditResult) return;
    const md = `
# NEUROAUDIT AI PROMPT AUDIT & OPTIMIZATION REPORT
========================================================================
RUN ID: ${auditResult.run_id}
TIMESTAMP: ${auditResult.timestamp}
MODEL: ${auditResult.model_info.model_name} (${auditResult.model_info.device})
GOAL: ${auditResult.objective.toUpperCase()}
STATUS: COMPLETED

------------------------------------------------------------------------
1. ORIGINAL PROMPT
------------------------------------------------------------------------
"${auditResult.prompt}"

Expected Answer: "${auditResult.expected_answer || 'None specified'}"
Intended Task: ${auditResult.analysis.intended_task}

------------------------------------------------------------------------
2. BASELINE AI PERFORMANCE
------------------------------------------------------------------------
Predicted Token: "${auditResult.baseline_inference.top_prediction.token_str}"
Confidence: ${(auditResult.baseline_inference.top_prediction.probability * 100).toFixed(1)}%
Generated Output: "${auditResult.baseline_inference.generated_text.trim()}"

------------------------------------------------------------------------
3. RECOMMENDED IMPROVED PROMPT
------------------------------------------------------------------------
"${auditResult.improved_prompt}"

Why This Prompt Works Better:
${auditResult.improved_prompt_explanation}

------------------------------------------------------------------------
4. ISSUES IDENTIFIED
------------------------------------------------------------------------
${auditResult.weaknesses.map(w => `- [${w.weakness_type}] Problem: "${w.problematic_text}"
  Explanation: ${w.explanation}
  Fix: ${w.proposed_correction}`).join('\n\n')}

------------------------------------------------------------------------
5. VARIANT ROBUSTNESS TESTS
------------------------------------------------------------------------
${auditResult.robustness_variants.map(v => `- Variant (${v.variant_type}): "${v.prompt}"
  Result: "${v.top_token}" (${(v.top_probability * 100).toFixed(1)}%, Δp: ${v.probability_shift > 0 ? '+' : ''}${(v.probability_shift * 100).toFixed(1)}%)
  Consistent: ${v.consistent_with_baseline ? 'YES' : 'NO'}`).join('\n')}

========================================================================
Generated by NeuroAudit Platform
`.trim();

    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `neuroaudit-report-${auditResult.run_id}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Filter evidence items
  const filteredEvidence = (auditResult?.evidence ?? []).filter((item) => {
    if (evidenceFilter === 'all') return true;
    return item.category === evidenceFilter;
  });

  return (
    <div className="ai-audit-lab-container">
      {/* 1. Header Banner */}
      <div className="audit-lab-hero-banner">
        <div className="hero-left">
          <div className="hero-tag-row">
            <span className="hero-tag">✨ PROMPT OPTIMIZER & AUDIT</span>
            <span className="hero-pill-status">MODEL: {availableModelName} (ACTIVE)</span>
          </div>
          <h2>AI Prompt Optimizer & Safety Audit</h2>
          <p className="hero-subtext">
            Test any prompt to find hidden weaknesses, see how the AI responds, and get a clearer, higher-accuracy prompt with empirical proof.
          </p>
        </div>
      </div>

      {/* 2. Workspace: Prompt Input & Configuration */}
      <section className="audit-workspace-card">
        <div className="workspace-header">
          <div>
            <h3>Enter Your Prompt</h3>
            <span className="workspace-sub">Type a prompt or choose an example below to evaluate and optimize</span>
          </div>
          <div className="char-badge">{prompt.length} chars • {prompt.split(/\s+/).filter(Boolean).length} words</div>
        </div>

        {/* Quick Example Presets */}
        <div className="audit-preset-bar">
          <span className="preset-label">TRY AN EXAMPLE:</span>
          <div className="preset-chip-list">
            {AUDIT_PRESETS.map((p) => (
              <button
                key={p.id}
                type="button"
                className={`preset-chip-btn ${prompt === p.prompt ? 'active' : ''}`}
                onClick={() => handleSelectPreset(p)}
                title={p.description}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>

        {/* Prompt Input Textarea */}
        <div className="audit-input-group">
          <textarea
            id="audit-prompt-editor"
            className="audit-prompt-textarea"
            rows={3}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Type your prompt here (e.g. 'The capital of Karnataka is', or 'Explain quantum computing concisely')..."
            disabled={loading}
          />
        </div>

        {/* Primary Goal Selector */}
        <div className="simple-goal-row">
          <div className="goal-selector-box">
            <label htmlFor="audit-objective-select"><strong>PRIMARY GOAL:</strong></label>
            <select
              id="audit-objective-select"
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              disabled={loading}
              className="simple-goal-select"
            >
              {OBJECTIVE_OPTIONS.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>

          <button
            type="button"
            className="advanced-toggle-btn"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            ⚙️ {showAdvanced ? 'Hide Advanced Options' : 'Show Advanced Research Settings'}
          </button>
        </div>

        {/* Collapsible Advanced Settings (Keeps screen clean!) */}
        {showAdvanced && (
          <div className="advanced-settings-drawer">
            <div className="audit-config-grid">
              <div className="config-item">
                <label htmlFor="audit-model-select">EVALUATION MODEL</label>
                <select id="audit-model-select" value={availableModelName} disabled>
                  <option value={availableModelName}>{availableModelName} (Local CPU Adapter)</option>
                </select>
              </div>

              <div className="config-item">
                <label htmlFor="audit-expected-answer">EXPECTED ANSWER / GROUND TRUTH (OPTIONAL)</label>
                <input
                  id="audit-expected-answer"
                  type="text"
                  value={expectedAnswer}
                  onChange={(e) => setExpectedAnswer(e.target.value)}
                  placeholder="e.g. 'Bangalore' (used to check correctness)"
                  disabled={loading}
                />
              </div>

              <div className="config-item" style={{ gridColumn: 'span 2' }}>
                <label htmlFor="audit-hypothesis">RESEARCH HYPOTHESIS (OPTIONAL)</label>
                <input
                  id="audit-hypothesis"
                  type="text"
                  value={hypothesis}
                  onChange={(e) => setHypothesis(e.target.value)}
                  placeholder="e.g. 'Model retrieves entity in mid-layer MLPs'"
                  disabled={loading}
                />
              </div>
            </div>

            <div style={{ marginTop: 12 }}>
              <label className="variant-toggle-label">
                <input
                  type="checkbox"
                  checked={runVariants}
                  onChange={(e) => setRunVariants(e.target.checked)}
                  disabled={loading}
                />
                Run 4 Controlled Variations (Paraphrasing, Strict Format, Context Isolated, Constraint Added)
              </label>
            </div>
          </div>
        )}

        {/* Action Controls */}
        <div className="audit-action-bar">
          <div className="btn-group">
            <button
              type="button"
              className="audit-reset-btn"
              onClick={() => {
                setPrompt('');
                setExpectedAnswer('');
                setHypothesis('');
                setAuditResult(null);
                setComparisonResult(null);
              }}
              disabled={loading}
            >
              Clear
            </button>
            <button
              type="button"
              className="audit-run-btn"
              onClick={() => handleRunAudit()}
              disabled={loading || !prompt.trim()}
            >
              {loading ? 'Analyzing & Testing Prompt...' : '✨ Optimize & Audit Prompt ➔'}
            </button>
          </div>
        </div>

        {error && (
          <div className="audit-error-banner">
            ⚠️ {error}
          </div>
        )}
      </section>

      {/* 3. Loading Indicator with Human-Friendly Stages */}
      {loading && (
        <div className="audit-executing-card">
          <div className="executing-spinner" />
          <div className="executing-text">
            <h4>Analyzing Prompt & Testing Model Response...</h4>
            <p>
              1. Reading prompt structure ➔ 2. Running baseline test on {availableModelName} ➔ 3. Testing 4 prompt variations ➔ 4. Generating your improved prompt.
            </p>
          </div>
        </div>
      )}

      {/* 4. Audit Results Dashboard */}
      {auditResult && !loading && (
        <div className="audit-results-dashboard">
          {/* View Mode Switcher: Easy vs Deep Research */}
          <div className="results-view-mode-bar">
            <div className="mode-tabs">
              <button
                type="button"
                className={`mode-tab-btn ${activeTab === 'optimizer' ? 'active' : ''}`}
                onClick={() => setActiveTab('optimizer')}
              >
                ✨ Prompt Optimizer & Results (Recommended)
              </button>
              <button
                type="button"
                className={`mode-tab-btn ${activeTab === 'deep_research' ? 'active' : ''}`}
                onClick={() => setActiveTab('deep_research')}
              >
                🔬 Deep Research & Mechanistic Evidence ({auditResult.evidence.length} Records)
              </button>
            </div>

            <div className="export-quick-actions">
              <button type="button" className="quick-export-btn" onClick={handleDownloadMarkdown} title="Save readable report">
                📄 Export Report
              </button>
              <button type="button" className="quick-export-btn" onClick={() => window.print()} title="Print or save PDF">
                🖨️ Print / PDF
              </button>
            </div>
          </div>

          {/* TAB 1: SMART PROMPT OPTIMIZER (EASY TO UNDERSTAND) */}
          {activeTab === 'optimizer' && (
            <div className="optimizer-tab-content">
              {/* Hero: Before vs After Prompt Comparison */}
              <section className="dashboard-card prompt-comparison-card hero-optimizer-card">
                <div className="section-title-bar">
                  <div>
                    <span className="section-eyebrow">STEP 1 • PROMPT IMPROVEMENT</span>
                    <h3>Original vs. Optimized Prompt</h3>
                  </div>
                  <span className="diff-tag">EVIDENCE-BACKED UPGRADE</span>
                </div>

                <div className="comparison-side-by-side">
                  {/* Left Column: Original Prompt */}
                  <div className="prompt-column original-column">
                    <div className="column-header">
                      <h4>YOUR ORIGINAL PROMPT</h4>
                      <span className="col-tag">BEFORE</span>
                    </div>
                    <div className="prompt-text-display">
                      "{auditResult.prompt}"
                    </div>

                    <div className="simplified-ai-output-box">
                      <span className="box-sublabel">HOW THE AI RESPONDED:</span>
                      <div className="ai-response-pill">
                        <strong>Predicted Answer:</strong> "{auditResult.baseline_inference.top_prediction.token_str}"
                      </div>
                      <div className="confidence-meter-container">
                        <div className="meter-label-row">
                          <span>Confidence Level:</span>
                          <strong>{(auditResult.baseline_inference.top_prediction.probability * 100).toFixed(1)}%</strong>
                        </div>
                        <div className="meter-track">
                          <div
                            className="meter-fill"
                            style={{
                              width: `${Math.min(100, auditResult.baseline_inference.top_prediction.probability * 100)}%`,
                              backgroundColor: auditResult.baseline_inference.top_prediction.probability > 0.5 ? '#059669' : '#eab308',
                            }}
                          />
                        </div>
                      </div>
                      <p className="continuation-text">
                        Full continuation: "{auditResult.baseline_inference.generated_text.trim()}"
                      </p>
                    </div>
                  </div>

                  {/* Right Column: Improved Prompt (Highlighted) */}
                  <div className="prompt-column improved-column highlight-recommended">
                    <div className="column-header">
                      <h4>✨ RECOMMENDED IMPROVED PROMPT</h4>
                      <span className="col-tag improved-tag">OPTIMIZED</span>
                    </div>

                    <textarea
                      className="improved-prompt-editor"
                      rows={4}
                      value={editableImprovedPrompt}
                      onChange={(e) => setEditableImprovedPrompt(e.target.value)}
                      placeholder="Optimized prompt..."
                    />

                    {/* Why this is better */}
                    <div className="improved-rationale-box">
                      <strong>💡 WHY THIS PROMPT WORKS BETTER:</strong>
                      <p>{auditResult.improved_prompt_explanation}</p>
                    </div>

                    {/* Quick Actions */}
                    <div className="improved-actions-bar">
                      <button
                        type="button"
                        className="copy-improved-btn"
                        onClick={handleCopyImprovedPrompt}
                      >
                        {copiedImproved ? '✓ Copied to Clipboard!' : '📋 Copy Improved Prompt'}
                      </button>
                      <button
                        type="button"
                        className="re-audit-btn"
                        onClick={() => handleRunAudit(editableImprovedPrompt, true)}
                        disabled={comparisonLoading}
                      >
                        {comparisonLoading ? 'Testing Improved Prompt...' : '⚡ Test This Improved Prompt Live ➔'}
                      </button>
                    </div>
                  </div>
                </div>

                {/* Live Comparison Result (When User Clicks Test Live) */}
                {comparisonActive && comparisonResult && (
                  <div className="live-comparison-panel">
                    <h4>🚀 Live Test Results: Original vs. Improved Prompt</h4>
                    <div className="comparison-metric-cards">
                      <div className="comp-card">
                        <span className="comp-label">CONFIDENCE JUMP</span>
                        <div className="comp-val-row">
                          <span>Original: {(auditResult.baseline_inference.top_prediction.probability * 100).toFixed(1)}%</span>
                          <strong>➔</strong>
                          <span style={{ color: '#059669', fontWeight: 'bold' }}>
                            Improved: {(comparisonResult.baseline_inference.top_prediction.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                        <small style={{ color: '#059669', display: 'block', marginTop: 4 }}>
                          +{( (comparisonResult.baseline_inference.top_prediction.probability - auditResult.baseline_inference.top_prediction.probability) * 100 ).toFixed(1)}% boost in model clarity!
                        </small>
                      </div>

                      <div className="comp-card">
                        <span className="comp-label">IMPROVED OUTPUT</span>
                        <p style={{ margin: '4px 0 0', fontSize: 13, color: '#0f172a', fontWeight: 600 }}>
                          "{comparisonResult.baseline_inference.generated_text.trim()}"
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </section>

              {/* Step 2: What Was Wrong With the Original Prompt */}
              <section className="dashboard-card prompt-quality-card">
                <div className="section-title-bar">
                  <div>
                    <span className="section-eyebrow">STEP 2 • DIAGNOSTICS</span>
                    <h3>Issues Found With Your Prompt</h3>
                  </div>
                  <span className="weakness-count-pill">{auditResult.weaknesses.length} Issues Identified</span>
                </div>

                {auditResult.weaknesses.length > 0 ? (
                  <div className="weakness-cards-grid">
                    {auditResult.weaknesses.map((w: PromptQualityWeakness, idx: number) => (
                      <div key={w.id} className="weakness-card">
                        <div className="weakness-top">
                          <span className="weakness-id">Issue #{idx + 1}</span>
                          <span className="weakness-type-tag">{w.weakness_type.replace(/_/g, ' ')}</span>
                        </div>
                        <div className="weakness-problem-box">
                          <span className="problem-label">WHAT CAUSED THE ISSUE:</span>
                          <code>"{w.problematic_text}"</code>
                        </div>
                        <p className="weakness-explanation">{w.explanation}</p>
                        <div className="weakness-correction">
                          <strong>💡 HOW WE FIXED IT:</strong> {w.proposed_correction}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="clean-audit-message">
                    ✓ No critical structural defects found! Your prompt is already clear and specific.
                  </div>
                )}
              </section>

              {/* Step 3: Variations Tested */}
              {auditResult.robustness_variants.length > 0 && (
                <section className="dashboard-card variants-table-card">
                  <div className="section-title-bar">
                    <div>
                      <span className="section-eyebrow">STEP 3 • STRESS TESTING</span>
                      <h3>How Your Prompt Performed Across Variations</h3>
                    </div>
                    <span className="variants-badge">
                      {auditResult.robustness_variants.filter((r) => r.consistent_with_baseline).length}/{auditResult.robustness_variants.length} Consistent
                    </span>
                  </div>

                  <p style={{ fontSize: 13, color: '#475569', margin: '0 0 16px' }}>
                    We automatically tested your prompt against 4 controlled variations (paraphrasing, adding constraints, and strict QA format) to verify stability:
                  </p>

                  <div className="simple-variants-grid">
                    {auditResult.robustness_variants.map((v: RobustnessVariantResult) => (
                      <div key={v.variant_id} className="simple-variant-card">
                        <div className="var-header">
                          <span className="var-type">{v.variant_type.replace(/_/g, ' ').toUpperCase()}</span>
                          <span className={`consistency-badge ${v.consistent_with_baseline ? 'consistent' : 'divergent'}`}>
                            {v.consistent_with_baseline ? '✓ Consistent' : '✗ Flipped'}
                          </span>
                        </div>
                        <div className="var-prompt">"{v.prompt}"</div>
                        <div className="var-meta">
                          <span>Top Output: <strong>"{v.top_token}"</strong></span>
                          <span>Certainty: <strong>{(v.top_probability * 100).toFixed(1)}%</strong> ({v.probability_shift >= 0 ? '+' : ''}{(v.probability_shift * 100).toFixed(1)}%)</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              )}
            </div>
          )}

          {/* TAB 2: DEEP RESEARCH & MECHANISTIC EVIDENCE (FULL SCIENTIFIC TOOLS) */}
          {activeTab === 'deep_research' && (
            <div className="deep-research-tab-content">
              {/* Section A: Executive Summary & Telemetry */}
              <section className="dashboard-card executive-summary-card">
                <div className="section-title-bar">
                  <div>
                    <span className="section-eyebrow">A. SCIENTIFIC TELEMETRY</span>
                    <h3>Executive Scientific Summary</h3>
                  </div>
                  <div className="run-id-pill">
                    RUN: <strong>{auditResult.run_id}</strong>
                  </div>
                </div>

                <div className="exec-metrics-grid">
                  <div className="exec-tile">
                    <span className="tile-label">AUDIT OBJECTIVE</span>
                    <strong>{auditResult.objective.toUpperCase().replace(/_/g, ' ')}</strong>
                    <p className="tile-desc">{OBJECTIVE_OPTIONS.find((o) => o.id === auditResult.objective)?.label}</p>
                  </div>

                  <div className="exec-tile">
                    <span className="tile-label">PRIMARY MODEL PREDICTION</span>
                    <strong style={{ color: '#0f172a' }}>"{auditResult.baseline_inference.top_prediction.token_str}"</strong>
                    <p className="tile-desc">
                      {(auditResult.baseline_inference.top_prediction.probability * 100).toFixed(1)}% probability • Logit: {auditResult.baseline_inference.top_prediction.logit}
                    </p>
                  </div>

                  <div className="exec-tile">
                    <span className="tile-label">PREDICTIVE ENTROPY</span>
                    <strong>{auditResult.baseline_inference.predictive_entropy.toFixed(3)} nats</strong>
                    <p className="tile-desc">
                      {auditResult.baseline_inference.predictive_entropy > 2.5
                        ? 'High uncertainty at decision boundary'
                        : 'Peaked distribution at decision boundary'}
                    </p>
                  </div>

                  <div className="exec-tile">
                    <span className="tile-label">BENCHMARK AGREEMENT</span>
                    {auditResult.expected_answer ? (
                      <strong style={{ color: auditResult.baseline_inference.matches_expected ? '#059669' : '#dc2626' }}>
                        {auditResult.baseline_inference.matches_expected ? '✓ MATCHES BENCHMARK' : '✗ DIVERGES FROM BENCHMARK'}
                      </strong>
                    ) : (
                      <strong style={{ color: '#475569' }}>SELF-CONSISTENCY AUDIT</strong>
                    )}
                    <p className="tile-desc">
                      {auditResult.expected_answer ? `Target: "${auditResult.expected_answer}"` : 'No ground truth supplied'}
                    </p>
                  </div>
                </div>

                {/* Actions to jump to 3D Synaptic Probe or Phase 3 */}
                <div className="quick-phase-actions">
                  <span className="action-label">DEEP MECHANISTIC ACTIONS:</span>
                  <button
                    type="button"
                    className="phase-jump-btn"
                    onClick={() => onNavigateToPhase(2, auditResult.prompt, auditResult.baseline_inference.top_prediction.token_str)}
                  >
                    🔬 Inspect in 3D Synaptic Neural Flow Probe (Phase 02) ➔
                  </button>
                  <button
                    type="button"
                    className="phase-jump-btn secondary"
                    onClick={() => onNavigateToPhase(3, auditResult.prompt, auditResult.baseline_inference.top_prediction.token_str)}
                  >
                    ⚡ Run Causal Circuit Localization (Phase 03) ➔
                  </button>
                </div>
              </section>

              {/* Section B: Evidence Explorer Table */}
              <section className="dashboard-card evidence-explorer-card">
                <div className="section-title-bar">
                  <div>
                    <span className="section-eyebrow">B. EMPIRICAL OBSERVATIONS</span>
                    <h3>Evidence Explorer</h3>
                  </div>
                  <div className="evidence-filter-tabs">
                    {['all', 'directly_measured', 'derived_metric', 'hypothesis_interpretation', 'externally_verified', 'unavailable'].map((cat) => (
                      <button
                        key={cat}
                        type="button"
                        className={`evidence-filter-btn ${evidenceFilter === cat ? 'active' : ''}`}
                        onClick={() => setEvidenceFilter(cat)}
                      >
                        {cat.toUpperCase().replace(/_/g, ' ')}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="evidence-table-container">
                  <table className="evidence-table">
                    <thead>
                      <tr>
                        <th>EVIDENCE ID</th>
                        <th>CATEGORY</th>
                        <th>OBSERVATION</th>
                        <th>MEASUREMENT SOURCE & METHOD</th>
                        <th>CONFIDENCE</th>
                        <th>SCIENTIFIC LIMITATIONS</th>
                      </tr>
                    </thead>
                    <tbody>
                      {filteredEvidence.map((item) => {
                        const isSelected = selectedEvidenceItem?.evidence_id === item.evidence_id;
                        return (
                          <tr
                            key={item.evidence_id}
                            className={`evidence-row ${isSelected ? 'selected-row' : ''}`}
                            onClick={() => setSelectedEvidenceItem(item)}
                          >
                            <td className="evidence-id-cell">
                              <code>{item.evidence_id}</code>
                            </td>
                            <td>
                              <span className={`evidence-category-badge ${item.category}`}>
                                {item.category.replace(/_/g, ' ')}
                              </span>
                            </td>
                            <td className="evidence-observation-cell">
                              <strong>{item.observation}</strong>
                            </td>
                            <td className="evidence-source-cell">
                              <div>{item.source}</div>
                              <small className="method-sub">{item.method}</small>
                            </td>
                            <td className="evidence-confidence-cell">
                              {item.confidence_or_uncertainty || 'Standard'}
                            </td>
                            <td className="evidence-limits-cell">
                              <em>{item.limitations}</em>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                {/* Selected Evidence Item Details Box */}
                {selectedEvidenceItem && selectedEvidenceItem.raw_data && (
                  <div className="evidence-raw-viewer">
                    <div className="raw-header">
                      <strong>RAW TELEMETRY FOR {selectedEvidenceItem.evidence_id}: {selectedEvidenceItem.source}</strong>
                      <span>Category: {selectedEvidenceItem.category.toUpperCase().replace(/_/g, ' ')}</span>
                    </div>
                    <pre className="raw-json">
                      {JSON.stringify(selectedEvidenceItem.raw_data, null, 2)}
                    </pre>
                  </div>
                )}
              </section>

              {/* Section F: Detailed Controlled Robustness Table */}
              <section className="dashboard-card variants-table-card">
                <div className="section-title-bar">
                  <div>
                    <span className="section-eyebrow">F. VARIANT SENSITIVITY TESTING</span>
                    <h3>Full Controlled Robustness Experiments</h3>
                  </div>
                </div>

                <div className="variants-table-scroll">
                  <table className="variants-table">
                    <thead>
                      <tr>
                        <th>VARIANT ID</th>
                        <th>VARIATION TYPE</th>
                        <th>EXACT PROMPT</th>
                        <th>TOP TOKEN</th>
                        <th>PROBABILITY (Δp)</th>
                        <th>CONSISTENCY</th>
                      </tr>
                    </thead>
                    <tbody>
                      {auditResult.robustness_variants.map((v: RobustnessVariantResult) => (
                        <tr key={v.variant_id}>
                          <td><code>{v.variant_id}</code></td>
                          <td>
                            <span className="variant-type-pill">{v.variant_type.replace(/_/g, ' ')}</span>
                          </td>
                          <td className="variant-prompt-cell">"{v.prompt}"</td>
                          <td className="variant-token-cell">
                            <strong>"{v.top_token}"</strong>
                          </td>
                          <td className="variant-prob-cell">
                            <span>{(v.top_probability * 100).toFixed(1)}%</span>
                            <small style={{ color: v.probability_shift >= 0 ? '#059669' : '#dc2626' }}>
                              ({v.probability_shift >= 0 ? '+' : ''}{(v.probability_shift * 100).toFixed(1)}%)
                            </small>
                          </td>
                          <td>
                            <span className={`consistency-badge ${v.consistent_with_baseline ? 'consistent' : 'divergent'}`}>
                              {v.consistent_with_baseline ? '✓ Consistent' : '✗ Flipped'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              {/* Section G: Full Exportable Audit Dossier */}
              <section className="dashboard-card export-dossier-card">
                <div className="section-title-bar">
                  <div>
                    <span className="section-eyebrow">G. EXPORT & SYNTHESIS</span>
                    <h3>8-Question Scientific Synthesis & Regulatory Dossier</h3>
                  </div>
                  <div className="export-btn-group">
                    <button type="button" className="export-download-btn json-btn" onClick={handleDownloadJSON}>
                      💾 Download JSON Ledger
                    </button>
                    <button type="button" className="export-download-btn md-btn" onClick={handleDownloadMarkdown}>
                      📄 Download Markdown Dossier
                    </button>
                  </div>
                </div>

                <div className="dossier-summary-box">
                  <div className="synthesis-questions-grid">
                    <div className="q-card">
                      <strong>1. What did original prompt cause model to produce?</strong>
                      <p>{auditResult.synthesis.original_behavior_summary}</p>
                    </div>
                    <div className="q-card">
                      <strong>2. What evidence was collected?</strong>
                      <p>{auditResult.synthesis.evidence_collected_count} empirical records across BPE tokens, logit lens, and attention entropy.</p>
                    </div>
                    <div className="q-card">
                      <strong>3. Which issues were directly observed?</strong>
                      <ul>
                        {auditResult.synthesis.directly_observed_issues.map((iss, i) => (
                          <li key={i}>{iss}</li>
                        ))}
                      </ul>
                    </div>
                    <div className="q-card">
                      <strong>4. Which conclusions are supported by experiments?</strong>
                      <ul>
                        {auditResult.synthesis.empirically_supported_conclusions.map((c, i) => (
                          <li key={i}>{c}</li>
                        ))}
                      </ul>
                    </div>
                    <div className="q-card">
                      <strong>5. Which hypotheses remain unverified?</strong>
                      <ul>
                        {auditResult.synthesis.unverified_hypotheses.map((h, i) => (
                          <li key={i}>{h}</li>
                        ))}
                      </ul>
                    </div>
                    <div className="q-card">
                      <strong>6. What weaknesses were discovered?</strong>
                      <p>{auditResult.synthesis.discovered_weaknesses.join(', ') || 'No critical structural weaknesses found.'}</p>
                    </div>
                    <div className="q-card">
                      <strong>7. How could the prompt be improved?</strong>
                      <ul>
                        {auditResult.synthesis.recommended_improvements.map((r, i) => (
                          <li key={i}>{r}</li>
                        ))}
                      </ul>
                    </div>
                    <div className="q-card">
                      <strong>8. Follow-up experiments:</strong>
                      <ul>
                        {auditResult.synthesis.recommended_followup_experiments.map((f, i) => (
                          <li key={i}>{f}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
