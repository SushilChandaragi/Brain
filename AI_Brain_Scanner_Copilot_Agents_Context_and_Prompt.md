# AI Brain Scanner — GitHub Copilot Agents Context & Execution Prompt

## 1. Project Context

You are working on a final-year Computer Science Senior Development Project (SDP) named:

**AI Brain Scanner: Interactive Mechanistic Interpretability of Gemma 4**

The project is an educational and research-oriented software platform for inspecting and experimentally studying the internal computation of a pretrained Transformer language model.

The primary model is **Gemma 4 E2B** (with the architecture designed so that additional compatible models can be added later).

The system is NOT intended to recreate or train Gemma from scratch, and it must NOT claim to reveal hidden human-like “thoughts.” The project studies observable internal model states and experimentally tests their influence on specific model behaviors.

### Core research idea

The platform must support this loop:

```text
Input
  ↓
Real Gemma inference
  ↓
Record selected internal computation
  ↓
Visualize internal states
  ↓
Identify candidate components
  ↓
Intervene on selected components
  ↓
Run a counterfactual forward pass
  ↓
Compare baseline vs intervention
  ↓
Quantify causal effect
  ↓
Save/replay the experiment
```

The central distinction is:

- **Observation:** something is active or receives attention.
- **Candidate identification:** the component appears related to a behavior.
- **Causal testing:** changing the component changes the behavior.

The system must never equate high attention or high activation with causal importance without intervention evidence.

---

## 2. Primary Goals

Build a working platform that can:

1. Load and run a supported local Gemma model.
2. Record selected intermediate states during inference.
3. Visualize tokens, layers, attention, hidden states/residual states, MLP activations and output logits.
4. Let a user select a model component or internal state for investigation.
5. Perform controlled interventions without permanently modifying model weights for the core experiments.
6. Rerun the same input under the intervention.
7. Compare baseline and intervention outputs quantitatively.
8. Support activation patching between controlled runs.
9. Save experiment metadata and selected tensors for replay and comparison.
10. Provide a clean web UI that makes complex model internals understandable.
11. Make the experiment engine modular enough to support other Transformer models later.
12. Include a **small educational Transformer simulator** that shows the math of a forward pass in a transparent way.

---

## 3. What “Simulation” Means in This Project

Do NOT implement a fake visual-only “AI brain.”

The project has two complementary simulation concepts:

### A. Counterfactual Model Simulation — Core

Use the real pretrained Gemma model and alter selected internal activations during a forward pass.

Examples:

- zero an attention-head output;
- zero an MLP/neuron activation;
- scale an activation;
- replace an activation with one from another controlled run;
- mask selected attention connections where technically and scientifically appropriate.

Then rerun and measure the change.

This is the project's primary simulation concept.

### B. Educational Mini-Transformer Simulator — Supporting

Build a tiny Transformer with deliberately small dimensions, for example:

- 1–2 layers;
- 2 attention heads;
- small vocabulary;
- small embedding size;
- small MLP.

It should allow the user to step through:

```text
Tokens
 → Embeddings
 → Q/K/V
 → Attention scores
 → Softmax
 → Weighted values
 → Residual connection
 → MLP
 → Activation function
 → Output logits
```

This mini-model exists to explain Transformer mechanics and demonstrate interventions at a human-understandable scale. It must be clearly labelled as an educational simulator, not as Gemma itself.

---

## 4. Core Scientific Scope

The initial experimental behavior should be **controlled factual recall**.

Example:

```text
“The capital of France is”
```

The system should capture a baseline prediction and then test selected internal interventions.

This controlled scope is intentional. The project is not attempting to explain every aspect of language generation.

Future behavior sets may include:

- factual recall;
- simple entity relationships;
- controlled contrastive prompts;
- selected multimodal tasks if the chosen Gemma checkpoint and infrastructure support them.

Do not expand scope until the core text pipeline is stable.

---

## 5. Core Interventions

Implement these in this order:

### Phase 1

1. Attention-head ablation.
2. MLP/neuron or intermediate activation ablation.
3. Layer-level ablation/bypass where technically safe.

### Phase 2

4. Activation patching between clean/corrupted or paired prompts.
5. Activation scaling.

### Phase 3 — Optional

6. Attention masking.
7. Feature-level intervention using SAE representations.
8. Automated circuit/component scanning.
9. Steering experiments.

Never implement advanced interventions merely for visual effect. Every intervention must have a stated experimental question and a measurable outcome.

---

## 6. Required Quantitative Outputs

For each experiment, compute as applicable:

- baseline top-k token probabilities;
- intervention top-k token probabilities;
- probability difference;
- target-token logit difference;
- target-token rank change;
- final prediction change;
- recovery after patching, where applicable;
- component/experiment identifier;
- runtime;
- memory usage where useful.

Example experiment summary:

```text
Experiment #0042

Model: Gemma 4 E2B
Prompt: The capital of France is
Target token: Paris

Intervention: Layer 17 / Head 3 ablation

Baseline probability:       0.78
Intervention probability:  0.24
Probability change:        -0.54
Baseline rank:              1
Intervention rank:          5
```

Do not use invented values in demos or tests. Clearly label synthetic examples as mock data.

---

## 7. Internal States to Support

The instrumentation layer should be able to expose, as the model implementation permits:

### Input-level

- raw text;
- token strings;
- token IDs;
- token positions;
- attention masks.

### Representation-level

- embeddings;
- per-layer hidden states;
- residual-stream states where accessible;
- selected intermediate states.

### Attention-level

- attention tensors where supported;
- per-head attention patterns;
- selected head outputs;
- Q/K/V tensors when technically accessible and useful.

### MLP-level

- MLP outputs;
- intermediate activations;
- selected neuron/unit activations;
- activation statistics.

### Output-level

- logits;
- softmax probabilities;
- top-k predictions.

The system should support **selective capture** rather than always retaining every tensor from every layer. Memory-aware capture is a first-class requirement.

---

## 8. Visualization Requirements

The UI should provide multiple abstraction levels.

### Level 1 — Overview

```text
Prompt
 ↓
Tokens
 ↓
Layer progression
 ↓
Prediction
```

### Level 2 — Component inspection

Allow selection of:

- layer;
- attention head;
- MLP;
- selected activation/neuron.

### Level 3 — Deep inspection

Show:

- attention heatmaps;
- token-to-token relationships;
- activation heatmaps;
- layer-wise activation summaries;
- vector statistics;
- logits/probabilities;
- baseline vs intervention differences.

### Level 4 — Causal view

Show a graph or dependency view such as:

```text
Input token
   ↓
Layer 7 / Head 3
   ↓
Residual state
   ↓
Layer 8 / MLP
   ↓
Target logit
```

This graph represents the experimental hypothesis/evidence and must not imply certainty beyond the measured evidence.

---

## 9. Experiment Trace / Recording Format

Every experiment should have a reproducible record.

Recommended structure:

```json
{
  "experiment_id": "...",
  "timestamp": "...",
  "model": "...",
  "model_revision": "...",
  "prompt": "...",
  "tokens": [...],
  "baseline": {
    "top_tokens": [...],
    "logits": [...],
    "probabilities": [...]
  },
  "intervention": {
    "type": "head_ablation",
    "layer": 7,
    "head": 3,
    "parameters": {...}
  },
  "intervention_result": {
    "top_tokens": [...],
    "logits": [...],
    "probabilities": [...]
  },
  "metrics": {
    "probability_delta": ...,
    "logit_delta": ...,
    "rank_delta": ...
  },
  "captured_components": [...]
}
```

Do not store huge raw tensors in the relational database. Store metadata in SQLite/PostgreSQL and selected tensor artifacts separately using a documented format.

---

## 10. Recommended Architecture

Use a clean separation of concerns.

```text
                    WEB FRONTEND
                         │
                         ▼
                      API LAYER
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Model Service   Experiment API  Trace API
          │              │              │
          └──────────────┼──────────────┘
                         ▼
               INTERPRETABILITY CORE
                         │
           ┌─────────────┼─────────────┐
           ▼             ▼             ▼
      Model Adapter   Capture       Analysis
           │          Engine          Engine
           ▼             │             │
         Gemma           ▼             ▼
                    Activation    Candidate ranking
                      Cache
                         │
                         ▼
                  Intervention Engine
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
        Ablate         Patch           Scale
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                  Counterfactual Run
                         │
                         ▼
                    Comparison
                         │
                         ▼
                    Experiment
                       Trace
```

Recommended implementation stack unless the repository already dictates otherwise:

- Python
- PyTorch
- Hugging Face Transformers
- FastAPI
- React + TypeScript
- Plotly/D3 or another suitable visualization library
- SQLite initially
- PyTest
- optionally TransformerLens where compatibility with the selected model and experiment requires it

Do not add libraries without a reason. Prefer a small, maintainable dependency set.

---

## 11. Repository Structure

Use or evolve toward a structure similar to:

```text
ai-brain-scanner/
├── README.md
├── docs/
│   ├── architecture.md
│   ├── experiments.md
│   ├── scientific-scope.md
│   └── model-support.md
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── models/
│   │   ├── instrumentation/
│   │   ├── interventions/
│   │   ├── analysis/
│   │   ├── experiments/
│   │   ├── storage/
│   │   └── schemas/
│   └── tests/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── features/
│   │   ├── visualizations/
│   │   └── api/
│   └── tests/
├── simulator/
│   ├── core/
│   ├── models/
│   ├── experiments/
│   └── tests/
├── scripts/
└── experiments/
```

The exact structure may be adapted if an existing repository already has a better organization. Avoid unnecessary rewrites.

---

## 12. Model Adapter Requirement

Do not couple the entire codebase directly to Gemma internals.

Define a model-adapter abstraction similar to:

```python
class ModelAdapter(Protocol):
    def load(self) -> None: ...
    def tokenize(self, text: str) -> ...: ...
    def run(self, inputs, capture_config=None) -> ...: ...
    def get_supported_components(self) -> ...: ...
    def apply_intervention(self, intervention) -> ...: ...
```

Gemma is the primary implementation.

A second model is optional.

The architecture should make it possible to add another Transformer without rewriting the frontend and experiment engine.

---

## 13. Hardware / Execution Strategy

Primary development environment may be a local laptop with limited VRAM.

Heavier experiments should support execution on a college GPU server.

The software must therefore:

- detect device capabilities;
- support CPU/GPU selection;
- avoid assuming a large VRAM budget;
- capture tensors selectively;
- release tensors promptly when not needed;
- support configurable sequence lengths;
- support quantization/offloading only when compatible with required instrumentation;
- report useful runtime/memory diagnostics.

Do not optimize prematurely. First produce a correct instrumented forward pass on a manageable setup.

---

## 14. Multimodal Support

Multimodal input may be an advanced extension.

Do not make the initial core dependent on image/audio support.

Once text-only instrumentation is reliable, the architecture may be extended to:

```text
Text
Image
Audio
   ↓
Gemma multimodal input
   ↓
internal states
   ↓
comparison/intervention
```

Potential experiments include:

- text-only vs image+text;
- clean image vs corrupted image;
- modality ablation;
- comparing internal representations across modalities.

Do not claim full multimodal mechanistic dissection unless the implementation actually supports it and the relevant internal pathways can be observed rigorously.

---

## 15. Scientific Guardrails

This project is an interpretability research/engineering project. Every feature must respect the following:

1. **No unsupported causal claims.**
2. Attention is evidence for a hypothesis, not proof of causation.
3. High activation is not automatically feature importance.
4. Intervention effects may be distributed and nonlinear.
5. Results must be described as behavior-specific/experiment-specific unless replicated more broadly.
6. Use repeated controlled prompts for evaluation.
7. Record model revision/configuration for reproducibility.
8. Distinguish real model output from mock/demo data.
9. Do not describe the system as revealing literal human-like reasoning or consciousness.
10. Do not permanently modify pretrained weights unless a dedicated experiment explicitly requires it.

---

## 16. Development Phases

### Phase 0 — Repository understanding

Before changing code:

- inspect the repository;
- identify current stack;
- identify existing files and TODOs;
- read README and architecture documents;
- run the existing project;
- document what works and what does not.

Do not rewrite working infrastructure without a reason.

### Phase 1 — Model bootstrap

Goal:

```text
load Gemma → tokenize → inference → output
```

Tasks:

- environment setup;
- model-loading abstraction;
- baseline inference endpoint/script;
- top-k probability extraction;
- tests.

Acceptance criterion:

A reproducible prompt produces a valid baseline result.

### Phase 2 — Instrumentation

Goal:

```text
model → selected internal states → trace
```

Tasks:

- hook/capture layer;
- configurable capture policy;
- hidden-state recording;
- attention recording where supported;
- MLP/intermediate recording;
- trace schema;
- memory checks.

Acceptance criterion:

A single experiment can be replayed from its stored trace metadata and selected artifacts.

### Phase 3 — Visualization

Goal:

Make the trace understandable.

Implement:

- token viewer;
- layer explorer;
- attention heatmap;
- activation heatmap/plot;
- logits/probability visualization;
- before/after comparison.

### Phase 4 — Causal interventions

Implement:

- head ablation;
- neuron/MLP ablation;
- layer intervention;
- activation patching;
- activation scaling.

Every intervention must be unit tested with deterministic/small test cases where possible.

### Phase 5 — Experiment engine

Implement:

- experiment configuration;
- baseline run;
- intervention run;
- metrics;
- persistence;
- replay;
- comparison;
- export.

### Phase 6 — Candidate identification

Implement a transparent ranking system using signals such as:

- activation magnitude;
- attention statistics;
- causal effect size.

Label these separately. Do not collapse them into one “importance” score without explaining the methodology.

### Phase 7 — Mini-Transformer simulator

Implement the educational model and step-through interface.

Include:

- Q/K/V visualization;
- attention calculation;
- residual connections;
- MLP;
- activation function visualization;
- controlled perturbations.

### Phase 8 — Advanced research features

Optional and only after the core is stable:

- automated circuit search;
- SAE feature visualization;
- prompt-specific circuit comparison;
- multimodal experiments;
- model-to-model comparison.

---

## 17. Testing Strategy

Every major subsystem needs tests.

### Unit tests

Test:

- token handling;
- intervention configuration;
- tensor shapes;
- metric calculations;
- experiment serialization;
- patching index alignment;
- model adapter behavior.

### Integration tests

Test:

```text
prompt → model → capture → intervention → rerun → metrics → stored trace
```

### Scientific sanity checks

Where appropriate:

- intervention set to “none” should match baseline within expected numerical tolerance;
- repeated deterministic runs should be consistent under controlled settings;
- zeroing a deliberately chosen synthetic component in the mini-model should produce an expected effect;
- activation patching must preserve correct tensor shapes and positions.

### Frontend tests

Test:

- experiment creation;
- component selection;
- visualization state;
- loading/error states;
- comparison view;
- experiment replay.

---

## 18. Agent Working Rules

GitHub Copilot agents must follow these rules:

### Rule 1 — Inspect before modifying

Never assume the existing architecture. Read the relevant code first.

### Rule 2 — Work in small milestones

Implement one coherent milestone at a time.

### Rule 3 — Preserve scientific validity

If a requested feature would encourage unsupported causal claims, implement the underlying measurement but change the wording/UI to remain scientifically accurate.

### Rule 4 — Prefer real experiments

Do not build mock visualizations where actual model data is available.

### Rule 5 — No fake results

Never hard-code impressive probabilities, causal scores or graphs and present them as model results.

### Rule 6 — Document assumptions

For model internals that depend on Gemma's implementation, document the exact module path, model revision and assumptions.

### Rule 7 — Fail clearly

If a component cannot be instrumented with the current model/tooling, return a clear capability error rather than silently producing misleading output.

### Rule 8 — Keep performance visible

Track runtime, memory and tensor sizes for expensive operations.

### Rule 9 — Keep the backend and frontend decoupled

The frontend should consume stable API schemas rather than importing model internals.

### Rule 10 — Update documentation with architectural changes

Whenever a core behavior changes, update the relevant docs and tests.

---

# 19. Master Prompt for GitHub Copilot Coding Agent

Copy the following prompt into the Copilot coding-agent task.

```text
You are the lead software engineer working on my final-year SDP:

AI Brain Scanner: Interactive Mechanistic Interpretability of Gemma 4

PROJECT MISSION
Build an interactive software platform that treats a pretrained Transformer as an experimentally observable computational system. The platform must let a user run a controlled prompt through Gemma, record selected internal computation, inspect those states visually, intervene on selected components, rerun the same model under the intervention, and quantitatively compare the resulting behavior.

The project's central loop is:

INPUT
→ REAL GEMMA INFERENCE
→ RECORD SELECTED INTERNAL STATES
→ VISUALIZE
→ IDENTIFY CANDIDATES
→ INTERVENE
→ COUNTERFACTUAL RERUN
→ MEASURE EFFECT
→ SAVE / REPLAY / COMPARE

PRIMARY MODEL
Gemma 4 E2B is the first-class model target. Do not hard-code the entire application around a single module path if a model-adapter abstraction can avoid it.

The project does NOT:
- train Gemma from scratch;
- claim to reveal literal human-like thoughts;
- equate attention with causality;
- permanently alter model weights for the core experiments;
- fabricate experimental results.

CORE EXPERIMENTS
1. Attention-head ablation.
2. MLP/neuron or intermediate activation ablation.
3. Layer-level intervention where safe.
4. Activation patching.
5. Activation scaling.

CORE OBSERVABLES
- tokens and token IDs;
- embeddings where available;
- per-layer hidden states;
- attention tensors/patterns where supported;
- attention-head outputs;
- residual/intermediate activations where accessible;
- MLP outputs/intermediate activations;
- logits;
- probabilities;
- top-k predictions.

CORE METRICS
- probability change;
- logit change;
- rank change;
- prediction change;
- patching recovery where applicable;
- runtime and memory diagnostics.

SCIENTIFIC PRINCIPLE
Use attention/activation information to form hypotheses. Use interventions to test those hypotheses. Report the results as experiment-specific causal evidence rather than universal claims.

EDUCATIONAL SIMULATOR
In addition to the real-model instrumentation path, build a tiny Transformer simulator. It must expose step-by-step computation including tokenization, embeddings, Q/K/V, attention scores, softmax, weighted values, residual connections, MLP, activation functions and logits. This simulator is for explanation and controlled teaching; clearly distinguish it from real Gemma.

FIRST TASK
Before writing code:
1. Inspect the entire repository structure.
2. Read the README and existing architecture documentation.
3. Identify the current stack and execution commands.
4. Find existing model-loading, API, frontend and testing code.
5. Run the existing test suite/build if possible.
6. Produce a concise implementation plan tailored to the current repository.
7. Do not rewrite existing working code without justification.

IMPLEMENTATION ORDER
Phase 1: model adapter + baseline Gemma inference.
Phase 2: instrumentation/capture layer.
Phase 3: experiment trace storage.
Phase 4: visualizations.
Phase 5: ablation interventions.
Phase 6: activation patching.
Phase 7: quantitative comparison/replay.
Phase 8: mini-Transformer simulator.
Phase 9: automated component scanning.
Phase 10: optional advanced extensions.

ENGINEERING REQUIREMENTS
- Python/PyTorch/Hugging Face for model execution unless the repository uses another justified stack.
- FastAPI backend unless an existing API framework should be retained.
- React + TypeScript frontend unless the existing frontend uses another justified stack.
- SQLite initially for experiment metadata unless the repository requires another database.
- PyTest and appropriate frontend tests.
- Device-aware execution and memory-conscious tensor capture.
- Selective activation capture; never capture every tensor by default.
- Stable typed API schemas.
- Clear error handling.
- Reproducible experiment IDs and model/config metadata.

PERFORMANCE REQUIREMENTS
The development laptop may have limited VRAM. The application must therefore support selective capture and avoid unnecessary copies of large tensors. The college GPU server may be used for heavier runs. Do not build the architecture around a single hardware configuration.

UI REQUIREMENTS
Provide these major views:
1. Model / prompt setup.
2. Prediction overview.
3. Layer explorer.
4. Attention explorer.
5. Activation explorer.
6. Intervention configuration.
7. Before-vs-after experiment comparison.
8. Experiment trace/replay.
9. Causal/component graph.
10. Educational Transformer simulator.

For expensive or unavailable visualizations, show capability/status information instead of fake data.

DATA REQUIREMENTS
Each experiment must preserve:
- model identifier/revision;
- prompt/input;
- tokenization information;
- baseline output;
- intervention configuration;
- intervention output;
- quantitative metrics;
- selected captured components;
- execution metadata.

Do not store huge tensors directly inside the relational database.

TEST REQUIREMENTS
For every core feature, add unit/integration tests. In particular verify:
- no-intervention result matches baseline within numerical tolerance;
- intervention changes the intended tensor/module only;
- patching aligns token positions and shapes correctly;
- metric calculations are correct;
- experiment traces can be saved and reloaded.

SCIENTIFIC VALIDATION
Use controlled factual-recall tasks initially. For example:
“The capital of France is”

Build experiments where the expected target token is explicit and measurable. Evaluate across multiple prompts rather than relying on one example.

NOVELTY POSITIONING
Do not claim that the project invented mechanistic interpretability, ablation, activation patching, circuit discovery or sparse autoencoders. The contribution is an integrated, interactive software environment that combines observation, internal-state recording, causal intervention, counterfactual experimentation, quantitative comparison and accessible visualization.

WHEN IMPLEMENTING
- First explain what files you will modify and why.
- Then implement the smallest coherent change.
- Run tests.
- Fix failures.
- Update documentation.
- Report what was completed, what remains, and any model-specific limitations.

Do not move to the next major phase until the current phase has a working testable baseline.
```

---

# 20. Suggested Copilot Agent Tasks

Use these as separate agent issues rather than asking one agent to build everything at once.

## Agent 01 — Repository Audit

```text
Inspect the repository and document the current architecture, stack, run commands, test commands, and existing implementation. Do not make large code changes. Produce docs/implementation-audit.md and recommend the smallest plan for integrating AI Brain Scanner.
```

## Agent 02 — Gemma Model Adapter

```text
Implement a model adapter for Gemma 4 E2B. Add configuration, device detection, tokenizer support, baseline inference and top-k probability extraction. Add unit/integration tests and document model loading requirements. Do not implement interventions yet.
```

## Agent 03 — Instrumentation Engine

```text
Implement an instrumentation layer that can selectively capture supported Gemma internal states during forward execution. Support configurable layer/component selection and memory-conscious capture. Add tensor-shape validation and tests.
```

## Agent 04 — Experiment Trace

```text
Implement experiment metadata, serialization, IDs, baseline/intervention result schemas and selected tensor artifact storage. Keep large tensors out of the relational DB. Add replay loading and tests.
```

## Agent 05 — Attention/Activation UI

```text
Build the first frontend exploration interface for tokens, layers, attention patterns, activation summaries and logits. Use real backend data. Do not use invented values except explicitly labelled UI placeholders.
```

## Agent 06 — Head Ablation

```text
Implement attention-head ablation as a controlled intervention. The original weights must remain unchanged. Add baseline-vs-ablation metrics, experiment trace recording and tests verifying the intervention target.
```

## Agent 07 — Neuron/MLP Ablation

```text
Implement selected MLP/intermediate activation ablation. Make component indexing explicit, validate tensor shapes and record the intervention. Add quantitative comparison and tests.
```

## Agent 08 — Activation Patching

```text
Implement activation patching between two controlled runs with explicit source/target layer, component and token-position selection. Add strong validation for shape and position compatibility, record the experiment, and provide before/after metrics.
```

## Agent 09 — Experiment Comparison / Replay

```text
Build the experiment replay and comparison UI. Given two saved experiments, display differences in predictions, logits, probabilities and selected activations. Make the causal/intervention status explicit.
```

## Agent 10 — Mini Transformer Simulator

```text
Build a small educational Transformer simulator with step-by-step execution. Expose tokenization, embeddings, Q/K/V, attention scores, softmax, residuals, MLP and activation functions. Add safe perturbations such as zeroing/scaling and make clear that it is a teaching model, not Gemma.
```

## Agent 11 — Candidate Component Scanner

```text
Implement a transparent candidate-ranking pipeline that scans selected components using observational and causal signals. Keep activation/attention evidence separate from causal intervention evidence. Do not label a component causally important unless intervention data supports that claim.
```

## Agent 12 — Validation and Documentation

```text
Run the entire test suite and audit the project against the scientific guardrails in this document. Identify unsupported claims, fake/demo values, memory risks, reproducibility issues and missing documentation. Fix or report each issue.
```

---

# 21. Definition of Done

The project reaches a strong MVP when a user can:

1. Select Gemma 4 E2B.
2. Enter a controlled prompt.
3. Run baseline inference.
4. See tokens and prediction probabilities.
5. Explore selected layers and internal activations.
6. Inspect attention patterns where supported.
7. Select a component.
8. Run an ablation or activation patching experiment.
9. Rerun the same model/input under intervention.
10. See before-vs-after probabilities/logits/ranks.
11. Save the experiment.
12. Reload and replay it.
13. View a causal/evidence-oriented component graph.
14. Run the educational mini-Transformer simulator.
15. Export an experiment report.

A strong final release additionally supports:

- automated component scans;
- prompt-specific experiments;
- selected SAE feature analysis;
- multimodal experiments where supported;
- model-adapter extensibility;
- GPU-server execution profiles.

---

# 22. Final Instruction to All Agents

Build this as a **research-oriented engineering platform**, not a flashy demo.

Every visualization should answer a question.
Every intervention should test a hypothesis.
Every experiment should produce measurable evidence.
Every causal claim should be tied to an actual intervention.
Every large tensor should have a memory strategy.
Every major feature should be reproducible.

The project's core promise is:

> **Observe → Record → Intervene → Rerun → Measure → Understand**

Keep the implementation faithful to that promise.
