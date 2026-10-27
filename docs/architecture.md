# Architecture Overview

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React + TypeScript)             │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │   Setup UI   │  │  Inference   │  │ Experiment   │           │
│  │              │  │  Dashboard   │  │ Replay       │           │
│  └──────────────┘  └──────────────┘  └──────────────┘           │
│                                                                   │
│  └─────────────────────── API Client ──────────────────────────┘
│
└─────────────────────────────────────────────────────────────────┘
                              ▲
                        HTTP / REST
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI + Python)                  │
│                                                                   │
│  ┌──────────────┐        ┌──────────────┐                       │
│  │ API Routes   │  ────▶ │  Model       │                       │
│  │              │        │  Manager     │                       │
│  └──────────────┘        └──────────────┘                       │
│         │                       │                                │
│         ▼                       ▼                                │
│  ┌──────────────┐        ┌──────────────┐                       │
│  │ Schemas      │        │ Model        │                       │
│  │ (Pydantic)   │        │ Adapters     │                       │
│  └──────────────┘        └──────────────┘                       │
│                                 │                                │
│                                 ▼                                │
│                         ┌──────────────┐                       │
│                         │ Gemma Model  │                       │
│                         │ (HuggingFace)│                       │
│                         └──────────────┘                       │
│                                                                   │
│  Future Components:                                              │
│  - Instrumentation layer (Phase 2)                               │
│  - Intervention system (Phase 4)                                 │
│  - Experiment engine (Phase 5)                                   │
│  - Storage/Database (Phase 3)                                    │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                      PyTorch / GPU
                              │
                  ┌───────────────────────┐
                  │  Gemma 4 E2B (Local)  │
                  │  CUDA/CPU Device      │
                  └───────────────────────┘
```

## Component Breakdown

### Frontend (`frontend/`)

**Key Files:**
- `src/App.tsx` - Main application component
- `src/api/client.ts` - API client for backend communication
- `src/App.css` - Styling
- `vite.config.ts` - Build configuration

**Current Capabilities (Phase 1):**
- Health check and model status display
- Text input for prompts
- Baseline inference execution
- Token visualization
- Top-K prediction display

### Backend (`backend/`)

**Key Directories:**
- `app/` - Main application code
  - `main.py` - FastAPI app and routes
  - `config.py` - Configuration management
  - `models/` - Model adapter abstraction
    - `adapter.py` - Protocol definition
    - `gemma.py` - Gemma implementation
  - `schemas/` - Pydantic models for validation
  - `instrumentation/` - Capture layer (Phase 2)
  - `interventions/` - Ablation logic (Phase 4)
  - `experiments/` - Experiment engine (Phase 5)
  - `storage/` - Database layer (Phase 3)
- `tests/` - PyTest test suite

**Current Capabilities (Phase 1):**
- Model loading from HuggingFace
- Tokenization
- Forward pass inference
- Probability/logit computation
- Top-K token ranking

## API Endpoints

### Health & Status

```
GET /health
```
Check backend health and model loading status.

```
GET /model/info
```
Get information about the loaded model (name, device, vocab size, parameters).

### Inference

```
POST /infer
Content-Type: application/json

{
  "prompt": "The capital of France is",
  "max_new_tokens": 20,
  "temperature": 1.0
}
```
Run inference on a prompt and return baseline predictions.

```
POST /test-inference
```
Quick test endpoint for debugging.

## Data Flow

### Baseline Inference (Phase 1)

```
1. User enters prompt
   ↓
2. Frontend sends POST /infer
   ↓
3. Backend tokenizes prompt
   ↓
4. Gemma forward pass runs
   ↓
5. Logits → Probabilities computed
   ↓
6. Top-K tokens selected
   ↓
7. Response serialized to JSON
   ↓
8. Frontend displays results
```

## Technology Stack

### Backend
- **Framework:** FastAPI
- **Model:** PyTorch + Transformers (HuggingFace)
- **Async:** Uvicorn
- **Validation:** Pydantic
- **Testing:** PyTest
- **Model:** Gemma 1.1 2B IT

### Frontend
- **Framework:** React 18
- **Language:** TypeScript
- **Build:** Vite
- **API Client:** Axios
- **Visualization:** Plotly.js (ready for Phase 3)

## Device Support

- **Primary:** CUDA GPU (float16 for efficiency)
- **Fallback:** CPU (float32)
- **Auto-detection:** Checks torch.cuda.is_available()

## Configuration

Environment variables (`.env` or defaults in `backend/app/config.py`):

```
MODEL_NAME=google/gemma-1.1-2b-it
MODEL_DEVICE=cuda  # or "cpu" or "auto"
MAX_SEQUENCE_LENGTH=512
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True
```

## Next Phases

**Phase 2 - Instrumentation:**
- Add hooks to capture intermediate layers
- Store hidden states, attention tensors, MLP activations
- Build trace recording infrastructure

**Phase 3 - Persistence:**
- SQLite database setup
- Experiment metadata storage
- Trace artifact serialization

**Phase 4 - Interventions:**
- Attention head ablation
- Neuron/MLP ablation
- Activation patching hooks

**Phase 5 - Experiment Engine:**
- Baseline + intervention runs
- Metric computation
- Comparison and replay

**Phase 6 - Advanced Visualizations:**
- Attention heatmaps
- Activation distributions
- Causal graphs

## Scientific Principles

1. **No Unsupported Claims:** Attention ≠ causation
2. **Interventions Test Hypotheses:** Evidence comes from controlled experiments
3. **Reproducible Experiments:** Full metadata for replay
4. **Selective Capture:** Memory-conscious, not everything by default
5. **Baseline Comparisons:** Always compare intervention to no-intervention

## Development Workflow

```bash
# Terminal 1: Backend
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m app.main

# Terminal 2: Frontend
cd frontend
npm install
npm run dev

# Terminal 3: Testing
cd backend
pytest tests/ -v
```

Access frontend at `http://localhost:5173`
Access API docs at `http://localhost:8000/docs`
