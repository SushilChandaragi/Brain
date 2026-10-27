# AI Brain Scanner: Interactive Mechanistic Interpretability

An interactive platform for inspecting, visualizing, and experimentally testing the internal computations of Transformer language models through causal interventions.

---

## 🎯 What This Project Does

Instead of treating Large Language Models as opaque black boxes or confusing attention weights with causal importance, **AI Brain Scanner** enables **causal interpretability**:
1. **Traces internal states** during forward inference (attention matrices, residual stream hidden states, MLP activations).
2. **Visualizes model internals** interactively via heatmaps, activation bars, and token probability distributions.
3. **Applies controlled interventions** (e.g. attention head ablation) without permanently modifying model weights.
4. **Measures causal effects** quantitatively by comparing baseline vs. intervened predictions (logit deltas, probability drops, rank changes).

---

## 🚀 Current Project Status (What Is Achieved Right Now)

| Component | Status | Details |
|---|---|---|
| **Phase 1: Model Loading & Inference** | ✅ **Done** | Causal LM adapter supporting `TinyLlama/TinyLlama-1.1B-Chat-v1.0` and `google/gemma-1.1-2b-it` on CPU/CUDA. |
| **Phase 2: Instrumentation & Tracing** | ✅ **Done** | PyTorch forward hooks record exact attention weights, hidden states, and MLP outputs per layer. |
| **Phase 3: Interactive Web Dashboard** | ✅ **Done** | React + TypeScript UI with real-time prompt scanning, presets, attention heatmaps, and probability fields. |
| **Phase 4: Causal Head Ablation** | ✅ **Done** | Pre-hook interventions temporarily zero out specific attention heads to measure direct causal impact on predictions. |
| **Phase 5: Experiment Engine & Storage** | 🔄 *Next* | SQLite persistence for saving, tagging, and comparing multi-run experiment benchmarks. |
| **Phase 6: Educational Mini-Transformer** | 🔄 *Planned* | 1–2 layer visual transformer showing step-by-step mathematical tensor transformations. |

---

## 📋 Prerequisites

- **Python:** 3.11 or higher
- **Node.js:** 18 or higher (with `npm`)
- **Memory/Hardware:** 
  - Standard CPU with ~4 GB free RAM (runs the default 1.1B model smoothly).
  - Optional: NVIDIA GPU with CUDA for faster generation.

---

## ⚡ Quick Setup Guide (Step-by-Step)

### 1. Clone the Repository
```bash
git clone https://github.com/SushilChandaragi/Brain.git
cd Brain
```

---

### 2. Backend Setup (FastAPI + PyTorch)

Open a terminal and run:

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment:
# Windows (PowerShell):
venv\Scripts\activate
# Windows (Command Prompt):
venv\Scripts\activate.bat
# macOS / Linux:
source venv/bin/activate

# Copy example environment configuration
# Windows:
copy .env.example .env
# macOS / Linux:
cp .env.example .env

# Install backend dependencies
pip install -r requirements.txt

# Start backend server
# Windows (PowerShell with UTF-8):
$env:PYTHONUTF8 = "1"; python -m app.main
# macOS / Linux:
python -m app.main
```

> **Backend URLs:**
> - API Server: `http://localhost:8000`
> - Interactive Swagger Docs: `http://localhost:8000/docs`
> - Health Check: `http://localhost:8000/health`

---

### 3. Frontend Setup (React + Vite + TypeScript)

Open a **separate terminal** and run:

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start Vite dev server
npm run dev
```

> **Frontend URL:** Open `http://localhost:5173` in your browser.

---

## ⚙️ Environment Variables (`.env`)

Configuration lives in `backend/.env` (created from `backend/.env.example`):

| Variable | Default Value | Description |
|---|---|---|
| `MODEL_NAME` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Hugging Face model identifier. Default is public and requires no authentication. |
| `MODEL_DEVICE` | `cpu` | Execution hardware: `cpu`, `cuda`, or `auto`. |
| `MAX_SEQUENCE_LENGTH` | `512` | Maximum sequence context length. |
| `API_HOST` | `0.0.0.0` | Host IP for FastAPI. |
| `API_PORT` | `8000` | Port for FastAPI. |
| `API_RELOAD` | `false` | Hot-reloading (`true` during active development, `false` for stable memory usage). |
| `DATABASE_URL` | `sqlite:///./experiments.db` | Local database path for storing experiment records. |
| `HF_TOKEN` | *(empty)* | Optional Hugging Face token. **Only needed if switching to gated models like Google Gemma.** |
| `CAPTURE_ATTENTION` | `true` | Capture self-attention matrices during forward passes. |
| `CAPTURE_HIDDEN_STATES` | `true` | Capture residual stream representations. |
| `CAPTURE_MLP_ACTIVATIONS`| `true` | Capture feed-forward intermediate layer activations. |

---

## 🧪 Testing

To run the backend test suite:

```bash
cd backend
pytest
```

To validate the frontend build:

```bash
cd frontend
npm run build
```

---

## 📁 Repository Structure

```
Brain/
├── .env.example                         # Root template for environment variables
├── .gitignore                           # Excludes venvs, node_modules, and weight checkpoints
├── README.md                            # Project overview and setup instructions
├── GETTING_STARTED.md                   # Detailed walkthrough guide
├── backend/
│   ├── .env.example                     # Backend env template
│   ├── requirements.txt                 # Python dependencies
│   ├── app/
│   │   ├── main.py                      # FastAPI application & API endpoints
│   │   ├── config.py                    # Settings loader from environment
│   │   ├── instrumentation.py           # PyTorch hooks (TraceRecorder, HeadAblation)
│   │   ├── models/
│   │   │   ├── adapter.py               # Abstract base adapter protocol
│   │   │   └── gemma.py                 # Transformer causal LM adapter
│   │   └── schemas/
│   │       └── inference.py             # Pydantic request/response schemas
│   └── tests/
│       ├── test_model_adapter.py        # Model loading & inference tests
│       └── test_phase2_instrumentation.py# Hook & ablation tests
├── frontend/
│   ├── package.json                     # Node dependencies and scripts
│   ├── vite.config.ts                   # Vite configuration with proxy to :8000
│   └── src/
│       ├── App.tsx                      # Main interactive UI dashboard
│       ├── App.css                      # UI styles, theme, and animations
│       └── api/
│           └── client.ts                # TypeScript Axios API client
└── docs/
    └── architecture.md                  # Detailed system architecture notes
```

---

## 💡 Tips & Common Gotchas for Teammates

1. **Why `TinyLlama` instead of `Gemma` by default?**
   Google's Gemma models require signing an agreement on Hugging Face and configuring an `HF_TOKEN`. TinyLlama runs immediately on any teammate's machine on CPU without login barriers. To test Gemma, add your `HF_TOKEN` to `.env` and set `MODEL_NAME=google/gemma-1.1-2b-it`.
2. **Temperature & Deterministic Output:**
   - **Temperature = 0.0 (Greedy):** The model deterministically selects the top-ranked candidate (e.g. *"The capital of Karnataka is"* $\rightarrow$ *"Bangalore"*).
   - **Temperature > 0.0 (Sampling):** The model samples probabilistically; generated text may choose lower-ranked tokens.
3. **Windows PowerShell UTF-8:**
   If running on Windows PowerShell, ensure `$env:PYTHONUTF8 = "1"` is set before starting Python to avoid encoding issues with special characters.
