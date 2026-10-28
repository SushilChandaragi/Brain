# Quickstart Guide: NeuroAudit

Welcome to **NeuroAudit** — the interactive mechanistic interpretability and AI safety platform. This guide provides a rapid 3-minute setup to get both the FastAPI backend and React frontend running locally.

---

## ⚡ 3-Minute Quickstart

### Prerequisites
- **Python:** 3.11 or higher
- **Node.js:** 18 or higher (with `npm`)
- **System Memory:** Minimum 4 GB RAM (runs on standard CPU; CUDA GPU optional)

---

### Step 1: Start the Backend (FastAPI + PyTorch)

Open a terminal at the repository root:

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (if not already created)
python -m venv venv

# Activate virtual environment:
# Windows (PowerShell):
venv\Scripts\activate
# Windows (Command Prompt):
venv\Scripts\activate.bat
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend server
# Windows (PowerShell):
$env:PYTHONUTF8 = "1"; python -m app.main
# macOS / Linux:
python -m app.main
```

> **Verification:**
> Open `http://localhost:8000/health` in your browser or curl. You should receive:
> `{"status":"healthy","model_loaded":true,"device":"cpu","message":"Model is ready"}`

---

### Step 2: Start the Frontend (React + Vite + Three.js)

Open a **separate terminal** at the repository root:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

> **Access the Web Dashboard:**
> Open `http://localhost:5173` in your browser.

---

## 🔍 Guided Forensic Audit Walkthrough

Once the application loads in your browser:

1. **Step 1: Target Specification & Benchmark Suites**
   - Click the **"Factual Recall Circuit"** preset card (`"The capital of France is"` $\rightarrow$ `"Paris"`).
   - Click **"▶ Run Diagnostic Scan"** to perform the baseline forward pass and hook capture.
2. **Step 2: 3D Synaptic Flow Deep Probe**
   - Click **"Proceed to Step 2: 3D Synaptic Flow"**.
   - Interact with the 3D WebGL Transformer canvas: left-click to orbit, right-click to pan, scroll to zoom.
   - Click any Attention or MLP node to view live HUD activation telemetry.
   - Inspect the **Logit Lens Projection** panel to observe how token predictions crystallize across layers.
3. **Step 3: Causal Circuit Localization**
   - Click **"Proceed to Step 3: Causal Circuit Localization"**.
   - Review the **Causal Head Sweep** to see which attention heads have the strongest influence on the target prediction.
   - Explore the **Multi-Head Attention Matrix** and **Activation Patching** tabs.
4. **Step 4: Surgical Verification & Compliance Audit Dossier**
   - Click **"Proceed to Step 4: Executive Audit Dossier"**.
   - Select a target layer and head to ablate, then click **"Apply Surgical Ablation"**.
   - Observe the causal impact: $\Delta\text{logit}$, $\Delta\text{probability}$, and rank shift.
   - Click **"📄 View Audit Certificate"** to view and print the official verification certificate.

---

## 🔬 Academic Research Observatory Mode

Toggle the switcher in the top navigation bar to access the **Research Observatory**:
- **Mini-Transformer Simulator:** Walk through step-by-step tensor mathematics (Q/K/V projections, softmax attention, residual addition, output logits).
- **Experiment Registry:** Search, filter, inspect, and export historical audit runs stored in SQLite (`backend/experiments.db`).

---

## 🧪 Running Automated Tests

Run backend tests from the `backend/` directory:
```bash
cd backend
python -m pytest tests/ -v
```

Validate frontend production build from `frontend/`:
```bash
cd frontend
npm run build
```

---

## 📚 Complete Documentation

For full architectural blueprints, API documentation, and configuration options, see:
- [README.md](README.md) — Master project documentation
- [docs/architecture.md](docs/architecture.md) — Detailed system design & tensor pipeline
- Interactive API Swagger UI: `http://localhost:8000/docs`
