# Getting Started Guide - AI Brain Scanner

## Quick Setup (5-10 minutes)

### Prerequisites
- Python 3.11+ installed
- Node.js 18+ installed
- ~8GB+ VRAM (for Gemma 4 E2B)

### Step 1: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies (this downloads Gemma on first import)
pip install -r requirements.txt

# Start backend server
python -m app.main
```

**Expected output:**
```
🚀 Loading model: google/gemma-1.1-2b-it
...downloading model...
✅ Model loaded successfully. Device: cuda
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The backend is now running on `http://localhost:8000`

**Test it:**
Open another terminal and run:
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy","model_loaded":true,...}
```

### Step 2: Frontend Setup

Open a **new terminal** and:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

**Expected output:**
```
  VITE v5.0.8  ready in 123 ms

  ➜  Local:   http://localhost:5173/
```

Open `http://localhost:5173` in your browser.

## What You Should See

### Frontend (React App)

1. **Status Panel** - Shows ✅ Model Loaded with device info
2. **Inference Panel** - Text area with default prompt: "The capital of France is"
3. **Run Inference Button** - Click to generate completions
4. **Results** - Shows generated tokens and top-K predictions

### Example Flow

1. Keep the default prompt or change it
2. Click "▶ Run Inference"
3. Wait ~2-5 seconds (depending on your GPU)
4. See:
   - Input tokens tokenized
   - Generated text (next token)
   - Top 5 predictions with probabilities

**Example output for "The capital of France is":**
```
Rank #1: Paris     89.34%  (logit: 4.231)
Rank #2: France    6.21%   (logit: 1.892)
Rank #3: the       2.10%   (logit: -0.145)
Rank #4: French    1.45%   (logit: -0.891)
Rank #5: home      0.90%   (logit: -1.234)
```

## Troubleshooting

### Issue: "Connection refused" or "Cannot connect to backend"

**Solution:**
1. Check backend is running: `curl http://localhost:8000/health`
2. Make sure it printed "✅ Model loaded successfully"
3. Check port 8000 is not in use: `lsof -i :8000` (macOS/Linux) or `netstat -ano | findstr :8000` (Windows)

### Issue: "CUDA out of memory"

**Solution:**
1. Reduce `MAX_SEQUENCE_LENGTH` in `backend/app/config.py` (default 512)
2. Or use CPU: Set `MODEL_DEVICE=cpu` (slower but always works)
3. Or restart and clear cache: Delete `.cache/huggingface/`

### Issue: "Model not found" or download fails

**Solution:**
1. First run requires downloading Gemma (2-3GB)
2. Ensure internet connection is active
3. Check you have ~5GB free disk space
4. If stuck, manually download:
   ```bash
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('google/gemma-1.1-2b-it')"
   ```

### Issue: Inference very slow

**Solution:**
1. Check device: Open `/model/info` API (see Architecture docs)
2. If device is "cpu", consider switching to GPU
3. Reduce sequence length in config

### Issue: Node/npm errors

**Solution:**
```bash
# Clear cache and reinstall
rm -rf node_modules package-lock.json
npm install
npm run dev
```

## Project Structure at This Point (Phase 1)

```
ai-brain-scanner/
├── README.md
├── .gitignore
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              ✅ Inference routes
│   │   ├── config.py            ✅ Settings
│   │   ├── models/
│   │   │   ├── adapter.py       ✅ Protocol
│   │   │   ├── gemma.py         ✅ Gemma implementation
│   │   │   └── __init__.py
│   │   └── schemas/
│   │       ├── inference.py     ✅ Pydantic models
│   │       └── __init__.py
│   ├── tests/
│   │   └── test_model_adapter.py ✅ Basic tests
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx              ✅ Main UI
│   │   ├── App.css
│   │   ├── index.css
│   │   └── api/
│   │       └── client.ts        ✅ API client
│   ├── vite.config.ts
│   ├── tsconfig.json
│   └── package.json
└── docs/
    ├── architecture.md          ✅ This document
    ├── experiments.md           (coming Phase 4)
    └── scientific-scope.md      (reference docs)
```

## Next Steps (Phase 2)

After confirming Phase 1 works:

1. **Instrumentation Layer** - Capture internal states
   - Hook into Gemma layers
   - Record hidden states, attention, MLP activations
   - Add to traces

2. **Trace Storage** - Save experiment data
   - SQLite setup
   - Experiment metadata schema
   - Selective tensor saving

3. **Visualizations** - Interactive dashboards
   - Layer explorer
   - Attention heatmaps
   - Activation distributions

4. **Interventions** - Ablation experiments
   - Attention head ablation
   - Neuron ablation
   - Compare baseline vs intervention

## API Reference for Phase 1

See `/docs` Swagger UI at `http://localhost:8000/docs` while backend is running.

### Available Endpoints

```
GET  /health              → Check system status
GET  /model/info          → Model metadata
POST /infer               → Run inference
POST /test-inference      → Quick test
```

## Development Commands

```bash
# Backend
cd backend
python -m app.main                 # Run app
pytest tests/ -v                   # Run tests
python -m pytest tests/ --cov      # With coverage

# Frontend
cd frontend
npm run dev                        # Dev server
npm run build                      # Production build
npm run preview                    # Preview build
npm run lint                       # ESLint check

# Full project
python -m app.main & npm run dev   # Both in one command
```

## Scientific Validation (Phase 1)

To verify baseline inference works correctly:

```bash
# Test with controlled prompts
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is"}'
```

Expected: Model outputs "Paris" as top prediction

## Performance Expectations

**Baseline Inference Times (Phase 1):**
- GPU (CUDA): 2-5 seconds per prompt
- CPU: 30-120 seconds per prompt

**Memory Usage:**
- Model: ~4-6GB with float16
- Per-inference: ~1-2GB

**Throughput:**
- ~10-20 tokens/second on modern GPU

## Questions?

Refer to:
- [Architecture.md](architecture.md) - System design
- [README.md](../README.md) - Project overview
- Backend API docs: http://localhost:8000/docs
- PyTest output: `pytest tests/ -v`

Next: **Phase 2 - Instrumentation Layer**
