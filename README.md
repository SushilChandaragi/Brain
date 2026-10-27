# AI Brain Scanner: Interactive Mechanistic Interpretability of Gemma 4

An interactive platform for inspecting and experimentally studying the internal computation of Gemma 4 through controlled interventions and causal analysis.

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- GPU with CUDA support (or CPU-only fallback)
- ~8GB+ VRAM for Gemma 4 E2B (adjust sequence length for lower VRAM)

### Setup

**Backend:**
```bash
cd backend
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python -m app.main
```

Backend runs on `http://localhost:8000`
API docs: `http://localhost:8000/docs`

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

Frontend runs on `http://localhost:5173`

## Architecture

```
ai-brain-scanner/
├── backend/
│   ├── app/
│   │   ├── api/                 # FastAPI routes
│   │   ├── models/              # Model adapter abstraction
│   │   ├── instrumentation/     # Hook & capture layer
│   │   ├── interventions/       # Ablation, patching, scaling
│   │   ├── experiments/         # Experiment engine
│   │   ├── storage/             # Database & serialization
│   │   ├── schemas/             # Pydantic models
│   │   └── main.py              # FastAPI app
│   ├── tests/                   # PyTest suite
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   ├── pages/               # App pages
│   │   ├── features/            # Feature modules
│   │   ├── visualizations/      # Plotly/D3 components
│   │   ├── api/                 # API client
│   │   └── App.tsx
│   └── package.json
└── docs/
    ├── architecture.md
    ├── experiments.md
    ├── scientific-scope.md
    └── model-support.md
```

## Development Phases

- **Phase 1:** Model loading & baseline inference ✨ *Currently Here*
- Phase 2: Instrumentation (capture internal states)
- Phase 3: Visualization (UI for inspection)
- Phase 4: Ablation interventions
- Phase 5: Experiment engine
- Phase 6: Activation patching
- Phase 7: Mini-Transformer simulator
- Phase 8+: Advanced features

## Key Design Decisions

1. **Model Adapter:** Abstraction layer allows supporting multiple Transformers
2. **Selective Capture:** Memory-conscious capture of internal states
3. **No Permanent Weight Changes:** Core experiments don't modify model weights
4. **Reproducible Experiments:** All runs include full metadata for replay
5. **Scientific Rigor:** No unsupported causal claims; interventions provide evidence

## Documentation

- [Architecture](docs/architecture.md)
- [Experiments & Methodology](docs/experiments.md)
- [Scientific Scope & Guardrails](docs/scientific-scope.md)
- [Model Support](docs/model-support.md)
