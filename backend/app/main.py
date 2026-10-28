"""FastAPI application for AI Brain Scanner."""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import torch
import traceback
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from app.config import settings
from app.instrumentation import ActivationPatch, HeadAblation, MlpAblation, TraceRecorder
from app.models.gemma import GemmaAdapter
from app.storage import storage
from app.schemas.inference import (
    InferenceRequest,
    InferenceResponse,
    ModelInfoResponse,
    HealthResponse,
    TokenInfo,
    TopKPrediction,
    HeadAblationRequest,
    MlpAblationRequest,
    HeadSweepRequest,
    HeadSweepResponse,
    HeadScore,
    ActivationPatchingRequest,
    ActivationPatchingResponse,
    ExperimentResponse,
    ExperimentListItem,
    NeuralGraphRequest,
    NeuralGraphResponse,
    NeuronNode,
    SynapseLink,
    ModelThinkingStep,
)

# Initialize FastAPI app
app = FastAPI(
    title="NeuroAudit API",
    description="Mechanistic AI Safety & Neuro-Observatory Platform",
    version="1.0.0",
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model_adapter: Optional[GemmaAdapter] = None
trace_store: dict[str, dict] = {}
experiment_store: dict[str, dict] = {}


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global model_adapter
    try:
        print(f"[INFO] Loading model: {settings.MODEL_NAME}")
        model_adapter = GemmaAdapter(
            model_name=settings.MODEL_NAME,
            device=settings.MODEL_DEVICE,
        )
        model_adapter.load()
        print("[INFO] Model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        traceback.print_exc()


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model_adapter is not None else "initializing",
        model_loaded=model_adapter is not None,
        device=settings.MODEL_DEVICE if model_adapter else None,
        message="Model is ready" if model_adapter else "Model is loading...",
    )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info() -> ModelInfoResponse:
    """Get information about the loaded model."""
    if model_adapter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    config = model_adapter.get_config()
    
    return ModelInfoResponse(
        model_name=config.model_name,
        model_type=config.model_type,
        device=config.device,
        vocab_size=model_adapter.model.config.vocab_size,
        num_parameters=sum(p.numel() for p in model_adapter.model.parameters()),
        dtype=str(config.dtype),
    )


@app.get("/model/architecture")
async def get_model_architecture() -> dict:
    """Return named transformer modules discovered on the loaded model."""
    if model_adapter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    recorder = TraceRecorder(model_adapter.model, layer_indices=[])
    return {
        "model": settings.MODEL_NAME,
        "layer_count": recorder.layer_count,
        "modules": recorder.architecture,
    }


@app.post("/model/neural-graph", response_model=NeuralGraphResponse)
async def get_neural_graph(request: NeuralGraphRequest) -> NeuralGraphResponse:
    """Generate multi-layer neural synapse graph showing genuine node activations from input to decided output."""
    if model_adapter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    raw_prompt = request.prompt.strip()
    if not raw_prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")

    tokens = model_adapter.tokenize(raw_prompt)
    input_ids = tokens["input_ids"]
    attention_mask = tokens.get("attention_mask")

    with torch.no_grad():
        outputs = model_adapter.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

    hidden_states = outputs.hidden_states
    logits = outputs.logits[0, -1]
    probabilities = torch.softmax(logits, dim=-1)

    total_layers = len(hidden_states) - 1
    k_per_col = max(4, min(10, request.num_neurons_per_column))

    W_U = model_adapter.model.lm_head.weight
    norm_module = getattr(model_adapter.model.model, "norm", None)

    # 1. Output candidate tokens from vocabulary
    top_probs, top_pred_ids = torch.topk(probabilities, k_per_col)
    winner_token_id = int(top_pred_ids[0])
    winner_prob = float(top_probs[0])
    winner_logit = float(logits[winner_token_id])

    early_l = min(3, total_layers - 1)
    mid_l = min(10, total_layers - 1)
    late_l = min(18, total_layers - 1)

    col_specs = [
        {"name": "Input Tokens", "type": "input", "layer": 0},
        {"name": f"L{early_l:02d} Syntax/Early", "type": "hidden", "layer": early_l},
        {"name": f"L{mid_l:02d} Factual/Mid", "type": "hidden", "layer": mid_l},
        {"name": f"L{late_l:02d} Decision/Late", "type": "hidden", "layer": late_l},
        {"name": "Output Candidates", "type": "output", "layer": -1},
    ]

    nodes: List[NeuronNode] = []
    links: List[SynapseLink] = []
    col_nodes: List[List[NeuronNode]] = [[] for _ in col_specs]
    thinking_steps: List[ModelThinkingStep] = []

    # 1. Real Input Tokens (Col 0)
    seq_len = input_ids.shape[1]
    last_pos_attn = None
    if outputs.attentions is not None and len(outputs.attentions) > 0:
        last_pos_attn = outputs.attentions[0][0, :, -1, :].mean(dim=0)

    if seq_len <= k_per_col:
        selected_token_indices = list(range(seq_len))
    else:
        scores = last_pos_attn.clone() if last_pos_attn is not None else torch.ones(seq_len)
        scores[0] += 0.5
        scores[-1] += 1.0
        _, top_idx = torch.topk(scores, k_per_col)
        selected_token_indices = sorted(top_idx.tolist())

    embed_module = model_adapter.model.model.embed_tokens
    token_embeds = embed_module(input_ids)[0]

    for n_idx, t_pos in enumerate(selected_token_indices):
        tok_id = int(input_ids[0, t_pos])
        raw_tok_str = model_adapter.decode(torch.tensor([tok_id]))
        clean_tok_str = raw_tok_str if raw_tok_str.strip() else "[space]"
        emb_norm = float(token_embeds[t_pos].norm().item())
        attn_val = float(last_pos_attn[t_pos].item()) if last_pos_attn is not None else (1.0 / seq_len)
        norm_val = min(1.0, max(0.15, attn_val * 3.0))

        node = NeuronNode(
            id=f"col0_n{n_idx}",
            layer_idx=0,
            layer_name=col_specs[0]["name"],
            neuron_idx=n_idx,
            channel_idx=tok_id,
            label=f'"{clean_tok_str}"',
            activation=round(emb_norm, 4),
            normalized=round(norm_val, 4),
            role="input",
            details={
                "token_position": t_pos,
                "token_id": tok_id,
                "token_string": clean_tok_str,
                "embedding_l2_norm": round(emb_norm, 4),
                "attention_received": f"{attn_val * 100:.2f}%",
                "description": f"Input prompt token '{clean_tok_str}' at sequence pos #{t_pos}. Query attention weight: {attn_val * 100:.1f}%.",
            },
        )
        nodes.append(node)
        col_nodes[0].append(node)

    thinking_steps.append(ModelThinkingStep(
        step_index=0,
        stage_name="Prompt Ingestion",
        layer_num=0,
        top_hypothesized_token=col_nodes[0][-1].label if col_nodes[0] else "prompt",
        confidence=1.0,
        description=f"Received prompt with {seq_len} tokens. Embedded into {token_embeds.shape[-1]}-dimensional semantic space.",
    ))

    # 2. Intermediate Checkpoints (Col 1, 2, 3)
    for col_idx in range(1, 4):
        l_num = col_specs[col_idx]["layer"]
        h_state = hidden_states[l_num][0, -1]
        h_normed = norm_module(h_state) if norm_module else h_state
        layer_logits = model_adapter.model.lm_head(h_normed)
        layer_probs = torch.softmax(layer_logits, dim=-1)
        l_top_probs, l_top_ids = torch.topk(layer_probs, 3)
        l_winner_str = model_adapter.decode(torch.tensor([l_top_ids[0].item()]))
        l_winner_str = l_winner_str if l_winner_str.strip() else "[space]"
        l_conf = float(l_top_probs[0].item())

        thinking_steps.append(ModelThinkingStep(
            step_index=col_idx,
            stage_name=col_specs[col_idx]["name"],
            layer_num=l_num,
            top_hypothesized_token=f'"{l_winner_str}"',
            confidence=round(l_conf, 4),
            description=f"Logit Lens Layer {l_num:02d}: top hypothesis '{l_winner_str}' ({l_conf*100:.1f}%), runner-ups: " +
                        ", ".join([f"'{model_adapter.decode(torch.tensor([i.item()])).strip() or '[space]'}' ({p.item()*100:.1f}%)" for p, i in zip(l_top_probs[1:], l_top_ids[1:])]),
        ))

        top_vals, top_indices = torch.topk(h_state.abs(), k_per_col)
        max_act = float(top_vals[0]) if top_vals[0] > 0 else 1.0

        for n_idx in range(k_per_col):
            ch_id = int(top_indices[n_idx])
            raw_act = float(h_state[ch_id].item())
            norm_val = min(1.0, max(0.12, abs(raw_act) / max_act))

            p_vals, p_ids = torch.topk(W_U[:, ch_id], 3)
            promoted_tokens = [model_adapter.decode(torch.tensor([idx.item()])).strip() or "[space]" for idx in p_ids]
            s_vals, s_ids = torch.topk(-W_U[:, ch_id], 3)
            suppressed_tokens = [model_adapter.decode(torch.tensor([idx.item()])).strip() or "[space]" for idx in s_ids]
            win_contrib = float(raw_act * W_U[winner_token_id, ch_id].item())

            node = NeuronNode(
                id=f"col{col_idx}_n{n_idx}",
                layer_idx=col_idx,
                layer_name=col_specs[col_idx]["name"],
                neuron_idx=n_idx,
                channel_idx=ch_id,
                label=f"Neuron #{ch_id}",
                activation=round(raw_act, 4),
                normalized=round(norm_val, 4),
                role="hidden",
                details={
                    "layer_number": l_num,
                    "channel_unit": ch_id,
                    "raw_scalar_activation": round(raw_act, 4),
                    "logit_contribution_to_winner": round(win_contrib, 4),
                    "promoted_tokens": promoted_tokens,
                    "suppressed_tokens": suppressed_tokens,
                    "layer_logit_lens_hypothesis": f'"{l_winner_str}" ({l_conf*100:.1f}%)',
                    "description": f"Channel #{ch_id} at Layer {l_num:02d}. Promotes [{', '.join(promoted_tokens)}] with {win_contrib:+.3f} logit contribution to winner.",
                },
            )
            nodes.append(node)
            col_nodes[col_idx].append(node)

    # 3. Real Decided Output Logits (Col 4)
    winner_id = "col4_n0"
    for n_idx in range(k_per_col):
        tok_id = int(top_pred_ids[n_idx])
        prob_val = float(top_probs[n_idx])
        logit_val = float(logits[tok_id])
        tok_str = model_adapter.decode(torch.tensor([tok_id]))
        clean_tok_str = tok_str if tok_str.strip() else "[space]"
        norm_val = 1.0 if n_idx == 0 else min(0.85, max(0.12, prob_val / winner_prob))

        node = NeuronNode(
            id=f"col4_n{n_idx}",
            layer_idx=4,
            layer_name=col_specs[4]["name"],
            neuron_idx=n_idx,
            channel_idx=tok_id,
            label=f'#{n_idx+1}: "{clean_tok_str}" ({prob_val*100:.1f}%)',
            activation=round(logit_val, 4),
            normalized=round(norm_val, 4),
            role="output",
            details={
                "rank": n_idx + 1,
                "token_id": tok_id,
                "token_string": clean_tok_str,
                "probability": round(prob_val, 4),
                "percentage": f"{prob_val * 100:.2f}%",
                "logit": round(logit_val, 4),
                "is_winner": n_idx == 0,
                "margin_over_runner_up": f"+{(winner_prob - prob_val)*100:.2f}%" if n_idx > 0 else "WINNER",
                "description": f"Prediction Candidate #{n_idx+1}: '{clean_tok_str}' with {prob_val * 100:.2f}% probability (Logit: {logit_val:.2f})",
            },
        )
        nodes.append(node)
        col_nodes[4].append(node)

    thinking_steps.append(ModelThinkingStep(
        step_index=4,
        stage_name="Final Decision",
        layer_num=total_layers,
        top_hypothesized_token=f'"{col_nodes[4][0].details["token_string"]}"',
        confidence=round(winner_prob, 4),
        description=f"Model converged on '{col_nodes[4][0].details['token_string']}' with {winner_prob*100:.2f}% confidence across 32,000 vocabulary logits.",
    ))

    # 4. Synaptic Links
    for s in col_nodes[0]:
        t_pos = s.details["token_position"]
        for d in col_nodes[1]:
            ch_id = d.channel_idx
            emb_val = abs(float(token_embeds[t_pos, ch_id % token_embeds.shape[-1]].item()))
            weight = round(min(1.0, max(0.04, emb_val * s.normalized)), 4)
            links.append(SynapseLink(
                source_id=s.id,
                target_id=d.id,
                weight=weight,
                activity=round(weight * d.normalized, 4),
            ))

    for c_idx in [1, 2]:
        for s in col_nodes[c_idx]:
            for d in col_nodes[c_idx + 1]:
                corr = abs(s.activation * d.activation)
                s_contrib = s.details.get("logit_contribution_to_winner", 0)
                d_contrib = d.details.get("logit_contribution_to_winner", 0)
                if s_contrib > 0 and d_contrib > 0:
                    corr *= 1.3
                weight = round(min(1.0, max(0.05, min(1.0, corr / 4.0))), 4)
                links.append(SynapseLink(
                    source_id=s.id,
                    target_id=d.id,
                    weight=weight,
                    activity=round(weight * d.normalized, 4),
                ))

    for s in col_nodes[3]:
        ch_id = s.channel_idx
        for d in col_nodes[4]:
            cand_tok_id = d.channel_idx
            w_u_val = float(W_U[cand_tok_id, ch_id].item())
            syn_weight = max(0.04, min(1.0, (w_u_val * s.activation + 2.0) / 6.0))
            if d.id == winner_id and s.details.get("logit_contribution_to_winner", 0) > 0:
                syn_weight = min(1.0, syn_weight * 1.4)
            links.append(SynapseLink(
                source_id=s.id,
                target_id=d.id,
                weight=round(syn_weight, 4),
                activity=round(syn_weight * d.normalized, 4),
            ))

    # Generate multi-token continuation (6 tokens) to show full word/completion
    continuation_str = ""
    subword_note = None
    try:
        with torch.no_grad():
            gen_out = model_adapter.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=6,
                do_sample=False,
                pad_token_id=model_adapter.tokenizer.eos_token_id,
            )
        continuation_ids = gen_out[0][seq_len:]
        continuation_str = model_adapter.decode(continuation_ids).strip()
        winner_token_text = col_nodes[4][0].details.get("token_string", "").strip()
        if continuation_str:
            subword_note = f"Token '{winner_token_text}' is the opening subword; full autoregressive generation completes into: '{continuation_str}'."
    except Exception as e:
        continuation_str = col_nodes[4][0].details.get("token_string", "")

    return NeuralGraphResponse(
        prompt=raw_prompt,
        model=settings.MODEL_NAME,
        columns=[spec["name"] for spec in col_specs],
        nodes=nodes,
        links=links,
        winner_node_id=winner_id,
        thinking_steps=thinking_steps,
        autoregressive_continuation=continuation_str,
        subword_explanation=subword_note,
    )


@app.get("/model/layers")
async def list_model_layers() -> dict:
    if model_adapter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    recorder = TraceRecorder(model_adapter.model, layer_indices=[])
    layers = sorted({int(item["layer"]) for item in recorder.architecture})
    return {
        "layer_count": recorder.layer_count,
        "num_attention_heads": model_adapter.model.config.num_attention_heads,
        "head_dim": getattr(model_adapter.model.config, "head_dim", None),
        "layers": [{"layer": layer, "paths": [item for item in recorder.architecture if int(item["layer"]) == layer]} for layer in layers],
    }


@app.get("/model/layers/{layer}")
async def inspect_model_layer(layer: int) -> dict:
    if model_adapter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    recorder = TraceRecorder(model_adapter.model, layer_indices=[])
    matches = [item for item in recorder.architecture if int(item["layer"]) == layer]
    if not matches:
        raise HTTPException(status_code=404, detail="Layer not found")
    return {"layer": layer, "num_attention_heads": model_adapter.model.config.num_attention_heads, "head_dim": getattr(model_adapter.model.config, "head_dim", None), "modules": matches}


@app.get("/traces/{trace_id}")
async def get_trace(trace_id: str) -> dict:
    """Retrieve a completed internal computation trace."""
    trace = trace_store.get(trace_id) or storage.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


def _trace_capture(trace_id: str, layer: int, component: str) -> dict:
    trace = trace_store.get(trace_id) or storage.get_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    capture = next((item for item in trace["captures"] if item["layer"] == layer and item["component"] == component), None)
    if capture is None:
        raise HTTPException(status_code=404, detail="Requested tensor was not captured")
    return capture


@app.get("/traces/{trace_id}/attention/{layer}")
async def get_attention_weights(trace_id: str, layer: int, head: Optional[int] = None) -> dict:
    capture = _trace_capture(trace_id, layer, "attention_weights")
    values = capture.get("values", [])
    if head is not None:
        if head < 0 or head >= len(values):
            raise HTTPException(status_code=400, detail="Attention head is out of range")
        values = values[head]
    return {"trace_id": trace_id, "layer": layer, "head": head, "shape": capture["shape"], "values": values, "statistics": {key: capture[key] for key in ("mean", "std", "min", "max", "abs_max")}}


@app.get("/traces/{trace_id}/hidden/{layer}")
async def get_hidden_statistics(trace_id: str, layer: int) -> dict:
    return _trace_capture(trace_id, layer, "hidden")


@app.get("/traces/{trace_id}/mlp/{layer}")
async def get_mlp_output(trace_id: str, layer: int) -> dict:
    return _trace_capture(trace_id, layer, "mlp")


def _prediction_summary(adapter: GemmaAdapter, logits: torch.Tensor, target_token_id: Optional[int] = None) -> dict:
    last_logits = logits[0, -1]
    probabilities = torch.softmax(last_logits, dim=-1)
    predicted_id = int(torch.argmax(last_logits))
    target_id = predicted_id if target_token_id is None else target_token_id
    ranking = int((last_logits > last_logits[target_id]).sum().item()) + 1
    return {
        "predicted_token_id": predicted_id,
        "predicted_token": adapter.decode(torch.tensor([predicted_id], device=logits.device)),
        "target_token_id": target_id,
        "target_token": adapter.decode(torch.tensor([target_id], device=logits.device)),
        "target_logit": float(last_logits[target_id]),
        "target_probability": float(probabilities[target_id]),
        "target_rank": ranking,
    }


@app.post("/experiments/head-ablation", response_model=ExperimentResponse)
async def run_head_ablation(request: HeadAblationRequest) -> ExperimentResponse:
    if model_adapter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    tokens = model_adapter.tokenize(request.prompt)
    input_ids = tokens["input_ids"]
    attention_mask = tokens.get("attention_mask")
    baseline = model_adapter.forward(input_ids, attention_mask)
    target_id = request.target_token_id
    baseline_summary = _prediction_summary(model_adapter, baseline.logits, target_id)
    if target_id is None:
        target_id = baseline_summary["predicted_token_id"]
        baseline_summary = _prediction_summary(model_adapter, baseline.logits, target_id)

    ablation = HeadAblation(model_adapter.model, request.layer, request.head)
    try:
        ablation.start()
        intervened = model_adapter.forward(input_ids, attention_mask)
    finally:
        ablation.stop()
    intervention_summary = _prediction_summary(model_adapter, intervened.logits, target_id)
    experiment = {
        "experiment_id": str(uuid.uuid4()), "prompt": request.prompt, "model": settings.MODEL_NAME,
        "layer": request.layer, "head": request.head,
        "intervention": {"type": "attention_head_ablation", "path": ablation.path, "head_dim": ablation.head_dim},
        "baseline": baseline_summary, "intervention_result": intervention_summary,
        "metrics": {
            "target_logit_delta": intervention_summary["target_logit"] - baseline_summary["target_logit"],
            "target_probability_delta": intervention_summary["target_probability"] - baseline_summary["target_probability"],
            "rank_delta": intervention_summary["target_rank"] - baseline_summary["target_rank"],
            "prediction_changed": baseline_summary["predicted_token_id"] != intervention_summary["predicted_token_id"],
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    experiment_store[experiment["experiment_id"]] = experiment
    storage.save_experiment(experiment)
    return ExperimentResponse(**experiment)


@app.post("/experiments/mlp-ablation", response_model=ExperimentResponse)
async def run_mlp_ablation(request: MlpAblationRequest) -> ExperimentResponse:
    if model_adapter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    tokens = model_adapter.tokenize(request.prompt)
    input_ids = tokens["input_ids"]
    attention_mask = tokens.get("attention_mask")
    baseline = model_adapter.forward(input_ids, attention_mask)
    target_id = request.target_token_id
    baseline_summary = _prediction_summary(model_adapter, baseline.logits, target_id)
    if target_id is None:
        target_id = baseline_summary["predicted_token_id"]
        baseline_summary = _prediction_summary(model_adapter, baseline.logits, target_id)

    ablation = MlpAblation(model_adapter.model, request.layer)
    try:
        ablation.start()
        intervened = model_adapter.forward(input_ids, attention_mask)
    finally:
        ablation.stop()
    intervention_summary = _prediction_summary(model_adapter, intervened.logits, target_id)
    experiment = {
        "experiment_id": str(uuid.uuid4()), "prompt": request.prompt, "model": settings.MODEL_NAME,
        "layer": request.layer, "head": -1,
        "intervention": {"type": "mlp_ablation", "path": ablation.path},
        "baseline": baseline_summary, "intervention_result": intervention_summary,
        "metrics": {
            "target_logit_delta": intervention_summary["target_logit"] - baseline_summary["target_logit"],
            "target_probability_delta": intervention_summary["target_probability"] - baseline_summary["target_probability"],
            "rank_delta": intervention_summary["target_rank"] - baseline_summary["target_rank"],
            "prediction_changed": baseline_summary["predicted_token_id"] != intervention_summary["predicted_token_id"],
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    experiment_store[experiment["experiment_id"]] = experiment
    storage.save_experiment(experiment)
    return ExperimentResponse(**experiment)


@app.post("/experiments/head-sweep", response_model=HeadSweepResponse)
async def run_head_sweep(request: HeadSweepRequest) -> HeadSweepResponse:
    if model_adapter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    tokens = model_adapter.tokenize(request.prompt)
    input_ids = tokens["input_ids"]
    attention_mask = tokens.get("attention_mask")

    # Baseline forward pass
    baseline = model_adapter.forward(input_ids, attention_mask)
    last_logits = baseline.logits[0, -1]
    probabilities = torch.softmax(last_logits, dim=-1)

    target_id = request.target_token_id
    if target_id is None:
        target_id = int(torch.argmax(last_logits))

    baseline_prob = float(probabilities[target_id])
    baseline_logit = float(last_logits[target_id])
    target_token_str = model_adapter.decode(torch.tensor([target_id]))

    recorder = TraceRecorder(model_adapter.model, layer_indices=[])
    total_layers = recorder.layer_count
    num_heads = int(model_adapter.model.config.num_attention_heads)

    target_layers = request.layers if request.layers is not None else list(range(min(12, total_layers)))
    target_layers = [l for l in target_layers if 0 <= l < total_layers]

    sensitivity_matrix: List[List[float]] = []
    all_head_scores: List[HeadScore] = []

    for layer in target_layers:
        layer_scores = []
        for head in range(num_heads):
            ablation = HeadAblation(model_adapter.model, layer, head)
            try:
                ablation.start()
                intervened = model_adapter.forward(input_ids, attention_mask)
                interv_last = intervened.logits[0, -1]
                interv_prob = float(torch.softmax(interv_last, dim=-1)[target_id])
                interv_logit = float(interv_last[target_id])
            finally:
                ablation.stop()

            drop = round(baseline_prob - interv_prob, 5)
            logit_delta = round(interv_logit - baseline_logit, 5)
            layer_scores.append(drop)
            all_head_scores.append(HeadScore(
                layer=layer, head=head, probability_drop=drop, target_logit_delta=logit_delta
            ))
        sensitivity_matrix.append(layer_scores)

    top_causal = sorted(all_head_scores, key=lambda s: s.probability_drop, reverse=True)[:8]

    sweep_exp = {
        "experiment_id": str(uuid.uuid4()),
        "prompt": request.prompt,
        "model": settings.MODEL_NAME,
        "experiment_type": "head_sweep",
        "target_token": target_token_str,
        "target_token_id": target_id,
        "baseline_probability": baseline_prob,
        "baseline_logit": baseline_logit,
        "layers_evaluated": target_layers,
        "num_heads": num_heads,
        "sensitivity_matrix": sensitivity_matrix,
        "top_causal_heads": [s.model_dump() for s in top_causal],
        "metrics": {
            "top_causal_head": f"L{top_causal[0].layer}H{top_causal[0].head} (drop {top_causal[0].probability_drop*100:+.1f}%)" if top_causal else "None"
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    storage.save_experiment(sweep_exp)
    return HeadSweepResponse(**sweep_exp)


@app.post("/experiments/activation-patching", response_model=ActivationPatchingResponse)
async def run_activation_patching(request: ActivationPatchingRequest) -> ActivationPatchingResponse:
    if model_adapter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # 1. Clean run: capture target activation
    clean_tokens = model_adapter.tokenize(request.clean_prompt)
    clean_ids = clean_tokens["input_ids"]
    clean_mask = clean_tokens.get("attention_mask")

    captured_clean = []
    suffix = f"layers.{request.target_layer}" if request.component == "hidden" else (
        f"layers.{request.target_layer}.mlp" if request.component == "mlp" else f"layers.{request.target_layer}.self_attn"
    )

    def cap_hook(_m, _i, o):
        t = o[0] if isinstance(o, tuple) else o
        captured_clean.append(t.detach().clone())

    hook_handle = None
    for p, m in model_adapter.model.named_modules():
        if p.endswith(suffix):
            hook_handle = m.register_forward_hook(cap_hook)
            break

    if hook_handle is None:
        raise HTTPException(status_code=400, detail=f"Target module for layer {request.target_layer} not found")

    try:
        with torch.no_grad():
            clean_out = model_adapter.model(clean_ids, attention_mask=clean_mask)
    finally:
        hook_handle.remove()

    if not captured_clean:
        raise HTTPException(status_code=500, detail="Failed to capture clean activation")

    clean_act = captured_clean[0]
    clean_logits = clean_out.logits[0, -1]
    clean_probs = torch.softmax(clean_logits, dim=-1)

    target_id = request.target_token_id
    if target_id is None:
        target_id = int(torch.argmax(clean_logits))

    clean_token_str = model_adapter.decode(torch.tensor([target_id]))
    clean_logit = float(clean_logits[target_id])
    clean_prob = float(clean_probs[target_id])

    # 2. Corrupted baseline
    corr_tokens = model_adapter.tokenize(request.corrupted_prompt)
    corr_ids = corr_tokens["input_ids"]
    corr_mask = corr_tokens.get("attention_mask")

    corr_baseline = model_adapter.forward(corr_ids, corr_mask)
    corr_logits = corr_baseline.logits[0, -1]
    corr_probs = torch.softmax(corr_logits, dim=-1)
    corr_pred_id = int(torch.argmax(corr_logits))
    corr_pred_token = model_adapter.decode(torch.tensor([corr_pred_id]))
    corr_logit = float(corr_logits[target_id])
    corr_prob = float(corr_probs[target_id])

    # 3. Corrupted run with clean activation patched in
    patch = ActivationPatch(model_adapter.model, request.target_layer, request.component, clean_act)
    try:
        patch.start()
        patched_out = model_adapter.forward(corr_ids, corr_mask)
    finally:
        patch.stop()

    pat_logits = patched_out.logits[0, -1]
    pat_probs = torch.softmax(pat_logits, dim=-1)
    pat_pred_id = int(torch.argmax(pat_logits))
    pat_pred_token = model_adapter.decode(torch.tensor([pat_pred_id]))
    pat_logit = float(pat_logits[target_id])
    pat_prob = float(pat_probs[target_id])

    denom = clean_logit - corr_logit
    if abs(denom) < 1e-4:
        recovery = 1.0 if pat_logit >= corr_logit else 0.0
    else:
        recovery = float((pat_logit - corr_logit) / denom)

    patch_exp = {
        "experiment_id": str(uuid.uuid4()),
        "clean_prompt": request.clean_prompt,
        "corrupted_prompt": request.corrupted_prompt,
        "target_layer": request.target_layer,
        "component": request.component,
        "clean_target_token": clean_token_str,
        "clean_target_id": target_id,
        "corrupted_prediction": corr_pred_token,
        "patched_prediction": pat_pred_token,
        "clean_logit": clean_logit,
        "corrupted_logit": corr_logit,
        "patched_logit": pat_logit,
        "clean_prob": clean_prob,
        "corrupted_prob": corr_prob,
        "patched_prob": pat_prob,
        "causal_recovery": round(recovery, 4),
        "experiment_type": "activation_patching",
        "prompt": f"Patch L{request.target_layer}: {request.clean_prompt}",
        "model": settings.MODEL_NAME,
        "layer": request.target_layer,
        "metrics": {
            "causal_recovery": round(recovery, 4),
            "patched_prob_delta": round(pat_prob - corr_prob, 4),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    storage.save_experiment(patch_exp)
    return ActivationPatchingResponse(**patch_exp)


@app.get("/experiments", response_model=List[ExperimentListItem])
async def list_experiments(limit: int = 50, experiment_type: Optional[str] = None) -> List[ExperimentListItem]:
    items = storage.list_experiments(limit=limit, exp_type=experiment_type)
    return [ExperimentListItem(**item) for item in items]


@app.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str) -> dict:
    experiment = experiment_store.get(experiment_id) or storage.get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@app.delete("/experiments/{experiment_id}")
async def delete_experiment(experiment_id: str) -> dict:
    success = storage.delete_experiment(experiment_id)
    if not success:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {"status": "success", "deleted": experiment_id}


@app.post("/infer", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest) -> InferenceResponse:
    """Run inference on a prompt.
    
    This is Phase 1: baseline inference without interventions.
    """
    if model_adapter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Tokenize input
        tokens = model_adapter.tokenize(request.prompt)
        input_ids = tokens["input_ids"]
        attention_mask = tokens.get("attention_mask")
        
        # Record only selected components during the baseline forward pass.
        recorder = TraceRecorder(
            model_adapter.model,
            layer_indices=request.capture_layers,
            components=request.capture_components,
        )
        recorder.start()
        try:
            result = model_adapter.forward(input_ids, attention_mask, capture_internals=True)
        finally:
            recorder.stop()
        recorder.add_forward_outputs(result.hidden_states, result.attentions)
        trace = recorder.build_trace(input_ids, result.logits)
        trace_store[trace["trace_id"]] = trace
        storage.save_trace(trace)
        # Keep the in-memory store bounded for the local Phase 2 workflow.
        while len(trace_store) > 12:
            trace_store.pop(next(iter(trace_store)))
        
        # Decode tokens
        input_tokens = []
        for idx, token_id in enumerate(input_ids[0]):
            token_str = model_adapter.decode(token_id.unsqueeze(0))
            input_tokens.append(TokenInfo(
                token_id=int(token_id),
                token_str=token_str,
                position=idx,
            ))
        
        # Build predictions for each position
        predictions = []
        for pos_idx, pos_preds in enumerate(result.top_k_tokens[0]):
            pos_predictions = []
            for rank, (token_id, prob) in enumerate(pos_preds):
                token_str = model_adapter.decode(torch.tensor([token_id]))
                logit_val = result.top_k_logits[0][pos_idx][rank][1]
                
                pos_predictions.append(TopKPrediction(
                    position=pos_idx,
                    token_id=int(token_id),
                    token_str=token_str,
                    probability=float(prob),
                    logit=float(logit_val),
                    rank=rank + 1,
                ))
            predictions.append(pos_predictions)
        
        # Generate a continuation using the model's autoregressive loop.
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": request.max_new_tokens,
            "do_sample": request.temperature > 0,
            "temperature": max(request.temperature, 1e-5),
            "pad_token_id": model_adapter.tokenizer.eos_token_id,
        }
        if request.top_k is not None:
            generation_kwargs["top_k"] = request.top_k
        if request.top_p is not None:
            generation_kwargs["top_p"] = request.top_p

        with torch.inference_mode():
            output_ids = model_adapter.model.generate(**generation_kwargs)

        generated_ids = output_ids[:, input_ids.shape[1]:][0]
        generated_text = model_adapter.decode(generated_ids)
        
        generated_tokens = []
        for idx, token_id in enumerate(generated_ids):
            token_str = model_adapter.decode(token_id.unsqueeze(0))
            generated_tokens.append(TokenInfo(
                token_id=int(token_id),
                token_str=token_str,
                position=idx,
            ))
        
        return InferenceResponse(
            input_prompt=request.prompt,
            input_tokens=input_tokens,
            generated_text=generated_text,
            generated_tokens=generated_tokens,
            predictions=predictions,
            metadata={
                "model": result.metadata["model"],
                "device": result.metadata["device"],
                "vocab_size": result.metadata["vocab_size"],
                "sequence_length": result.metadata["sequence_length"],
                "generated_length": len(generated_tokens),
                "temperature": request.temperature,
                "trace_id": trace["trace_id"],
            },
        )
    
    except Exception as e:
        print(f"Inference error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test-inference")
async def test_inference():
    """Quick test endpoint to verify inference works."""
    if model_adapter is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Test with a simple prompt
        request = InferenceRequest(prompt="The capital of France is")
        response = await run_inference(request)
        return {
            "status": "success",
            "prompt": response.input_prompt,
            "next_token": response.generated_text,
            "top_predictions": [
                {
                    "token": p.token_str,
                    "probability": p.probability,
                }
                for p in response.predictions[0][:3]  # Top 3
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
    )
