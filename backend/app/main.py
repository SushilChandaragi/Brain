"""FastAPI application for AI Brain Scanner."""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import torch
import traceback
import uuid
from typing import Optional

from app.config import settings
from app.instrumentation import HeadAblation, TraceRecorder
from app.models.gemma import GemmaAdapter
from app.schemas.inference import (
    InferenceRequest,
    InferenceResponse,
    ModelInfoResponse,
    HealthResponse,
    TokenInfo,
    TopKPrediction,
    HeadAblationRequest,
    ExperimentResponse,
)

# Initialize FastAPI app
app = FastAPI(
    title="AI Brain Scanner API",
    description="Interactive Mechanistic Interpretability of Gemma 4",
    version="0.1.0",
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
    trace = trace_store.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace


def _trace_capture(trace_id: str, layer: int, component: str) -> dict:
    trace = trace_store.get(trace_id)
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
    }
    experiment_store[experiment["experiment_id"]] = experiment
    return ExperimentResponse(**experiment)


@app.get("/experiments/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(experiment_id: str) -> ExperimentResponse:
    experiment = experiment_store.get(experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ExperimentResponse(**experiment)


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
