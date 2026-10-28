"""Tests for AI Audit Lab & Prompt Intelligence service."""
import pytest
from app.schemas.inference import AuditPipelineRequest
from app.services.audit_service import (
    _analyze_prompt_linguistics,
    _generate_controlled_variants,
    _build_improved_prompt,
    run_audit_pipeline,
)
from app.models.gemma import GemmaAdapter
from app.config import settings


def test_analyze_prompt_linguistics():
    prompt = "The capital of Karnataka is"
    analysis, weaknesses = _analyze_prompt_linguistics(prompt, "factual_recall")
    assert "Factual Knowledge Retrieval" in analysis.intended_task or "Autoregressive" in analysis.intended_task
    assert len(weaknesses) >= 1  # Should flag unconstrained format or missing refusal guardrail


def test_generate_controlled_variants():
    prompt = "The capital of Karnataka is"
    variants = _generate_controlled_variants(prompt, "factual_recall", "Bangalore")
    assert len(variants) >= 3
    # Check that variant types are generated
    types = [v[0] for v in variants]
    assert "paraphrase_imperative" in types or "constrained_format" in types


def test_build_improved_prompt():
    prompt = "The capital of Karnataka is"
    analysis, weaknesses = _analyze_prompt_linguistics(prompt, "factual_recall")
    improved, explanation = _build_improved_prompt(prompt, analysis, weaknesses, "Bangalore")
    assert "Karnataka" in improved
    assert "Output" in improved or "capital" in improved
    assert len(explanation) > 10


@pytest.mark.asyncio
async def test_run_audit_pipeline_integration():
    adapter = GemmaAdapter(model_name=settings.MODEL_NAME, device=settings.MODEL_DEVICE)
    adapter.load()
    req = AuditPipelineRequest(
        prompt="The capital of Karnataka is",
        objective="factual_recall",
        expected_answer="Bangalore",
        run_robustness_variants=False,  # Fast run for unit test
        max_new_tokens=10,
    )
    res = await run_audit_pipeline(req, adapter, settings.MODEL_NAME)
    assert res.run_id.startswith("AUDIT-")
    assert res.baseline_inference["top_prediction"]["token_str"].strip() in ["Bang", "Bangalore"]
    assert len(res.evidence) >= 5
    assert len(res.synthesis.empirically_supported_conclusions) >= 1
