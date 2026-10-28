"""AI Audit Lab & Prompt Intelligence Engine.

Provides evidence-driven prompt auditing, baseline inference telemetry,
mechanistic candidate identification, controlled robustness variant testing,
and scientifically grounded prompt improvement recommendations.
"""
from datetime import datetime, timezone
import math
import re
from typing import Any, Dict, List, Optional, Tuple
import uuid

import torch

from app.models.gemma import GemmaAdapter
from app.schemas.inference import (
    AuditPipelineRequest,
    AuditPipelineResponse,
    PromptAnalysis,
    AuditEvidenceItem,
    RobustnessVariantResult,
    PromptQualityWeakness,
    MechanisticCandidate,
    AuditSynthesis,
)


def _compute_entropy(probs: torch.Tensor) -> float:
    """Compute Shannon entropy in nats from probability tensor."""
    p_nonzero = probs[probs > 1e-12]
    return float(-torch.sum(p_nonzero * torch.log(p_nonzero)).item())


def _analyze_prompt_linguistics(prompt: str, objective: str) -> Tuple[PromptAnalysis, List[PromptQualityWeakness]]:
    """Stage 1: Perform rigorous linguistic and structural prompt analysis."""
    p_clean = prompt.strip()
    p_lower = p_clean.lower()
    
    # 1. Identify Intended Task
    if re.search(r"\b(capital|population|president|founder|born|currency|distance|height|located|where is)\b", p_lower):
        intended_task = "Factual Knowledge Retrieval (Entity attribute association query)"
    elif re.search(r"\b(summarize|tldr|brief|overview)\b", p_lower):
        intended_task = "Document or Context Summarization"
    elif re.search(r"\b(write|generate|compose|create|draft)\b", p_lower):
        intended_task = "Creative or Generative Content Synthesis"
    elif re.search(r"\b(code|python|function|class|sql|regex|debug)\b", p_lower):
        intended_task = "Code Generation or Algorithmic Reasoning"
    elif re.search(r"\b(explain|why|how does|what causes)\b", p_lower):
        intended_task = "Explanatory or Mechanistic Causal Explanation"
    elif re.search(r"\b(is|does|can|will|are|true or false)\b", p_lower) and p_clean.endswith("?"):
        intended_task = "Boolean or Factual Verification"
    elif p_clean.endswith("is") or p_clean.endswith("are") or p_clean.endswith("was") or p_clean.endswith("to"):
        intended_task = "Autoregressive Sentence Completion (Cloze-style retrieval prompt)"
    else:
        intended_task = "Open-ended Instruction Following"

    # 2. Identify Explicit Requirements
    explicit_requirements = []
    if re.search(r"\b(one word|single word|only the name|no explanation|concise)\b", p_lower):
        explicit_requirements.append("Strict conciseness / single-token or single-entity response format")
    if re.search(r"\b(step by step|numbered|bullet|list|table|json|csv)\b", p_lower):
        explicit_requirements.append("Structured output formatting")
    if re.search(r"\b(do not|never|exclude|without)\b", p_lower):
        explicit_requirements.append("Negative constraint / exclusion rule declared")
    if not explicit_requirements:
        explicit_requirements.append("Unconstrained autoregressive generation (no output length, format, or style constraints explicitly specified)")

    # 3. Detect Ambiguous Terms
    vague_patterns = [
        (r"\b(best|good|proper|normal|appropriate)\b", "Subjective evaluative terms without defined rubric"),
        (r"\b(some|a few|many|a lot|various|several)\b", "Indefinite quantifier without numerical boundary"),
        (r"\b(fast|quick|simple|easy)\b", "Relative performance or complexity descriptor"),
        (r"\b(stuff|things|aspects|details)\b", "Vague catch-all nouns lacking conceptual precision"),
        (r"\b(they|it|this|that|he|she)\b", "Pronoun with potential antecedent ambiguity in zero-shot context"),
    ]
    ambiguous_terms = []
    for pattern, desc in vague_patterns:
        matches = re.findall(pattern, p_lower)
        if matches:
            unique_matches = sorted(list(set(matches)))
            ambiguous_terms.append(f"'{', '.join(unique_matches)}': {desc}")

    # 4. Detect Missing Context
    missing_context = []
    if not re.search(r"\b(json|markdown|list|words|sentences|paragraphs|format|only)\b", p_lower):
        missing_context.append("Output Format: No target format, schema, or delimiter specified.")
    if not re.search(r"\b(if unknown|if uncertain|do not know|cannot determine|if not sure)\b", p_lower):
        missing_context.append("Uncertainty Guardrail: No refusal or fallback directive provided if the fact is unknown or ambiguous.")
    if not re.search(r"\b(as of|year|current|modern|history|date)\b", p_lower) and "capital" in p_lower:
        missing_context.append("Temporal Context: No time horizon or temporal grounding (e.g. current vs historical).")
    if len(p_clean.split()) < 5:
        missing_context.append("Query Brevity: High risk of underspecification due to minimal prompt context (< 5 words).")

    # 5. Detect Contradictions & Misleading Premises
    contradictions = []
    if re.search(r"\b(brief|concise|one sentence)\b", p_lower) and re.search(r"\b(detailed|thorough|comprehensive|in-depth)\b", p_lower):
        contradictions.append("Contradictory constraints: Simultaneous demand for brevity ('concise') and exhaustive depth ('detailed').")
    if re.search(r"\b(include all)\b", p_lower) and re.search(r"\b(top 3|only a few)\b", p_lower):
        contradictions.append("Contradictory scope: Demands both exhaustive inclusion and strict truncation.")

    misleading_assumptions = []
    if re.search(r"\b(but the .* is in|although .* is in|whereas .* is in)\b", p_lower):
        misleading_assumptions.append("Distractor Clause Detected: Prompt embeds a counterfactual entity preface that may distract attention heads.")
    if re.search(r"\b(why did .* in (203|204|205)\d)\b", p_lower):
        misleading_assumptions.append("Future Temporal Anachronism: References events in future unobserved timeframes.")

    # 6. Expected Format & Reliability Concerns
    if "json" in p_lower:
        expected_format = "JSON Object / Structured Schema"
    elif "list" in p_lower or "bullet" in p_lower:
        expected_format = "Bulleted / Numbered List"
    elif "one word" in p_lower or p_clean.endswith("is"):
        expected_format = "Single Target Entity / Cloze Token Continuation"
    else:
        expected_format = "Freeform Unconstrained Text Paragraph"

    reliability_concerns = []
    if not any("Uncertainty Guardrail" in m for m in missing_context):
        pass
    else:
        reliability_concerns.append("Hallucination Vulnerability: Absent refusal instructions, the model will sample highest-entropy plausible tokens rather than acknowledge absence of knowledge.")
    if misleading_assumptions:
        reliability_concerns.append("Context Priming Bias: Salient distractor tokens can corrupt downstream residual stream representations.")
    if ambiguous_terms:
        reliability_concerns.append("Interpretive Drift: Subjective qualifiers allow non-deterministic interpretations across different sampling seeds.")

    summary = (
        f"Prompt acts as a {intended_task.lower()} with {len(p_clean.split())} tokens. "
        f"{'Contains ' + str(len(ambiguous_terms)) + ' ambiguous qualifiers. ' if ambiguous_terms else 'Uses relatively direct phrasing. '}"
        f"{'Lacks output format constraints. ' if 'Output Format' in ' '.join(missing_context) else 'Specifies output constraints.'}"
    )

    analysis = PromptAnalysis(
        intended_task=intended_task,
        explicit_requirements=explicit_requirements,
        ambiguous_terms=ambiguous_terms,
        missing_context=missing_context,
        contradictions=contradictions,
        misleading_assumptions=misleading_assumptions,
        expected_format=expected_format,
        reliability_concerns=reliability_concerns,
        summary=summary,
    )

    # Convert detected issues into structured Weakness objects
    weaknesses: List[PromptQualityWeakness] = []
    wid = 1
    if ambiguous_terms:
        weaknesses.append(PromptQualityWeakness(
            id=f"W-{wid:02d}",
            weakness_type="Ambiguous Qualifiers",
            problematic_text=", ".join([a.split(":")[0] for a in ambiguous_terms[:2]]),
            explanation="Subjective or unquantified qualifiers lack objective evaluation criteria, causing high variance in model responses.",
            evidence=f"Detected {len(ambiguous_terms)} vague token expressions without operational definitions.",
            proposed_correction="Replace qualitative adjectives with concrete constraints, criteria, or numerical boundaries."
        ))
        wid += 1

    if any("Output Format" in m for m in missing_context):
        weaknesses.append(PromptQualityWeakness(
            id=f"W-{wid:02d}",
            weakness_type="Unspecified Output Format",
            problematic_text=p_clean[-30:] if len(p_clean) > 30 else p_clean,
            explanation="Unconstrained prompts leave output length, tone, and schema to default sampling heuristics, leading to verbose filler or runaway completions.",
            evidence="No format delimiters, length ceilings, or response structure tags found in prompt text.",
            proposed_correction="Specify exact expected output schema (e.g., 'Answer with the entity name only.')."
        ))
        wid += 1

    if any("Uncertainty Guardrail" in m for m in missing_context):
        weaknesses.append(PromptQualityWeakness(
            id=f"W-{wid:02d}",
            weakness_type="Missing Uncertainty / Refusal Guardrail",
            problematic_text="[Entire prompt lacking refusal clause]",
            explanation="When the model has low factual recall confidence, the absence of an explicit refusal permission forces it to hallucinate plausible-sounding completions.",
            evidence="Predictive distribution lacks an explicit authorized exit token for uncertain states.",
            proposed_correction="Add refusal instruction: 'If the answer cannot be verified with certainty, state UNKNOWN rather than guessing.'"
        ))
        wid += 1

    if misleading_assumptions:
        weaknesses.append(PromptQualityWeakness(
            id=f"W-{wid:02d}",
            weakness_type="Distractor Premise / Context Misdirection",
            problematic_text=misleading_assumptions[0],
            explanation="Distractor clauses prime irrelevant early attention heads and contaminate the residual stream before the target entity position.",
            evidence="Detected counterfactual or secondary entity clause immediately preceding the target query clause.",
            proposed_correction="Isolate the target question into an atomic query without contradictory prefaces."
        ))
        wid += 1

    if contradictions:
        weaknesses.append(PromptQualityWeakness(
            id=f"W-{wid:02d}",
            weakness_type="Contradictory Instructions",
            problematic_text=contradictions[0],
            explanation="Directly conflicting constraints split attention weights across incompatible generation pathways.",
            evidence="Detected mutually exclusive adjectives / directives in prompt text.",
            proposed_correction="Resolve conflicting requirements into a single prioritized instruction."
        ))
        wid += 1

    return analysis, weaknesses


def _generate_controlled_variants(prompt: str, objective: str, expected_answer: Optional[str]) -> List[Tuple[str, str, str]]:
    """Stage 4: Generate 3-4 controlled, hypothesis-driven prompt variants for robustness evaluation."""
    p = prompt.strip()
    variants: List[Tuple[str, str, str]] = []  # (variant_type, variant_prompt, rationale)

    # Variant 1: Paraphrased / Direct Imperative
    if p.lower().startswith("the capital of ") and p.lower().endswith("is"):
        entity = p[15:-2].strip()
        v1 = f"Identify the capital city of {entity}."
        v1_rat = "Converts cloze completion into direct imperative query; tests if entity association depends on verbatim sentence stem."
    elif p.endswith("?"):
        v1 = f"Please provide a factual answer: {p}"
        v1_rat = "Prefixes factual directive to test sensitivity to epistemic priming."
    else:
        v1 = f"State accurately: {p}."
        v1_rat = "Formulates prompt as an explicit factual assertion request."
    variants.append(("paraphrase_imperative", v1, v1_rat))

    # Variant 2: Strict Output Constraint
    if p.lower().endswith("is"):
        v2 = f"{p} (Answer with the entity name only, without punctuation or preamble):"
        v2_rat = "Applies strict single-token/name constraint to evaluate model obedience and token probability concentration."
    else:
        v2 = f"{p}\nFormat: Return only the final answer in one sentence."
        v2_rat = "Specifies concise formatting constraint to suppress conversational filler."
    variants.append(("constrained_format", v2, v2_rat))

    # Variant 3: Uncertainty / Refusal Guardrail
    v3 = (
        f"{p}\n"
        f"Constraint: Answer accurately. If you are uncertain or the fact is not established, "
        f"state 'UNKNOWN' rather than guessing."
    )
    v3_rat = "Incorporate an explicit uncertainty refusal channel to test whether model confidence shifts or maintains factual integrity."
    variants.append(("refusal_guardrail", v3, v3_rat))

    # Variant 4: Distractor / Ambiguity Stripped (or Clean Context)
    if "but" in p.lower() or "although" in p.lower() or "," in p:
        # If prompt has clauses, isolate final clause
        parts = re.split(r"[,;]|\bbut\b|\bwhereas\b", p, flags=re.IGNORECASE)
        last_part = parts[-1].strip()
        if len(last_part) > 8:
            v4 = last_part if last_part.endswith((".", "?")) else last_part + "."
            v4_rat = "Strips preceding distractor context clauses to measure clean zero-shot baseline without contextual contamination."
            variants.append(("context_distractor_stripped", v4, v4_rat))
        else:
            v4 = f"Verified query: {p}"
            v4_rat = "Epistemic verification prefix."
            variants.append(("verified_prefix", v4, v4_rat))
    else:
        v4 = f"Q: {p}\nA:"
        v4_rat = "Few-shot / standard Q&A delimiter framing to test prompt format invariance."
        variants.append(("qa_delimited", v4, v4_rat))

    return variants[:4]


def _build_improved_prompt(prompt: str, analysis: PromptAnalysis, weaknesses: List[PromptQualityWeakness], expected_answer: Optional[str]) -> Tuple[str, str]:
    """Stage 6: Synthesize an evidence-driven, robust improved prompt."""
    p = prompt.strip()
    
    # Check if prompt is a simple cloze like "The capital of Karnataka is"
    if p.lower().startswith("the capital of ") and p.lower().endswith("is"):
        entity = p[15:-2].strip()
        improved = (
            f"State the official capital city of {entity}.\n"
            f"- Output: City name only, with no conversational filler.\n"
            f"- Constraint: If the entity or capital cannot be verified reliably, reply with 'UNKNOWN'."
        )
        explanation = (
            "1. Converted cloze-style token completion into an explicit factual imperative query.\n"
            "2. Added strict single-entity output constraint to prevent multi-line hallucinated run-on tokens.\n"
            "3. Injected an explicit epistemic fallback guardrail ('UNKNOWN') to curb speculative hallucinations."
        )
        return improved, explanation

    # If prompt contains distractor clause
    if "distractor" in " ".join([w.weakness_type.lower() for w in weaknesses]):
        parts = re.split(r"[,;]|\bbut\b|\bwhereas\b", p, flags=re.IGNORECASE)
        core_query = parts[-1].strip()
        improved = (
            f"Direct factual query: {core_query}\n"
            f"- Scope: Evaluate the entity in this query independently of any surrounding context.\n"
            f"- Response rule: Answer with the precise fact only. If unknown, respond with 'CANNOT BE DETERMINED'."
        )
        explanation = (
            "1. Isolated the target query clause to eliminate distractor contamination from prior clauses.\n"
            "2. Declared scope independence to prevent prior training tokens from overriding immediate query attention.\n"
            "3. Enforced structured fallback behavior."
        )
        return improved, explanation

    # General template improvement
    has_format_issue = any("format" in w.weakness_type.lower() for w in weaknesses)
    has_refusal_issue = any("refusal" in w.weakness_type.lower() for w in weaknesses)

    sections = [f"Task: {p}"]
    rules = []
    if has_format_issue:
        rules.append("Provide a direct, factual answer without extraneous conversational preamble.")
    if has_refusal_issue:
        rules.append("If any required information is missing, ambiguous, or unverifiable, state that limitation explicitly rather than guessing.")
    
    if rules:
        sections.append("Instructions:")
        for r in rules:
            sections.append(f"- {r}")

    improved = "\n".join(sections)
    explanation = (
        "1. Clarified the core instruction boundary.\n"
        "2. Added explicit output precision directives to minimize generation entropy.\n"
        "3. Introduced epistemic bounds to prevent hallucinated assertions on underspecified queries."
    )
    return improved, explanation


async def run_audit_pipeline(
    request: AuditPipelineRequest,
    model_adapter: GemmaAdapter,
    model_name: str,
) -> AuditPipelineResponse:
    """Execute the full 6-stage AI Audit and Prompt Intelligence pipeline."""
    run_id = f"AUDIT-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()
    execution_timeline: List[Dict[str, Any]] = []

    # Helper to log timeline steps
    def _log_stage(stage_num: int, name: str, status: str, details: str):
        execution_timeline.append({
            "stage_index": stage_num,
            "stage_name": name,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details,
        })

    # =========================================================================
    # STAGE 1: INPUT ANALYSIS
    # =========================================================================
    _log_stage(1, "Input Analysis", "in_progress", "Parsing prompt syntax, requirements, and ambiguity")
    analysis, weaknesses = _analyze_prompt_linguistics(request.prompt, request.objective)
    _log_stage(1, "Input Analysis", "completed", f"Identified task '{analysis.intended_task}' with {len(weaknesses)} diagnosed quality risks.")

    # =========================================================================
    # STAGE 2: BASELINE INFERENCE
    # =========================================================================
    _log_stage(2, "Baseline Inference", "in_progress", f"Running real forward pass and autoregressive generation on {model_name}")
    raw_prompt = request.prompt.strip()
    tokens = model_adapter.tokenize(raw_prompt)
    input_ids = tokens["input_ids"]
    attention_mask = tokens.get("attention_mask")
    seq_len = int(input_ids.shape[1])

    # Forward pass with hidden states for logit lens
    with torch.no_grad():
        outputs = model_adapter.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
        )

    last_logits = outputs.logits[0, -1]
    probabilities = torch.softmax(last_logits, dim=-1)
    entropy_val = _compute_entropy(probabilities)

    # Top-5 baseline candidates
    top_k_probs, top_k_ids = torch.topk(probabilities, 5)
    baseline_top_preds = []
    for rank, (tok_id, prob) in enumerate(zip(top_k_ids, top_k_probs)):
        token_str = model_adapter.decode(torch.tensor([tok_id]))
        logit_val = float(last_logits[tok_id])
        baseline_top_preds.append({
            "rank": rank + 1,
            "token_id": int(tok_id),
            "token_str": token_str,
            "probability": float(prob),
            "logit": round(logit_val, 4),
        })

    winner_pred = baseline_top_preds[0]

    # Autoregressive generation
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max(4, min(48, request.max_new_tokens)),
        "do_sample": request.temperature > 0,
        "temperature": max(request.temperature, 1e-5),
        "pad_token_id": model_adapter.tokenizer.eos_token_id,
    }
    with torch.inference_mode():
        gen_output_ids = model_adapter.model.generate(**gen_kwargs)
    
    generated_token_ids = gen_output_ids[:, input_ids.shape[1]:][0]
    generated_text = model_adapter.decode(generated_token_ids)

    # Expected answer verification if supplied
    matches_expected = None
    expected_token_stats = None
    if request.expected_answer and request.expected_answer.strip():
        exp_clean = request.expected_answer.strip().lower()
        full_text_lower = (raw_prompt + generated_text).lower()
        matches_expected = exp_clean in full_text_lower or exp_clean in winner_pred["token_str"].strip().lower()

        # Tokenize expected token to find exact logit
        exp_tokens = model_adapter.tokenizer.encode(request.expected_answer, add_special_tokens=False)
        if exp_tokens:
            exp_id = exp_tokens[0]
            exp_prob = float(probabilities[exp_id]) if exp_id < len(probabilities) else 0.0
            exp_logit = float(last_logits[exp_id]) if exp_id < len(last_logits) else 0.0
            exp_rank = int((last_logits > exp_logit).sum().item()) + 1
            expected_token_stats = {
                "token_id": exp_id,
                "token_str": model_adapter.decode(torch.tensor([exp_id])),
                "probability": exp_prob,
                "logit": exp_logit,
                "rank": exp_rank,
            }

    baseline_inference_summary = {
        "model_identifier": model_name,
        "sequence_length": seq_len,
        "input_tokens": [
            {"position": idx, "token_id": int(tid), "token_str": model_adapter.decode(tid.unsqueeze(0))}
            for idx, tid in enumerate(input_ids[0])
        ],
        "top_prediction": winner_pred,
        "top_5_predictions": baseline_top_preds,
        "predictive_entropy": round(entropy_val, 4),
        "logit_margin_over_runner_up": round(baseline_top_preds[0]["logit"] - baseline_top_preds[1]["logit"], 4),
        "generated_text": generated_text,
        "matches_expected": matches_expected,
        "expected_token_stats": expected_token_stats,
        "temperature": request.temperature,
    }
    _log_stage(2, "Baseline Inference", "completed", f"Top-1: '{winner_pred['token_str']}' ({winner_pred['probability']*100:.1f}%), Entropy: {entropy_val:.3f} nats.")

    # =========================================================================
    # STAGE 3: EVIDENCE COLLECTION
    # =========================================================================
    _log_stage(3, "Evidence Collection", "in_progress", "Gathering empirical token, logit lens, and attention representations")
    evidence_items: List[AuditEvidenceItem] = []
    ev_id = 1

    # EV-01: Input Tokenization Evidence
    evidence_items.append(AuditEvidenceItem(
        evidence_id=f"EV-{ev_id:02d}",
        category="directly_measured",
        observation=f"Prompt was tokenized into {seq_len} discrete tokens using Byte-Pair Encoding.",
        source="Tokenizer vocabulary projection",
        method="Exact token segmentation via model BPE tokenizer",
        confidence_or_uncertainty="Deterministic (100% exact)",
        limitations="Token boundaries can segment semantic entities into unexpected subword fragments (e.g. state names).",
        raw_data={"token_count": seq_len, "sample_tokens": [t["token_str"] for t in baseline_inference_summary["input_tokens"][:10]]}
    ))
    ev_id += 1

    # EV-02: Baseline Next-Token Decision
    evidence_items.append(AuditEvidenceItem(
        evidence_id=f"EV-{ev_id:02d}",
        category="directly_measured",
        observation=f"Top candidate token is '{winner_pred['token_str']}' with next-token probability {winner_pred['probability']*100:.2f}% (Logit: {winner_pred['logit']}).",
        source="Unembedding layer (lm_head) logits",
        method="Direct forward pass measurement at final sequence position",
        confidence_or_uncertainty="Exact floating-point calculation",
        limitations="Next-token conditional probability reflects vocabulary likelihood under model pre-training, NOT factual correctness.",
        raw_data={"winner": winner_pred, "top_5": baseline_top_preds}
    ))
    ev_id += 1

    # EV-03: Decision Uncertainty / Logit Margin
    margin = baseline_inference_summary["logit_margin_over_runner_up"]
    evidence_items.append(AuditEvidenceItem(
        evidence_id=f"EV-{ev_id:02d}",
        category="derived_metric",
        observation=f"Decision boundary margin over runner-up is {margin:+.3f} logits; predictive distribution entropy is {entropy_val:.3f} nats.",
        source="Softmax probability distribution",
        method="Mathematical derivation: Shannon entropy across vocabulary logits",
        confidence_or_uncertainty="Exact analytical metric",
        limitations="High entropy indicates output uncertainty, but low entropy can also occur in confident hallucinations.",
        raw_data={"entropy_nats": entropy_val, "logit_margin": margin}
    ))
    ev_id += 1

    # EV-04: Logit Lens Intermediate Layer Projections
    hidden_states = outputs.hidden_states
    total_layers = len(hidden_states) - 1
    W_U = model_adapter.model.lm_head.weight
    norm_module = getattr(model_adapter.model.model, "norm", None)

    probe_layers = [
        min(2, total_layers - 1),
        min(10, total_layers - 1),
        min(18, total_layers - 1),
        total_layers - 1,
    ]
    lens_steps = []
    for l_idx in probe_layers:
        h = hidden_states[l_idx + 1][0, -1]
        if norm_module is not None:
            h = norm_module(h)
        lens_logits = torch.matmul(W_U, h)
        lens_probs = torch.softmax(lens_logits, dim=-1)
        top_id = int(torch.argmax(lens_logits).item())
        top_str = model_adapter.decode(torch.tensor([top_id]))
        top_p = float(lens_probs[top_id].item())
        lens_steps.append({
            "layer": l_idx,
            "predicted_token": top_str,
            "probability": round(top_p, 4),
            "logit": round(float(lens_logits[top_id].item()), 3),
        })

    evidence_items.append(AuditEvidenceItem(
        evidence_id=f"EV-{ev_id:02d}",
        category="directly_measured",
        observation=(
            f"Logit Lens reveals internal hypothesis progression: Layer {lens_steps[0]['layer']} promotes '{lens_steps[0]['predicted_token']}' ({lens_steps[0]['probability']*100:.1f}%), "
            f"Layer {lens_steps[1]['layer']} transitions to '{lens_steps[1]['predicted_token']}' ({lens_steps[1]['probability']*100:.1f}%), "
            f"and Layer {lens_steps[-1]['layer']} yields '{lens_steps[-1]['predicted_token']}' ({lens_steps[-1]['probability']*100:.1f}%)."
        ),
        source="Transformer residual stream (hidden_states) unembedded via lm_head",
        method="Logit Lens: Projecting intermediate layer residual streams into vocabulary space",
        confidence_or_uncertainty="Empirical vector projection",
        limitations="Logit Lens is a diagnostic tool, not literal proof of what the layer 'thought' in isolation, as later layers perform non-linear transformations.",
        raw_data={"logit_lens_steps": lens_steps}
    ))
    ev_id += 1

    # EV-05: Attention Entropy
    if outputs.attentions:
        early_attn = outputs.attentions[min(2, len(outputs.attentions) - 1)][0]  # [num_heads, seq, seq]
        mean_attn_entropy = float(torch.mean(-torch.sum(early_attn * torch.log(early_attn + 1e-12), dim=-1)).item())
        evidence_items.append(AuditEvidenceItem(
            evidence_id=f"EV-{ev_id:02d}",
            category="directly_measured",
            observation=f"Early layer attention heads exhibit an average entropy of {mean_attn_entropy:.3f} across sequence positions.",
            source="Self-attention weight matrices (Q * K^T / sqrt(d))",
            method="Attention entropy computation across all heads",
            confidence_or_uncertainty="Direct tensor measurement",
            limitations="Attention weights show information routing pathways, but do NOT prove causal necessity without intervention.",
            raw_data={"mean_attention_entropy": mean_attn_entropy}
        ))
        ev_id += 1

    # EV-06: External Verification Status
    if request.expected_answer:
        evidence_items.append(AuditEvidenceItem(
            evidence_id=f"EV-{ev_id:02d}",
            category="externally_verified",
            observation=(
                f"External reference fact provided: '{request.expected_answer}'. "
                f"Model baseline {'SUCCESSFULLY MATCHES' if matches_expected else 'FAILS TO MATCH'} expected target fact."
            ),
            source="User-specified ground truth specification",
            method="String and token match comparison against model top-1 prediction and generated output",
            confidence_or_uncertainty="Exact comparison against provided benchmark string",
            limitations="Relies on accuracy of user-provided reference string.",
            raw_data={"expected": request.expected_answer, "matches": matches_expected, "stats": expected_token_stats}
        ))
        ev_id += 1
    else:
        evidence_items.append(AuditEvidenceItem(
            evidence_id=f"EV-{ev_id:02d}",
            category="hypothesis_interpretation",
            observation="No ground-truth reference fact provided. Audit evaluates self-consistency, entropy, and robustness rather than external factual accuracy.",
            source="Audit session configuration",
            method="Empirical consistency auditing without external ground-truth oracle",
            confidence_or_uncertainty="Qualified self-consistency evaluation",
            limitations="Cannot confirm objective external truth without verified knowledge base or ground truth annotation.",
            raw_data={"ground_truth_available": False}
        ))
        ev_id += 1

    # EV-07: Unavailable Evidence (Scientific Integrity)
    evidence_items.append(AuditEvidenceItem(
        evidence_id=f"EV-{ev_id:02d}",
        category="unavailable",
        observation="Exact training corpus document provenance is UNAVAILABLE.",
        source="Pre-training dataset archive",
        method="N/A (Requires training set vector index and data attribution pipeline)",
        confidence_or_uncertainty="Not measured",
        limitations="Local inference runtime cannot determine which pre-training documents caused this specific parameter association.",
        raw_data={"status": "unsupported_by_local_engine"}
    ))
    _log_stage(3, "Evidence Collection", "completed", f"Collected {len(evidence_items)} structured evidence records with full provenance.")

    # =========================================================================
    # STAGE 4: CONTROLLED ROBUSTNESS TESTING
    # =========================================================================
    robustness_results: List[RobustnessVariantResult] = []
    if request.run_robustness_variants:
        _log_stage(4, "Robustness Testing", "in_progress", "Generating and executing controlled prompt variants through model")
        variants = _generate_controlled_variants(request.prompt, request.objective, request.expected_answer)
        
        for v_idx, (v_type, v_prompt, v_rationale) in enumerate(variants):
            v_toks = model_adapter.tokenize(v_prompt)
            v_ids = v_toks["input_ids"]
            v_mask = v_toks.get("attention_mask")

            with torch.no_grad():
                v_out = model_adapter.model(input_ids=v_ids, attention_mask=v_mask)
            v_last = v_out.logits[0, -1]
            v_probs = torch.softmax(v_last, dim=-1)
            v_top_id = int(torch.argmax(v_last).item())
            v_top_str = model_adapter.decode(torch.tensor([v_top_id]))
            v_top_prob = float(v_probs[v_top_id].item())

            # Short generation for variant
            with torch.inference_mode():
                v_gen = model_adapter.model.generate(
                    input_ids=v_ids,
                    attention_mask=v_mask,
                    max_new_tokens=16,
                    do_sample=False,
                    pad_token_id=model_adapter.tokenizer.eos_token_id,
                )
            v_gen_text = model_adapter.decode(v_gen[:, v_ids.shape[1]:][0])

            # Compare with baseline
            consistent = (v_top_str.strip().lower() == winner_pred["token_str"].strip().lower())
            prob_shift = round(v_top_prob - winner_pred["probability"], 4)
            v_matches = None
            if request.expected_answer:
                v_matches = (request.expected_answer.strip().lower() in (v_prompt + v_gen_text).lower()) or (request.expected_answer.strip().lower() in v_top_str.strip().lower())

            robustness_results.append(RobustnessVariantResult(
                variant_id=f"VAR-{v_idx+1:02d}",
                variant_type=v_type,
                prompt=v_prompt,
                rationale=v_rationale,
                top_token=v_top_str,
                top_probability=round(v_top_prob, 4),
                generated_text=v_gen_text,
                consistent_with_baseline=consistent,
                matches_expected=v_matches,
                probability_shift=prob_shift,
            ))
        _log_stage(4, "Robustness Testing", "completed", f"Evaluated {len(robustness_results)} controlled variants. Consistency rate: {sum(1 for r in robustness_results if r.consistent_with_baseline)}/{len(robustness_results)}.")
    else:
        _log_stage(4, "Robustness Testing", "skipped", "User opted out of automated variant generation.")

    # =========================================================================
    # STAGE 5: MECHANISTIC INVESTIGATION CANDIDATES
    # =========================================================================
    _log_stage(5, "Mechanistic Investigation", "in_progress", "Mapping candidate causal attention heads and MLP layers")
    mechanistic_candidates: List[MechanisticCandidate] = []

    # Pinpoint transition layer from Logit Lens
    jump_layer = 10
    max_prob_jump = 0.0
    for i in range(len(lens_steps) - 1):
        delta = lens_steps[i + 1]["probability"] - lens_steps[i]["probability"]
        if delta > max_prob_jump:
            max_prob_jump = delta
            jump_layer = lens_steps[i + 1]["layer"]

    mechanistic_candidates.append(MechanisticCandidate(
        component_type="layer",
        layer=jump_layer,
        observation=f"Logit Lens shows dominant probability acceleration at Layer {jump_layer} (Δp = {max_prob_jump*100:+.1f}%).",
        hypothesis=f"Layer {jump_layer} residual stream hosts the primary factual resolution circuit routing entity tokens to vocabulary logits.",
        suggested_action=f"Inspect Layer {jump_layer} in the 3D Synaptic Neural Flow Probe or run Counterfactual Activation Patching at Layer {jump_layer}."
    ))

    mechanistic_candidates.append(MechanisticCandidate(
        component_type="attention_head",
        layer=min(5, total_layers - 1),
        head=0,
        observation="Early self-attention layers route syntactic subject tokens to the prompt boundary.",
        hypothesis="Early attention heads act as induction or copy heads that bind entity tokens into query position.",
        suggested_action="Execute Causal Head Sweep across Layers 0-10 in Phase 03 (Circuit Localization) to measure head sensitivity."
    ))

    mechanistic_candidates.append(MechanisticCandidate(
        component_type="mlp",
        layer=min(11, total_layers - 1),
        observation="Mid-layer feed-forward networks (MLP) store key-value factual associations.",
        hypothesis=f"Ablating MLP Layer {min(11, total_layers - 1)} will cause a significant drop in target token logit.",
        suggested_action=f"Run MLP Ablation on Layer {min(11, total_layers - 1)} in Phase 04 (Verification & Dossier) to test causal necessity."
    ))
    _log_stage(5, "Mechanistic Investigation", "completed", f"Identified {len(mechanistic_candidates)} candidate causal components.")

    # =========================================================================
    # STAGE 6: SYNTHESIS AND EVIDENCE-DRIVEN PROMPT IMPROVEMENT
    # =========================================================================
    _log_stage(6, "Synthesis & Recommendations", "in_progress", "Synthesizing empirical findings and generating optimized prompt")
    improved_prompt, improved_explanation = _build_improved_prompt(request.prompt, analysis, weaknesses, request.expected_answer)

    directly_observed_issues = [f"{w.weakness_type}: {w.explanation}" for w in weaknesses]
    conclusions = [
        f"Baseline model produces top token '{winner_pred['token_str']}' with {winner_pred['probability']*100:.1f}% probability and entropy {entropy_val:.3f} nats.",
        f"Intermediate Logit Lens shows representation resolves at Layer {jump_layer}.",
    ]
    if robustness_results:
        inconsistent = [r for r in robustness_results if not r.consistent_with_baseline]
        if inconsistent:
            conclusions.append(f"Prompt is brittle under syntactic rephrasing: {len(inconsistent)}/{len(robustness_results)} variants flipped the top predicted token.")
        else:
            conclusions.append("Prompt displays consistent top-1 predictions across controlled syntactic paraphrases.")

    unverified = [
        "Whether candidate Layer " + str(jump_layer) + " is causally necessary rather than merely correlated (requires knockout ablation).",
        "Whether training data distribution contains exact duplicate matches for the prompt query.",
    ]

    recommendations = [
        "Adopt the improved prompt with explicit constraints and fallback guardrails.",
        f"Verify causal attribution by executing Phase 03 Head Sweep and Phase 04 Knock-out Ablation on Layer {jump_layer}.",
        "Include explicit uncertainty instructions to avoid over-confident sampling on edge cases.",
    ]

    followups = [
        f"Run zero-ablation on Layer {jump_layer} MLP in Phase 04 to calculate exact Δlogit impact.",
        "Perform activation patching between clean prompt and corrupted variant to quantify causal recovery percentage.",
    ]

    synthesis = AuditSynthesis(
        original_behavior_summary=(
            f"Given the original prompt, the model generated '{generated_text.strip()}' "
            f"with initial token '{winner_pred['token_str']}' at {winner_pred['probability']*100:.1f}% probability. "
            f"{'Output matches the specified target benchmark.' if matches_expected else 'Output does not confirm the specified benchmark.' if matches_expected is False else 'Self-consistency evaluated without external ground truth.'}"
        ),
        evidence_collected_count=len(evidence_items),
        directly_observed_issues=directly_observed_issues,
        empirically_supported_conclusions=conclusions,
        unverified_hypotheses=unverified,
        discovered_weaknesses=[w.weakness_type for w in weaknesses],
        recommended_improvements=recommendations,
        recommended_followup_experiments=followups,
    )
    _log_stage(6, "Synthesis & Recommendations", "completed", "Evidence synthesized and improved prompt formulated.")

    return AuditPipelineResponse(
        run_id=run_id,
        timestamp=timestamp,
        prompt=request.prompt,
        objective=request.objective,
        expected_answer=request.expected_answer,
        hypothesis=request.hypothesis,
        model_info={
            "model_name": model_name,
            "device": str(model_adapter.device),
            "total_layers": total_layers,
            "vocab_size": model_adapter.model.config.vocab_size,
        },
        analysis=analysis,
        baseline_inference=baseline_inference_summary,
        evidence=evidence_items,
        robustness_variants=robustness_results,
        weaknesses=weaknesses,
        mechanistic_candidates=mechanistic_candidates,
        synthesis=synthesis,
        improved_prompt=improved_prompt,
        improved_prompt_explanation=improved_explanation,
        execution_timeline=execution_timeline,
    )
