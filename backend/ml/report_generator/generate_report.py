"""
generate_report.py — Step 5: Orchestrates retrieval + LLM (or fallback) report generation.

This is the main entry point for generating clinical reports from
PD-VoiceNet dashboard payloads. It:
  1. Builds a retrieval query from the payload
  2. Retrieves relevant knowledge base chunks
  3. Checks for an available LLM provider
  4. Falls back to template-based generation if no LLM is available
  5. Returns a fully auditable response dict
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from retriever import retrieve
from llm_provider import get_provider
from report_templates import generate_template_report


# Safety constraints, repeated verbatim in the system prompt
SAFETY_CONSTRAINTS = """
SAFETY CONSTRAINTS — You MUST follow ALL of these without exception:

1. You must NEVER state a clinical fact, statistic, or claim that is not
   either (a) directly present in the RETRIEVED CONTEXT provided below,
   or (b) a value taken directly from the STRUCTURED PAYLOAD provided
   below. If you are tempted to include a claim not traceable to one of
   these two sources, OMIT it entirely.

2. You must NEVER present the model's prediction as a diagnosis. Every
   report must include language framing this as a decision-support signal
   requiring clinical correlation, not a diagnostic conclusion.

3. If the payload's refer_to_clinician flag is True (meaning the prediction
   set is not a singleton — it is {PD,HC} or empty), you MUST explicitly
   state this uncertainty and recommend clinical follow-up. This is
   mandatory, not optional.

4. For patient-mode reports: Do NOT use language that could cause alarm
   disproportionate to the model's actual confidence. Low-confidence or
   uncertain results should be framed calmly (e.g., "your results were
   inconclusive and your doctor should review them with you"), not
   dramatically.

5. NEVER fabricate a citation or reference a source not actually present
   in the RETRIEVED CONTEXT below.
"""


def _build_retrieval_query(payload: dict) -> str:
    """Build a natural language query from the payload for retrieval."""
    parts = []
    
    prediction = payload.get("prediction", "")
    dominance = payload.get("modality_dominance", "")
    refer = payload.get("refer_to_clinician", False)
    
    if refer:
        parts.append("conformal prediction set uncertain non-singleton")
    
    if dominance == "Biomarker":
        parts.append("jitter shimmer HNR acoustic biomarkers")
    else:
        parts.append("deep audio features SSL Wav2Vec2")
    
    parts.append("alpha modality dominance")
    parts.append("decision support not diagnosis")
    
    return " ".join(parts)


def _build_system_prompt(audience: str, retrieved_chunks: list[dict]) -> str:
    """Construct the full system prompt with safety constraints and context."""
    
    if audience == "clinician":
        audience_instruction = (
            "You are writing a clinical decision-support report for a CLINICIAN. "
            "Use technical language. You may reference biomarker names (jitter, "
            "shimmer, HNR, MFCCs) directly. Be precise and data-driven."
        )
    else:
        audience_instruction = (
            "You are writing a voice analysis results summary for a PATIENT. "
            "Use plain, accessible language. Do NOT use technical jargon. "
            "Be calm and reassuring in tone, even for uncertain or flagged results. "
            "Frame everything in a way that is informative without causing "
            "disproportionate alarm."
        )
    
    # Format retrieved context
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[{i}] {chunk['title']}\n"
            f"Source: {chunk['source']}\n"
            f"Content: {chunk['text']}"
        )
    retrieved_context = "\n\n".join(context_parts)
    
    return f"""{audience_instruction}

{SAFETY_CONSTRAINTS}

RETRIEVED CONTEXT (the ONLY background/definitional content you may draw from):

{retrieved_context}

INSTRUCTIONS:
- Use ONLY the numeric values from the STRUCTURED PAYLOAD to state results.
  Never estimate, round creatively, or invent numbers.
- Use ONLY the RETRIEVED CONTEXT above for any definitional or explanatory
  content. Do not add medical facts, statistics, or claims from your own
  knowledge.
- Always include a clear statement that this is a decision-support signal,
  not a diagnosis.
- Keep the report concise (150-300 words for clinician, 100-200 for patient).
- Do not include section headers or formatting — produce flowing prose.
"""


def _build_user_prompt(payload: dict) -> str:
    """Build the user prompt containing the structured payload."""
    
    lines = ["Generate a report for the following voice analysis result:"]
    lines.append("")
    lines.append("STRUCTURED PAYLOAD:")
    lines.append(f"  Subject ID: {payload.get('subject_id', 'Unknown')}")
    lines.append(f"  Primary Classification: {payload.get('prediction', 'Unknown')}")
    lines.append(f"  Prediction Set: {payload.get('prediction_set', [])}")
    lines.append(f"  Refer to Clinician: {payload.get('refer_to_clinician', False)}")
    lines.append(f"  Modality Gate Alpha: {payload.get('modality_gate_alpha', 0.0):.3f}")
    lines.append(f"  Modality Dominance: {payload.get('modality_dominance', 'Unknown')}")
    
    prob_hc = payload.get("probability_hc")
    prob_pd = payload.get("probability_pd")
    if prob_hc is not None and prob_pd is not None:
        lines.append(f"  P(HC): {prob_hc:.4f}")
        lines.append(f"  P(PD): {prob_pd:.4f}")
    
    attn = payload.get("attention_weights", [])
    if attn:
        lines.append(f"  Attention Weights: {attn}")
    
    return "\n".join(lines)


def generate_report(payload: dict, audience: str = "clinician") -> dict:
    """Generate a clinical report from a dashboard payload.
    
    Args:
        payload: Dashboard payload dict from build_dashboard_payload().
        audience: "clinician" or "patient".
    
    Returns:
        dict with keys:
            - report: The generated report text
            - generation_method: "llm" or "template_fallback"
            - source_payload: The input payload (for auditability)
            - retrieved_chunks: List of retrieved context chunks (if LLM used)
            - model_used: LLM model name (if LLM used)
    """
    # Step 1: Build retrieval query
    query = _build_retrieval_query(payload)
    
    # Step 2: Retrieve relevant chunks
    try:
        retrieved_chunks = retrieve(query, k=3)
    except Exception:
        retrieved_chunks = []
    
    # Step 3: Check for LLM provider
    provider = get_provider()
    
    # Step 4: If no provider, use template fallback
    if provider is None:
        report_text = generate_template_report(payload, audience)
        return {
            "report": report_text,
            "generation_method": "template_fallback",
            "source_payload": payload,
            "retrieved_chunks": [],
            "model_used": None
        }
    
    # Step 5: LLM generation
    try:
        system_prompt = _build_system_prompt(audience, retrieved_chunks)
        user_prompt = _build_user_prompt(payload)
        report_text = provider.generate(system_prompt, user_prompt)
        
        return {
            "report": report_text,
            "generation_method": "llm",
            "source_payload": payload,
            "retrieved_chunks": [
                {"text": c["text"], "source": c["source"], "title": c["title"]}
                for c in retrieved_chunks
            ],
            "model_used": getattr(provider, "model", "unknown")
        }
    except Exception as e:
        # LLM call failed — fall back to template
        report_text = generate_template_report(payload, audience)
        return {
            "report": report_text,
            "generation_method": "template_fallback",
            "source_payload": payload,
            "retrieved_chunks": [],
            "model_used": None,
            "llm_error": str(e)
        }
