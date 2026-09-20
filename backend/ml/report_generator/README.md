# PD-VoiceNet Report Generator

RAG + LLM clinical report generator for PD-VoiceNet dashboard payloads.

## Architecture

```
report_generator/
├── knowledge_base/
│   ├── seed_documents.md       # Verified reference text (8 chunks)
│   ├── build_index.py          # Embeds + indexes seed_documents.md
│   ├── vector_index.npz        # Pre-built embeddings (generated)
│   └── chunks.json             # Chunk metadata (generated)
├── retriever.py                # Local vector retrieval (sentence-transformers)
├── llm_provider.py             # Swappable LLM interface (Anthropic default)
├── report_templates.py         # Deterministic fallback (no LLM needed)
├── generate_report.py          # Main orchestrator
├── app_endpoint.py             # Standalone Flask endpoint (port 5050)
├── tests/
│   └── test_safety_constraints.py
└── README.md
```

## How to Run

### 1. Build the vector index (first time only)
```bash
cd backend/ml/report_generator
python knowledge_base/build_index.py
```

### 2. Start the standalone endpoint
```bash
python app_endpoint.py
# Runs on http://localhost:5050
```

### 3. Make a request
```bash
# Clinician mode
curl -X POST http://localhost:5050/generate_report \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "subject_id": "SUBJ_042",
      "prediction": "PD",
      "prediction_set": ["PD"],
      "refer_to_clinician": false,
      "modality_gate_alpha": 0.73,
      "modality_dominance": "SSL",
      "attention_weights": [0.20, 0.80],
      "probability_hc": 0.08,
      "probability_pd": 0.92
    },
    "audience": "clinician"
  }'

# Patient mode
curl -X POST http://localhost:5050/generate_report \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "subject_id": "SUBJ_042",
      "prediction": "Uncertain",
      "prediction_set": ["HC", "PD"],
      "refer_to_clinician": true,
      "modality_gate_alpha": 0.42,
      "modality_dominance": "Biomarker",
      "attention_weights": [0.35, 0.65],
      "probability_hc": 0.52,
      "probability_pd": 0.48
    },
    "audience": "patient"
  }'
```

### 4. Run safety tests
```bash
python tests/test_safety_constraints.py
```

## Knowledge Base Scope

> **Important:** The knowledge base is intentionally minimal and scoped to
> explaining the MODEL'S OUTPUT, not Parkinson's disease as a condition.
> It contains only:
> - Definitions of MFCC, Jitter, Shimmer, HNR, and SHAP from the base
>   paper (Shen, Mortezaagha & Rahgozar, 2025, Scientific Reports 15:11687)
> - Explanations of PD-VoiceNet's alpha (modality dominance) and conformal
>   prediction set outputs from the architecture specification
> - A decision-support disclaimer
>
> No PD symptom lists, treatment information, severity scales, or
> epidemiological statistics are included. Adding such content requires
> explicit sourcing from a cited reference document.

## LLM Fallback Behavior

If `ANTHROPIC_API_KEY` is not set (or the API is unavailable), the system
automatically falls back to a deterministic, template-based report. This
template satisfies all safety constraints without any external dependency.

Set the API key to enable LLM-generated reports:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Safety Constraints

All generated reports (both LLM and template) enforce these constraints:

1. **No ungrounded claims** — Only facts from the retrieval corpus or payload values
2. **Not a diagnosis** — Every report includes decision-support framing
3. **Uncertainty handling** — Non-singleton prediction sets always trigger referral language
4. **Patient-safe tone** — Patient-mode reports avoid disproportionate alarm
5. **No fabricated citations** — Only references from the actual retrieval corpus
6. **Always-available fallback** — Template-based reports work with zero external dependencies

## Example Generated Reports

### Example 1: High-Confidence PD (Template Fallback, Clinician)

```
============================================================
PD-VoiceNet Clinical Decision-Support Report
============================================================

Subject ID: TEST_002
Prediction Set: {PD}
Primary Classification: PD
Calibrated Probabilities: P(HC) = 8.0%, P(PD) = 92.0%

--- Modality Analysis ---
Modality Gate Alpha (α): 0.730
Dominant Modality: SSL
Interpretation: This prediction relied primarily on deep audio
patterns extracted by the Wav2Vec2 SSL branch (α > 0.5).

--- Attention Weights (per recording) ---
  Recording 1: 0.2000
  Recording 2: 0.8000

--- Disclaimer ---
IMPORTANT: This is a clinical decision-support signal, NOT a
diagnosis. PD-VoiceNet outputs are intended to assist qualified
clinicians in their assessment and must not be used as a standalone
basis for diagnosis or treatment decisions. All results should be
reviewed in the context of a full clinical evaluation.
```

### Example 2: Uncertain Result (Template Fallback, Patient)

```
============================================================
Your Voice Analysis Results
============================================================

Subject ID: TEST_001

Result: Your voice analysis results were inconclusive.

What this means: Our voice analysis tool was not able to reach a
clear conclusion from your recording. This can happen for many
reasons and does not necessarily indicate a health concern.

Next steps: We recommend that you discuss these results with your
doctor, who can provide a full evaluation and determine whether
any follow-up is needed. Your doctor is the best person to put
these results in context with the rest of your health information.

How the analysis worked: The tool primarily analyzed specific
measurable characteristics of your voice (such as steadiness and
clarity) to reach its conclusion.

Please remember: This voice analysis is a decision-support tool,
not a medical diagnosis. It is designed to help your doctor, not
to replace their judgment. Always consult with a qualified
healthcare professional about your health.
```

### Example 3: Uncertain Result (Template Fallback, Clinician)

```
============================================================
PD-VoiceNet Clinical Decision-Support Report
============================================================

Subject ID: TEST_001
Prediction Set: {HC, PD}
Primary Classification: Uncertain
Calibrated Probabilities: P(HC) = 52.0%, P(PD) = 48.0%

--- Modality Analysis ---
Modality Gate Alpha (α): 0.420
Dominant Modality: Biomarker
Interpretation: This prediction relied primarily on classical
acoustic biomarkers (jitter, shimmer, HNR, MFCCs) from the
biomarker branch (α ≤ 0.5).

--- Attention Weights (per recording) ---
  Recording 1: 0.3500
  Recording 2: 0.6500

*** UNCERTAINTY FLAG ***
The conformal prediction set is {HC, PD}, indicating the model
could not confidently distinguish between Parkinson's disease and
healthy control for this subject. Clinical follow-up and additional
assessment are recommended.

--- Disclaimer ---
IMPORTANT: This is a clinical decision-support signal, NOT a
diagnosis. PD-VoiceNet outputs are intended to assist qualified
clinicians in their assessment and must not be used as a standalone
basis for diagnosis or treatment decisions. All results should be
reviewed in the context of a full clinical evaluation.
```

## Test Results

All 4 safety constraint tests pass:

```
============================================================
SUMMARY
============================================================
  [PASS] Test 1: Uncertain → referral language: PASSED
  [PASS] Test 2: High-confidence → still disclaims: PASSED
  [PASS] Test 3: No fabricated content (denylist): PASSED
  [PASS] Test 4: Template fallback w/o API key: PASSED

ALL TESTS PASSED
```

**Test 1** verified that uncertain payloads (`refer_to_clinician=True`) produce
reports containing referral keywords (follow-up, clinician, review, etc.) in
both clinician and patient modes.

**Test 2** verified that even high-confidence PD predictions still include
"not a diagnosis" / decision-support framing in both modes.

**Test 3** ran a denylist check across all 6 payload×audience combinations,
confirming no fabricated medical content (levodopa, prevalence stats, treatment
names, severity scales, etc.) appears in any report.

**Test 4** temporarily unset `ANTHROPIC_API_KEY` and confirmed
`generate_report()` correctly fell back to `template_fallback` with
`model_used=None`.
