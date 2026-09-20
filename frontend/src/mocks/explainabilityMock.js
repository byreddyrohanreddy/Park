/**
 * explainabilityMock.js — Mock payloads for frontend development.
 *
 * Shapes match build_extended_dashboard_payload() + report generator response.
 * Two variants: confident PD and uncertain (non-singleton prediction set).
 */

// ─── Mock: Confident PD Prediction ──────────────────────────────────────────
export const mockPayloadPD = {
  // Base dashboard payload (from calibration.py build_dashboard_payload)
  subject_id: "B1ABNINSAC46F240120171753",
  prediction: "PD",
  prediction_set: ["PD"],
  refer_to_clinician: false,
  probability_hc: 0.006,
  probability_pd: 0.994,
  modality_gate_alpha: 0.62,
  modality_dominance: "SSL",
  attention_weights: [0.35, 0.28, 0.22, 0.15],

  // Explainability extensions (from integrate_dashboard.py)
  shap_top5: [
    { feature: "mfcc_5", value: 0.1017 },
    { feature: "f1_bw", value: 0.0945 },
    { feature: "f3_bw", value: 0.0414 },
    { feature: "ppq5_jitter", value: 0.04 },
    { feature: "mfcc_4", value: 0.0375 },
  ],

  grad_cam_cnn: [
    {
      recording_id: "rec_001",
      timestamps_sec: Array.from({ length: 50 }, (_, i) => +(i * 0.02).toFixed(3)),
      importance_values: Array.from({ length: 50 }, () =>
        +(Math.random() * 0.6 + 0.2).toFixed(4)
      ),
    },
    {
      recording_id: "rec_002",
      timestamps_sec: Array.from({ length: 40 }, (_, i) => +(i * 0.02).toFixed(3)),
      importance_values: Array.from({ length: 40 }, () =>
        +(Math.random() * 0.5 + 0.1).toFixed(4)
      ),
    },
  ],

  transformer_saliency: [
    {
      recording_id: "rec_001",
      timestamps_sec: Array.from({ length: 50 }, (_, i) => +(i * 0.02).toFixed(3)),
      importance_values: Array.from({ length: 50 }, () =>
        +(Math.random() * 0.7 + 0.1).toFixed(4)
      ),
    },
    {
      recording_id: "rec_002",
      timestamps_sec: Array.from({ length: 40 }, (_, i) => +(i * 0.02).toFixed(3)),
      importance_values: Array.from({ length: 40 }, () =>
        +(Math.random() * 0.4 + 0.05).toFixed(4)
      ),
    },
  ],
};


// ─── Mock: Uncertain Prediction (non-singleton set) ─────────────────────────
export const mockPayloadUncertain = {
  subject_id: "VE1ABNINSAC46F240120171800",
  prediction: "Uncertain",
  prediction_set: ["HC", "PD"],
  refer_to_clinician: true,
  probability_hc: 0.524,
  probability_pd: 0.476,
  modality_gate_alpha: 0.48,
  modality_dominance: "Biomarker",
  attention_weights: [0.52, 0.48],

  shap_top5: [
    { feature: "mfcc_5", value: 0.112 },
    { feature: "f3_bw", value: -0.0948 },
    { feature: "mfcc_4", value: 0.0882 },
    { feature: "hnr", value: 0.0832 },
    { feature: "rap_jitter", value: -0.0697 },
  ],

  grad_cam_cnn: [
    {
      recording_id: "rec_010",
      timestamps_sec: Array.from({ length: 45 }, (_, i) => +(i * 0.02).toFixed(3)),
      importance_values: Array.from({ length: 45 }, () =>
        +(Math.random() * 0.4 + 0.1).toFixed(4)
      ),
    },
  ],

  transformer_saliency: [
    {
      recording_id: "rec_010",
      timestamps_sec: Array.from({ length: 45 }, (_, i) => +(i * 0.02).toFixed(3)),
      importance_values: Array.from({ length: 45 }, () =>
        +(Math.random() * 0.3 + 0.05).toFixed(4)
      ),
    },
  ],
};


// ─── Mock: Report Generator Responses ───────────────────────────────────────
export const mockReportClinician = {
  report:
    `## Clinical Voice Analysis Report

**Subject:** B1ABNINSAC46F240120171753
**Analysis Date:** ${new Date().toLocaleDateString()}
**Model:** PD-VoiceNet v1.0 (Leave-One-Site-Out validated)

### Prediction Summary
The voice analysis indicates findings **consistent with Parkinson's Disease** with high confidence (P(PD) = 0.994, 95% CI: [0.97, 1.00]).

### Modality Analysis
The prediction is driven primarily by the **SSL (self-supervised learning) branch** (α = 0.62), indicating that spectro-temporal patterns in the raw waveform contributed more than deterministic acoustic biomarkers.

### Key Biomarker Attributions (SHAP)
The following acoustic features most strongly influenced the PD classification:
1. **MFCC-5** (+0.102): Elevated mel-frequency coefficient suggests altered vocal tract resonance
2. **F1 Bandwidth** (+0.095): Widened first formant bandwidth consistent with articulatory imprecision
3. **F3 Bandwidth** (+0.041): Third formant instability
4. **PPQ5 Jitter** (+0.040): Elevated pitch perturbation quotient
5. **MFCC-4** (+0.038): Additional spectral envelope deviation

### Conformal Prediction
The prediction set is a **singleton {PD}**, indicating the model is confident in this classification at the calibrated significance level.

### Recommendation
These findings suggest vocal biomarkers consistent with Parkinson's Disease. Standard clinical follow-up with neurological assessment is recommended to confirm.`,

  generation_method: "llm",
  model_used: "gemini-2.0-flash",
  source_payload: mockPayloadPD,
  retrieved_chunks: [
    "PD patients typically exhibit increased jitter and shimmer values...",
    "MFCC features capture spectral envelope characteristics that differ in PD..."
  ],
};


export const mockReportPatient = {
  report:
    `## Your Voice Analysis Results

**Date:** ${new Date().toLocaleDateString()}

### What the Analysis Found
Your voice recording was analyzed by our AI system, which looks at many different features of your voice — including pitch stability, vocal quality, and speech patterns.

The analysis found patterns in your voice that are **sometimes associated with Parkinson's Disease**. The system is quite confident in this finding (99.4%).

### What This Means
- This is a **screening tool**, not a diagnosis
- The AI detected some changes in your voice quality and pitch control
- These changes can have many causes and don't necessarily mean you have Parkinson's Disease

### What You Should Do Next
We recommend discussing these results with your doctor, who can:
- Review these findings in the context of your overall health
- Perform additional clinical assessments if needed
- Determine whether further testing is appropriate

### Important Note
This analysis is a **decision-support tool** designed to assist healthcare professionals. It is not a substitute for professional medical evaluation.`,

  generation_method: "llm",
  model_used: "gemini-2.0-flash",
  source_payload: mockPayloadPD,
  retrieved_chunks: [],
};


export const mockReportUncertainClinician = {
  report:
    `## Clinical Voice Analysis Report — Inconclusive

**Subject:** VE1ABNINSAC46F240120171800
**Analysis Date:** ${new Date().toLocaleDateString()}

### Prediction Summary
The voice analysis produced an **inconclusive result**. The conformal prediction set is non-singleton {HC, PD}, with near-equal posterior probabilities (P(PD) = 0.476, P(HC) = 0.524).

### Clinical Recommendation
Given the inconclusive result, **clinical referral is recommended**. A longitudinal voice monitoring protocol with repeat assessment in 3-6 months may help resolve ambiguity.`,

  generation_method: "template_fallback",
  model_used: null,
  source_payload: mockPayloadUncertain,
  retrieved_chunks: [],
};
