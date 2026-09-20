"""
report_templates.py — Deterministic template-based clinical report engine.

Generates rich, comprehensive, and tailored clinical decision-support reports 
for both Clinicians (technical, quantitative) and Patients (plain-language, 
educational, reassuring, and actionable).
"""

def generate_template_report(payload: dict, audience: str = "clinician") -> str:
    """Generate a deterministic report from the dashboard payload."""
    if audience == "clinician":
        return _clinician_report(payload)
    elif audience == "patient":
        return _patient_report(payload)
    else:
        raise ValueError(f"Unknown audience: {audience}. Must be 'clinician' or 'patient'.")


FEATURE_EXPLANATIONS = {
    "local_jitter": ("Pitch Stability (Micro-Tremors)", "Measures involuntary cycle-to-cycle frequency variations in vocal fold vibration."),
    "ppq5_jitter": ("Sustained Pitch Steadiness", "Evaluates 5-point pitch period stability over sustained vowel phonation."),
    "rap_jitter": ("Relative Pitch Perturbation", "Detects rapid micro-fluctuations in vocal pitch frequency."),
    "local_shimmer": ("Loudness & Amplitude Steadiness", "Measures cycle-to-cycle sound wave volume consistency as air passes the vocal folds."),
    "apq3_shimmer": ("Short-Term Amplitude Stability", "Quantifies 3-cycle variations in vocal intensity."),
    "apq5_shimmer": ("Mid-Term Volume Stability", "Evaluates loudness stability across 5 consecutive acoustic wave cycles."),
    "apq11_shimmer": ("Long-Term Volume Stability", "Assesses wider amplitude modulation and steadiness in vocal projection."),
    "dda_shimmer": ("Amplitude Differential Perturbation", "Reflects subtle irregularity in vocal cord contact and air control."),
    "hnr": ("Vocal Clarity (Harmonics-to-Noise Ratio)", "Measures the balance of pure acoustic vocal tone against turbulent breathiness or hoarseness."),
    "nhr": ("Breathiness / Noise Ratio", "Assesses acoustic turbulence caused by incomplete vocal cord closure during phonation."),
    "mean_f0": ("Average Vocal Pitch", "Represents the fundamental frequency of the voice."),
    "std_f0": ("Pitch Inflection & Dynamics", "Evaluates vocal expressiveness versus a monotone or rigid vocal cadence."),
    "min_f0": ("Lowest Vocal Frequency", "The baseline bottom of the subject's dynamic pitch range."),
    "max_f0": ("Highest Vocal Frequency", "The top reach of the subject's dynamic pitch range."),
    "mfcc_1": ("Vocal Power Envelope", "Captures overall spectral energy and phonation intensity."),
    "mfcc_2": ("Low-Frequency Vocal Resonance", "Reflects pharyngeal resonance and lower vocal tract shaping."),
    "mfcc_3": ("Mid-Vowel Spectral Balance", "Measures tongue placement and mid-frequency acoustic resonance."),
    "mfcc_4": ("Higher Formant Distribution", "Evaluates articulation sharpness and upper vocal tract dynamics."),
    "mfcc_5": ("High-Frequency Acoustic Detail", "Tracks subtle high-frequency consonant and vowel harmonics."),
    "mfcc_6": ("Fine Spectral Articulation", "Captures micro-articulation clarity in speech production."),
    "f1_freq": ("First Formant Frequency", "Related to tongue height and vertical vocal tract volume."),
    "f2_freq": ("Second Formant Frequency", "Related to tongue forward/backward positioning and vowel differentiation."),
    "f3_freq": ("Third Formant Frequency", "Related to lip rounding and oral cavity dimensions."),
    "f1_bw": ("First Formant Bandwidth", "Reflects damping and energy distribution in the primary resonance band."),
    "f2_bw": ("Second Formant Bandwidth", "Quantifies clarity of secondary acoustic vowel resonances."),
    "f3_bw": ("Third Formant Bandwidth", "Measures acoustic dispersion across upper articulation frequencies."),
}


def _patient_report(payload: dict) -> str:
    """Comprehensive, empathetic, and actionable plain-language report for patients."""
    subject_id = payload.get("subject_id", "Patient Recording")
    raw_pred = str(payload.get("prediction", "")).strip()
    is_pd = "parkinson" in raw_pred.lower() or raw_pred.upper() == "PD"
    is_hc = "healthy" in raw_pred.lower() or "control" in raw_pred.lower() or raw_pred.upper() == "HC"
    refer = bool(payload.get("refer_to_clinician", False))
    prob_pd = payload.get("probability_pd")
    if prob_pd is None and payload.get("probability_hc") is not None:
        prob_pd = 1.0 - payload.get("probability_hc")
    alpha = payload.get("modality_gate_alpha", 0.5)
    shap_top5 = payload.get("shap_top5", [])

    lines = []
    lines.append("## Voice Analysis Assessment Summary")
    lines.append(f"- **Patient / Recording ID**: `{subject_id}`")

    if refer:
        lines.append("- **Overall Status**: **Inconclusive / Neutral (Follow-up Recommended)**")
        lines.append("- **Confidence**: The tool observed overlapping patterns that require doctor review.")
    elif is_pd:
        lines.append("- **Overall Status**: **Acoustic Patterns Associated with Vocal Rigidity or Tremor Detected**")
        if prob_pd is not None:
            lines.append(f"- **Analysis Pattern Association**: **{prob_pd:.0%}** statistical association with characteristic voice changes.")
    else:
        lines.append("- **Overall Status**: **Typical Healthy Speech Characteristics Observed**")
        if prob_pd is not None:
            lines.append(f"- **Analysis Pattern Association**: **{(1.0 - prob_pd):.0%}** consistency with typical baseline speech patterns.")

    lines.append("- **Analysis Type**: Combined High-Resolution Acoustic Biomarkers & Deep Speech Pattern Recognition")
    lines.append("")

    lines.append("## What This Means For You")
    if refer:
        lines.append(
            "Our voice analysis tool observed mixed characteristics during your phonation. "
            "Because human speech fluctuates naturally depending on how well-rested, hydrated, or relaxed you are, "
            "the system could not confidently assign a single outcome. This does **not** mean you have a medical problem—it "
            "simply indicates that your recording should be interpreted by a healthcare provider alongside a regular clinical check-up."
        )
    elif is_pd:
        lines.append(
            "The voice analyzer identified subtle variations in your vocal tone, pitch stability, or volume consistency "
            "that resemble acoustic changes sometimes observed in individuals with Parkinson's disease (such as mild vocal cord rigidity or subtle tremor). "
            "**Crucially, this is NOT a medical diagnosis.** Voice changes can happen for many harmless, temporary reasons, including vocal strain, "
            "allergies, fatigue, acid reflux, or everyday stress."
        )
    else:
        lines.append(
            "Your voice recording demonstrated steady pitch control, consistent loudness, and clear vocal fold vibration. "
            "The automated system found no significant indicators of vocal tremor, roughness, or rigidity. "
            "While this is an encouraging result, voice screening is only one small aspect of overall wellness, and you should always "
            "continue regular health check-ups with your doctor."
        )
    lines.append("")

    lines.append("## Detailed Vocal Dimensions Evaluated")
    lines.append("- **1. Vocal Stability & Micro-Tremors (Frequency Jitter)**: We evaluated whether the microscopic vibration rate of your vocal cords remained steady from cycle to cycle.")
    lines.append("- **2. Loudness Consistency & Breath Control (Amplitude Shimmer)**: We analyzed whether your vocal volume remained smooth and consistent as air passed through your vocal tract.")
    lines.append("- **3. Voice Clarity & Tone Purity (Harmonics-to-Noise Ratio)**: We checked the ratio of clear vocal tone versus breathiness or hoarseness in your sustained vowel.")
    lines.append("- **4. Articulation & Deep Vocal Resonance**: Neural speech networks examined the subtle acoustics shaped by your tongue, lips, and pharynx.")
    lines.append("")

    if shap_top5:
        lines.append("## Primary Factors Influencing Your Result")
        lines.append("The artificial intelligence identified the following specific acoustic traits as having the highest impact on your analysis:")
        for item in shap_top5:
            feat_name = item.get("feature", "")
            friendly_name, friendly_desc = FEATURE_EXPLANATIONS.get(feat_name, (feat_name.replace("_", " ").title(), "Acoustic speech parameter."))
            val = item.get("value", 0.0)
            impact = "leaned toward voice alteration" if val > 0 else "reflected steady vocal stability"
            lines.append(f"- **{friendly_name}**: {friendly_desc} (*This feature {impact}*).")
        lines.append("")

    lines.append("## How the Analysis Worked")
    if alpha > 0.5:
        lines.append(
            f"The system relied primarily ({alpha:.0%}) on **deep neural audio representations** (Wav2Vec2), "
            "which listen to continuous temporal flow, rhythm, and acoustic texture across your whole recording."
        )
    else:
        lines.append(
            f"The system relied primarily ({(1.0 - alpha):.0%}) on **direct acoustic measurements** "
            "(such as frequency jitter, loudness shimmer, and vocal harmonics), quantifying physical voice parameters."
        )
    lines.append("")

    lines.append("## Common Factors That Can Affect Your Voice")
    lines.append("Before jumping to conclusions, keep in mind that vocal acoustics are sensitive to many everyday circumstances:")
    lines.append("- **Physical Fatigue or Stress**: Tired vocal muscles or shallow breathing can introduce temporary vocal tremor.")
    lines.append("- **Hydration & Dry Air**: Dry vocal folds vibrate with greater irregularity and increased shimmer.")
    lines.append("- **Colds, Allergies, or Post-Nasal Drip**: Congestion alters vocal tract resonance and harmonics.")
    lines.append("- **Acid Reflux (GERD)**: Stomach acid can irritate vocal cords, causing temporary hoarseness.")
    lines.append("- **Medications**: Certain allergy, blood pressure, or asthma medications can cause dry mouth or mild vocal changes.")
    lines.append("")

    lines.append("## Suggested Questions to Discuss with Your Doctor")
    lines.append("If you decide to share this analysis with your healthcare provider, here are helpful questions you can ask:")
    lines.append("1. *'Could any of my current medications, allergies, or vocal habits be contributing to these voice characteristics?'*")
    lines.append("2. *'Do you notice any subtle changes in my speech volume, speed, or facial expressiveness during our visit?'*")
    lines.append("3. *'Are there any motor or coordination tests you recommend to establish a comprehensive baseline?'*")
    lines.append("4. *'Would consulting a speech-language pathologist (SLP) or neurologist be helpful for peace of mind?'*")
    lines.append("")

    lines.append("## Recommended Next Steps")
    lines.append("- **Share with your physician**: Bring a printed or digital copy of this report to your next appointment.")
    lines.append("- **Track your symptoms over time**: Notice if your voice changes after rest or at different times of day.")
    lines.append("- **Practice good vocal hygiene**: Drink plenty of water and avoid excessive throat clearing.")
    lines.append("")

    lines.append("## Important Medical Disclaimer")
    lines.append(
        "**This analysis is a digital screening and decision-support tool -- NOT a medical diagnosis.** "
        "It cannot confirm, rule out, or diagnose Parkinson's disease or any other medical condition. "
        "A formal neurological diagnosis requires a thorough in-person examination by a physician, including physical mobility, "
        "reflex tests, medical history, and clinical observation. Always seek the advice of a qualified physician."
    )

    return "\n".join(lines)


def _clinician_report(payload: dict) -> str:
    """Quantitative, technical decision-support report for healthcare providers."""
    subject_id = payload.get("subject_id", "Unknown")
    prediction = payload.get("prediction", "Unknown")
    prediction_set = payload.get("prediction_set", [])
    refer = payload.get("refer_to_clinician", False)
    alpha = payload.get("modality_gate_alpha", 0.0)
    dominance = payload.get("modality_dominance", "Unknown")
    prob_pd = payload.get("probability_pd")
    prob_hc = payload.get("probability_hc")
    if prob_hc is None and prob_pd is not None:
        prob_hc = 1.0 - prob_pd
    elif prob_pd is None and prob_hc is not None:
        prob_pd = 1.0 - prob_hc
    attn = payload.get("attention_weights", [])
    shap_top5 = payload.get("shap_top5", [])

    lines = []
    lines.append("## Clinical Decision-Support Summary")
    lines.append(f"- **Subject / Recording Identifier**: `{subject_id}`")
    lines.append(f"- **Primary Classification**: **{prediction}**")
    lines.append(f"- **Conformal Prediction Set**: **{{{', '.join(prediction_set)}}}**" if prediction_set else "- **Conformal Prediction Set**: Empty Set")

    if prob_pd is not None and prob_hc is not None:
        lines.append(f"- **Calibrated Posterior Probabilities**: P(HC) = **{prob_hc:.1%}**, P(PD) = **{prob_pd:.1%}**")

    flag_str = "[!] REFERRAL RECOMMENDED (Uncertain / Inconclusive set)" if refer else "Standard Confidence Set (Singleton)"
    lines.append(f"- **Conformal Coverage Status**: {flag_str}")
    lines.append("")

    lines.append("## Multimodal Fusion & Modality Gate Analysis")
    lines.append(f"- **Modality Gate Alpha (alpha)**: **{alpha:.3f}**")
    lines.append(f"- **Dominant Model Branch**: **{dominance}**")
    if alpha > 0.5:
        lines.append(
            f"- **Interpretation**: The prediction was driven predominantly by the **Wav2Vec2 Self-Supervised Learning (SSL) branch** (alpha = {alpha:.3f} > 0.5), "
            "indicating that continuous acoustic dynamics and latent speech representations provided stronger discriminative signal than static handcrafted features."
        )
    else:
        lines.append(
            f"- **Interpretation**: The prediction was driven predominantly by the **Acoustic Biomarker branch** (alpha = {alpha:.3f} <= 0.5), "
            "indicating that classical physical perturbation metrics (perturbation in frequency/amplitude, spectral noise ratios) dominated model classification."
        )
    lines.append("")

    if shap_top5:
        lines.append("## Top Influential Acoustic Biomarkers (SHAP)")
        lines.append("Feature contributions to model log-odds derived via Kernel SHAP:")
        for item in shap_top5:
            feat_name = item.get("feature", "")
            friendly_name, friendly_desc = FEATURE_EXPLANATIONS.get(feat_name, (feat_name.replace("_", " ").title(), "Acoustic parameter."))
            val = item.get("value", 0.0)
            direction = "-> Toward PD" if val > 0 else "-> Toward HC"
            lines.append(f"- **{friendly_name}** (`{feat_name}`): SHAP = **{val:+.4f}** ({direction}) -- {friendly_desc}")
        lines.append("")

    if attn:
        lines.append("## Cross-Recording Attention Weights")
        lines.append("Attention distribution over subject phonation segments:")
        for i, w in enumerate(attn):
            lines.append(f"- Recording #{i+1}: weight = **{w:.4f}**")
        lines.append("")

    if refer:
        lines.append("## Uncertainty & Conformal Risk Assessment")
        if len(prediction_set) == 2:
            lines.append(
                "- **Set {HC, PD}**: The model's conformal prediction set spans both healthy control and Parkinson's disease. "
                "The calibrated risk threshold guarantees coverage at the 90% confidence level, meaning the sample falls in an ambiguous boundary region. "
                "A full motor assessment (UPDRS Part III) and formal clinical evaluation are strongly indicated."
            )
        elif len(prediction_set) == 0:
            lines.append(
                "- **Empty Set**: The conformal prediction set is empty, indicating that neither hypothesis met the non-conformity threshold. "
                "This anomalous response suggests atypical phonation, recording noise, or outlier acoustic properties. Repeat acquisition is advised."
            )
        lines.append("")

    lines.append("## Clinical Correlation & Medical Disclaimer")
    lines.append(
        "**NOTE**: PD-VoiceNet is an investigational decision-support instrument designed to supplement, not supplant, "
        "comprehensive neurological evaluation. Vocal biomarkers exhibit sensitivity to non-parkinsonian etiologies (e.g., laryngitis, "
        "presbyphonia, essential vocal tremor, medication-induced xerostomia, respiratory compromise). "
        "Results must be correlated with clinical history, motor examination, and validated neurological rating scales."
    )

    return "\n".join(lines)
