# PD-VoiceNet Research Paper & Documentation

This directory contains the IEEE-formatted research paper manuscript, automated paper generator script, references, and drafts for **PD-VoiceNet**.

---

## 📑 Contents

1. **`PD_VoiceNet_IEEE_Paper.docx`**
   - The complete, camera-ready IEEE conference manuscript.
   - Formatted in two-column IEEE format with:
     - Abstract & IEEE Keywords
     - Section I: Introduction & Clinical Motivation (Substantia nigra degeneration, hypokinetic dysarthria)
     - Section II: Related Work (Classical phonation features vs. modern SSL representations)
     - Section III: Methodology (Dual-branch architecture, Wav2Vec 2.0 Layer 9, 21 Praat biomarkers, adaptive gating $\alpha$)
     - Section IV: Conformal Prediction & Calibration (Temperature scaling $T=1.6276$, coverage guarantee $\alpha_{\text{err}}=0.05$)
     - Section V: Experimental Results & Leave-One-Database-Out (LODO) validation across Italian Parkinson's Voice dataset
     - Section VI: Tripartite Explainability (SHAP, Grad-CAM, Saliency)
     - Section VII: Discussion & Limitations
     - Section VIII: Conclusion
     - 3 detailed tables & comprehensive references

2. **`generate_paper.py`**
   - The Python script that programmatically synthesizes `PD_VoiceNet_IEEE_Paper.docx`.
   - Injects the full paper text, mathematical notations, experimental tables, and IEEE style formatting into the base layout.
   - Run via:
     ```bash
     python paper/generate_paper.py
     ```

3. **`drafts/park_draft.docx`**
   - Initial working notes, methodology outlines, and experimental draft text.

4. **`references/`**
   - `IEEE-SAMPLE PAPER.docx`: Standard IEEE template reference.
   - `conference-template-a4.docx`: Official IEEE A4 conference layout.
   - `schizobrain (1).docx`: Layout base used for exact 2-column margin and typography matching.
