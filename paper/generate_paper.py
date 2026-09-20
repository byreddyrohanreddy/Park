"""
Generate PD-VoiceNet IEEE paper using SchizoBrain DOCX as the layout base.
SchizoBrain uses: Title / Body Text / List Paragraph styles, A4 page,
0.30in L/R margins, 2-column body sections.
We load SchizoBrain, wipe its content, and inject PD-VoiceNet content
using the same styles so the column/margin layout is preserved exactly.
"""
import copy, os
from docx import Document
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(BASE_DIR, 'references', 'schizobrain (1).docx')
DEST = os.path.join(BASE_DIR, 'PD_VoiceNet_IEEE_Paper.docx')

doc = Document(SRC)
body = doc.element.body
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

# Save the final sectPr (page layout for last section)
final_sectPr = body.find(f'{{{W}}}sectPr')
final_sectPr_copy = copy.deepcopy(final_sectPr) if final_sectPr is not None else None

# Wipe all body children
for child in list(body):
    body.remove(child)

# Restore a minimal dummy paragraph + final sectPr so page layout is kept
dummy = OxmlElement('w:p')
body.append(dummy)
if final_sectPr_copy is not None:
    body.append(final_sectPr_copy)

# ── Helpers ──────────────────────────────────────────────────────────────────
def bt(text='', bold=False, italic=False, align=None):
    """Add a Body Text paragraph."""
    p = doc.add_paragraph(style='Body Text')
    if align is not None:
        p.alignment = align
    if text:
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
    return p

def bt2(label, text):
    """Body Text with bold-italic label then normal text (Abstract/Keywords)."""
    p = doc.add_paragraph(style='Body Text')
    r = p.add_run(label)
    r.bold = True
    r.italic = True
    p.add_run(text)
    return p

def ttl(text):
    p = doc.add_paragraph(style='Title')
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(text)
    return p

def h1(text):
    p = doc.add_paragraph(style='Body Text')
    run = p.add_run(text.upper())
    run.bold = True
    return p

def h2(text):
    p = doc.add_paragraph(style='Body Text')
    run = p.add_run(text)
    run.bold = True
    run.italic = True
    return p

def h3(text):
    p = doc.add_paragraph(style='Body Text')
    run = p.add_run(text)
    run.bold = True
    return p

def bl(text):
    p = doc.add_paragraph(style='List Paragraph')
    p.add_run(u'\u2022 ' + text)
    return p

def caption(text):
    p = doc.add_paragraph(style='Body Text')
    run = p.add_run(text)
    run.bold = True
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return p

def add_table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = 'Table Normal'
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = h
        for p in hdr[i].paragraphs:
            for r in p.runs:
                r.bold = True
    for row_data in rows:
        row = t.add_row().cells
        for i, v in enumerate(row_data):
            row[i].text = str(v)
    return t

# ============================================================================
# TITLE
# ============================================================================
ttl(
    "PD-VoiceNet: Multi-Site Parkinson\u2019s Disease Voice Screening via "
    "LoRA-Adapted Self-Supervised Speech Representations, Biomarker Fusion, "
    "and Site-Leakage Auditing"
)

# ── Authors (Body Text, centred, as in SchizoBrain) ──────────────────────────
for name, email in [
    ("Pavani", ""),
    ("Rohan Reddy", "byreddyrohanreddy@gmail.com"),
    ("Sri Vaishnavi", ""),
    ("SreeHarshini", ""),
    ("Sravika", ""),
    ("Raj Kumar", ""),
]:
    p = doc.add_paragraph(style='Body Text')
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(name)
    run.bold = True

for line in [
    "Department of Computer Science and Engineering,",
    "Keshav Memorial Institute of Technology,",
    "Narayanaguda, Hyderabad\u2013500029, Telangana, India.",
]:
    p = doc.add_paragraph(style='Body Text')
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run(line)

# ── Abstract ─────────────────────────────────────────────────────────────────
bt2(
    "Abstract\u2014",
    "Voice-based screening for Parkinson\u2019s disease (PD) offers a non-invasive and "
    "low-cost clinical tool, yet the field is plagued by a pervasive confound: recordings "
    "from different clinical sites differ in microphone hardware, room acoustics, and "
    "recording protocols, causing models to partially learn site identity rather than genuine "
    "disease acoustics. We introduce PD-VoiceNet, a dual-branch architecture fusing "
    "self-supervised speech representations from Wav2Vec2-base with low-rank adaptation "
    "(LoRA, r=4, \u03b1=16) on transformer layers 9\u201311, with 26 Praat-extracted acoustic "
    "biomarkers, through a subject attention pooling mechanism and a biomarker-driven scalar "
    "modality gate. Evaluated under a strict 11-fold Leave-One-Site-Out (LODO) protocol "
    "across 827 subjects from 11 clinical hospital sites, PD-VoiceNet achieves 92.19% macro "
    "accuracy and 91.47% macro F1. A systematic site-leakage audit with dimensionality-matched "
    "controls reveals both branches encode site identity at >95% at matched 16D, and "
    "non-linear probes recover 68.57% site accuracy from the 2D fusion space. Tri-modal "
    "interpretability (biomarker SHAP, CNN Grad-CAM, Transformer saliency) and Mondrian "
    "conformal calibration at 90% class-conditional coverage complete the pipeline."
)
bt2(
    "Keywords\u2014",
    "Parkinson\u2019s disease; voice biomarkers; self-supervised learning; wav2vec2; LoRA; "
    "leave-one-site-out; site leakage; conformal prediction; explainability; acoustic features"
)

# ============================================================================
# I. INTRODUCTION
# ============================================================================
h1("I. Introduction")
bt(
    "Parkinson\u2019s disease (PD) is the second most prevalent neurodegenerative disorder "
    "worldwide, affecting over 10 million people and projected to double in prevalence by "
    "2040 [1]. Clinical diagnosis relies on expert neurological examination, a process "
    "vulnerable to inter-rater variability that frequently misses subtle early-stage "
    "impairment. Voice analysis offers a compelling non-invasive alternative: PD-related "
    "neurodegeneration of the basal ganglia produces measurable dysphonia\u2014reduced vocal "
    "loudness, tremor-related pitch instability, aperiodicity, breathiness, and articulatory "
    "imprecision\u2014all detectable from simple sustained phonation recordings [4][17]."
)
bt(
    "Despite rapid progress in deep learning for PD voice analysis, a critical confound "
    "undermines virtually all published benchmarks: most studies pool recordings from "
    "multiple clinical sites and evaluate with random train-test splits, allowing subjects "
    "from the same hospital to appear in both training and test sets. Because different "
    "acquisition environments impose site-specific spectral colorations, models can achieve "
    "high accuracy by learning to recognize hospital acoustics rather than disease pathology. "
    "This recording-condition confound means published metrics substantially overestimate "
    "performance on a truly unseen clinical site\u2014the exact scenario that matters for deployment."
)
bt(
    "We address this through PD-VoiceNet, evaluated under the only credible protocol: "
    "strict 11-fold Leave-One-Site-Out (LODO). We additionally conduct a systematic "
    "site-leakage audit with dimensionality-matched controls to quantify how much site "
    "identity is encoded at each representation stage. Main contributions:"
)
bl("PD-VoiceNet: LoRA-adapted Wav2Vec2 SSL branch fused with a Praat biomarker MLP through subject attention pooling and a biomarker-driven modality gate, producing a 2D z_fused representation.")
bl("Strict 11-fold LODO on N=827 subjects, 11 clinical sites: 92.19% macro accuracy, 91.47% macro F1, with 9 of 11 sites exceeding 91%.")
bl("First systematic site-leakage audit with dimensionality-matched controls: symmetric SSL and biomarker saturation (>95% at matched 16D) and 68.57% non-linear residual leakage in z_fused.")
bl("Tri-modal interpretability (SHAP, Grad-CAM, Transformer saliency) + Mondrian conformal calibration with class-conditional 90% coverage and correct abstention under distributional shift.")

# ============================================================================
# II. RELATED WORKS
# ============================================================================
h1("II. Related Works")
bt(
    "A previous landmark study by Tsanas et al. [5] explored accurate telemonitoring of PD "
    "progression using 16 Praat-derived features\u2014jitter, shimmer, HNR, and RPDE\u2014"
    "achieving strong correlation with UPDRS scores and establishing that acoustic biomarkers "
    "have clinical-grade discriminative power for PD. Their follow-up [6] reported >98% "
    "two-class accuracy using SVMs on a 17-feature set. Sakar et al. [7] collected the "
    "widely-used Turkish multi-task PD voice dataset with 26 acoustic features, reporting "
    "up to 77.3% accuracy. Amato et al. [1] surveyed 102 ML and statistical PD voice "
    "studies, concluding that while detection potential is genuine, non-standardized "
    "evaluation protocols make inter-study comparisons unreliable. Ma et al. [4][17] "
    "reviewed articulatory and phonatory biomarkers, establishing clinical basis for "
    "MFCC, jitter, shimmer, and formant-based analysis. None of these studies employed "
    "site-stratified evaluation, making their estimates susceptible to the "
    "recording-condition confound."
)
bt(
    "Deep learning methods subsequently superseded traditional approaches. Gunduz [9] "
    "applied CNNs to Mel-spectrograms achieving 86.4% accuracy; Quan et al. [10] proposed "
    "an end-to-end CNN+BiLSTM model capturing temporal dysrhythmia; Hires et al. [11] "
    "demonstrated that a CNN ensemble across multiple acoustic tasks significantly "
    "outperformed single-task models. Van Gelderen and Tejedor-Garcia [2] reviewed 33 "
    "deep learning PD voice studies (2020\u20132024), flagging poor cross-dataset "
    "generalization and absence of site stratification as critical unsolved challenges."
)
bt(
    "Self-supervised pretrained speech models opened a new frontier. Tougui et al. [12] "
    "applied the Audio Spectrogram Transformer to mPower voice recordings, achieving "
    "95.89% AUC, demonstrating that large-scale pretraining dramatically reduces data "
    "requirements. Postma and Tejedor-Garcia [14] benchmarked wav2vec2, HuBERT, and WavLM, "
    "finding embedding quality is the primary driver of downstream accuracy. Critically, "
    "no prior SSL-based PD voice work audited whether representations encode site identity "
    "or evaluated under LODO\u2014gaps PD-VoiceNet directly addresses."
)
bt(
    "On interpretability, Shen et al. [15] applied SHAP to acoustic features for PD; "
    "Aladhadh [16] combined SHAP, LIME, and Grad-CAM in an ensemble framework. Tracy et al. [8] "
    "combined voice with accelerometer and typing data for early screening. None of these "
    "works addresses conformal uncertainty quantification\u2014which provides statistical "
    "coverage guarantees\u2014or systematically audits recording-site leakage. "
    "PD-VoiceNet is the first PD voice system combining all three."
)

# ============================================================================
# III. PROPOSED METHODOLOGY
# ============================================================================
h1("III. Proposed Methodology")

h2("Dataset Description:")
bt(
    "We evaluate on a pooled multi-site clinical corpus of 827 subjects from 11 hospital "
    "sites (B1, B2, D1, D2, FB, PR, VA, VE, VI, VO, VU), contributing between 47 (FB) and "
    "99 (VA, VE, VI, VO) subjects per site, each providing one or more sustained /a/ phonation "
    "recordings. Sites differ in microphone model, room acoustics, and recording software, "
    "making this a genuinely heterogeneous multi-site corpus. An additional held-out "
    "benchmark (AH, N=81) is reported separately as out-of-distribution: a cross-validated "
    "linear probe achieves 99.67% accuracy distinguishing AH from the 11 clinical sites "
    "using h_subject alone, confirming irreducible distributional shift."
)
bt(
    "We adopt a strict 11-fold LODO protocol as the sole evaluation scheme. In each fold, "
    "one complete site is withheld as the test partition; the model is trained and calibrated "
    "on the remaining 10 sites. No subject from the test site appears in any training or "
    "calibration split. This directly mirrors the real deployment scenario. Random "
    "subject-level splits, used by most prior work, allow site-correlated data to leak "
    "into training and artificially inflate reported test accuracy."
)

h2("Preprocessing Pipeline:")
bt(
    "Raw recordings undergo a three-stage pipeline before being fed into the model. "
    "First, voice activity detection (VAD) trims leading and trailing silence, ensuring "
    "only voiced segments contribute to feature extraction. Second, all recordings are "
    "resampled to 16 kHz, the native sample rate of Wav2Vec2-base [3], using sinc-based "
    "resampling. Third, per-recording mean-variance normalization is applied to reduce gross "
    "amplitude differences between sites. Wav2Vec2-base consists of a convolutional feature "
    "encoder followed by 12 transformer encoder layers. Layers 0\u20138 are fully frozen; "
    "their (T\u00d7768) activations are extracted offline and cached as PyTorch tensors, "
    "eliminating repeated forward passes through the 90M-parameter backbone and reducing "
    "training time by approximately 8\u00d7. Only layers 9\u201311 participate in training "
    "via LoRA adaptation."
)
bt(
    "Concurrently, 26 Praat acoustic biomarkers are extracted per recording: F0 statistics "
    "(mean, std), jitter (local, RAP, PPQ5), shimmer (local, APQ3, APQ5), harmonics-to-noise "
    "ratio (HNR), MFCC coefficients 1\u201313, and formant bandwidths F1\u2013F3. These "
    "collectively capture the multi-dimensional dysphonia profile associated with PD: "
    "F0 instability reflects laryngeal tremor, jitter and shimmer quantify cycle-to-cycle "
    "perturbations, HNR measures the noise component in the glottal signal, and formant "
    "bandwidths reflect vocal tract resonance changes due to reduced articulatory mobility [5][17]."
)

h2("SSL Branch with LoRA Adaptation:")
bt(
    "Cached layer-8 activations (T\u00d7768) pass through LoRA-adapted deep copies of "
    "Wav2Vec2 transformer layers 9, 10, and 11. LoRA [18] wraps the query projection "
    "(q_proj) and value projection (v_proj) of each self-attention block with low-rank "
    "update matrices: output = W_frozen(x) + (B\u00b7A)(x)\u00d7(\u03b1/r), where "
    "A\u2208\u211d^{d\u00d7r}, B\u2208\u211d^{r\u00d7d}, rank r=4, scaling \u03b1=16, "
    "input dropout=0.1. Only ~48K LoRA parameters are trained versus 90M frozen backbone "
    "parameters, providing highly parameter-efficient task-specific fine-tuning. "
    "Frame-level outputs are mean-pooled and projected: Linear(768\u2192128) "
    "\u2192 h_ssl\u2208\u211d^{128}."
)

h2("Biomarker Branch:")
bt(
    "The 26 Praat features are processed by a three-layer MLP: "
    "Linear(26\u219232, ReLU, BN) \u2192 Linear(32\u219216, ReLU, BN) "
    "\u2192 h_bio\u2208\u211d^{16}. Batch normalization before each activation stabilizes "
    "training across the wide dynamic range of heterogeneous acoustic biomarkers. "
    "The 16-dimensional output h_bio is subsequently used as the sole input to "
    "the biomarker-driven modality gate, ensuring the gating decision is grounded "
    "in interpretable acoustic features."
)

h2("Subject Attention Pooling:")
bt(
    "Each subject contributes N recording embeddings (h_ssl, h_bio) per recording. "
    "A learned-query cross-attention mechanism aggregates these N embeddings into a "
    "single subject-level representation via scaled dot-product attention, producing "
    "h_sub_ssl\u2208\u211d^{128} and h_sub_bio\u2208\u211d^{16}. This design natively "
    "handles variable N without padding, truncation, or recording selection heuristics, "
    "which is critical because subjects across sites contributed different numbers of recordings."
)

h2("Modality Gate, Fusion, and Conformal Prediction:")
bt(
    "A scalar gate \u03b1 = \u03c3(Linear(h_sub_bio\u21921))\u2208[0,1] is computed from "
    "the biomarker subject embedding, allowing the model to dynamically weight the two "
    "branches. Fusion: z_fused = \u03b1\u00b7W_ssl\u00b7h_sub_ssl + "
    "(1\u2212\u03b1)\u00b7W_bio\u00b7h_sub_bio \u2208 \u211d^{2}. A linear classification "
    "head with per-fold temperature scaling T (calibrated on in-fold validation) produces "
    "calibrated logits. Mondrian conformal prediction [19] with class-conditional calibration "
    "at 90% target coverage generates set-valued predictions: the prediction set for each "
    "subject includes all classes whose non-conformity score falls below the class-specific "
    "threshold, providing coverage guarantees even when PD/HC base rates vary across sites."
)

h2("Training Strategy and Optimization:")
bt(
    "Models are trained with cross-entropy loss and AdamW optimization "
    "(lr=1\u00d710^{-4} for LoRA parameters, lr=1\u00d710^{-3} for all others; "
    "weight decay=0.01). Batches are assembled at the subject level, enforcing "
    "subject-level prediction from the first training step. Temperature T and conformal "
    "calibration thresholds are computed on in-fold validation only, with strict "
    "separation from the test site. Early stopping on validation loss (patience=20 epochs) "
    "prevents overfitting on the small per-fold training sets."
)

h2("Site-Leakage Auditing Protocol:")
bt(
    "Out-of-fold subject embeddings (h_sub_bio, h_sub_ssl, h_subject=[h_sub_ssl; h_sub_bio], "
    "z_fused) are extracted for all 827 clinical subjects. Linear probes (L2-regularized "
    "logistic regression, C=1.0, 5-fold CV) classify the 11-class site label. "
    "A 500-iteration permutation null distribution provides statistical significance "
    "at the 99th percentile (p<0.01). Dimensionality-matched controls disentangle "
    "the contribution of dimension from learned site-invariance: (i) PCA compression "
    "and random projection of h_subject to 2D, matching z_fused; (ii) PCA of h_sub_ssl "
    "to 16D, matching h_sub_bio; (iii) a 1-hidden-layer MLP (32 units, ReLU) and "
    "RBF-kernel SVM applied to z_fused as non-linear probes."
)

h2("Tri-Modal Interpretability:")
bt(
    "Three complementary explanation modalities are computed per subject at inference time. "
    "Biomarker SHAP computes exact additive Shapley values over the 26 Praat features "
    "(\u0394<10^{-6}, verified), providing signed attribution per biomarker. CNN Grad-CAM "
    "produces gradient-weighted class activation maps from CNN feature encoder layer 6 "
    "at ~20 ms temporal resolution, highlighting which speech frames drove CNN-level features. "
    "Transformer Saliency computes gradient-weighted attention scores from LoRA-adapted "
    "layer 11, providing frame-level attribution over the SSL branch. The three modalities "
    "are complementary by design: SHAP explains the biomarker branch in feature space, "
    "Grad-CAM localizes discriminative frames acoustically, and Transformer saliency "
    "traces deep SSL feature attribution."
)

# ============================================================================
# IV. RESULTS AND DISCUSSION
# ============================================================================
h1("IV. Results and Discussion")
bt(
    "The proposed PD-VoiceNet framework integrates a LoRA-adapted Wav2Vec2 SSL branch "
    "with a Praat biomarker MLP branch, fused through subject attention pooling and a "
    "biomarker-driven modality gate. In addition, clinical confidence is quantified via "
    "Mondrian conformal calibration and residual site bias is systematically audited "
    "through dimensionality-matched probe experiments. Table I summarizes per-site "
    "classification performance under strict LODO."
)

add_table(
    headers=['Site', 'N', 'Acc (%)', 'Prec (%)', 'Rec (%)', 'F1 (%)'],
    rows=[
        ('B1','64','98.44','100.00','96.43','98.18'),
        ('B2','60','98.33','96.00','100.00','97.96'),
        ('D1','50','96.00','93.33','100.00','96.55'),
        ('D2','50','96.00','96.43','96.43','96.43'),
        ('FB','47','100.00','100.00','100.00','100.00'),
        ('PR','65','95.38','90.32','100.00','94.92'),
        ('VA','99','91.92','87.30','100.00','93.22'),
        ('VE*','99','64.65','100.00','36.36','53.33'),
        ('VI','99','88.89','97.83','81.82','89.11'),
        ('VO','99','92.93','96.15','90.91','93.46'),
        ('VU','95','91.58','86.89','100.00','92.98'),
        ('Macro','827','92.19','-','-','91.47'),
        ('Micro','827','90.57','-','-','-'),
        ('AH (OOD)','81','49.38','48.78','50.00','49.38'),
    ]
)
caption("TABLE I. Per-Site LODO Results (N=827 clinical). AH (OOD, N=81) separate. VE*=outlier.")

bt(
    "PD-VoiceNet achieves 92.19% macro accuracy and 91.47% macro F1. Nine of eleven sites "
    "exceed 91% accuracy; FB reaches 100%, B1 and B2 reach 98.44% and 98.33%, and D1, D2, "
    "and PR exceed 95%. The integration of the LoRA-adapted SSL branch with the biomarker "
    "MLP and modality gate enables the model to leverage complementary signal sources, "
    "improving performance over either branch alone as confirmed by the ablation study."
)
bt(
    "Site VE is the sole outlier (64.65% accuracy, 36.36% recall, 100% precision). "
    "This high-precision/low-recall pattern indicates the model systematically predicts HC "
    "for VE PD subjects while never producing a false positive. This is consistent with a "
    "subpopulation or acquisition shift: VE PD patients may present atypically mild dysphonia, "
    "or the site\u2019s recording equipment may attenuate the frequency bands most predictive "
    "of PD. Mondrian conformal calibration reflects this uncertainty correctly: 6 of 99 VE "
    "subjects receive empty {PD, HC} abstention sets, demonstrating correct abstention under "
    "distributional shift rather than overconfident misclassification. AH at 49.38% "
    "(near chance) confirms genuine OOD status."
)

bt(
    "Table II presents the site-leakage audit and dimensionality-matched control results. "
    "All probes are statistically significant (p<0.01) against the permutation null. "
    "Three key findings emerge. Finding 1: the apparent gap between h_sub_ssl (128D, 99.76%) "
    "and z_fused (2D, 33.37%) is largely a dimensionality artifact\u2014PCA compression of "
    "h_subject to 2D yields 35.43%, nearly identical to z_fused\u2019s 33.37%. The 2D "
    "bottleneck, not learned site-invariance, explains the probe protection. "
    "Finding 2: at matched 16D, both branches saturate equally (h_ssl-PCA-16D=95.52% vs. "
    "h_bio=97.46%), demonstrating that Wav2Vec2 pretraining does not prevent site-identity "
    "encoding. Finding 3: non-linear probes on z_fused recover 68.57% (MLP) and 50.78% "
    "(SVM) site accuracy, far above null thresholds, confirming residual leakage."
)

add_table(
    headers=['Representation', 'Dim', 'Probe', 'Acc (%)', 'Null 99th (%)'],
    rows=[
        ('h_sub_bio','16','Linear','97.46','14.26'),
        ('h_sub_ssl','128','Linear','99.76','13.54'),
        ('h_subject','144','Linear','99.88','13.30'),
        ('z_fused','2','Linear','33.37','14.52'),
        ('h_subj-PCA-2D','2','Lin (PCA)','35.43','14.63'),
        ('h_subj-Rand-2D','2','Lin (Rand)','24.67','14.63'),
        ('z_fused-MLP','2','MLP (32u)','68.57','14.75'),
        ('z_fused-SVM','2','SVC (RBF)','50.78','15.12'),
        ('h_ssl-PCA-16D','16','Lin (PCA)','95.52','14.14'),
    ]
)
caption("TABLE II. Site-Leakage Probes and Controls (827 subjects, 500-iter permutation, p<0.01).")

bt(
    "Table III provides an ablation study assessing the contribution of each architectural "
    "component. The biomarker-only variant (74.31%) establishes the handcrafted-feature "
    "baseline. Frozen SSL only (81.56%) demonstrates the value of pretrained representations "
    "without fine-tuning. Adding LoRA and biomarker fusion without a gate (88.93%) "
    "substantially closes the gap. The full model with LoRA and modality gate (92.19%) "
    "confirms both components are independently valuable. The progressive improvement "
    "validates each design decision in the PD-VoiceNet architecture."
)

add_table(
    headers=['Model Variant', 'Macro Acc (%)', 'Macro F1 (%)', 'Notes'],
    rows=[
        ('Biomarker MLP only','74.31','72.88','No SSL; handcrafted features only'),
        ('Frozen SSL only (Wav2Vec2)','81.56','80.12','No LoRA; no biomarker branch'),
        ('SSL + Bio, no gate (alpha=0.5)','88.93','88.01','Fixed fusion; no modality gate'),
        ('SSL + Bio + gate, no LoRA','89.47','88.76','Frozen SSL; gate only'),
        ('Full PD-VoiceNet (proposed)','92.19','91.47','LoRA + gate + conformal calib.'),
    ]
)
caption("TABLE III. Ablation Study. 11-fold LODO macro accuracy and F1, 827 subjects.")

bt(
    "Biomarker SHAP analysis verifies exact additive attributions (\u0394=0.000000). "
    "Top discriminative Praat features are MFCC-5, F1 Bandwidth, F3 Bandwidth, and "
    "Jitter PPQ5\u2014consistent with established dysphonia literature [5][17]: MFCC-5 "
    "captures mid-frequency spectral shape changes associated with vocal fold closure "
    "asymmetry, F1 and F3 bandwidths reflect vocal tract resonance changes from reduced "
    "articulatory mobility, and Jitter PPQ5 quantifies pitch cycle perturbations linked "
    "to laryngeal tremor. CNN Grad-CAM heatmaps (encoder layer 6) highlight phonation "
    "onset and sustained vowel segments at ~20 ms resolution; Transformer saliency "
    "(LoRA layer 11) shows elevated attribution at utterance boundaries where tremor-related "
    "instability is most pronounced. Mondrian conformal calibration achieves \u226590% "
    "empirical class-conditional coverage in 10 of 11 folds."
)

# ============================================================================
# V. CONCLUSION AND FUTURE SCOPE
# ============================================================================
h1("V. Conclusion and Future Scope")
bt(
    "The present study proposed PD-VoiceNet, a dual-branch deep learning framework for "
    "multi-site Parkinson\u2019s disease voice screening evaluated under a strict 11-fold "
    "Leave-One-Site-Out protocol across 827 subjects from 11 clinical hospital sites. "
    "The architecture combined a LoRA-adapted Wav2Vec2-base SSL branch with a Praat "
    "biomarker MLP branch fused through subject attention pooling and a biomarker-driven "
    "modality gate, achieving 92.19% macro accuracy and 91.47% macro F1 with 9 of 11 sites "
    "exceeding 91%. Only ~48K LoRA parameters are trained versus 90M frozen backbone "
    "parameters, providing highly parameter-efficient task-specific adaptation."
)
bt(
    "The site-leakage audit with dimensionality-matched controls revealed an important "
    "symmetric finding: both the SSL (Wav2Vec2+LoRA) and biomarker (Praat) branches encode "
    "site identity at >95% linear probe accuracy at matched 16D, and non-linear probes "
    "recover 68.57% site accuracy from z_fused. Wav2Vec2 pretraining does not provide "
    "site-invariance, and the 2D fusion bottleneck offers only geometric protection. "
    "Future architectures for clinical deployment must incorporate explicit site-invariance "
    "objectives. Supervised cross-site contrastive regularization\u2014using same-label "
    "different-site pairs as positives\u2014is the most principled next step to directly "
    "minimize mutual information between embeddings and site identity."
)
bt(
    "Future work will extend evaluation to connected speech and read-speech tasks to test "
    "whether the symmetric saturation finding generalizes beyond sustained phonation; "
    "add severity estimation via UPDRS regression; incorporate longitudinal disease "
    "progression tracking through repeated voice assessments; and resolve the VE site "
    "performance gap through targeted data collection or site-specific adapter modules. "
    "The tri-modal interpretability pipeline and Mondrian conformal calibration form a "
    "solid clinical trust foundation for responsible AI-assisted PD screening across "
    "heterogeneous real-world environments."
)

# ============================================================================
# REFERENCES
# ============================================================================
h1("References")
refs = [
    'F. Amato, G. Saggio, V. Cesarini, G. Olmo, and G. Costantini, "Machine learning- and statistical-based voice analysis of Parkinson\u2019s disease patients: A survey," Expert Syst. Appl., vol. 219, Art. no. 119651, 2023.',
    'L. van Gelderen and C. Tejedor-Garcia, "Innovative speech-based deep learning approaches for Parkinson\u2019s disease classification: A systematic review," Appl. Sci., vol. 14, no. 17, Art. no. 7873, 2024.',
    'A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," in NeurIPS, vol. 33, pp. 12449\u201312460, 2020.',
    'A. Ma, K. K. Lau, and D. Thyagarajan, "Voice changes in Parkinson\u2019s disease: What are they telling us?" J. Clin. Neurosci., vol. 72, pp. 1\u20137, 2020.',
    'A. Tsanas, M. A. Little, P. E. McSharry, and L. O. Ramig, "Accurate telemonitoring of Parkinson\u2019s disease progression by noninvasive speech tests," IEEE Trans. Biomed. Eng., vol. 57, no. 4, pp. 884\u2013893, 2010.',
    'A. Tsanas, M. A. Little, P. E. McSharry, J. Spielman, and L. O. Ramig, "Novel speech signal processing algorithms for high-accuracy classification of Parkinson\u2019s disease," IEEE Trans. Biomed. Eng., vol. 59, no. 5, pp. 1264\u20131271, 2012.',
    'B. E. Sakar et al., "Collection and analysis of a Parkinson speech dataset with multiple types of sound recordings," IEEE J. Biomed. Health Inform., vol. 17, no. 4, pp. 828\u2013834, 2013.',
    'J. M. Tracy, Y. Ozkanca, D. C. Atkins, and R. Hosseini Ghomi, "Investigating voice as a biomarker: Deep phenotyping for early detection of Parkinson\u2019s disease," J. Biomed. Inform., vol. 104, Art. no. 103362, 2020.',
    'H. Gunduz, "Deep learning-based Parkinson\u2019s disease classification using vocal feature sets," IEEE Access, vol. 7, pp. 115540\u2013115551, 2019.',
    'C. Quan, K. Ren, Z. Luo, Z. Chen, and Y. Ling, "End-to-end deep learning approach for Parkinson\u2019s disease detection from speech signals," Biocybern. Biomed. Eng., vol. 42, no. 2, pp. 556\u2013574, 2022.',
    'M. Hires, M. Gazda, P. Drotar, N. D. Pah, M. A. Motin, and D. K. Kumar, "Convolutional neural network ensemble for Parkinson\u2019s disease detection from voice recordings," Comput. Biol. Med., vol. 141, Art. no. 105021, 2022.',
    'I. Tougui et al., "Transformer-based transfer learning on self-reported voice recordings for Parkinson\u2019s disease diagnosis," Sci. Rep., vol. 14, Art. no. 30131, 2024.',
    'N. Madusanka and B.-I. Lee, "Vocal biomarkers for Parkinson\u2019s disease classification using audio spectrogram transformers," J. Voice, 2024.',
    'E. Postma and C. Tejedor-Garcia, "Evaluating pre-trained audio embeddings for Parkinson\u2019s disease speech classification," in Proc. Interspeech 2025, pp. 4603\u20134607, 2025.',
    'M. Shen, P. Mortezaagha, and A. Rahgozar, "Explainable artificial intelligence to diagnose early Parkinson\u2019s disease via voice analysis," Sci. Rep., vol. 15, Art. no. 11687, 2025.',
    'S. Aladhadh, "An explainable ensemble and deep learning framework for Parkinson\u2019s disease detection from voice biomarkers," Diagnostics, vol. 15, no. 22, Art. no. 2892, 2025.',
    'A. Ma, K. K. Lau, and D. Thyagarajan, "Advances in Parkinson\u2019s disease detection using voice and speech: A review," Biomed. Signal Process. Control, vol. 66, Art. no. 102418, 2021.',
    'E. J. Hu et al., "LoRA: Low-rank adaptation of large language models," in Proc. ICLR 2022. [Online]. Available: https://arxiv.org/abs/2106.09685.',
    'A. N. Angelopoulos and S. Bates, "A gentle introduction to conformal prediction," Found. Trends Mach. Learn., vol. 16, no. 4, pp. 494\u2013591, 2023.',
    'R. Chiaramonte and M. Bonfiglio, "Acoustic analysis of voice in Parkinson\u2019s disease: A systematic review," Neurologia, vol. 35, no. 11, pp. 393\u2013405, 2020.',
]
for i, r in enumerate(refs, 1):
    bt(f"[{i}] {r}")

doc.save(DEST)
size = os.path.getsize(DEST)
print(f"Saved: {DEST}")
print(f"Paragraphs: {len(doc.paragraphs)} | Tables: {len(doc.tables)} | Size: {size/1024:.1f} KB")
