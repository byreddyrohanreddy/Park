# PD-VoiceNet: Subject-Gated Multimodal Fusion

## Architecture

PD-VoiceNet is a subject-level multimodal fusion model for Parkinson's Disease 
voice classification. It combines:

1. **SSL Branch**: Frozen Wav2Vec2-base (layers 0-8, cached) + LoRA-adapted 
   transformer layers 9-11 (r=4, α=16)
2. **Biomarker Branch**: 26 acoustic features → MLP (26→32→16)
3. **Subject Attention Pooling**: Learned-query attention over variable-N recordings
4. **Modality Gate**: Scalar gate from biomarker sub-embedding only (α ∈ [0,1])
5. **Fusion Head**: α·z_ssl + (1-α)·z_bio → classification logits

Post-training calibration: temperature scaling + Mondrian conformal prediction 
at 90% coverage.

## How to Run

### Prerequisites
```bash
conda activate ml
pip install torch transformers librosa praat-parselmouth scikit-learn pandas scipy
```

### Step 1: Cache frozen features (run once)
```bash
cd PD/backend/ml/pd_voicenet
python cache_features.py --dataset-dir ../../../../files/dataset
```
This extracts Wav2Vec2-base layers 0-8 activations for all 908 recordings and 
saves them to `artifacts_pdvoicenet/cache/`. Takes ~5 minutes on GPU.

### Step 2: Run 17-fold LODO training
```bash
python train_lodo.py --dataset-dir ../../../../files/dataset --epochs 15
```
This trains 12 separate models (one per valid site, sites with <5 recordings or 
single-class sites are skipped). Each fold:
- Completely excludes the held-out site from training
- Re-initializes ALL weights from scratch
- Fits biomarker scaler on training subjects only
- Uses internal subject-level GroupKFold for early stopping + calibration

## File Structure
```
pd_voicenet/
├── cache_features.py       # Step 1: frozen Wav2Vec2 caching
├── model.py                # Step 2: full architecture (6 nn.Module classes)
├── calibration.py          # Step 3: temperature scaling + conformal
├── train_lodo.py           # Step 4: 17-fold LODO loop + biomarker extraction
├── data.py                 # Shared: manifest, parsing, GroupKFold helpers
├── artifacts_pdvoicenet/   # Output: per-fold weights + results
│   ├── cache/              # Cached Wav2Vec2 activations
│   ├── fold_AH/            # Per-fold model weights + calibration
│   ├── fold_B1/
│   ├── ...
│   ├── biomarkers_cache.npy
│   └── lodo_results.json
└── README.md               # This file
```

## Known Ambiguities / Design Decisions

### MFCC Feature Count
The spec lists "MFCC 1-12 + delta" as part of exactly 26 total features. 
MFCC 1-12 alone = 12 features; with deltas = 24, which combined with the other 
feature groups would far exceed 26. 

**Decision**: Used MFCC 1-6 means (6 features) to hit exactly 26 total:
- Jitter: local, ppq5, rap (3)
- Shimmer: local, apq3, apq5, apq11, dda (5)
- HNR, NHR (2)
- F0: mean, std, min, max (4)
- MFCC 1-6 mean (6)
- Formants F1-F3 + BW (6)

### Single-Recording Subjects
All 908 recordings map to 908 unique subjects. SubjectAttentionPooling always 
receives N=1, meaning the attention weight is trivially 1.0. The code correctly 
handles arbitrary N, but this dataset does not exercise the multi-recording path.

### Wav2Vec2 Layer Numbering
"Layers 0-8" = CNN feature encoder + transformer layers 0-8 (9 transformer layers).
"Layers 9-11" = final 3 transformer layers with LoRA adapters.
The cached tensor is `hidden_states[9]` from HuggingFace's output (output of 
transformer layer 8, zero-indexed).

## Self-Verification

### Check 1: Subject Overlap
Asserted zero subject-ID overlap between each fold's train set and held-out site.

### Check 2: Fresh Initialization
Logged MD5 hash of initial LoRA weights per fold. All hashes should differ 
(PyTorch random init with different seeds per instantiation).

## Results Table

*(Populated after running train_lodo.py)*

| Site | N | HC | PD | Accuracy | Precision | Recall | F1 | α | T* | {PD} | {HC} | {PD,HC} | ∅ |
|------|---|----|----|----------|-----------|--------|-----|---|-----|------|------|---------|---|
| B1   | 64| 36 | 28 | 98.44%   | 100.0%    | 96.4%  | 98.2%| 0.339| 1.260 | 27 | 37 | 0 | 0 |
| B2   | 60| 36 | 24 | 98.33%   | 96.0%     | 100.0% | 98.0%| 0.409| 1.155 | 23 | 35 | 0 | 2 |
| D1   | 50| 22 | 28 | 96.00%   | 93.3%     | 100.0% | 96.6%| 0.519| 0.770 | 29 | 19 | 0 | 2 |
| D2   | 50| 22 | 28 | 96.00%   | 96.4%     | 96.4%  | 96.4%| 0.393| 0.622 | 28 | 22 | 0 | 0 |
| FB   | 47| 21 | 26 | 100.00%  | 100.0%    | 100.0% | 100.0%|0.407| 1.628 | 26 | 21 | 0 | 0 |
| PR   | 65| 37 | 28 | 95.38%   | 90.3%     | 100.0% | 94.9%| 0.425| 1.193 | 29 | 33 | 0 | 3 |
| VA   | 99| 44 | 55 | 91.92%   | 87.3%     | 100.0% | 93.2%| 0.397| 1.034 | 60 | 39 | 0 | 0 |
| VE   | 99| 44 | 55 | 64.65%   | 100.0%    | 36.4%  | 53.3%| 0.327| 1.336 | 20 | 73 | 6 | 0 |
| VI   | 99| 44 | 55 | 88.89%   | 97.8%     | 81.8%  | 89.1%| 0.449| 1.670 | 41 | 51 | 0 | 7 |
| VO   | 99| 44 | 55 | 92.93%   | 96.2%     | 90.9%  | 93.5%| 0.309| 1.843 | 52 | 47 | 0 | 0 |
| VU   | 95| 42 | 53 | 91.58%   | 86.9%     | 100.0% | 93.0%| 0.320| 1.244 | 63 | 30 | 0 | 2 |

**11-Site Clinical Average:** Accuracy: 92.19% | F1: 91.47%

### Out-of-Distribution Benchmark (AH)

Site AH is the original Iyer et al. public benchmark dataset, not a clinical site.

| Site | N | HC | PD | Accuracy | Precision | Recall | F1 | α | T* | {PD} | {HC} | {PD,HC} | ∅ |
|------|---|----|----|----------|-----------|--------|-----|---|-----|------|------|---------|---|
| AH   | 81| 41 | 40 | 49.38%   | 48.8%     | 50.0%  | 49.4%| 0.280| 1.047 | 25 | 20 | 0 | 36 |

> [!WARNING]
> **Performance Anomalies on Held-Out Sites**
> - **Site AH**: Accuracy is 49.38% (chance level). This suggests a severe domain shift or demographic difference at site AH that the model fails to generalize to when AH is completely held out during training. The large empty-set rate on AH is consistent with appropriate caution under distribution shift, though the formal 90% coverage guarantee does not strictly apply outside the exchangeability assumption, which AH violates by construction.
> - **Site VE**: Accuracy is 64.65% with 100% precision but very low recall (36.4%). The model is overly conservative in predicting PD for site VE.

> [!NOTE]
> **Legacy DANN Probe Caveat**
> The leakage-probe results (74.11% → 39.98% → 47.58%) from the earlier DANN phase were computed under a buggy 17-"site" labeling scheme (where sites like VA and VA1 were erroneously split). If revisiting that pipeline, those numbers must be re-run under this corrected 12-site grouping before being cited.

## Device/Site-Leakage Probe (Step 5)

To evaluate whether the trained PD-VoiceNet representations inadvertently encode site identity, a 500-iteration permutation test was performed on the out-of-fold embeddings of the 11 clinical sites (827 subjects). 

### Leakage Results Table

| Representation | Dim | Probe Accuracy | Null 95th %ile | Null 99th %ile | p < 0.01? |
|----------------|-----|----------------|----------------|----------------|-----------|
| `h_sub_bio`    | 16  | 97.46%         | 13.66%         | 14.26%         | **YES**   |
| `h_sub_ssl`    | 128 | 99.76%         | 12.34%         | 13.54%         | **YES**   |
| `h_subject`    | 144 | 99.88%         | 12.21%         | 13.30%         | **YES**   |
| `z_fused`      | 2   | 33.37%         | 13.91%         | 14.52%         | **YES**   |

### Interpretation
- **Universal Leakage**: All intermediate representations encode site identity at a highly statistically significant level (p < 0.01), vastly outperforming the random chance null distributions (~9% for 11 sites).
- **Modality Gate Protection (`z_fused` vs `h_subject`)**: The ModalityGate and FusionHead provide a massive protective bottleneck. By compressing the highly-leaky 144-dimensional `h_subject` (99.88% leak) down to a 2D scalar logit (`z_fused`), the probe accuracy is crippled down to 33.37%. While still statistically significant, the architecture structurally scrubs the vast majority of accessible site identity before the final prediction.
- **Biomarker vs SSL Leakage**: Surprisingly, the raw biomarker branch (`h_sub_bio` at 97.46%) leaks slightly *less* than the SSL branch (`h_sub_ssl` at 99.76%). This contradicts the prior hypothesis that acoustic features (like HNR) were the sole/primary driver of site leakage; the deep Wav2Vec2 embeddings are essentially maximally leaky.

### AH Sanity Check
A binary classification probe (AH vs 11 Clinical Sites) using `h_subject` achieved **99.67%** cross-validated accuracy. 
*Note: This is a sanity check, not a scientific finding. It confirms that the probe methodology correctly and trivially separates the vastly out-of-distribution public benchmark dataset (AH) from the real clinical cohort.*

## Step 5d: Dimensionality-Matched Controls

To rule out linear-probing dimensionality artifacts from the Step 5 results, controlled probes were evaluated on matched-dimension representations and with nonlinear capacities.

| Representation        | Dim | Probe Type                  | Accuracy | Null 95th %ile | Null 99th %ile | p<0.01? |
|-----------------------|-----|-----------------------------|----------|----------------|----------------|---------|
| `h_subject`-PCA-2D    | 2   | LogisticRegression (PCA)    | 35.43%   | 13.66%         | 14.63%         | **YES** |
| `h_subject`-random-2D | 2   | LogisticRegression (Random) | 24.67%   | 13.91%         | 14.63%         | **YES** |
| `z_fused`-MLP         | 2   | MLP (32 units)              | 68.57%   | 13.79%         | 14.75%         | **YES** |
| `z_fused`-RBF-SVM     | 2   | SVC (RBF)                   | 50.78%   | 14.15%         | 15.12%         | **YES** |
| `h_sub_ssl`-PCA-16D   | 16  | LogisticRegression (PCA)    | 95.52%   | 13.42%         | 14.14%         | **YES** |

### Control Verdicts on Step 5 Claims

1. **Claim: "The ModalityGate provides a massive protective effect"**
   **Verdict: NOT SUPPORTED (likely dimensionality artifact)**
   The apparent protection observed previously (33.37% linear accuracy on `z_fused`) was entirely an artifact of linear compression into a 2D space. When the full 144-dimensional `h_subject` is generically compressed to 2D via PCA without any learned gating, it leaks just as much (35.43%). Furthermore, when a nonlinear probe (MLP) is applied to the actual `z_fused` representation, it extracts 68.57% site accuracy. The ModalityGate does not scrub the site identity; it merely tangles it nonlinearly.

2. **Claim: "SSL embeddings are leakier than biomarker features"**
   **Verdict: NOT SUPPORTED (likely dimensionality artifact)**
   The previously observed gap (SSL 99.76% vs Bio 97.46%) was an artifact of the SSL branch having 128 dimensions to the Bio branch's 16. When `h_sub_ssl` is compressed to a matching 16 dimensions via PCA, its leakage drops to 95.52% — landing *below* the biomarker branch's 97.46%. This confirms that the original 128-vs-16 dimension gap explained the difference, and intrinsically, the acoustic biomarker branch remains slightly more leaky per-dimension (aligning with the original HNR hypothesis).
