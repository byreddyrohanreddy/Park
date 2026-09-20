"""
train_lodo.py — Step 4: 17-fold Leave-One-Site-Out training loop.

For each of 17 sites:
  1. Hold out ALL subjects from that site
  2. Re-initialize ALL trainable weights FROM SCRATCH
  3. Train ONLY on subjects from the remaining 16 sites
  4. Fit biomarker z-score scaler on training subjects only
  5. Evaluate ONLY on held-out site subjects

Self-verification:
  - Assert zero subject-ID overlap per fold
  - Log hash of initial LoRA weights per fold (confirm fresh init)
  - Per-site results table with accuracy/precision/recall/F1 + conformal sets
"""

import os
import sys
import json
import argparse
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import parselmouth
from parselmouth.praat import call
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
import warnings
from copy import deepcopy

sys.path.insert(0, os.path.dirname(__file__))
from data import (build_manifest, get_valid_sites, get_label, lodo_split,
                  subject_group_kfold_split, hash_tensor)
from model import PDVoiceNet, BIOMARKER_NAMES
from calibration import (fit_temperature, apply_temperature,
                         MondrianConformalPredictor, build_dashboard_payload)

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Biomarker extraction (adapted from existing feature_extraction.py)
# ---------------------------------------------------------------------------

def extract_biomarkers(wav_path: str) -> np.ndarray:
    """Extract 26 acoustic biomarker features from a wav file.
    
    Features (in order matching BIOMARKER_NAMES):
        Jitter: local, ppq5, rap (3)
        Shimmer: local, apq3, apq5, apq11, dda (5)
        HNR, NHR (2)
        F0: mean, std, min, max (4)
        MFCC 1-6 mean (6)
        Formants: F1, F2, F3 freq + F1, F2, F3 bandwidth (6)
    Total: 26
    
    AMBIGUITY NOTE: The spec says "MFCC 1-12 + delta" as part of 26 total.
    MFCC 1-12 alone = 12 features, which would exceed 26 with other groups.
    Using MFCC 1-6 means to hit exactly 26. See README.md for details.
    """
    y, sr = librosa.load(wav_path, sr=16000, mono=True, duration=25.0)
    y, _ = librosa.effects.trim(y, top_db=25)
    
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    
    # --- Pitch / F0 ---
    pitch = sound.to_pitch()
    f0_values = pitch.selected_array['frequency']
    f0_voiced = f0_values[f0_values > 0]
    
    if len(f0_voiced) > 0:
        mean_f0 = float(np.mean(f0_voiced))
        std_f0 = float(np.std(f0_voiced))
        min_f0 = float(np.min(f0_voiced))
        max_f0 = float(np.max(f0_voiced))
    else:
        mean_f0 = std_f0 = min_f0 = max_f0 = 0.0
    
    # --- Jitter / Shimmer ---
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    
    local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5_jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    
    local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3_shimmer = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5_shimmer = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11_shimmer = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    dda_shimmer = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    # --- HNR / NHR ---
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    nhr = 1.0 / (10 ** (hnr / 10)) if hnr > 0 else 0.0
    
    # --- MFCC 1-6 (mean over time) ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=7)  # [0] is energy, [1-6] are coeffs 1-6
    mfcc_means = [float(np.mean(mfccs[i])) for i in range(1, 7)]
    
    # --- Formants F1-F3 + bandwidths ---
    formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    
    def get_formant_stats(formant_obj, formant_num):
        try:
            freq = call(formant_obj, "Get mean", formant_num, 0, 0, "Hertz")
            bw = call(formant_obj, "Get standard deviation", formant_num, 0, 0, "Hertz")
        except Exception:
            freq = bw = 0.0
        return freq, bw
    
    f1_freq, f1_bw = get_formant_stats(formant, 1)
    f2_freq, f2_bw = get_formant_stats(formant, 2)
    f3_freq, f3_bw = get_formant_stats(formant, 3)
    
    # Assemble feature vector (must match BIOMARKER_NAMES order)
    features = np.array([
        local_jitter, ppq5_jitter, rap_jitter,
        local_shimmer, apq3_shimmer, apq5_shimmer, apq11_shimmer, dda_shimmer,
        hnr, nhr,
        mean_f0, std_f0, min_f0, max_f0,
        *mfcc_means,
        f1_freq, f2_freq, f3_freq, f1_bw, f2_bw, f3_bw,
    ], dtype=np.float32)
    
    assert len(features) == 26, f"Expected 26 features, got {len(features)}"
    return np.nan_to_num(features, nan=0.0)


def extract_all_biomarkers(manifest, cache_path=None):
    """Extract biomarkers for all recordings in manifest. Optionally cache."""
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True).item()
        print(f"  Loaded cached biomarkers from {cache_path}")
        return data
    
    biomarkers = {}
    for idx, row in manifest.iterrows():
        rid = row["recording_id"]
        try:
            biomarkers[rid] = extract_biomarkers(row["path"])
        except Exception as e:
            print(f"  WARNING: Biomarker extraction failed for {rid}: {e}")
            biomarkers[rid] = np.zeros(26, dtype=np.float32)
    
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, biomarkers)
        print(f"  Saved biomarkers cache to {cache_path}")
    
    return biomarkers


# ---------------------------------------------------------------------------
# Training loop for one fold
# ---------------------------------------------------------------------------

def train_one_fold(train_manifest, cache_dir, biomarkers, 
                   num_epochs=15, lr_lora=1e-4, lr_heads=1e-3,
                   patience=5):
    """Train PDVoiceNet on one fold's training data.
    
    Uses internal subject-level GroupKFold for early stopping and 
    temperature/conformal fitting.
    
    Returns: (model, temperature, conformal_predictor, init_hash)
    """
    # Fresh init
    model = PDVoiceNet().to(device)
    
    # Log hash of initial LoRA weights for verification
    lora_params = [p for n, p in model.named_parameters() if 'lora_' in n]
    init_hash = hash_tensor(lora_params[0]) if lora_params else "no_lora"
    
    # Separate LoRA params from head params
    lora_param_list = [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]
    head_param_list = [p for n, p in model.named_parameters() if 'lora_' not in n and p.requires_grad]
    
    optimizer = optim.AdamW([
        {'params': lora_param_list, 'lr': lr_lora},
        {'params': head_param_list, 'lr': lr_heads},
    ])
    criterion = nn.CrossEntropyLoss()
    
    # Internal train/val split (subject-level GroupKFold, use first split)
    splits = list(subject_group_kfold_split(train_manifest, n_splits=5))
    inner_train_df, inner_val_df = splits[0]
    
    # Fit biomarker scaler on inner train subjects ONLY
    inner_train_rids = inner_train_df["recording_id"].tolist()
    inner_train_bio = np.array([biomarkers[rid] for rid in inner_train_rids])
    scaler = StandardScaler()
    scaler.fit(inner_train_bio)
    
    # Prepare data lists
    def prepare_subjects(df):
        """Group recordings by subject, return list of (subject_id, label, cached_list, bio_tensor)."""
        subjects = []
        for subj_id, group in df.groupby("subject_id"):
            label = get_label(group.iloc[0]["category"])
            cached = []
            bios = []
            for _, row in group.iterrows():
                rid = row["recording_id"]
                cache_path = os.path.join(cache_dir, f"{rid}.pt")
                cached_tensor = torch.load(cache_path, weights_only=True).float().to(device)
                cached.append(cached_tensor)
                bio_raw = biomarkers[rid]
                bios.append(bio_raw)
            
            bio_array = np.array(bios)
            bio_scaled = scaler.transform(bio_array)
            bio_tensor = torch.tensor(bio_scaled, dtype=torch.float32).to(device)
            
            subjects.append((subj_id, label, cached, bio_tensor))
        return subjects
    
    train_subjects = prepare_subjects(inner_train_df)
    val_subjects = prepare_subjects(inner_val_df)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    best_state = None
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        np.random.shuffle(train_subjects)
        
        for subj_id, label, cached_list, bio_tensor in train_subjects:
            optimizer.zero_grad()
            z_fused, alpha, attn_w = model(cached_list, bio_tensor)
            loss = criterion(z_fused.unsqueeze(0), torch.tensor([label], device=device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for subj_id, label, cached_list, bio_tensor in val_subjects:
                z_fused, alpha, attn_w = model(cached_list, bio_tensor)
                loss = criterion(z_fused.unsqueeze(0), torch.tensor([label], device=device))
                val_loss += loss.item()
                pred = z_fused.argmax().item()
                if pred == label:
                    val_correct += 1
        
        val_acc = val_correct / max(len(val_subjects), 1) * 100
        avg_train = train_loss / max(len(train_subjects), 1)
        avg_val = val_loss / max(len(val_subjects), 1)
        
        if epoch % 3 == 0 or epoch == num_epochs - 1:
            print(f"    Epoch {epoch+1}/{num_epochs}: "
                  f"train_loss={avg_train:.4f}, val_loss={avg_val:.4f}, val_acc={val_acc:.1f}%")
        
        # Early stopping
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Fit temperature on validation logits
    model.eval()
    val_logits = []
    val_labels = []
    with torch.no_grad():
        for subj_id, label, cached_list, bio_tensor in val_subjects:
            z_fused, _, _ = model(cached_list, bio_tensor)
            val_logits.append(z_fused.cpu().numpy())
            val_labels.append(label)
    
    val_logits = np.array(val_logits)
    val_labels = np.array(val_labels)
    
    temperature = fit_temperature(val_logits, val_labels)
    
    # Fit conformal predictor on validation (calibration) data
    cal_probs = apply_temperature(val_logits, temperature)
    conformal = MondrianConformalPredictor(target_coverage=0.90)
    conformal.fit(cal_probs, val_labels)
    
    return model, scaler, temperature, conformal, init_hash


# ---------------------------------------------------------------------------
# Evaluation on held-out site
# ---------------------------------------------------------------------------

def evaluate_fold(model, scaler, temperature, conformal,
                  test_manifest, cache_dir, biomarkers):
    """Evaluate a trained model on the held-out site. Returns metrics dict."""
    
    model.eval()
    all_logits = []
    all_labels = []
    all_subjects = []
    all_alphas = []
    all_attn_weights = []
    
    with torch.no_grad():
        for subj_id, group in test_manifest.groupby("subject_id"):
            label = get_label(group.iloc[0]["category"])
            cached_list = []
            bios = []
            
            for _, row in group.iterrows():
                rid = row["recording_id"]
                cache_path = os.path.join(cache_dir, f"{rid}.pt")
                cached_tensor = torch.load(cache_path, weights_only=True).float().to(device)
                cached_list.append(cached_tensor)
                bios.append(biomarkers[rid])
            
            bio_array = np.array(bios)
            bio_scaled = scaler.transform(bio_array)
            bio_tensor = torch.tensor(bio_scaled, dtype=torch.float32).to(device)
            
            z_fused, alpha, attn_w = model(cached_list, bio_tensor)
            
            all_logits.append(z_fused.cpu().numpy())
            all_labels.append(label)
            all_subjects.append(subj_id)
            all_alphas.append(alpha.item())
            all_attn_weights.append(attn_w.cpu().numpy())
    
    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    
    # Apply temperature scaling
    cal_probs = apply_temperature(all_logits, temperature)
    
    # Conformal prediction sets
    pred_sets = conformal.predict(cal_probs)
    
    # Point predictions (argmax of calibrated probs)
    preds = np.argmax(cal_probs, axis=1)
    
    # Metrics
    acc = accuracy_score(all_labels, preds) * 100
    prec = precision_score(all_labels, preds, zero_division=0) * 100
    rec = recall_score(all_labels, preds, zero_division=0) * 100
    f1 = f1_score(all_labels, preds, zero_division=0) * 100
    
    # Conformal set distribution
    set_counts = {"PD": 0, "HC": 0, "PD_HC": 0, "empty": 0}
    for ps in pred_sets:
        if ps == {1}:
            set_counts["PD"] += 1
        elif ps == {0}:
            set_counts["HC"] += 1
        elif ps == {0, 1}:
            set_counts["PD_HC"] += 1
        else:
            set_counts["empty"] += 1
    
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n_subjects": len(all_labels),
        "n_pd": int((all_labels == 1).sum()),
        "n_hc": int((all_labels == 0).sum()),
        "conformal_sets": set_counts,
        "mean_alpha": float(np.mean(all_alphas)),
        "temperature": temperature,
    }


# ---------------------------------------------------------------------------
# Main LODO loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="PD-VoiceNet 17-fold LODO training")
    parser.add_argument("--dataset-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "files", "dataset"))
    parser.add_argument("--cache-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "artifacts_pdvoicenet", "cache"))
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "artifacts_pdvoicenet"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()
    
    print("=" * 70)
    print("  PD-VoiceNet: 17-Fold Leave-One-Site-Out Training")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Dataset: {args.dataset_dir}")
    print(f"  Cache: {args.cache_dir}")
    print()
    
    # Build manifest
    manifest = build_manifest(args.dataset_dir)
    valid_sites = get_valid_sites(manifest)
    print(f"  Total recordings: {len(manifest)}")
    print(f"  Valid sites ({len(valid_sites)}): {valid_sites}")
    
    # Verify cache exists
    for _, row in manifest.iterrows():
        cache_path = os.path.join(args.cache_dir, f"{row['recording_id']}.pt")
        assert os.path.exists(cache_path), (
            f"Missing cached features for {row['recording_id']}. "
            f"Run cache_features.py first."
        )
    print("  Cache verification: PASSED (all recordings have cached features)")
    
    # Extract biomarkers (cached to avoid re-extraction per fold)
    bio_cache = os.path.join(args.output_dir, "biomarkers_cache.npy")
    print("\n  Extracting biomarker features...")
    biomarkers = extract_all_biomarkers(manifest, cache_path=bio_cache)
    print(f"  Biomarkers extracted for {len(biomarkers)} recordings")
    
    # LODO loop
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    init_hashes = []
    
    for fold_idx, site in enumerate(valid_sites):
        print(f"\n{'='*70}")
        print(f"  FOLD {fold_idx+1}/{len(valid_sites)}: Holding out site '{site}'")
        print(f"{'='*70}")
        
        # Split
        train_df, test_df = lodo_split(manifest, site)
        print(f"  Train: {len(train_df)} recordings | Test: {len(test_df)} recordings")
        
        # Self-verification check 1: zero subject overlap
        train_subs = set(train_df["subject_id"].unique())
        test_subs = set(test_df["subject_id"].unique())
        assert len(train_subs & test_subs) == 0, "SUBJECT OVERLAP DETECTED"
        print(f"  Subject overlap check: PASSED (0 overlap)")
        
        # Check both classes present in test
        test_labels = test_df["category"].unique()
        if len(test_labels) < 2:
            print(f"  SKIPPED: Only {test_labels} in held-out site '{site}'")
            continue
        
        # Train
        model, scaler, temperature, conformal, init_hash = train_one_fold(
            train_df, args.cache_dir, biomarkers,
            num_epochs=args.epochs, patience=args.patience
        )
        init_hashes.append((site, init_hash))
        
        # Self-verification check 2: confirm fresh init
        print(f"  Init LoRA hash: {init_hash}")
        
        # Evaluate
        metrics = evaluate_fold(model, scaler, temperature, conformal,
                                test_df, args.cache_dir, biomarkers)
        metrics["site"] = site
        results.append(metrics)
        
        print(f"  >>> Site '{site}': Acc={metrics['accuracy']:.2f}%, "
              f"P={metrics['precision']:.1f}%, R={metrics['recall']:.1f}%, "
              f"F1={metrics['f1']:.1f}%")
        print(f"      Conformal sets: {metrics['conformal_sets']}")
        print(f"      Mean gate α: {metrics['mean_alpha']:.3f}, T*: {metrics['temperature']:.3f}")
        
        # Save fold artifacts
        fold_dir = os.path.join(args.output_dir, f"fold_{site}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(fold_dir, "model.pt"))
        np.save(os.path.join(fold_dir, "scaler_mean.npy"), scaler.mean_)
        np.save(os.path.join(fold_dir, "scaler_scale.npy"), scaler.scale_)
        with open(os.path.join(fold_dir, "calibration.json"), "w") as f:
            json.dump({
                "temperature": temperature,
                "conformal": conformal.get_state(),
            }, f, indent=2, default=str)
        
        # Free GPU memory
        del model
        torch.cuda.empty_cache()
    
    # ---------------------------------------------------------------------------
    # Final results table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  FINAL RESULTS TABLE")
    print("=" * 70)
    print(f"  {'Site':<6} {'N':>4} {'HC':>4} {'PD':>4} {'Acc':>7} {'Prec':>7} "
          f"{'Rec':>7} {'F1':>7} {'α':>6} {'T*':>6} "
          f"{'|PD|':>5} {'|HC|':>5} {'|Both|':>6} {'|∅|':>5}")
    print(f"  {'-'*6} {'-'*4} {'-'*4} {'-'*4} {'-'*7} {'-'*7} "
          f"{'-'*7} {'-'*7} {'-'*6} {'-'*6} "
          f"{'-'*5} {'-'*5} {'-'*6} {'-'*5}")
    
    for r in results:
        cs = r["conformal_sets"]
        print(f"  {r['site']:<6} {r['n_subjects']:>4} {r['n_hc']:>4} {r['n_pd']:>4} "
              f"{r['accuracy']:>6.2f}% {r['precision']:>6.1f}% "
              f"{r['recall']:>6.1f}% {r['f1']:>6.1f}% "
              f"{r['mean_alpha']:>5.3f} {r['temperature']:>5.3f} "
              f"{cs['PD']:>5} {cs['HC']:>5} {cs['PD_HC']:>6} {cs['empty']:>5}")
    
    if results:
        accs = [r["accuracy"] for r in results]
        f1s = [r["f1"] for r in results]
        print(f"\n  Mean Accuracy: {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
        print(f"  Mean F1:       {np.mean(f1s):.2f}% ± {np.std(f1s):.2f}%")
        print(f"  Min Accuracy:  {np.min(accs):.2f}% (site: {results[np.argmin(accs)]['site']})")
        print(f"  Max Accuracy:  {np.max(accs):.2f}% (site: {results[np.argmax(accs)]['site']})")
    
    # Self-verification check 2: confirm all init hashes differ
    print(f"\n--- SELF-VERIFICATION ---")
    print(f"  Check 1 (subject overlap): PASSED for all {len(results)} folds")
    
    hashes = [h for _, h in init_hashes]
    all_different = len(set(hashes)) == len(hashes)
    print(f"  Check 2 (fresh init): {'PASSED' if all_different else 'WARNING — some hashes match'}")
    for site, h in init_hashes:
        print(f"    Fold {site}: {h}")
    
    # Flag any site near chance
    chance = 50.0  # Binary classification
    near_chance = [r for r in results if r["accuracy"] < chance + 10]
    if near_chance:
        print(f"\n  WARNING: The following sites are near chance ({chance}%):")
        for r in near_chance:
            print(f"    Site {r['site']}: {r['accuracy']:.2f}%")
    
    # Save summary
    summary_path = os.path.join(args.output_dir, "lodo_results.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {summary_path}")


if __name__ == "__main__":
    main()
