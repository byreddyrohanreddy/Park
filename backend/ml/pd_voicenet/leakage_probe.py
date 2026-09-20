"""
leakage_probe.py — Step 5: Device/Site-Leakage Probing on PD-VoiceNet

Extracts out-of-fold embeddings from trained models, then trains
a Logistic Regression probe to predict site identity, establishing
statistical significance via permutation testing.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from data import build_manifest, get_site
from model import PDVoiceNet
from train_lodo import extract_all_biomarkers

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_fold_model(site: str, output_dir: str):
    """Load model and scaler for the fold where `site` was held out."""
    fold_dir = os.path.join(output_dir, f"fold_{site}")
    model = PDVoiceNet().to(device)
    model.load_state_dict(torch.load(os.path.join(fold_dir, "model.pt"), map_location=device, weights_only=True))
    model.eval()
    
    scaler_mean = np.load(os.path.join(fold_dir, "scaler_mean.npy"))
    scaler_scale = np.load(os.path.join(fold_dir, "scaler_scale.npy"))
    
    return model, scaler_mean, scaler_scale


def extract_subject_embeddings(model, scaler_mean, scaler_scale, 
                               group_df, cache_dir, biomarkers):
    """Manually run forward pass to extract intermediate representations for a subject."""
    
    cached_list = []
    bios = []
    
    for _, row in group_df.iterrows():
        rid = row["recording_id"]
        cache_path = os.path.join(cache_dir, f"{rid}.pt")
        cached_tensor = torch.load(cache_path, map_location=device, weights_only=True).float()
        cached_list.append(cached_tensor)
        bios.append(biomarkers[rid])
        
    bio_array = np.array(bios)
    # Apply standard scaling
    bio_scaled = (bio_array - scaler_mean) / scaler_scale
    bio_tensor = torch.tensor(bio_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # Replicate PDVoiceNet forward pass to intercept intermediates
        recording_vectors = []
        for i in range(len(cached_list)):
            h_ssl = model.ssl_branch(cached_list[i])
            h_bio = model.bio_branch(bio_tensor[i])
            h_concat = torch.cat([h_ssl, h_bio], dim=0)
            recording_vectors.append(h_concat)
            
        recording_vectors = torch.stack(recording_vectors, dim=0)
        h_subject, _ = model.attention_pool(recording_vectors)
        
        h_sub_ssl = h_subject[:128]
        h_sub_bio = h_subject[128:]
        
        alpha = model.modality_gate(h_sub_bio)
        z_fused, _ = model.fusion_head(h_sub_ssl, h_sub_bio, alpha)
        
    return {
        "h_sub_bio": h_sub_bio.cpu().numpy(),
        "h_sub_ssl": h_sub_ssl.cpu().numpy(),
        "h_subject": h_subject.cpu().numpy(),
        "z_fused": z_fused.cpu().numpy()
    }


def permutation_test(X, y, n_permutations=500, n_splits=5):
    """Run permutation test using StratifiedKFold on subject-level features."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    
    # Actual score
    actual_score = np.mean(cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1))
    
    # Null distribution
    null_scores = []
    y_array = np.array(y)
    
    # We shuffle labels directly since each row is a distinct subject
    for seed in tqdm(range(n_permutations), desc="Permutations", leave=False):
        np.random.seed(seed)
        y_perm = np.random.permutation(y_array)
        score = np.mean(cross_val_score(clf, X, y_perm, cv=cv, scoring='accuracy', n_jobs=-1))
        null_scores.append(score)
        
    p95 = np.percentile(null_scores, 95)
    p99 = np.percentile(null_scores, 99)
    
    return actual_score, p95, p99


def main():
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "files", "dataset")
    cache_dir = os.path.join(os.path.dirname(__file__), "artifacts_pdvoicenet", "cache")
    output_dir = os.path.join(os.path.dirname(__file__), "artifacts_pdvoicenet")
    
    print("=" * 70)
    print("  PD-VoiceNet Device-Leakage Probe (Step 5)")
    print("=" * 70)
    
    manifest = build_manifest(dataset_dir)
    bio_cache = os.path.join(output_dir, "biomarkers_cache.npy")
    biomarkers = extract_all_biomarkers(manifest, cache_path=bio_cache)
    
    # Identify clinical sites vs AH
    sites = [get_site(f) for f in manifest["path"]]
    clinical_sites = sorted(list(set(sites) - {"AH"}))
    
    # Filter clinical subjects
    clinical_df = manifest[manifest["path"].apply(lambda p: get_site(p) in clinical_sites)]
    
    print(f"\nStep 5a: Extracting out-of-fold embeddings for {len(clinical_df)} clinical subjects...")
    
    # Cache out-of-fold embeddings
    embeddings_cache_dir = os.path.join(output_dir, "leakage_embeddings")
    os.makedirs(embeddings_cache_dir, exist_ok=True)
    
    extracted_data = defaultdict(list)
    targets = []
    
    # Cache models in memory to avoid reloading from disk 900 times
    model_cache = {}
    
    for subj_id, group_df in tqdm(clinical_df.groupby("subject_id"), desc="Extracting"):
        site = get_site(group_df.iloc[0]["path"])
        
        # We load the model where this site was held out
        if site not in model_cache:
            model_cache[site] = load_fold_model(site, output_dir)
        model, scaler_mean, scaler_scale = model_cache[site]
        
        reps = extract_subject_embeddings(
            model, scaler_mean, scaler_scale, group_df, cache_dir, biomarkers
        )
        
        for rep_name, vec in reps.items():
            extracted_data[rep_name].append(vec)
        targets.append(site)
        
        # Optionally save to disk (as requested by spec)
        np.savez(
            os.path.join(embeddings_cache_dir, f"{subj_id}.npz"),
            site=site,
            **reps
        )
    
    # Convert lists to arrays
    X_reps = {name: np.stack(vecs) for name, vecs in extracted_data.items()}
    y_clinical = np.array(targets)
    
    print(f"\nStep 5b: Permutation Tests (500 iterations, 5-fold CV)")
    print(f"{'Representation':<15} | {'Dim':<5} | {'Accuracy':<10} | {'95th %ile':<10} | {'99th %ile':<10} | p<0.01?")
    print("-" * 75)
    
    results = {}
    for rep_name in ["h_sub_bio", "h_sub_ssl", "h_subject", "z_fused"]:
        X = X_reps[rep_name]
        acc, p95, p99 = permutation_test(X, y_clinical, n_permutations=500)
        is_sig = "YES" if acc > p99 else "No"
        
        results[rep_name] = {
            "accuracy": acc, "p95": p95, "p99": p99, "significant": is_sig
        }
        
        print(f"{rep_name:<15} | {X.shape[1]:<5} | {acc:>9.2%} | {p95:>9.2%} | {p99:>9.2%} | {is_sig}")
        
    print("\nStep 5c: AH Sanity Check")
    # Extract AH embeddings using any clinical fold (e.g. fold_B1)
    # Since AH was not used to train any model, ANY model is out-of-fold for AH.
    ah_df = manifest[manifest["path"].apply(lambda p: get_site(p) == "AH")]
    ah_reps = []
    
    model_b1, scaler_mean_b1, scaler_scale_b1 = load_fold_model("B1", output_dir)
    
    for subj_id, group_df in tqdm(ah_df.groupby("subject_id"), desc="Extracting AH"):
        reps = extract_subject_embeddings(
            model_b1, scaler_mean_b1, scaler_scale_b1, group_df, cache_dir, biomarkers
        )
        ah_reps.append(reps["h_subject"])
        
        np.savez(
            os.path.join(embeddings_cache_dir, f"{subj_id}.npz"),
            site="AH",
            **reps
        )
        
    X_ah = np.stack(ah_reps)
    y_ah = np.ones(len(X_ah))  # AH = 1
    
    # Mix with clinical (using h_subject representation)
    X_clinical_h_subject = X_reps["h_subject"]
    y_clinical_zeros = np.zeros(len(X_clinical_h_subject))  # Clinical = 0
    
    X_mixed = np.concatenate([X_clinical_h_subject, X_ah], axis=0)
    y_mixed = np.concatenate([y_clinical_zeros, y_ah], axis=0)
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    ah_acc = np.mean(cross_val_score(clf, X_mixed, y_mixed, cv=cv, scoring='accuracy', n_jobs=-1))
    
    print(f"  Task: Binary classification (AH vs 11 Clinical Sites)")
    print(f"  Representation: h_subject")
    print(f"  Cross-validated Accuracy: {ah_acc:>9.2%}")
    if ah_acc > 0.95:
        print("  Sanity check PASSED: Probe trivially separates AH from clinical data.")
    else:
        print("  WARNING: Sanity check failed. Probe struggles to separate AH.")


if __name__ == "__main__":
    main()
