"""
leakage_probe_controls.py — Step 5d: Dimensionality-Matched Controls

Reuses embeddings from Step 5a to run controlled experiments, testing if
claims about ModalityGate and SSL leakage were dimensionality artifacts.
"""

import os
import sys
import numpy as np
import warnings
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

def run_permutation_test(clf, X, y, n_permutations=500, n_splits=5):
    """Run permutation test using StratifiedKFold."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Actual score
    actual_score = np.mean(cross_val_score(clf, X, y, cv=cv, scoring='accuracy', n_jobs=-1))
    
    # Null distribution
    null_scores = []
    y_array = np.array(y)
    
    for seed in tqdm(range(n_permutations), desc="Permutations", leave=False):
        np.random.seed(seed)
        y_perm = np.random.permutation(y_array)
        score = np.mean(cross_val_score(clf, X, y_perm, cv=cv, scoring='accuracy', n_jobs=-1))
        null_scores.append(score)
        
    p95 = np.percentile(null_scores, 95)
    p99 = np.percentile(null_scores, 99)
    
    return actual_score, p95, p99

def main():
    print("=" * 70)
    print("  PD-VoiceNet Leakage Probe Controls (Step 5d)")
    print("=" * 70)
    
    embeddings_dir = os.path.join(os.path.dirname(__file__), "artifacts_pdvoicenet", "leakage_embeddings")
    
    # Load all clinical subject embeddings (exclude AH)
    h_subject_list = []
    z_fused_list = []
    h_sub_ssl_list = []
    targets = []
    
    for fname in sorted(os.listdir(embeddings_dir)):
        if not fname.endswith(".npz"): continue
        data = np.load(os.path.join(embeddings_dir, fname))
        site = str(data["site"])
        if site == "AH":
            continue
            
        targets.append(site)
        h_subject_list.append(data["h_subject"])
        z_fused_list.append(data["z_fused"])
        h_sub_ssl_list.append(data["h_sub_ssl"])
        
    X_h_subject = np.stack(h_subject_list)
    X_z_fused = np.stack(z_fused_list)
    X_h_sub_ssl = np.stack(h_sub_ssl_list)
    y = np.array(targets)
    
    print(f"Loaded {len(y)} clinical embeddings from 11 sites.")
    print(f"Starting 500-iteration permutation tests...")
    
    # Define the 5 controls
    experiments = [
        {
            "name": "h_subject-PCA-2D",
            "dim": "2",
            "probe_type": "LogisticRegression (PCA)",
            "clf": Pipeline([('pca', PCA(n_components=2, random_state=42)), ('lr', LogisticRegression(max_iter=1000))]),
            "X": X_h_subject
        },
        {
            "name": "h_subject-random-2D",
            "dim": "2",
            "probe_type": "LogisticRegression (Random)",
            "clf": Pipeline([('rand', GaussianRandomProjection(n_components=2, random_state=42)), ('lr', LogisticRegression(max_iter=1000))]),
            "X": X_h_subject
        },
        {
            "name": "z_fused-MLP",
            "dim": "2",
            "probe_type": "MLP (32 units)",
            "clf": MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42),
            "X": X_z_fused
        },
        {
            "name": "z_fused-RBF-SVM",
            "dim": "2",
            "probe_type": "SVC (RBF)",
            "clf": SVC(kernel='rbf', random_state=42),
            "X": X_z_fused
        },
        {
            "name": "h_sub_ssl-PCA-16D",
            "dim": "16",
            "probe_type": "LogisticRegression (PCA)",
            "clf": Pipeline([('pca', PCA(n_components=16, random_state=42)), ('lr', LogisticRegression(max_iter=1000))]),
            "X": X_h_sub_ssl
        }
    ]
    
    print(f"\n{'Representation':<25} | {'Dim':<3} | {'Probe Type':<25} | {'Accuracy':<10} | {'95th %ile':<10} | {'99th %ile':<10} | p<0.01?")
    print("-" * 110)
    
    for exp in experiments:
        acc, p95, p99 = run_permutation_test(exp["clf"], exp["X"], y, n_permutations=500)
        is_sig = "YES" if acc > p99 else "No"
        print(f"{exp['name']:<25} | {exp['dim']:<3} | {exp['probe_type']:<25} | {acc:>9.2%} | {p95:>9.2%} | {p99:>9.2%} | {is_sig}")

if __name__ == "__main__":
    main()
