"""
shap_explainer.py — Step A: SHAP explainer for PD-VoiceNet BiomarkerBranch.

Wraps the full model path from raw features to scalar output (PD prob),
holding SSL contributions constant.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
import shap

import sys
sys.path.insert(0, os.path.dirname(__file__) + "/..")
from model import BIOMARKER_NAMES


def get_training_background_data(manifest_df, site_held_out, biomarkers_dict):
    """
    Get (M, 26) background data from training subjects only.
    Each subject's recordings are averaged to form a (26,) vector.
    """
    from data import get_site
    
    # Filter to only subjects NOT in the held-out site
    train_df = manifest_df[manifest_df["path"].apply(lambda p: get_site(p) != site_held_out)]
    
    background = []
    for subj_id, group_df in train_df.groupby("subject_id"):
        subj_bios = []
        for _, row in group_df.iterrows():
            rid = row["recording_id"]
            if rid in biomarkers_dict:
                subj_bios.append(biomarkers_dict[rid])
        if subj_bios:
            # Mean over recordings for this subject
            background.append(np.mean(subj_bios, axis=0))
            
    return np.array(background)


def explain_subject(subject_id, original_recordings, h_ssl_fixed, fold_model, 
                    scaler_mean, scaler_scale, background_data, device="cuda"):
    """
    Computes SHAP values for the 26 raw acoustic features for a specific subject.
    
    Args:
        subject_id: str
        original_recordings: (N, 26) raw biomarker features for this subject
        h_ssl_fixed: (N, 128) fixed SSL embeddings for this subject
        fold_model: PDVoiceNet model loaded for the appropriate fold
        scaler_mean: (26,) array
        scaler_scale: (26,) array
        background_data: (M, 26) array of background training data
        device: torch device
        
    Returns:
        dict: {feature_name: shap_value} for all 26, plus "top_5", plus "additivity_check"
    """
    fold_model.eval()
    
    N = original_recordings.shape[0]
    x_subj_mean = np.mean(original_recordings, axis=0)  # (26,)
    
    # We use a summary of the background to speed up KernelExplainer
    # 50 samples is typical for SHAP KernelExplainer background
    bg_summary = shap.kmeans(background_data, 50)
    
    def wrapper(X):
        # X is (K, 26)
        K = X.shape[0]
        probs = np.zeros(K)
        
        for k in range(K):
            x_k = X[k]
            # Reconstruct (N, 26) input
            x_in = np.zeros((N, 26))
            for j in range(26):
                # If the feature comes from the subject, restore the per-recording variance
                if np.isclose(x_k[j], x_subj_mean[j], rtol=1e-5, atol=1e-8):
                    x_in[:, j] = original_recordings[:, j]
                else:
                    # If it comes from background, broadcast the background value
                    x_in[:, j] = x_k[j]
                    
            # Normalize
            x_in_scaled = (x_in - scaler_mean) / scaler_scale
            x_in_tensor = torch.tensor(x_in_scaled, dtype=torch.float32).to(device)
            
            with torch.no_grad():
                h_bio = fold_model.bio_branch(x_in_tensor)  # (N, 16)
                h_concat = torch.cat([h_ssl_fixed, h_bio], dim=1)  # (N, 144)
                
                h_subject, _ = fold_model.attention_pool(h_concat)  # (144,)
                
                h_sub_ssl = h_subject[:128]
                h_sub_bio = h_subject[128:]
                
                alpha = fold_model.modality_gate(h_sub_bio)
                z_fused, _ = fold_model.fusion_head(h_sub_ssl, h_sub_bio, alpha)
                
                # Apply softmax to get PD probability
                prob = F.softmax(z_fused, dim=0)[1].item()
                probs[k] = prob
                
        return probs

    # Run KernelExplainer
    explainer = shap.KernelExplainer(wrapper, bg_summary)
    
    # Explain the subject's mean vector
    # We suppress the progress bar for clean logs
    shap_vals = explainer.shap_values(x_subj_mean, nsamples=500, silent=True)
    
    # Build results
    feature_impacts = {}
    for i, name in enumerate(BIOMARKER_NAMES):
        feature_impacts[name] = float(shap_vals[i])
        
    # Sort by absolute impact
    sorted_features = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
    
    top_5 = [{"feature": k, "value": v} for k, v in sorted_features[:5]]
    
    # Additivity check
    actual_output = wrapper(np.array([x_subj_mean]))[0]
    expected_value = float(explainer.expected_value)
    sum_shap = float(np.sum(shap_vals))
    
    additivity_check = {
        "model_output": float(actual_output),
        "expected_value": expected_value,
        "sum_shap": sum_shap,
        "difference": float(actual_output - (expected_value + sum_shap))
    }

    return {
        "features": feature_impacts,
        "top_5": top_5,
        "additivity_check": additivity_check
    }
