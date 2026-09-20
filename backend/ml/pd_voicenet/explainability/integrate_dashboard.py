"""
integrate_dashboard.py — Step C: Extended dashboard payload.

Wraps the existing build_dashboard_payload to add explainability outputs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/..")
from calibration import build_dashboard_payload


def build_extended_dashboard_payload(
    subject_id,
    z_fused,
    alpha,
    attn_weights,
    prediction_set,
    shap_results,
    grad_cam_results,
    transformer_saliency_results,
    calibrated_probs=None
):
    """
    Extends the base dashboard payload with explainability artifacts.
    
    Args:
        subject_id: str
        z_fused: (2,) np.ndarray
        alpha: float
        attn_weights: np.ndarray (N,)
        prediction_set: set of str (e.g. {"PD"}, {"HC", "PD"})
        shap_results: dict containing "top_5" list from shap_explainer
        grad_cam_results: list of dicts from grad_cam_cnn (one per recording)
        transformer_saliency_results: list of dicts from transformer_saliency (one per recording)
        calibrated_probs: optional (2,) np.ndarray
        
    Returns:
        dict: The extended payload payload
    """
    # Build base payload
    payload = build_dashboard_payload(
        subject_id=subject_id,
        z_fused=z_fused,
        alpha=alpha,
        attn_weights=attn_weights,
        prediction_set=prediction_set,
        calibrated_probs=calibrated_probs
    )
    
    # Extend with explainability outputs
    # Using the exact keys specified in the prompt
    payload["shap_top5"] = shap_results.get("top_5", [])
    payload["grad_cam_cnn"] = grad_cam_results
    payload["transformer_saliency"] = transformer_saliency_results
    
    return payload
