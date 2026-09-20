"""
calibration.py — Step 3: Temperature scaling + Mondrian conformal prediction.

Contains:
    - fit_temperature: Fit scalar T* on validation split (NLL minimization)
    - apply_temperature: Scale logits by T
    - MondrianConformalPredictor: Class-conditional split-conformal at 90% coverage
    - build_dashboard_payload: Structured output for clinical dashboard
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize_scalar


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit a single scalar temperature T* by minimizing NLL on validation data.
    
    Args:
        logits: (N, 2) array of raw logits from PDVoiceNet
        labels: (N,) array of integer labels (0=HC, 1=PD)
    
    Returns:
        T*: optimal temperature scalar (> 0)
    """
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    def nll_at_T(T):
        T = max(T, 1e-4)  # Prevent division by zero
        scaled = logits_tensor / T
        log_probs = F.log_softmax(scaled, dim=1)
        nll = F.nll_loss(log_probs, labels_tensor).item()
        return nll
    
    result = minimize_scalar(nll_at_T, bounds=(0.1, 10.0), method='bounded')
    return float(result.x)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling to logits and return calibrated probabilities.
    
    Args:
        logits: (N, 2) array of raw logits
        T: temperature scalar
    
    Returns:
        probs: (N, 2) array of calibrated probabilities
    """
    scaled = torch.tensor(logits, dtype=torch.float32) / max(T, 1e-4)
    probs = F.softmax(scaled, dim=1).numpy()
    return probs


# ---------------------------------------------------------------------------
# Mondrian Conformal Prediction
# ---------------------------------------------------------------------------

class MondrianConformalPredictor:
    """Class-conditional (Mondrian) split-conformal predictor.
    
    Nonconformity score: s(x, y) = 1 - p_calib(y | x)
    Target coverage: 90%
    Quantile: q_hat = ceil((n+1) * 0.9) / n per class
    
    Produces prediction sets: {PD}, {HC}, {PD, HC} (refer), or {} (contradictory).
    """
    
    def __init__(self, target_coverage: float = 0.90):
        self.target_coverage = target_coverage
        self.q_hat = {}  # Per-class quantiles
        self.fitted = False
    
    def fit(self, calib_probs: np.ndarray, calib_labels: np.ndarray):
        """Fit conformal quantiles on calibration data.
        
        Args:
            calib_probs: (N, 2) calibrated probabilities
            calib_labels: (N,) integer labels (0=HC, 1=PD)
        """
        for cls in [0, 1]:
            mask = calib_labels == cls
            if mask.sum() == 0:
                # Fallback: use a very permissive quantile
                self.q_hat[cls] = 1.0
                continue
            
            # Nonconformity scores for samples of this class
            scores = 1.0 - calib_probs[mask, cls]
            n = len(scores)
            
            # Mondrian quantile: ceil((n+1) * coverage) / n
            quantile_idx = int(np.ceil((n + 1) * self.target_coverage))
            quantile_idx = min(quantile_idx, n)  # Clamp to array length
            
            sorted_scores = np.sort(scores)
            self.q_hat[cls] = sorted_scores[quantile_idx - 1]  # -1 for 0-indexing
        
        self.fitted = True
    
    def predict(self, probs: np.ndarray) -> list:
        """Produce prediction sets for each sample.
        
        Args:
            probs: (N, 2) calibrated probabilities
            
        Returns:
            list of sets: each is one of {0}, {1}, {0,1}, or set()
        """
        assert self.fitted, "Must call fit() before predict()"
        
        prediction_sets = []
        for i in range(len(probs)):
            pred_set = set()
            for cls in [0, 1]:
                score = 1.0 - probs[i, cls]
                if score <= self.q_hat[cls]:
                    pred_set.add(cls)
            prediction_sets.append(pred_set)
        
        return prediction_sets
    
    def get_state(self) -> dict:
        """Serialize quantiles for saving."""
        return {"q_hat": self.q_hat, "target_coverage": self.target_coverage}
    
    def load_state(self, state: dict):
        """Restore from serialized state."""
        self.q_hat = state["q_hat"]
        self.target_coverage = state["target_coverage"]
        self.fitted = True


# ---------------------------------------------------------------------------
# Dashboard payload
# ---------------------------------------------------------------------------

LABEL_MAP = {0: "HC", 1: "PD"}

def build_dashboard_payload(subject_id: str, z_fused: np.ndarray, alpha: float,
                             attn_weights: np.ndarray, prediction_set: set,
                             calibrated_probs: np.ndarray = None) -> dict:
    """Build a structured output dict for the clinical dashboard.
    
    Args:
        subject_id: Subject identifier
        z_fused: (2,) raw logits
        alpha: Modality dominance scalar (0 = bio-only, 1 = ssl-only)
        attn_weights: (N,) per-recording attention weights
        prediction_set: Set of predicted class indices
        calibrated_probs: (2,) calibrated probabilities (optional)
    
    Returns:
        dict with prediction, confidence, modality info, uncertainty flag
    """
    # Map prediction set to labels
    pred_labels = {LABEL_MAP[c] for c in prediction_set}
    
    # Determine primary prediction
    if len(pred_labels) == 1:
        prediction = list(pred_labels)[0]
        refer = False
    elif len(pred_labels) == 2:
        prediction = "Uncertain"
        refer = True
    else:
        prediction = "Contradictory"
        refer = True
    
    payload = {
        "subject_id": subject_id,
        "prediction": prediction,
        "prediction_set": sorted(pred_labels) if pred_labels else [],
        "refer_to_clinician": refer,
        "modality_gate_alpha": float(alpha),
        "modality_dominance": "SSL" if alpha > 0.5 else "Biomarker",
        "attention_weights": attn_weights.tolist() if attn_weights is not None else [],
    }
    
    if calibrated_probs is not None:
        payload["probability_hc"] = float(calibrated_probs[0])
        payload["probability_pd"] = float(calibrated_probs[1])
    
    return payload
