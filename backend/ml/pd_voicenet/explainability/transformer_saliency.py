"""
transformer_saliency.py — Step B2: Saliency on LoRA-adapted layers.

Captures the (T, 768) hidden states from the fine-tuned Wav2Vec2 transformer 
layers (right before temporal mean pooling) and computes Grad-CAM style
saliency to show what the task-adapted part of the network attends to.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(__file__) + "/..")
from model import PDVoiceNet


def transformer_saliency_explain(subject_id, recording_id, fold_model, 
                                 cached_layer_8_activations, biomarker_features, 
                                 scaler_mean, scaler_scale, device="cuda"):
    """
    Computes saliency on the final LoRA-adapted transformer layer (layer 11).
    
    Args:
        subject_id: str
        recording_id: str
        fold_model: PDVoiceNet loaded for the appropriate fold
        cached_layer_8_activations: (T, 768) tensor
        biomarker_features: (26,) array
        scaler_mean: (26,) array
        scaler_scale: (26,) array
        device: torch device
        
    Returns:
        dict: {
            "recording_id": str,
            "timestamps_sec": list of float,
            "importance": list of float [0, 1]
        }
    """
    fold_model.eval()
    
    # We must ensure the parameters of the model (or at least the input) require grad
    # to allow backprop to the intermediate layers.
    cached_layer_8_activations = cached_layer_8_activations.to(device).clone().detach()
    cached_layer_8_activations.requires_grad_(True)
    
    bio_scaled = (biomarker_features - scaler_mean) / scaler_scale
    bio_tensor = torch.tensor(bio_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Register hook on the last transformer layer in the SSLBranch
    feature_sequence = []
    
    def forward_hook(module, input, output):
        # Handle both tuple (older transformers) and raw tensor outputs
        hidden_states = output[0] if isinstance(output, tuple) else output
        hidden_states.retain_grad()
        feature_sequence.append(hidden_states)
        
    target_layer = fold_model.ssl_branch.transformer_layers[-1]
    hook_handle = target_layer.register_forward_hook(forward_hook)
    
    try:
        # Forward pass
        z_fused, alpha, attn_weights = fold_model([cached_layer_8_activations], bio_tensor)
        
        # PD class logit
        pd_logit = z_fused[1]
        
        fold_model.zero_grad()
        pd_logit.backward()
        
        # Extract features and gradients
        target_activations = feature_sequence[0]  # (1, T, 768)
        gradients = target_activations.grad       # (1, T, 768)
        
        # Convert to (1, C, T) to match B1 style logic
        target_activations = target_activations.transpose(1, 2)  # (1, 768, T)
        gradients = gradients.transpose(1, 2)                    # (1, 768, T)
        
        # Global average pooling over the time dimension for gradients
        alpha_c = torch.mean(gradients, dim=2, keepdim=True)  # (1, C, 1)
        
        # Weighted combination of feature maps
        cam = torch.sum(alpha_c * target_activations, dim=1).squeeze(0)  # (T,)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)
            
        cam_np = cam.cpu().detach().numpy()
        T = cam_np.shape[0]
        
        # Map to seconds (~20ms per frame)
        timestamps = [i * 0.02 for i in range(T)]
        
        return {
            "recording_id": recording_id,
            "timestamps_sec": timestamps,
            "importance": cam_np.tolist()
        }
        
    finally:
        hook_handle.remove()
