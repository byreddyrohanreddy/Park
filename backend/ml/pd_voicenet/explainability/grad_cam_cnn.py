"""
grad_cam_cnn.py — Step B1: True Grad-CAM on the frozen CNN encoder.

Runs a forward pass from raw audio through the base Wav2Vec2 encoder
and the PD-VoiceNet fold model, hooks layer 6 of the CNN, and computes
Grad-CAM with respect to the PD-class probability.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import librosa
from transformers import Wav2Vec2Model, AutoFeatureExtractor

import sys
sys.path.insert(0, os.path.dirname(__file__) + "/..")
from model import PDVoiceNet


def grad_cam_explain(subject_id, recording_id, fold_model, raw_audio_path, 
                     biomarker_features, scaler_mean, scaler_scale, device="cuda"):
    """
    Computes Grad-CAM on the last CNN layer of Wav2Vec2.
    
    Args:
        subject_id: str
        recording_id: str
        fold_model: PDVoiceNet loaded for the appropriate fold
        raw_audio_path: str, path to the raw .wav file
        biomarker_features: (26,) raw biomarker features for this recording
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
    
    # 1. Load frozen Wav2Vec2
    base_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
    base_model.eval()
    processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    
    # We must explicitly allow gradients to flow back to the CNN activations
    for param in base_model.parameters():
        param.requires_grad = False
        
    # 2. Register hook on the last conv layer (layer 6)
    feature_maps = []
    
    def forward_hook(module, input, output):
        # output is (batch, channels, time) -> (1, 512, T)
        # We need to retain grad on this tensor
        output.retain_grad()
        feature_maps.append(output)
        
    target_layer = base_model.feature_extractor.conv_layers[6]
    hook_handle = target_layer.register_forward_hook(forward_hook)
    
    try:
        # 3. Load audio and process
        audio, sr = librosa.load(raw_audio_path, sr=16000, mono=True, duration=25.0)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        
        # Enable gradients on the input to ensure PyTorch tracks graph through the frozen network
        input_values.requires_grad_(True)
        
        # Forward through Wav2Vec2 (layers 0-8)
        outputs = base_model(input_values, output_hidden_states=True)
        layer_8_output = outputs.hidden_states[9]  # (1, T, 768)
        
        # 4. Prepare biomarker input for this recording
        bio_scaled = (biomarker_features - scaler_mean) / scaler_scale
        bio_tensor = torch.tensor(bio_scaled, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 26)
        
        # Forward through fold_model
        z_fused, alpha, attn_weights = fold_model([layer_8_output.squeeze(0)], bio_tensor)
        
        # PD class logit
        pd_logit = z_fused[1]
        
        # Backward pass
        fold_model.zero_grad()
        base_model.zero_grad()
        pd_logit.backward()
        
        # 5. Extract feature maps and gradients
        target_activations = feature_maps[0]  # (1, C, T)
        gradients = target_activations.grad  # (1, C, T)
        
        # Global average pooling over the time dimension for gradients -> alpha_c
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
        
        # Map to seconds (stride is 320 at 16kHz -> ~20ms per frame)
        timestamps = [i * 0.02 for i in range(T)]
        
        return {
            "recording_id": recording_id,
            "timestamps_sec": timestamps,
            "importance": cam_np.tolist()
        }
        
    finally:
        hook_handle.remove()
