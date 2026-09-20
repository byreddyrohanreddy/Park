"""
model.py — Step 2: PD-VoiceNet architecture.

Contains:
    - LoRALinear: Manual LoRA adapter wrapper for frozen Linear layers
    - SSLBranch: Wav2Vec2 layers 9-11 with LoRA, mean pooling, projection
    - BiomarkerBranch: MLP on 26 acoustic features
    - SubjectAttentionPooling: Learned-query attention over variable-N recordings
    - ModalityGate: Scalar gate from h_sub_bio only
    - FusionHead: Gated fusion of SSL and biomarker streams
    - PDVoiceNet: Full model wiring everything together
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model
import copy


# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for a frozen nn.Linear layer.
    
    Only the low-rank matrices A, B are trainable.
    Output: frozen_linear(x) + (B @ A)(x) * (alpha / r)
    
    Args:
        base_linear: The frozen nn.Linear to wrap
        r: Rank of the low-rank matrices (default: 4)
        alpha: Scaling factor (default: 16)
        dropout: Dropout on input before LoRA path (default: 0.1)
    """
    def __init__(self, base_linear: nn.Linear, r: int = 4, alpha: float = 16.0,
                 dropout: float = 0.1):
        super().__init__()
        self.base_linear = base_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        
        # Freeze the base linear layer
        for param in self.base_linear.parameters():
            param.requires_grad = False
        
        # LoRA matrices — only these are trainable
        self.lora_A = nn.Parameter(torch.randn(in_features, r) * (1.0 / math.sqrt(r)))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.lora_dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Frozen path
        base_out = self.base_linear(x)
        # LoRA path
        lora_out = self.lora_dropout(x) @ self.lora_A @ self.lora_B * self.scaling
        return base_out + lora_out


# ---------------------------------------------------------------------------
# SSLBranch
# ---------------------------------------------------------------------------

class SSLBranch(nn.Module):
    """SSL branch: takes cached layer 0-8 activations, runs through
    LoRA-adapted transformer layers 9-11, mean-pools, projects to 128d.
    
    Args:
        wav2vec2_model: A loaded Wav2Vec2Model (to extract layers 9-11)
        lora_r: LoRA rank (default: 4)
        lora_alpha: LoRA scaling (default: 16)
        lora_dropout: LoRA dropout (default: 0.1)
    """
    def __init__(self, wav2vec2_model: Wav2Vec2Model = None,
                 lora_r: int = 4, lora_alpha: float = 16.0, lora_dropout: float = 0.1):
        super().__init__()
        
        if wav2vec2_model is None:
            wav2vec2_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        
        # Extract transformer layers 9, 10, 11 (deep copy to decouple)
        self.transformer_layers = nn.ModuleList([
            copy.deepcopy(wav2vec2_model.encoder.layers[i]) for i in range(9, 12)
        ])
        
        # Freeze all base weights in layers 9-11
        for layer in self.transformer_layers:
            for param in layer.parameters():
                param.requires_grad = False
        
        # Apply LoRA to q_proj and v_proj in each layer
        for layer in self.transformer_layers:
            attn = layer.attention
            attn.q_proj = LoRALinear(attn.q_proj, r=lora_r, alpha=lora_alpha, 
                                     dropout=lora_dropout)
            attn.v_proj = LoRALinear(attn.v_proj, r=lora_r, alpha=lora_alpha,
                                     dropout=lora_dropout)
        
        # Projection: 768 -> 128
        self.projection = nn.Linear(768, 128)
    
    def forward(self, cached_activations):
        """
        Args:
            cached_activations: (T, 768) tensor — output of frozen layers 0-8
            
        Returns:
            h_ssl: (128,) tensor
        """
        # Add batch dimension if needed
        if cached_activations.dim() == 2:
            x = cached_activations.unsqueeze(0)  # (1, T, 768)
        else:
            x = cached_activations
        
        # Run through LoRA-adapted layers 9-11
        for layer in self.transformer_layers:
            # Newer transformers may return raw tensor instead of (hidden_states,) tuple
            layer_out = layer(x, attention_mask=None, output_attentions=False)
            x = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        
        # Temporal mean pooling: (1, T, 768) -> (1, 768)
        x = x.mean(dim=1)
        
        # Project: (1, 768) -> (1, 128)
        h_ssl = self.projection(x)
        
        # Remove batch dim
        return h_ssl.squeeze(0)  # (128,)


# ---------------------------------------------------------------------------
# BiomarkerBranch
# ---------------------------------------------------------------------------

# The 26 biomarker feature names (for documentation / ordering)
BIOMARKER_NAMES = [
    # Jitter (3)
    "local_jitter", "ppq5_jitter", "rap_jitter",
    # Shimmer (5)
    "local_shimmer", "apq3_shimmer", "apq5_shimmer", "apq11_shimmer", "dda_shimmer",
    # HNR / NHR (2)
    "hnr", "nhr",
    # F0 stats (4)
    "mean_f0", "std_f0", "min_f0", "max_f0",
    # MFCC 1-6 mean (6)
    "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6",
    # Formants (6)
    "f1_freq", "f2_freq", "f3_freq", "f1_bw", "f2_bw", "f3_bw",
]
assert len(BIOMARKER_NAMES) == 26, f"Expected 26 biomarker names, got {len(BIOMARKER_NAMES)}"


class BiomarkerBranch(nn.Module):
    """Biomarker MLP branch: 26 -> 32 -> 16, ReLU.
    
    NOTE: This module does NOT fit its own scaler. The z-score normalization 
    scaler must be fit externally per-fold on training subjects only, and 
    applied to the input before calling this module.
    """
    def __init__(self, input_dim: int = 26):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
    
    def forward(self, x):
        """
        Args:
            x: (26,) tensor — z-score normalized biomarker features
            
        Returns:
            h_bio: (16,) tensor
        """
        return self.mlp(x)


# ---------------------------------------------------------------------------
# SubjectAttentionPooling
# ---------------------------------------------------------------------------

class SubjectAttentionPooling(nn.Module):
    """Attention pooling over variable-length list of per-recording vectors.
    
    Uses a learned query q and key projection W_k:
        w_i = softmax(q^T W_k h_i / sqrt(d))
        h_subject = sum_i w_i * h_i
    
    Handles N=1 correctly (attention weight trivially = 1.0).
    Returns both h_subject and attention weights.
    """
    def __init__(self, dim: int = 144):
        super().__init__()
        self.dim = dim
        self.query = nn.Parameter(torch.randn(dim))  # Learned query
        self.key_proj = nn.Linear(dim, dim)
        self.scale = math.sqrt(dim)
    
    def forward(self, recording_vectors):
        """
        Args:
            recording_vectors: (N, 144) tensor — concatenated [h_ssl; h_bio] per recording
            
        Returns:
            h_subject: (144,) tensor
            attn_weights: (N,) tensor
        """
        N = recording_vectors.shape[0]
        
        # Project keys: (N, 144)
        keys = self.key_proj(recording_vectors)
        
        # Attention scores: q^T W_k h_i / sqrt(d)
        scores = torch.matmul(keys, self.query) / self.scale  # (N,)
        
        # Softmax (for N=1, this is trivially [1.0])
        attn_weights = F.softmax(scores, dim=0)  # (N,)
        
        # Weighted sum
        h_subject = torch.sum(attn_weights.unsqueeze(1) * recording_vectors, dim=0)  # (144,)
        
        return h_subject, attn_weights


# ---------------------------------------------------------------------------
# ModalityGate
# ---------------------------------------------------------------------------

class ModalityGate(nn.Module):
    """Scalar modality gate based on biomarker sub-embedding ONLY.
    
    alpha = sigmoid(W_g * h_sub_bio + b), scalar in [0, 1]
    
    NOTE: h_sub_ssl is deliberately NOT wired into this module.
    This is a spec requirement, not an oversight.
    """
    def __init__(self, bio_dim: int = 16):
        super().__init__()
        self.gate = nn.Linear(bio_dim, 1)
    
    def forward(self, h_sub_bio):
        """
        Args:
            h_sub_bio: (16,) tensor
            
        Returns:
            alpha: scalar tensor in [0, 1]
        """
        return torch.sigmoid(self.gate(h_sub_bio)).squeeze(-1)


# ---------------------------------------------------------------------------
# FusionHead
# ---------------------------------------------------------------------------

class FusionHead(nn.Module):
    """Gated fusion of SSL and biomarker sub-embeddings.
    
    z_ssl = MLP_ssl(h_sub_ssl)
    z_bio = MLP_bio(h_sub_bio)
    z_fused = alpha * z_ssl + (1 - alpha) * z_bio
    
    Returns z_fused (logits) and alpha.
    """
    def __init__(self, ssl_dim: int = 128, bio_dim: int = 16, n_classes: int = 2):
        super().__init__()
        self.mlp_ssl = nn.Sequential(
            nn.Linear(ssl_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )
        self.mlp_bio = nn.Sequential(
            nn.Linear(bio_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes),
        )
    
    def forward(self, h_sub_ssl, h_sub_bio, alpha):
        """
        Args:
            h_sub_ssl: (128,) tensor
            h_sub_bio: (16,) tensor
            alpha: scalar tensor in [0, 1]
            
        Returns:
            z_fused: (n_classes,) tensor — raw logits
            alpha: scalar (passed through for downstream use)
        """
        z_ssl = self.mlp_ssl(h_sub_ssl)
        z_bio = self.mlp_bio(h_sub_bio)
        z_fused = alpha * z_ssl + (1 - alpha) * z_bio
        return z_fused, alpha


# ---------------------------------------------------------------------------
# PDVoiceNet — full model
# ---------------------------------------------------------------------------

class PDVoiceNet(nn.Module):
    """Full PD-VoiceNet: Subject-Gated Multimodal Fusion.
    
    Takes a subject's cached SSL activations + biomarker features,
    returns (z_fused, alpha, attention_weights).
    """
    def __init__(self, wav2vec2_model=None):
        super().__init__()
        self.ssl_branch = SSLBranch(wav2vec2_model)
        self.bio_branch = BiomarkerBranch(input_dim=26)
        self.attention_pool = SubjectAttentionPooling(dim=144)
        self.modality_gate = ModalityGate(bio_dim=16)
        self.fusion_head = FusionHead(ssl_dim=128, bio_dim=16, n_classes=2)
    
    def forward(self, cached_recordings, biomarker_features):
        """
        Args:
            cached_recordings: list of (T_i, 768) tensors — one per recording
            biomarker_features: (N, 26) tensor — one row per recording,
                                already z-score normalized externally
        
        Returns:
            z_fused: (2,) tensor — raw logits [HC_score, PD_score]
            alpha: scalar — modality dominance
            attn_weights: (N,) tensor — per-recording attention weights
        """
        N = len(cached_recordings)
        assert biomarker_features.shape[0] == N
        
        # Process each recording through both branches
        recording_vectors = []
        for i in range(N):
            h_ssl = self.ssl_branch(cached_recordings[i])    # (128,)
            h_bio = self.bio_branch(biomarker_features[i])    # (16,)
            h_concat = torch.cat([h_ssl, h_bio], dim=0)       # (144,)
            recording_vectors.append(h_concat)
        
        recording_vectors = torch.stack(recording_vectors, dim=0)  # (N, 144)
        
        # Subject-level attention pooling
        h_subject, attn_weights = self.attention_pool(recording_vectors)  # (144,), (N,)
        
        # Split back into SSL and bio components
        h_sub_ssl = h_subject[:128]  # (128,)
        h_sub_bio = h_subject[128:]  # (16,)
        
        # Modality gate (from bio ONLY)
        alpha = self.modality_gate(h_sub_bio)  # scalar
        
        # Gated fusion
        z_fused, alpha = self.fusion_head(h_sub_ssl, h_sub_bio, alpha)
        
        return z_fused, alpha, attn_weights
