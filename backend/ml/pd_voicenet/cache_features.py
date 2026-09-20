"""
cache_features.py — Step 1: Frozen Wav2Vec2-base feature caching (layers 0-8).

Runs a single forward pass through frozen Wav2Vec2-base (CNN encoder + 
transformer layers 0-8) for every recording in the dataset, saving the 
output activation tensor to disk.

This is safe to run ONCE, dataset-wide, regardless of which site is later 
held out — frozen layers see no labels and get no gradients.

Usage:
    python cache_features.py --dataset-dir ../../files/dataset
"""

import os
import sys
import argparse
import torch
import librosa
import warnings
from transformers import Wav2Vec2Model, AutoFeatureExtractor

# Add parent to path for data.py import
sys.path.insert(0, os.path.dirname(__file__))
from data import build_manifest, get_recording_id

warnings.filterwarnings("ignore")


def cache_features(dataset_dir: str, cache_dir: str, device: str = None):
    """Extract and cache frozen Wav2Vec2 layers 0-8 activations for all recordings."""
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Build manifest
    manifest = build_manifest(dataset_dir)
    print(f"Dataset manifest: {len(manifest)} recordings")
    
    # Load frozen Wav2Vec2-base
    print("Loading facebook/wav2vec2-base (frozen)...")
    processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base", 
        output_hidden_states=True
    )
    model.to(device)
    model.eval()
    
    # Freeze everything — no gradients ever flow through this model
    for param in model.parameters():
        param.requires_grad = False
    
    print(f"Model loaded on {device}. Extracting activations...")
    
    cached_count = 0
    errors = []
    
    with torch.no_grad():
        for idx, row in manifest.iterrows():
            recording_id = row["recording_id"]
            cache_path = os.path.join(cache_dir, f"{recording_id}.pt")
            
            # Skip if already cached
            if os.path.exists(cache_path):
                cached_count += 1
                if cached_count % 100 == 0:
                    print(f"  Skipped (already cached): {cached_count}/{len(manifest)}")
                continue
            
            try:
                # Load audio at 16kHz
                audio, sr = librosa.load(row["path"], sr=16000, mono=True, duration=25.0)
                
                # Process through feature extractor
                inputs = processor(
                    audio, sampling_rate=16000, return_tensors="pt", padding=True
                )
                input_values = inputs.input_values.to(device)
                
                # Forward pass
                outputs = model(input_values)
                
                # hidden_states indexing:
                #   [0] = output of feature_projection (after CNN, before transformers)
                #   [1] = output of transformer layer 0
                #   ...
                #   [i+1] = output of transformer layer i
                #   [9] = output of transformer layer 8
                #
                # We cache hidden_states[9] = output after transformer layer 8.
                # SSLBranch (Step 2) will run layers 9-11 on top of this.
                
                layer_8_output = outputs.hidden_states[9]  # Shape: (1, T, 768)
                
                # Save to disk (squeeze batch dim, keep on CPU, float16 to save space)
                torch.save(
                    layer_8_output.squeeze(0).cpu().half(),
                    cache_path
                )
                
                cached_count += 1
                if cached_count % 50 == 0:
                    print(f"  Cached {cached_count}/{len(manifest)} recordings...")
                    
            except Exception as e:
                errors.append((recording_id, str(e)))
                print(f"  ERROR on {recording_id}: {e}")
    
    # Self-check: assert every recording has a cached file
    print(f"\n--- SELF-CHECK ---")
    print(f"Total recordings in manifest: {len(manifest)}")
    print(f"Successfully cached: {cached_count}")
    
    if errors:
        print(f"Errors: {len(errors)}")
        for rid, err in errors[:10]:
            print(f"  {rid}: {err}")
    
    missing = []
    for _, row in manifest.iterrows():
        cache_path = os.path.join(cache_dir, f"{row['recording_id']}.pt")
        if not os.path.exists(cache_path):
            missing.append(row["recording_id"])
    
    if missing:
        print(f"\nFAILED: {len(missing)} recordings missing from cache:")
        for rid in missing[:20]:
            print(f"  {rid}")
        sys.exit(1)
    else:
        print(f"PASSED: All {len(manifest)} recordings have cached features.")
    
    # Print a sample to verify shape
    sample_path = os.path.join(cache_dir, f"{manifest.iloc[0]['recording_id']}.pt")
    sample = torch.load(sample_path, weights_only=True)
    print(f"Sample cached tensor shape: {sample.shape} (dtype={sample.dtype})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache frozen Wav2Vec2 layer 0-8 activations")
    parser.add_argument("--dataset-dir", type=str, 
                        default=os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "files", "dataset"),
                        help="Path to dataset directory containing HC/ and PD/ subdirs")
    parser.add_argument("--cache-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "artifacts_pdvoicenet", "cache"),
                        help="Output directory for cached tensors")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu). Auto-detected if not specified.")
    
    args = parser.parse_args()
    cache_features(args.dataset_dir, args.cache_dir, args.device)
