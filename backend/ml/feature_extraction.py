"""
Acoustic feature extraction for Parkinson's Disease voice detection.

Replicates the feature set described in:
Shen, M., Mortezaagha, P. & Rahgozar, A. (2025).
"Explainable artificial intelligence to diagnose early Parkinson's disease
via voice analysis." Scientific Reports, 15:11687.

Extracts 26 features per recording:
  - Pitch: mean, min, max                       (3)
  - Jitter: local, local_absolute, rap, ppq5     (4)
  - Shimmer: local, apq3, apq5, apq11, dda       (5)
  - HNR: mean                                    (1)
  - MFCC: mean of coefficients 1-13              (13)
"""

import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa

FEATURE_NAMES = [
    "mean_pitch", "min_pitch", "max_pitch",
    "local_jitter", "local_absolute_jitter", "rap_jitter", "ppq5_jitter",
    "local_shimmer", "apq3_shimmer", "apq5_shimmer", "apq11_shimmer", "dda_shimmer",
    "mean_hnr",
] + [f"mfcc_{i}" for i in range(1, 14)]

assert len(FEATURE_NAMES) == 26


def extract_features(wav_path: str) -> np.ndarray:
    """Extract the 26-dimensional acoustic feature vector for one audio file."""
    # 1. Load audio natively and downsample to 16kHz for absolute consistency
    y, sr = librosa.load(wav_path, sr=16000)
    
    # 2. Trim silence (background noise at start/end)
    y, _ = librosa.effects.trim(y, top_db=25)
    
    # 3. Peak normalize to standardize volume across all microphones
    y = librosa.util.normalize(y)

    # 4. Initialize Parselmouth Sound using the purified numpy array
    sound = parselmouth.Sound(y, sampling_frequency=sr)

    # --- Pitch ---
    pitch = sound.to_pitch()
    mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
    min_pitch = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    max_pitch = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

    # --- Jitter / Shimmer (require a PointProcess) ---
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)

    local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    local_absolute_jitter = call(
        point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5_jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)

    local_shimmer = call(
        [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3_shimmer = call(
        [sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5_shimmer = call(
        [sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11_shimmer = call(
        [sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    dda_shimmer = call(
        [sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    # --- HNR ---
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    mean_hnr = call(harmonicity, "Get mean", 0, 0)

    # --- MFCCs (mean of each of the first 13 coefficients) ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfccs.mean(axis=1)

    features = np.array([
        mean_pitch, min_pitch, max_pitch,
        local_jitter, local_absolute_jitter, rap_jitter, ppq5_jitter,
        local_shimmer, apq3_shimmer, apq5_shimmer, apq11_shimmer, dda_shimmer,
        mean_hnr,
        *mfcc_means,
    ], dtype=np.float32)

    # Praat occasionally returns NaN on very short/quiet clips (e.g. undefined
    # jitter/shimmer when too few glottal pulses are detected) — guard against this.
    features = np.nan_to_num(features, nan=0.0)
    return features


def extract_dataset_features(file_label_pairs):
    """
    file_label_pairs: list of (wav_path, label) where label is 0 (HC) or 1 (PD)

    Returns:
        X: (N, 26) float32 array
        y: (N,) int64 array
        failed: list of (path, error_message) for files that could not be processed
    """
    X, y, failed = [], [], []
    for path, label in file_label_pairs:
        try:
            X.append(extract_features(path))
            y.append(label)
        except Exception as e:
            failed.append((path, str(e)))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), failed
