import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa
import warnings

FEATURE_NAMES = [
    "mean_pitch", "min_pitch", "max_pitch",
    "local_jitter", "local_absolute_jitter", "rap_jitter", "ppq5_jitter",
    "local_shimmer", "apq3_shimmer", "apq5_shimmer", "apq11_shimmer", "dda_shimmer",
    "mean_hnr",
]

assert len(FEATURE_NAMES) == 13

def extract_mel_spectrogram(y, sr, n_mels=64, max_pad_len=200):
    # Compute Mel-spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000)
    # Convert to log scale (dB)
    S_dB = librosa.power_to_db(S, ref=np.max)
    # Pad or truncate to max_pad_len frames to ensure uniform input shape for CNN
    if S_dB.shape[1] > max_pad_len:
        S_dB = S_dB[:, :max_pad_len]
    else:
        pad_width = max_pad_len - S_dB.shape[1]
        S_dB = np.pad(S_dB, pad_width=((0, 0), (0, pad_width)), mode='constant')
    # Return shape (max_pad_len, n_mels) to match standard time-series CNN input (Time, Features)
    return S_dB.T

def extract_features(wav_path: str) -> dict:
    warnings.filterwarnings("ignore")
    # 1. Load audio natively and downsample to 16kHz for absolute consistency
    y, sr = librosa.load(wav_path, sr=16000)
    
    # 2. Trim silence (background noise at start/end)
    y, _ = librosa.effects.trim(y, top_db=25)

    # 3. Extract Mel-Spectrogram (for CNN)
    mel_spec = extract_mel_spectrogram(y, sr)

    # 4. Extract Acoustic Features (for MKL Fusion)
    sound = parselmouth.Sound(y, sampling_frequency=sr)

    # --- Pitch ---
    pitch = sound.to_pitch()
    mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
    min_pitch = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    max_pitch = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")

    # --- Jitter / Shimmer (require a PointProcess) ---
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)

    local_jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    local_absolute_jitter = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rap_jitter = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5_jitter = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)

    local_shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3_shimmer = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5_shimmer = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11_shimmer = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    dda_shimmer = call([sound, point_process], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    # --- HNR ---
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    mean_hnr = call(harmonicity, "Get mean", 0, 0)

    acoustic = np.array([
        mean_pitch, min_pitch, max_pitch,
        local_jitter, local_absolute_jitter, rap_jitter, ppq5_jitter,
        local_shimmer, apq3_shimmer, apq5_shimmer, apq11_shimmer, dda_shimmer,
        mean_hnr,
    ], dtype=np.float32)

    acoustic = np.nan_to_num(acoustic, nan=0.0)
    
    return {
        "spectrogram": mel_spec,
        "acoustic": acoustic
    }

def extract_dataset_features(file_label_pairs):
    X_spec, X_acous, y, failed = [], [], [], []
    for path, label in file_label_pairs:
        try:
            feats = extract_features(path)
            X_spec.append(feats["spectrogram"])
            X_acous.append(feats["acoustic"])
            y.append(label)
        except Exception as e:
            failed.append((path, str(e)))
    return np.array(X_spec), np.array(X_acous), np.array(y), failed
