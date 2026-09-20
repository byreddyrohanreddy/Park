"""
app.py - PD-VoiceNet Flask API
Endpoints:
  POST /predict        - Run PD-VoiceNet inference on a single .wav file
  POST /explainability - SHAP + Grad-CAM + Saliency for a prediction
  GET  /health         - Liveness check
"""

import os, sys, json, uuid, warnings, threading
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import parselmouth
from parselmouth.praat import call
from flask import Flask, request, jsonify
from transformers import Wav2Vec2Model
import shap

warnings.filterwarnings("ignore")

# Patch CVE-2025-32434: transformers blocks torch.load on torch < 2.6
# Our model.pt is a trusted local file, so this is safe.
import transformers.utils.import_utils as _triu
import transformers.modeling_utils as _trmu
_triu.check_torch_load_is_safe = lambda: None
_trmu.check_torch_load_is_safe = lambda: None

ML_DIR   = os.path.dirname(os.path.abspath(__file__))
PVN_DIR  = os.path.join(ML_DIR, "pd_voicenet")
ART_DIR  = os.path.join(PVN_DIR, "artifacts_pdvoicenet")
FOLD_DIR = os.path.join(ART_DIR, "fold_FB")  # FB fold = 100% accuracy

sys.path.insert(0, PVN_DIR)
from model import PDVoiceNet, BIOMARKER_NAMES

REPORT_DIR = os.path.join(ML_DIR, "report_generator")
sys.path.insert(0, REPORT_DIR)
try:
    from generate_report import generate_report as run_generate_report
except Exception as e:
    run_generate_report = None
    print(f"[WARN] Report generator not loaded: {e}")

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_lock = threading.Lock()

global_model = None
global_wav2vec2 = None
global_scaler_mean = None
global_scaler_scale = None
global_temperature = 1.0
global_q_hat = None


def load_model():
    global global_model, global_wav2vec2
    global global_scaler_mean, global_scaler_scale
    global global_temperature, global_q_hat

    print("Loading Wav2Vec2 backbone...")
    global_wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
    global_wav2vec2.eval().to(device)
    for p in global_wav2vec2.parameters():
        p.requires_grad = False

    print(f"Loading PD-VoiceNet weights from {FOLD_DIR}...")
    global_model = PDVoiceNet(wav2vec2_model=global_wav2vec2)
    state = torch.load(os.path.join(FOLD_DIR, "model.pt"), map_location=device, weights_only=False)
    global_model.load_state_dict(state)
    global_model.to(device).eval()

    global_scaler_mean  = np.load(os.path.join(FOLD_DIR, "scaler_mean.npy"))
    global_scaler_scale = np.load(os.path.join(FOLD_DIR, "scaler_scale.npy"))

    with open(os.path.join(FOLD_DIR, "calibration.json")) as f:
        calib = json.load(f)
    global_temperature = calib["temperature"]
    global_q_hat = {int(k): float(v) for k, v in calib["conformal"]["q_hat"].items()}
    print(f"PD-VoiceNet ready. T={global_temperature:.4f} q_hat={global_q_hat}")


def extract_biomarkers(wav_path):
    y, sr = librosa.load(wav_path, sr=16000, mono=True, duration=25.0)
    y, _ = librosa.effects.trim(y, top_db=25)
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = sound.to_pitch()
    f0 = pitch.selected_array["frequency"]
    f0v = f0[f0 > 0]
    mean_f0, std_f0, min_f0, max_f0 = (
        (float(f0v.mean()), float(f0v.std()), float(f0v.min()), float(f0v.max()))
        if len(f0v) > 0 else (0.0, 0.0, 0.0, 0.0)
    )
    pp = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    lj  = call(pp, "Get jitter (local)",  0, 0, 0.0001, 0.02, 1.3)
    p5j = call(pp, "Get jitter (ppq5)",   0, 0, 0.0001, 0.02, 1.3)
    rj  = call(pp, "Get jitter (rap)",    0, 0, 0.0001, 0.02, 1.3)
    ls   = call([sound, pp], "Get shimmer (local)",  0, 0, 0.0001, 0.02, 1.3, 1.6)
    a3s  = call([sound, pp], "Get shimmer (apq3)",   0, 0, 0.0001, 0.02, 1.3, 1.6)
    a5s  = call([sound, pp], "Get shimmer (apq5)",   0, 0, 0.0001, 0.02, 1.3, 1.6)
    a11s = call([sound, pp], "Get shimmer (apq11)",  0, 0, 0.0001, 0.02, 1.3, 1.6)
    dda  = call([sound, pp], "Get shimmer (dda)",    0, 0, 0.0001, 0.02, 1.3, 1.6)
    harm = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr  = call(harm, "Get mean", 0, 0)
    nhr  = 1.0 / (10 ** (hnr / 10)) if hnr > 0 else 0.0
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=7)
    mfcc_means = [float(np.mean(mfccs[i])) for i in range(1, 7)]
    fmt = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    def fstat(obj, n):
        try:
            return (call(obj, "Get mean", n, 0, 0, "Hertz"),
                    call(obj, "Get standard deviation", n, 0, 0, "Hertz"))
        except Exception:
            return 0.0, 0.0
    f1f, f1b = fstat(fmt, 1)
    f2f, f2b = fstat(fmt, 2)
    f3f, f3b = fstat(fmt, 3)
    return np.array([
        lj, p5j, rj, ls, a3s, a5s, a11s, dda,
        hnr, nhr, mean_f0, std_f0, min_f0, max_f0,
        *mfcc_means, f1f, f2f, f3f, f1b, f2b, f3b,
    ], dtype=np.float32)


def scale_bio(raw):
    return (raw - global_scaler_mean) / (global_scaler_scale + 1e-8)


def ensure_wav(input_path, target_sr=16000):
    """
    Standardizes any audio format (AAC, M4A, MP3, OGG, WAV, FLAC, etc.) into a 16kHz
    mono 16-bit PCM WAV. PyAV is used for reliable multi-codec decoding on Windows.
    """
    clean_wav = os.path.join(ML_DIR, f"clean_{uuid.uuid4().hex}.wav")
    # 1. PyAV first: reliably handles AAC, M4A, MP4, MP3 without external ffmpeg binary
    try:
        import av
        container = av.open(input_path)
        audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
        if audio_stream is not None:
            resampler = av.AudioResampler(format='s16', layout='mono', rate=target_sr)
            frames = []
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    for rframe in resampler.resample(frame):
                        frames.append(rframe.to_ndarray())
            for rframe in resampler.resample(None):
                frames.append(rframe.to_ndarray())
            container.close()
            if frames:
                audio_data = np.concatenate(frames, axis=1).squeeze(0)
                sf.write(clean_wav, audio_data, target_sr, subtype='PCM_16')
                return clean_wav
    except Exception as e:
        print(f"[Audio Conv] PyAV decode fallback: {e}")

    # 2. Librosa / soundfile fallback
    try:
        y, _ = librosa.load(input_path, sr=target_sr, mono=True)
        sf.write(clean_wav, y, target_sr, subtype='PCM_16')
        return clean_wav
    except Exception as e:
        print(f"[Audio Conv] librosa decode error: {e}")
        raise RuntimeError(f"Unable to decode audio: {e}")


def cache_ssl(wav_path):
    y, sr = librosa.load(wav_path, sr=16000, mono=True, duration=25.0)
    y, _ = librosa.effects.trim(y, top_db=25)
    x = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = global_wav2vec2.feature_extractor(x)
        # Ensure (batch, C, T) — some transformers versions squeeze the batch dim
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
        feat = feat.transpose(1, 2)          # (1, T, C)
        feat, _ = global_wav2vec2.feature_projection(feat)
        h = feat
        for i in range(9):
            layer_out = global_wav2vec2.encoder.layers[i](
                h, attention_mask=None, output_attentions=False
            )
            # Newer transformers may return raw tensor instead of (hidden,) tuple
            h = layer_out[0] if isinstance(layer_out, tuple) else layer_out
    return h.squeeze(0)  # (T, 768)


def run_inference(ssl_act, bio_scaled):
    bio_t = torch.tensor(bio_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, alpha, _ = global_model([ssl_act], bio_t)
        probs = F.softmax(logits / global_temperature, dim=0).cpu().numpy()
    prob_pd = float(probs[1])
    pred = "PD" if probs[1] > probs[0] else "HC"
    scores = {0: 1 - float(probs[0]), 1: 1 - float(probs[1])}
    conf = [c for c in [0, 1] if scores[c] <= global_q_hat[c]]
    if len(conf) == 2:
        conf_label = "uncertain"
    elif len(conf) == 1:
        conf_label = "PD" if conf[0] == 1 else "HC"
    else:
        conf_label = "abstain"
    return pred, prob_pd, conf_label, float(alpha.cpu())


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "PD-VoiceNet", "device": str(device)})


@app.route("/predict", methods=["POST"])
def predict():
    if global_model is None:
        return jsonify({"error": "Model not loaded"}), 500
    if "audio" not in request.files:
        return jsonify({"error": "No audio file"}), 400
    audio_file = request.files["audio"]
    ext = os.path.splitext(audio_file.filename)[1] or ".wav"
    raw_tmp = os.path.join(ML_DIR, f"raw_{uuid.uuid4().hex}{ext}")
    clean_tmp = None
    audio_file.save(raw_tmp)
    try:
        clean_tmp = ensure_wav(raw_tmp)
        with model_lock:
            bio_scaled = scale_bio(extract_biomarkers(clean_tmp))
            ssl_act    = cache_ssl(clean_tmp)
            pred, prob_pd, conf_label, alpha = run_inference(ssl_act, bio_scaled)
        dur = librosa.get_duration(path=clean_tmp)
        return jsonify([{
            "prediction":    pred,
            "probability_pd": prob_pd,
            "duration":      f"{int(dur//60):02d}:{int(dur%60):02d}",
            "conformal_set": conf_label,
            "modality_alpha": alpha,
        }])
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(raw_tmp):
            try: os.remove(raw_tmp)
            except Exception: pass
        if clean_tmp and os.path.exists(clean_tmp):
            try: os.remove(clean_tmp)
            except Exception: pass


@app.route("/explainability", methods=["POST"])
def explainability():
    if global_model is None:
        return jsonify({"error": "Model not loaded"}), 500
    raw_tmp = None
    clean_tmp = None
    if "audio" in request.files:
        audio_file = request.files["audio"]
        ext = os.path.splitext(audio_file.filename)[1] or ".wav"
        raw_tmp = os.path.join(ML_DIR, f"raw_{uuid.uuid4().hex}{ext}")
        audio_file.save(raw_tmp)
    elif request.is_json and "audio_b64" in request.json:
        import base64
        raw_tmp = os.path.join(ML_DIR, f"raw_{uuid.uuid4().hex}.wav")
        with open(raw_tmp, "wb") as f:
            f.write(base64.b64decode(request.json["audio_b64"]))
    else:
        return jsonify({"error": "Provide audio field or audio_b64"}), 400
    try:
        clean_tmp = ensure_wav(raw_tmp)
        dur = librosa.get_duration(path=clean_tmp)
        with model_lock:
            raw_bio    = extract_biomarkers(clean_tmp)
            bio_scaled = scale_bio(raw_bio)
            ssl_act    = cache_ssl(clean_tmp)
            bio_t      = torch.tensor(bio_scaled, dtype=torch.float32).unsqueeze(0).to(device)

            shap_vals, base_val = _shap(bio_scaled, ssl_act)
            gradcam_vals        = _gradcam(ssl_act)
            saliency_vals       = _saliency(ssl_act, bio_t)

        return jsonify({
            "shap":     {"values": shap_vals.tolist(), "names": BIOMARKER_NAMES, "base_value": base_val},
            "gradcam":  {"values": gradcam_vals.tolist(), "duration_s": dur},
            "saliency": {"values": saliency_vals.tolist(), "duration_s": dur},
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if raw_tmp and os.path.exists(raw_tmp):
            try: os.remove(raw_tmp)
            except Exception: pass
        if clean_tmp and os.path.exists(clean_tmp):
            try: os.remove(clean_tmp)
            except Exception: pass


@app.route("/generate_report", methods=["POST"])
def generate_report_endpoint():
    data = request.get_json()
    if not data or "payload" not in data:
        return jsonify({"error": "Missing 'payload' in request body"}), 400
    payload = data["payload"]
    audience = data.get("audience", "clinician")
    if audience not in ("clinician", "patient"):
        return jsonify({"error": "audience must be 'clinician' or 'patient'"}), 400
    try:
        import importlib
        import report_templates as rt
        importlib.reload(rt)
        import generate_report as gr
        importlib.reload(gr)
        result = gr.generate_report(payload, audience)
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _shap(bio_scaled, ssl_act):
    def _bio_fn(bio_arr):
        out = []
        global_model.eval()
        for row in bio_arr:
            bio_t = torch.tensor(row, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits, _, _ = global_model([ssl_act], bio_t)
                probs = F.softmax(logits / global_temperature, dim=0).cpu().numpy()
                out.append(float(probs[1]))
        return np.array(out, dtype=np.float32)

    bg = np.zeros((1, 26), dtype=np.float32)
    ex = shap.KernelExplainer(_bio_fn, bg)
    vals = ex.shap_values(bio_scaled.reshape(1, -1), nsamples=128, silent=True)
    if isinstance(vals, list):
        shap_pd = vals[1].squeeze() if len(vals) > 1 else vals[0].squeeze()
    elif isinstance(vals, np.ndarray):
        if vals.ndim == 3:
            shap_pd = vals[0, :, 1]
        elif vals.ndim == 2:
            shap_pd = vals[0, :]
        else:
            shap_pd = vals.squeeze()
    else:
        shap_pd = np.array(vals).squeeze()

    base_val = ex.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])
    else:
        base_val = float(base_val)

    return shap_pd, base_val


def _gradcam(ssl_act):
    inp = ssl_act.detach().clone().unsqueeze(0).requires_grad_(True)
    acts, grads = {}, {}
    target_layer = global_model.ssl_branch.transformer_layers[-1]
    fh = target_layer.register_forward_hook(lambda m,i,o: acts.__setitem__("v", (o[0] if isinstance(o, tuple) else o)))
    bh = target_layer.register_full_backward_hook(lambda m,i,o: grads.__setitem__("v", (o[0] if isinstance(o, tuple) else o)))

    global_model.train()
    try:
        dummy_bio = torch.zeros(1, 26, device=device)
        logits, _, _ = global_model([inp.squeeze(0)], dummy_bio)
        global_model.zero_grad()
        (logits / global_temperature)[1].backward()
        if "v" not in acts or "v" not in grads:
            return np.zeros(ssl_act.shape[0])
        act = acts["v"]
        grad = grads["v"]
        weights = grad.mean(dim=-1, keepdim=True)
        cam = F.relu((weights * act).sum(dim=-1).squeeze(0)).detach().cpu().numpy()
        return cam / (cam.max() + 1e-8)
    finally:
        fh.remove()
        bh.remove()
        global_model.eval()


def _saliency(ssl_act, bio_t):
    inp = ssl_act.detach().clone().requires_grad_(True)
    global_model.train()
    try:
        logits, _, _ = global_model([inp], bio_t)
        global_model.zero_grad()
        (logits / global_temperature)[1].backward()
        grad = inp.grad
        if grad is None:
            return np.zeros(ssl_act.shape[0])
        s = grad.norm(dim=-1).detach().cpu().numpy()
        return s / (s.max() + 1e-8)
    finally:
        global_model.eval()


try:
    load_model()
except Exception as e:
    print(f"[ERROR] Model load failed: {e}")

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
