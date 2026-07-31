"""
Classify audio recordings as Healthy Control (HC) or Parkinson's Disease (PD)
using the trained hybrid MLP+CNN+RNN+MKL model.

Usage:
    python predict.py file1.wav file2.wav --model_dir ./artifacts
"""

import argparse
import json
import os

import joblib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

from feature_extraction import extract_features
from model import MKLFusionLayer  # required for loading the custom layer


def load_artifacts(model_dir):
    model = tf.keras.models.load_model(
        os.path.join(model_dir, "pd_voice_model.keras"),
        custom_objects={"MKLFusionLayer": MKLFusionLayer},
    )
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    return model, scaler


import librosa

def get_duration_formatted(path):
    try:
        duration_sec = librosa.get_duration(path=path)
        minutes = int(duration_sec // 60)
        seconds = int(duration_sec % 60)
        return f"{minutes:02d}:{seconds:02d}"
    except:
        return "00:00"

def classify(paths, model_dir="./artifacts"):
    model, scaler = load_artifacts(model_dir)
    results = []
    for path in paths:
        feats = extract_features(path).reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        prob_pd = float(model.predict(feats_scaled, verbose=0)[0, 1])
        label = "PD" if prob_pd >= 0.5 else "HC"
        results.append({
            "file": path,
            "prediction": label,
            "probability_pd": round(prob_pd, 4),
            "duration": get_duration_formatted(path)
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="One or more .wav files to classify")
    parser.add_argument("--model_dir", default="./artifacts")
    parser.add_argument("--json", action="store_true", help="Output strict JSON")
    args = parser.parse_args()

    results = classify(args.files, model_dir=args.model_dir)
    if args.json:
        # Output strictly the JSON for parsing
        print(json.dumps(results))
    else:
        print("\n" + "="*40)
        print("      LIVE PREDICTION RESULTS")
        print("="*40)
        for r in results:
            prob = r['probability_pd']
            filename = os.path.basename(r['file'])
            if r['prediction'] == 'PD':
                print(f"\n🎙️  Audio File: {filename}")
                print(f"🏥  Diagnosis: PARKINSON'S DISEASE")
                print(f"🎯  Confidence: {prob:.2%}")
            else:
                print(f"\n🎙️  Audio File: {filename}")
                print(f"✅  Diagnosis: HEALTHY CONTROL")
                print(f"🎯  Confidence: {(1.0 - prob):.2%}")
        print("\n" + "="*40 + "\n")


if __name__ == "__main__":
    main()
