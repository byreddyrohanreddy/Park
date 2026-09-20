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

from feature_extraction import extract_features

def load_artifacts(model_dir):
    model = joblib.load(os.path.join(model_dir, "pd_voice_model.joblib"))
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

def classify_with_model(paths, model, scaler, is_noisy_mic=False):
    results = []
    for path in paths:
        feats = extract_features(path, is_noisy_mic=is_noisy_mic).reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        
        prob_pd = float(model.predict_proba(feats_scaled)[0, 1])
        

        duration_str = get_duration_formatted(path)

        # Use the standard 0.5 threshold
        label = "PD" if prob_pd >= 0.5 else "HC"
        results.append({
            "file": path,
            "prediction": label,
            "probability_pd": round(prob_pd, 4),
            "duration": duration_str
        })
    return results

def classify(paths, model_dir="./artifacts"):
    model, scaler = load_artifacts(model_dir)
    return classify_with_model(paths, model, scaler)


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
