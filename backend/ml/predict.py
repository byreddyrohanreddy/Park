import argparse
import json
import os
import joblib
import numpy as np
import librosa
from feature_extraction import extract_features

def load_artifacts(model_dir):
    model = joblib.load(os.path.join(model_dir, "pd_voice_model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    return model, scaler

def get_duration_formatted(path):
    try:
        duration_sec = librosa.get_duration(path=path)
        minutes = int(duration_sec // 60)
        seconds = int(duration_sec % 60)
        return f"{minutes:02d}:{seconds:02d}"
    except:
        return "00:00"

def classify_with_model(paths, model, scaler):
    results = []
    for path in paths:
        # Get just the 13 acoustic features from the extraction dictionary
        feats = extract_features(path)["acoustic"].reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        
        prob_pd = float(model.predict_proba(feats_scaled)[0, 1])
        duration_str = get_duration_formatted(path)

        label = "PD" if prob_pd >= 0.5 else "HC"
        results.append({
            "file": path,
            "prediction": label,
            "probability_pd": round(prob_pd, 4),
            "duration": duration_str
        })
    return results

def classify(paths, model_dir="./artifacts_rf"):
    model, scaler = load_artifacts(model_dir)
    return classify_with_model(paths, model, scaler)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--model_dir", default="./artifacts_rf")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results = classify(args.files, model_dir=args.model_dir)
    if args.json:
        print(json.dumps(results))
    else:
        for r in results:
            print(r)

if __name__ == '__main__':
    main()
