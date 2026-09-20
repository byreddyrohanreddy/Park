from flask import Flask, request, jsonify
import os
from predict import load_artifacts, classify_with_model

app = Flask(__name__)

# Preload the model globally so it stays in RAM
MODEL_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
print(f"Starting ML API. Loading TensorFlow Model from {MODEL_DIR}...")
try:
    global_model, global_scaler = load_artifacts(MODEL_DIR)
    print("TensorFlow Model loaded successfully into memory!")
    
    import numpy as np
    print("Warming up Random Forest model...")
    dummy_feats = np.zeros((1, 13))
    dummy_scaled = global_scaler.transform(dummy_feats)
    global_model.predict_proba(dummy_scaled)
    print("Model warmup complete! Ready for instantaneous predictions.")
    
except Exception as e:
    print(f"Failed to load model: {e}")
    global_model, global_scaler = None, None

import uuid

@app.route('/predict', methods=['POST'])
def predict():
    if not global_model or not global_scaler:
        return jsonify({"error": "Model failed to load on server startup"}), 500

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided in request"}), 400
        
    file = request.files['audio']
    # Use unique filename to prevent WinError 32 lock conflicts
    temp_path = f"temp_upload_{uuid.uuid4().hex}.wav"
    file.save(temp_path)
    
    is_noisy_mic = request.form.get('isNoisyMic') == 'true'
    
    def safe_remove(path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    try:
        # Pass the preloaded model directly (no cold start!)
        results = classify_with_model([temp_path], global_model, global_scaler, is_noisy_mic=is_noisy_mic)
        safe_remove(temp_path)
        return jsonify(results)
    except Exception as e:
        safe_remove(temp_path)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use port 5001 so it doesn't conflict with Node.js on port 5000
    app.run(host='127.0.0.1', port=5001, debug=False)
