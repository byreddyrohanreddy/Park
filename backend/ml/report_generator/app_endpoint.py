"""
app_endpoint.py — Step 6: Standalone Flask endpoint for report generation.

Run standalone with: python app_endpoint.py
Exposes POST /generate_report on port 5050.

Do NOT wire this into the main app.py or Node backend.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify
from generate_report import generate_report

app = Flask(__name__)


@app.route("/generate_report", methods=["POST"])
def generate_report_endpoint():
    """Generate a clinical report from a dashboard payload.
    
    Request body:
        {
            "payload": { ...dashboard payload... },
            "audience": "clinician" | "patient"
        }
    
    Response:
        {
            "report": "...",
            "generation_method": "llm" | "template_fallback",
            "source_payload": { ... },
            "retrieved_chunks": [ ... ],
            "model_used": "..." | null
        }
    """
    data = request.get_json()
    
    if not data or "payload" not in data:
        return jsonify({"error": "Missing 'payload' in request body"}), 400
    
    payload = data["payload"]
    audience = data.get("audience", "clinician")
    
    if audience not in ("clinician", "patient"):
        return jsonify({"error": "audience must be 'clinician' or 'patient'"}), 400
    
    try:
        result = generate_report(payload, audience)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "report_generator"})


if __name__ == "__main__":
    print("Starting PD-VoiceNet Report Generator on port 5050...")
    print("POST /generate_report  — Generate a clinical report")
    print("GET  /health           — Health check")
    app.run(host="0.0.0.0", port=5050, debug=True)
