const express = require("express");
const router = express.Router();
const multer = require("multer");

const auth = require("../middleware/authMiddleware");

const FLASK_ML = process.env.FLASK_ML || "http://127.0.0.1:5001";
const FLASK_REPORT = process.env.FLASK_REPORT || "http://127.0.0.1:5001";

// multer: memory storage — gives us req.file.buffer to forward to Flask
const upload = multer({ storage: multer.memoryStorage() });


// =========================
// Generate Clinical Report
// =========================
router.post("/generate", auth, async (req, res) => {
  try {
    const { payload, audience } = req.body;
    if (!payload) {
      return res.status(400).json({ success: false, message: "Missing payload" });
    }

    const targets = [
      `${FLASK_REPORT}/generate_report`,
      "http://127.0.0.1:5001/generate_report",
      "http://127.0.0.1:5050/generate_report",
    ];
    const uniqueTargets = [...new Set(targets)];

    let lastError = null;
    for (const targetUrl of uniqueTargets) {
      try {
        const response = await fetch(targetUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ payload, audience: audience || "clinician" }),
        });
        const data = await response.json();
        if (response.ok) {
          return res.json(data);
        }
      } catch (err) {
        lastError = err;
      }
    }

    console.error("Report proxy error:", lastError ? lastError.message : "All targets failed");
    res.status(502).json({ success: false, message: "Report service unavailable: " + (lastError ? lastError.message : "All endpoints failed") });
  } catch (err) {
    console.error("Report proxy error:", err.message);
    res.status(500).json({ success: false, message: "Internal server error: " + err.message });
  }
});


// =============================================
// Explainability — SHAP + Grad-CAM + Saliency
// =============================================
// Uses Node 18+ built-in FormData + Blob (compatible with native fetch).
// The npm "form-data" package does NOT work with native fetch.
router.post("/explainability", auth, upload.single("audio"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, message: "No audio file provided." });
    }

    // Build form using native Node 18 FormData + Blob
    const form = new FormData();
    const blob = new Blob([req.file.buffer], { type: req.file.mimetype || "audio/wav" });
    form.append("audio", blob, req.file.originalname || "audio.wav");

    const flaskRes = await fetch(`${FLASK_ML}/explainability`, {
      method: "POST",
      body: form,
      // Do NOT set Content-Type manually — fetch auto-sets multipart/form-data with correct boundary
    });

    const rawText = await flaskRes.text();
    let data;
    try {
      const sanitizedText = rawText
        .replace(/:\s*NaN\b/g, ': null')
        .replace(/:\s*Infinity\b/g, ': 999999')
        .replace(/:\s*-Infinity\b/g, ': -999999');
      data = JSON.parse(sanitizedText);
    } catch (parseErr) {
      console.error("Flask explainability parse error:", rawText);
      return res.status(500).json({ success: false, message: "Invalid JSON response from ML service" });
    }

    if (!flaskRes.ok) {
      console.error("Flask explainability error:", JSON.stringify(data));
      return res.status(flaskRes.status).json({ success: false, message: (data && data.error) || "Explainability failed" });
    }

    res.json({ success: true, ...data });

  } catch (err) {
    console.error("Explainability proxy error:", err.message);
    res.status(502).json({ success: false, message: "ML service unavailable: " + err.message });
  }
});


// Legacy GET (410 Gone)
router.get("/explainability/:id", auth, (req, res) => {
  res.status(410).json({ success: false, message: "Use POST /api/report/explainability with audio file." });
});


module.exports = router;
