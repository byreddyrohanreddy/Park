import { useLocation, useNavigate, Navigate } from "react-router-dom";
import { useEffect, useState } from "react";
import { ArrowLeft } from "lucide-react";
import { motion } from "framer-motion";

import Sidebar from "../components/Sidebar";
import Button from "../components/Button";
import PredictionSummary from "../components/clinical/PredictionSummary";
import ClinicalReportSection from "../components/clinical/ClinicalReportSection";
import ExplainabilitySection from "../components/clinical/ExplainabilitySection";

import { generateReport, getExplainability } from "../services/api";

import {
  mockPayloadPD,
  mockPayloadUncertain,
  mockReportClinician,
  mockReportPatient,
  mockReportUncertainClinician,
} from "../mocks/explainabilityMock";

/**
 * ClinicalReport — Full results page.
 * Receives: location.state.recording + optional location.state.audioFile
 * If audioFile is present, calls real /explainability endpoint.
 */
function ClinicalReport() {
  const location = useLocation();
  const navigate = useNavigate();
  const recording = location.state?.recording;
  const audioFile = location.state?.audioFile ?? null;

  const [explainData, setExplainData] = useState(null);
  const [explainLoading, setExplainLoading] = useState(!!audioFile);
  const [explainError, setExplainError] = useState(null);

  useEffect(() => {
    if (!audioFile) return;
    (async () => {
      setExplainLoading(true);
      try {
        const data = await getExplainability(audioFile);
        if (data && data.shap && data.gradcam && data.saliency) {
          setExplainData(data);
        } else {
          setExplainError("Explainability service returned unexpected data.");
        }
      } catch (err) {
        console.warn("Explainability fetch failed:", err);
        setExplainError("ML service unavailable — showing mock data.");
      } finally {
        setExplainLoading(false);
      }
    })();
  }, [audioFile]);

  if (!recording) return <Navigate to="/dashboard" />;

  const isPD = (recording.prediction || "").toLowerCase().includes("parkinson") ||
               recording.prediction === "PD";
  const isUncertain = (recording.prediction || "").toLowerCase().includes("uncertain") ||
                      (recording.prediction || "").toLowerCase().includes("pending");

  const baseMock = isUncertain ? mockPayloadUncertain : mockPayloadPD;
  const confidence = (recording.confidence || 50) / 100;

  const payload = {
    subject_id: recording._id || recording.fileName || baseMock.subject_id,
    prediction: recording.prediction || (isPD ? "Parkinson's Disease" : "Healthy Control"),
    prediction_set: isUncertain ? ["HC", "PD"] : (isPD ? ["PD"] : ["HC"]),
    refer_to_clinician: isUncertain,
    probability_pd: recording.probability_pd ?? (isPD ? confidence : 1 - confidence),
    probability_hc: recording.probability_hc ?? (isPD ? 1 - confidence : confidence),
    modality_gate_alpha: recording.modality_alpha ?? baseMock.modality_gate_alpha,
    modality_dominance: recording.modality_dominance ?? baseMock.modality_dominance,
    attention_weights: recording.attention_weights ?? baseMock.attention_weights,
    shap_top5: explainData
      ? explainData.shap.names
          .map((name, i) => ({ feature: name, value: explainData.shap.values[i] }))
          .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
          .slice(0, 5)
      : baseMock.shap_top5,
    grad_cam_cnn: explainData?.gradcam?.values
      ? [
          {
            recording_id: recording._id || recording.fileName || "rec_001",
            timestamps_sec: explainData.gradcam.values.map((_, i) =>
              +((i / Math.max(1, explainData.gradcam.values.length - 1)) * (explainData.gradcam.duration_s || 1)).toFixed(3)
            ),
            importance_values: explainData.gradcam.values,
          },
        ]
      : baseMock.grad_cam_cnn,
    transformer_saliency: explainData?.saliency?.values
      ? [
          {
            recording_id: recording._id || recording.fileName || "rec_001",
            timestamps_sec: explainData.saliency.values.map((_, i) =>
              +((i / Math.max(1, explainData.saliency.values.length - 1)) * (explainData.saliency.duration_s || 1)).toFixed(3)
            ),
            importance_values: explainData.saliency.values,
          },
        ]
      : baseMock.transformer_saliency,
    shap_base_value: explainData?.shap?.base_value ?? null,
    shap_all_values: explainData ? explainData.shap : null,
  };

  const handleGenerateReport = async (audience) => {
    try {
      const response = await generateReport(payload, audience);
      if (response && !response.error && response.report) return response;
    } catch (err) {
      console.warn("Live report generator unavailable, using mock:", err);
    }
    await new Promise((r) => setTimeout(r, 1200));
    if (isUncertain) return mockReportUncertainClinician;
    return audience === "clinician" ? mockReportClinician : mockReportPatient;
  };

  return (
    <div className="flex min-h-screen bg-slate-50">
      <Sidebar />
      <main className="flex-1 p-6 md:p-10">
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-4xl mx-auto"
        >
          <button
            onClick={() => navigate("/result", { state: { recording } })}
            className="flex items-center gap-2 text-gray-500 hover:text-blue-600 transition mb-6 text-sm font-medium"
          >
            <ArrowLeft size={16} /> Back to Results
          </button>

          <h1 className="text-3xl font-bold text-gray-800 mb-6">
            Clinical Report &amp; Explainability
          </h1>

          {explainLoading && (
            <div className="mb-4 px-4 py-3 bg-blue-50 border border-blue-200 text-blue-700 rounded-xl text-sm">
              Computing SHAP, Grad-CAM, and Transformer Saliency...
            </div>
          )}
          {explainError && (
            <div className="mb-4 px-4 py-3 bg-yellow-50 border border-yellow-200 text-yellow-800 rounded-xl text-sm">
              {explainError}
            </div>
          )}
          {explainData && !explainLoading && (
            <div className="mb-4 px-4 py-3 bg-green-50 border border-green-200 text-green-700 rounded-xl text-sm">
              Real SHAP, Grad-CAM, and Transformer Saliency computed successfully.
            </div>
          )}

          <PredictionSummary recording={recording} payload={payload} />
          <ClinicalReportSection payload={payload} onGenerateReport={handleGenerateReport} />
          <ExplainabilitySection payload={payload} loading={explainLoading} />

          <div className="flex justify-center mt-8 mb-10">
            <Button onClick={() => navigate("/dashboard")} className="px-8 py-3 flex items-center">
              <ArrowLeft className="mr-2" size={20} /> Return to Dashboard
            </Button>
          </div>
        </motion.div>
      </main>
    </div>
  );
}

export default ClinicalReport;
