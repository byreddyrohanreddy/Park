import { Activity, ShieldCheck, AlertTriangle } from "lucide-react";
import Card from "../Card";

/**
 * PredictionSummary — Compact prediction display with uncertainty banner.
 *
 * Props:
 *   recording  — { prediction, confidence } from the existing upload flow
 *   payload    — extended dashboard payload (for refer_to_clinician, prediction_set, probabilities)
 */
function PredictionSummary({ recording, payload }) {
  const isPD = recording.prediction.toLowerCase().includes("parkinson");
  const isUncertain = payload?.refer_to_clinician;

  return (
    <Card className="mb-6">
      {/* Uncertainty Banner — above everything */}
      {isUncertain && (
        <div className="bg-amber-50 border-2 border-amber-300 rounded-xl p-4 mb-6 flex items-start gap-3">
          <AlertTriangle className="text-amber-500 flex-shrink-0 mt-0.5" size={24} />
          <div>
            <h3 className="font-bold text-amber-800 text-lg">
              Inconclusive Result
            </h3>
            <p className="text-amber-700 mt-1">
              ⚠ This result was inconclusive — clinical follow-up is recommended.
              The prediction set includes both{" "}
              <span className="font-semibold">
                {payload.prediction_set?.join(" and ")}
              </span>
              .
            </p>
          </div>
        </div>
      )}

      {/* Prediction Display */}
      <div
        className={`p-6 rounded-2xl flex flex-col sm:flex-row items-center justify-between border-2 ${
          isUncertain
            ? "bg-amber-50 border-amber-200"
            : isPD
            ? "bg-orange-50 border-orange-200"
            : "bg-green-50 border-green-200"
        }`}
      >
        <div className="flex items-center gap-4">
          {isUncertain ? (
            <AlertTriangle size={48} className="text-amber-500" />
          ) : isPD ? (
            <Activity size={48} className="text-orange-500" />
          ) : (
            <ShieldCheck size={48} className="text-green-500" />
          )}

          <div>
            <h2
              className={`text-2xl font-extrabold ${
                isUncertain
                  ? "text-amber-700"
                  : isPD
                  ? "text-orange-700"
                  : "text-green-700"
              }`}
            >
              {recording.prediction}
            </h2>
            <p
              className={`text-sm font-medium mt-1 ${
                isUncertain
                  ? "text-amber-600"
                  : isPD
                  ? "text-orange-600"
                  : "text-green-600"
              }`}
            >
              Confidence: {recording.confidence}%
            </p>
          </div>
        </div>

        {/* Probability breakdown */}
        {payload?.probability_pd != null && (
          <div className="mt-4 sm:mt-0 text-right text-sm text-gray-600">
            <p>
              P(PD):{" "}
              <span className="font-semibold">
                {(payload.probability_pd * 100).toFixed(1)}%
              </span>
            </p>
            <p>
              P(HC):{" "}
              <span className="font-semibold">
                {(payload.probability_hc * 100).toFixed(1)}%
              </span>
            </p>
            <p className="text-xs text-gray-400 mt-1">
              Modality: {payload.modality_dominance} (α={payload.modality_gate_alpha?.toFixed(2)})
            </p>
          </div>
        )}
      </div>

      {/* Persistent Disclaimer */}
      <div className="mt-4 bg-blue-50 border border-blue-200 rounded-xl p-3 text-center">
        <p className="text-sm text-blue-700">
          <span className="font-semibold">ℹ️ Disclaimer:</span> This is a decision-support
          tool, not a medical diagnosis. Please discuss these results with a qualified
          clinician.
        </p>
      </div>
    </Card>
  );
}

export default PredictionSummary;
