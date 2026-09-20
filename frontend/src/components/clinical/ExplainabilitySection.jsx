import { useState } from "react";
import { BarChart3, AudioWaveform, Brain, AlertCircle, Layers } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  AreaChart,
  Area,
  LineChart,
  Line,
  Legend,
} from "recharts";
import Card from "../Card";

const TAB_CONFIG = [
  { id: "shap", label: "Biomarker Attributions", icon: BarChart3 },
  { id: "gradcam", label: "Audio Attention (Grad-CAM)", icon: AudioWaveform },
  { id: "saliency", label: "Model Attention (Saliency)", icon: Brain },
];

/**
 * ExplainabilitySection — Tabbed explainability views.
 *
 * Props:
 *   payload — extended dashboard payload containing shap_top5, grad_cam_cnn, transformer_saliency
 */
function ExplainabilitySection({ payload }) {
  const [activeTab, setActiveTab] = useState("shap");

  return (
    <Card>
      <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
        <Layers size={22} className="text-blue-600" />
        Explainability
      </h2>

      {/* Tab Bar */}
      <div className="flex gap-1 bg-gray-100 rounded-xl p-1 mb-5 overflow-x-auto">
        {TAB_CONFIG.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setActiveTab(id)}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-semibold transition whitespace-nowrap flex-1 justify-center ${
              activeTab === id
                ? "bg-blue-600 text-white shadow-md"
                : "text-gray-600 hover:bg-gray-200"
            }`}
          >
            <Icon size={16} />
            <span className="hidden sm:inline">{label}</span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === "shap" && <ShapTab data={payload?.shap_top5} />}
      {activeTab === "gradcam" && <GradCamTab data={payload?.grad_cam_cnn} saliencyData={payload?.transformer_saliency} />}
      {activeTab === "saliency" && <SaliencyTab data={payload?.transformer_saliency} gradcamData={payload?.grad_cam_cnn} />}
    </Card>
  );
}


// ─── Tab 1: SHAP Biomarker Attributions ─────────────────────────────────────

function ShapTab({ data }) {
  if (!data || data.length === 0) {
    return <NotAvailable label="Biomarker attribution data" />;
  }

  // Prepare data for horizontal bar chart
  const chartData = data.map((item) => ({
    feature: formatFeatureName(item.feature),
    value: item.value,
    rawValue: item.value,
  }));

  return (
    <div>
      <p className="text-sm text-gray-500 mb-4">
        Top 5 acoustic features influencing the prediction. Positive values push toward
        PD; negative values push toward HC.
      </p>

      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 30, top: 5, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis type="number" tick={{ fontSize: 12 }} label={{ value: "SHAP Value", position: "insideBottom", offset: -2, fontSize: 12, fill: "#64748b" }} />
            <YAxis type="category" dataKey="feature" width={120} tick={{ fontSize: 12 }} />
            <Tooltip
              formatter={(value) => [value.toFixed(4), "SHAP Value"]}
              contentStyle={{ borderRadius: "12px", border: "1px solid #e2e8f0" }}
            />
            <Bar dataKey="value" radius={[0, 6, 6, 0]} maxBarSize={28}>
              {chartData.map((entry, index) => (
                <Cell
                  key={index}
                  fill={entry.rawValue >= 0 ? "#14b8a6" : "#f97316"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex justify-center gap-6 mt-3 text-xs text-gray-500">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm bg-teal-500" />
          <span>Pushes toward PD</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-3 rounded-sm bg-orange-500" />
          <span>Pushes toward HC</span>
        </div>
      </div>
    </div>
  );
}


function normalizeRecordings(data) {
  if (!data) return [];
  const list = Array.isArray(data) ? data : [data];
  return list
    .filter(Boolean)
    .map((item, idx) => {
      const vals = item.importance_values || item.values || [];
      const dur = item.duration_s || (vals.length > 0 ? vals.length * 0.02 : 1);
      const step = vals.length > 1 ? dur / (vals.length - 1) : 0.02;
      const timestamps = item.timestamps_sec || vals.map((_, i) => +(i * step).toFixed(3));
      return {
        recording_id: item.recording_id || `rec_${idx + 1}`,
        timestamps_sec: timestamps,
        importance_values: vals,
      };
    });
}

// ─── Tab 2: Grad-CAM (CNN) ──────────────────────────────────────────────────

function GradCamTab({ data, saliencyData }) {
  const [selectedRec, setSelectedRec] = useState(0);
  const [showCompare, setShowCompare] = useState(false);

  const normData = normalizeRecordings(data);
  const normSaliency = normalizeRecordings(saliencyData);

  if (!normData || normData.length === 0) {
    return <NotAvailable label="Grad-CAM attention data" />;
  }

  const rec = normData[Math.min(selectedRec, normData.length - 1)];
  const chartData = (rec?.timestamps_sec || []).map((t, i) => ({
    time: t,
    importance: rec.importance_values?.[i] ?? 0,
  }));

  // Add saliency data for comparison overlay
  const hasSaliencyMatch = normSaliency && normSaliency[selectedRec];
  if (showCompare && hasSaliencyMatch) {
    const salRec = normSaliency[selectedRec];
    chartData.forEach((point, i) => {
      point.saliency = i < (salRec.importance_values?.length ?? 0) ? salRec.importance_values[i] : null;
    });
  }

  return (
    <div>
      <p className="text-sm text-gray-500 mb-1 font-medium">
        Grad-CAM: Acoustic feature detector attention
      </p>
      <p className="text-xs text-gray-400 mb-4">
        Highlights temporal regions where Wav2Vec2 layers activated most strongly for the PD classification.
      </p>

      {/* Recording Selector */}
      {normData.length > 1 && (
        <div className="flex gap-2 mb-4">
          {normData.map((r, i) => (
            <button
              key={r.recording_id || i}
              onClick={() => setSelectedRec(i)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                selectedRec === i
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              Recording {i + 1}
            </button>
          ))}
        </div>
      )}

      {/* Compare toggle */}
      {hasSaliencyMatch && (
        <label className="flex items-center gap-2 mb-3 text-xs text-gray-500 cursor-pointer">
          <input
            type="checkbox"
            checked={showCompare}
            onChange={(e) => setShowCompare(e.target.checked)}
            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
          />
          Overlay transformer saliency for comparison
        </label>
      )}

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          {showCompare && hasSaliencyMatch ? (
            <LineChart data={chartData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="time" tick={{ fontSize: 11 }} label={{ value: "Time (seconds)", position: "insideBottom", offset: -2, fontSize: 11, fill: "#64748b" }} />
              <YAxis tick={{ fontSize: 11 }} label={{ value: "Importance", angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }} />
              <Tooltip contentStyle={{ borderRadius: "12px", border: "1px solid #e2e8f0" }} />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <Line type="monotone" dataKey="importance" stroke="#2563eb" strokeWidth={2} dot={false} name="Grad-CAM" />
              <Line type="monotone" dataKey="saliency" stroke="#14b8a6" strokeWidth={2} dot={false} name="Transformer Saliency" strokeDasharray="5 5" />
            </LineChart>
          ) : (
            <AreaChart data={chartData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
              <defs>
                <linearGradient id="gradcamGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#2563eb" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#2563eb" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="time" tick={{ fontSize: 11 }} label={{ value: "Time (seconds)", position: "insideBottom", offset: -2, fontSize: 11, fill: "#64748b" }} />
              <YAxis tick={{ fontSize: 11 }} label={{ value: "Importance", angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }} />
              <Tooltip contentStyle={{ borderRadius: "12px", border: "1px solid #e2e8f0" }} />
              <Area type="monotone" dataKey="importance" stroke="#2563eb" fill="url(#gradcamGradient)" strokeWidth={2} />
            </AreaChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}


// ─── Tab 3: Transformer Saliency ────────────────────────────────────────────

function SaliencyTab({ data, gradcamData }) {
  const [selectedRec, setSelectedRec] = useState(0);
  const [showCompare, setShowCompare] = useState(false);

  const normData = normalizeRecordings(data);
  const normGradcam = normalizeRecordings(gradcamData);

  if (!normData || normData.length === 0) {
    return <NotAvailable label="Transformer saliency data" />;
  }

  const rec = normData[Math.min(selectedRec, normData.length - 1)];
  const chartData = (rec?.timestamps_sec || []).map((t, i) => ({
    time: t,
    importance: rec.importance_values?.[i] ?? 0,
  }));

  // Add gradcam data for comparison overlay
  const hasGradcamMatch = normGradcam && normGradcam[selectedRec];
  if (showCompare && hasGradcamMatch) {
    const gcRec = normGradcam[selectedRec];
    chartData.forEach((point, i) => {
      point.gradcam = i < (gcRec.importance_values?.length ?? 0) ? gcRec.importance_values[i] : null;
    });
  }

  return (
    <div>
      <p className="text-sm text-gray-500 mb-1 font-medium">
        Fine-tuned model attention (LoRA layers 9-11)
      </p>
      <p className="text-xs text-gray-400 mb-4">
        Shows what the fine-tuned transformer self-attention layers prioritized in the speech signal for PD classification.
      </p>

      {/* Recording Selector */}
      {normData.length > 1 && (
        <div className="flex gap-2 mb-4">
          {normData.map((r, i) => (
            <button
              key={r.recording_id || i}
              onClick={() => setSelectedRec(i)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                selectedRec === i
                  ? "bg-blue-600 text-white"
                  : "bg-gray-100 text-gray-600 hover:bg-gray-200"
              }`}
            >
              Recording {i + 1}
            </button>
          ))}
        </div>
      )}

      {/* Compare toggle */}
      {hasGradcamMatch && (
        <label className="flex items-center gap-2 mb-3 text-xs text-gray-500 cursor-pointer">
          <input
            type="checkbox"
            checked={showCompare}
            onChange={(e) => setShowCompare(e.target.checked)}
            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
          />
          Overlay Grad-CAM for comparison
        </label>
      )}

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          {showCompare && hasGradcamMatch ? (
            <LineChart data={chartData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="time" tick={{ fontSize: 11 }} label={{ value: "Time (seconds)", position: "insideBottom", offset: -2, fontSize: 11, fill: "#64748b" }} />
              <YAxis tick={{ fontSize: 11 }} label={{ value: "Importance", angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }} />
              <Tooltip contentStyle={{ borderRadius: "12px", border: "1px solid #e2e8f0" }} />
              <Legend wrapperStyle={{ fontSize: "12px" }} />
              <Line type="monotone" dataKey="importance" stroke="#14b8a6" strokeWidth={2} dot={false} name="Transformer Saliency" />
              <Line type="monotone" dataKey="gradcam" stroke="#2563eb" strokeWidth={2} dot={false} name="Grad-CAM (CNN)" strokeDasharray="5 5" />
            </LineChart>
          ) : (
            <AreaChart data={chartData} margin={{ left: 10, right: 10, top: 5, bottom: 5 }}>
              <defs>
                <linearGradient id="saliencyGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#14b8a6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#14b8a6" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="time" tick={{ fontSize: 11 }} label={{ value: "Time (seconds)", position: "insideBottom", offset: -2, fontSize: 11, fill: "#64748b" }} />
              <YAxis tick={{ fontSize: 11 }} label={{ value: "Importance", angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }} />
              <Tooltip contentStyle={{ borderRadius: "12px", border: "1px solid #e2e8f0" }} />
              <Area type="monotone" dataKey="importance" stroke="#14b8a6" fill="url(#saliencyGradient)" strokeWidth={2} />
            </AreaChart>
          )}
        </ResponsiveContainer>
      </div>
    </div>
  );
}


// ─── Helpers ─────────────────────────────────────────────────────────────────

function NotAvailable({ label }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-gray-400">
      <AlertCircle size={40} className="mb-3" />
      <p className="text-sm font-medium">{label} is not available for this result.</p>
      <p className="text-xs mt-1">This data may not have been computed yet.</p>
    </div>
  );
}

function formatFeatureName(name) {
  const MAP = {
    mfcc_1: "MFCC-1", mfcc_2: "MFCC-2", mfcc_3: "MFCC-3",
    mfcc_4: "MFCC-4", mfcc_5: "MFCC-5", mfcc_6: "MFCC-6",
    local_jitter: "Jitter (local)", ppq5_jitter: "Jitter (PPQ5)",
    rap_jitter: "Jitter (RAP)", local_shimmer: "Shimmer (local)",
    apq3_shimmer: "Shimmer (APQ3)", apq5_shimmer: "Shimmer (APQ5)",
    apq11_shimmer: "Shimmer (APQ11)", dda_shimmer: "Shimmer (DDA)",
    hnr: "HNR", nhr: "NHR",
    mean_f0: "Mean F0", std_f0: "Std F0", min_f0: "Min F0", max_f0: "Max F0",
    f1_freq: "F1 Freq", f2_freq: "F2 Freq", f3_freq: "F3 Freq",
    f1_bw: "F1 Bandwidth", f2_bw: "F2 Bandwidth", f3_bw: "F3 Bandwidth",
  };
  return MAP[name] || name;
}

export default ExplainabilitySection;
