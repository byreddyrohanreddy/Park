import { useState } from "react";
import { FileText, AlertTriangle, Info } from "lucide-react";
import Card from "../Card";
import Button from "../Button";
import Loader from "../Loader";

/**
 * ClinicalReportSection — Report generation with audience toggle.
 *
 * Props:
 *   payload        — extended dashboard payload
 *   onGenerateReport — async (audience) => reportResponse
 */
function ClinicalReportSection({ payload, onGenerateReport }) {
  const [audience, setAudience] = useState("clinician");
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const isUncertain = payload?.refer_to_clinician;

  const handleGenerate = async () => {
    setLoading(true);
    setError("");
    setReport(null);

    try {
      const result = await onGenerateReport(audience);
      setReport(result);
    } catch (err) {
      setError(err.message || "Failed to generate report. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="mb-6">
      <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
        <FileText size={22} className="text-blue-600" />
        Clinical Report
      </h2>

      {/* Audience Toggle */}
      <div className="flex items-center gap-2 mb-5">
        <span className="text-sm text-gray-500 font-medium">Audience:</span>
        <div className="flex bg-gray-100 rounded-xl p-1">
          <button
            onClick={() => setAudience("clinician")}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition ${
              audience === "clinician"
                ? "bg-blue-600 text-white shadow-md"
                : "text-gray-600 hover:bg-gray-200"
            }`}
          >
            Clinician
          </button>
          <button
            onClick={() => setAudience("patient")}
            className={`px-4 py-2 rounded-lg text-sm font-semibold transition ${
              audience === "patient"
                ? "bg-blue-600 text-white shadow-md"
                : "text-gray-600 hover:bg-gray-200"
            }`}
          >
            Patient
          </button>
        </div>
      </div>

      {/* Generate Button */}
      {!report && !loading && (
        <Button onClick={handleGenerate} className="w-full sm:w-auto">
          <FileText className="inline mr-2" size={18} />
          Generate Clinical Report
        </Button>
      )}

      {/* Loading State */}
      {loading && (
        <div className="flex flex-col items-center py-6">
          <Loader />
          <p className="text-gray-500 mt-3 text-sm animate-pulse">
            Generating {audience === "clinician" ? "clinician" : "patient"} report...
            This may take a few seconds.
          </p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="mt-4 bg-red-100 border border-red-200 text-red-600 p-3 rounded-xl">
          <span className="font-semibold">{error}</span>
        </div>
      )}

      {/* Report Display */}
      {report && (
        <div className="mt-4 space-y-4">
          {/* Uncertainty warning ABOVE report text */}
          {isUncertain && (
            <div className="bg-amber-50 border-2 border-amber-300 rounded-xl p-4 flex items-start gap-3">
              <AlertTriangle className="text-amber-500 flex-shrink-0 mt-0.5" size={22} />
              <p className="text-amber-700 font-medium">
                ⚠ This result was inconclusive — clinical follow-up is recommended.
              </p>
            </div>
          )}

          {/* Generation method badge */}
          {report.generation_method === "template_fallback" && (
            <div className="flex items-center gap-2 text-xs text-gray-500 bg-gray-50 px-3 py-2 rounded-lg border border-gray-200">
              <Info size={14} />
              <span>Generated via standard template</span>
            </div>
          )}

          {/* Report text */}
          <div className="bg-gray-50 border border-gray-200 rounded-xl p-5 prose prose-sm max-w-none">
            {report.report.split("\n").map((line, idx) => {
              if (line.startsWith("## ")) {
                return (
                  <h2 key={idx} className="text-lg font-bold text-gray-800 mt-4 mb-2">
                    {line.replace("## ", "")}
                  </h2>
                );
              }
              if (line.startsWith("### ")) {
                return (
                  <h3 key={idx} className="text-base font-semibold text-gray-700 mt-3 mb-1">
                    {line.replace("### ", "")}
                  </h3>
                );
              }
              if (line.startsWith("- ")) {
                return (
                  <li key={idx} className="text-gray-600 ml-4 list-disc">
                    {renderBold(line.replace("- ", ""))}
                  </li>
                );
              }
              if (line.trim() === "") {
                return <br key={idx} />;
              }
              return (
                <p key={idx} className="text-gray-600 leading-relaxed">
                  {renderBold(line)}
                </p>
              );
            })}
          </div>

          {/* Regenerate button */}
          <div className="flex justify-end">
            <button
              onClick={handleGenerate}
              className="text-sm text-blue-600 hover:text-blue-800 font-medium transition"
            >
              ↻ Regenerate for {audience === "clinician" ? "clinician" : "patient"}
            </button>
          </div>

          {/* Persistent disclaimer */}
          <div className="bg-blue-50 border border-blue-200 rounded-xl p-3 text-center">
            <p className="text-sm text-blue-700">
              <span className="font-semibold">ℹ️ Disclaimer:</span> This is a
              decision-support tool, not a medical diagnosis. Please discuss these
              results with a qualified clinician.
            </p>
          </div>
        </div>
      )}
    </Card>
  );
}

/** Simple inline bold renderer for **text** patterns */
function renderBold(text) {
  const parts = text.split(/\*\*(.*?)\*\*/g);
  return parts.map((part, i) =>
    i % 2 === 1 ? (
      <span key={i} className="font-semibold text-gray-800">
        {part}
      </span>
    ) : (
      part
    )
  );
}

export default ClinicalReportSection;
