import { useLocation, useNavigate, Navigate } from "react-router-dom";
import { Activity, ShieldCheck, ArrowLeft, PlayCircle } from "lucide-react";
import Sidebar from "../components/Sidebar";
import Card from "../components/Card";
import Button from "../components/Button";

function Result() {
  const location = useLocation();
  const navigate = useNavigate();
  const recording = location.state?.recording;

  if (!recording) {
    return <Navigate to="/dashboard" />;
  }

  const isPD = recording.prediction.toLowerCase().includes("parkinson");
  const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
  const audioUrl = `${BASE_URL}/api/recordings/${recording._id}?token=${localStorage.getItem("token")}`;

  return (
    <div className="flex min-h-screen bg-slate-50">
      <Sidebar />
      <main className="flex-1 p-6 md:p-10 flex items-center justify-center">
        <Card className="w-full max-w-2xl text-center">
          
          <h1 className="text-3xl font-bold mb-8 text-gray-800">Analysis Complete</h1>

          <div className={`p-8 rounded-2xl mb-8 flex flex-col items-center justify-center border-2 ${isPD ? 'bg-orange-50 border-orange-200' : 'bg-green-50 border-green-200'}`}>
            {isPD ? (
              <Activity size={64} className="text-orange-500 mb-4" />
            ) : (
              <ShieldCheck size={64} className="text-green-500 mb-4" />
            )}
            
            <h2 className={`text-4xl font-extrabold mb-2 ${isPD ? 'text-orange-700' : 'text-green-700'}`}>
              {recording.prediction}
            </h2>
            
            <p className={`text-lg font-medium ${isPD ? 'text-orange-600' : 'text-green-600'}`}>
              Confidence: {recording.confidence}%
            </p>
          </div>

          <div className="bg-gray-50 rounded-xl p-6 mb-8 text-left border border-gray-100 shadow-sm">
            <h3 className="font-semibold text-gray-700 mb-4 flex items-center">
              <PlayCircle className="mr-2" size={20} />
              Playback Audio
            </h3>
            <audio controls src={audioUrl} className="w-full" />
            <p className="text-sm text-gray-400 mt-3 text-center">
              Analyzed {new Date(recording.createdAt).toLocaleDateString()} at {new Date(recording.createdAt).toLocaleTimeString()}
            </p>
          </div>

          <div className="flex justify-center">
            <Button onClick={() => navigate("/dashboard")} className="px-8 py-3 flex items-center">
              <ArrowLeft className="mr-2" size={20} />
              Return to Dashboard
            </Button>
          </div>

        </Card>
      </main>
    </div>
  );
}

export default Result;
