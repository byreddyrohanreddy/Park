import { useEffect, useState } from "react";
import axios from "axios";
import { Trash2 } from "lucide-react";

export default function History() {
  const [recordings, setRecordings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [successMsg, setSuccessMsg] = useState("");

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const token = localStorage.getItem("token");

      const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
      const res = await axios.get(
        `${BASE_URL}/api/recordings`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      setRecordings(res.data.recordings);
    } catch (err) {
      console.error(err.response?.data || err.message);
    } finally {
      setLoading(false);
    }
  };

  const deleteRecording = async (id) => {
    try {
      const token = localStorage.getItem("token");

      const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
      await axios.delete(
        `${BASE_URL}/api/recordings/${id}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      // Instantly remove from UI and show themed message
      setRecordings((prev) => prev.filter((r) => r._id !== id));
      setSuccessMsg("Recording successfully deleted.");
      setTimeout(() => setSuccessMsg(""), 3000);
    } catch (err) {
      console.error(err.response?.data || err.message);
    }
  };

  if (loading) {
    return (
      <div className="text-center mt-20 text-xl font-semibold">
        Loading...
      </div>
    );
  }

  if (recordings.length === 0) {
    return (
      <div className="text-center mt-20">
        <h2 className="text-3xl font-bold">No recordings yet</h2>
        <p className="text-gray-600 mt-2">
          Your uploaded and recorded audio will appear here.
        </p>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">
        Recording History
      </h1>

      {successMsg && (
        <div className="mb-6 bg-green-100 border border-green-200 text-green-700 p-4 rounded-xl flex items-center shadow-sm">
          <span className="font-semibold">{successMsg}</span>
        </div>
      )}

      {recordings.map((record) => (
        <div
          key={record._id}
          className="bg-white rounded-xl shadow-md border p-5 mb-5"
        >
          <h2 className="text-lg font-semibold">
            {record.fileName}
          </h2>

          <p className="mt-2">
            <strong>Duration:</strong> {record.duration}
          </p>

          <p>
            <strong>Prediction:</strong>{" "}
            {record.prediction || "Pending"}
          </p>

          <p>
            <strong>Confidence:</strong>{" "}
            {record.confidence ?? 0}%
          </p>

          <p>
            <strong>Uploaded:</strong>{" "}
            {new Date(record.createdAt).toLocaleString()}
          </p>

          <audio controls className="w-full mt-4">
            <source
              src={`${import.meta.env.VITE_API_URL || "http://localhost:5000"}/api/recordings/${record._id}?token=${localStorage.getItem("token")}`}
              type="audio/wav"
            />
            Your browser does not support audio.
          </audio>

          <button
            onClick={() => deleteRecording(record._id)}
            className="flex items-center gap-2 mt-4 bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
          >
            <Trash2 size={18} />
            Delete Recording
          </button>
        </div>
      ))}
    </div>
  );
}