import { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Mic,
  Pause,
  Play,
  Square,
  Trash2,
  CheckCircle
} from "lucide-react";

import Sidebar from "../components/Sidebar";
import Card from "../components/Card";
import Button from "../components/Button";

import { uploadAudio } from "../services/api";

const convertBlobToWav = async (blob) => {
  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
  
  const numOfChan = audioBuffer.numberOfChannels;
  const length = audioBuffer.length * numOfChan * 2 + 44;
  const buffer = new ArrayBuffer(length);
  const view = new DataView(buffer);
  
  const writeString = (view, offset, string) => {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  };
  
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + audioBuffer.length * 2, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, numOfChan, true);
  view.setUint32(24, audioBuffer.sampleRate, true);
  view.setUint32(28, audioBuffer.sampleRate * 2 * numOfChan, true);
  view.setUint16(32, numOfChan * 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, audioBuffer.length * 2 * numOfChan, true);
  
  let offset = 44;
  for (let i = 0; i < audioBuffer.length; i++) {
    for (let channel = 0; channel < numOfChan; channel++) {
      let sample = audioBuffer.getChannelData(channel)[i];
      sample = Math.max(-1, Math.min(1, sample));
      sample = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
      view.setInt16(offset, sample, true);
      offset += 2;
    }
  }
  
  return new Blob([buffer], { type: "audio/wav" });
};

function RecordVoice() {

  const navigate = useNavigate();

  const [recording, setRecording] = useState(false);

  const [paused, setPaused] = useState(false);

  const [audioURL, setAudioURL] = useState("");

  const [audioBlob, setAudioBlob] = useState(null);

  const [time, setTime] = useState(0);

  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [successMsg, setSuccessMsg] = useState("");

  const [isNoisyMic, setIsNoisyMic] = useState(true);

  const mediaRecorder = useRef(null);

  const chunks = useRef([]);

  const timerRef = useRef(null);

  useEffect(() => {

    if (recording && !paused) {

      timerRef.current = setInterval(() => {

        setTime((prev) => prev + 1);

      }, 1000);

    }

    else {

      clearInterval(timerRef.current);

    }

    return () => {

      clearInterval(timerRef.current);

    };

  }, [recording, paused]);



  const formatTime = () => {

    const minutes = Math.floor(time / 60);

    const seconds = time % 60;

    return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;

  };
    const startRecording = async () => {

    try {

      setError("");

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          noiseSuppression: false,
          echoCancellation: false,
          autoGainControl: false,
        }
      });

      const options = {};
      if (MediaRecorder.isTypeSupported('audio/webm;codecs=pcm')) {
        options.mimeType = 'audio/webm;codecs=pcm';
      }
      
      mediaRecorder.current = new MediaRecorder(stream, options);

      chunks.current = [];

      mediaRecorder.current.ondataavailable = (event) => {

        if (event.data.size > 0) {

          chunks.current.push(event.data);

        }

      };

      mediaRecorder.current.onstop = () => {

        const blob = new Blob(

          chunks.current,

          {

            type: "audio/webm"

          }

        );

        setAudioBlob(blob);

        const url = URL.createObjectURL(blob);

        setAudioURL(url);

        stream.getTracks().forEach(track => track.stop());

      };

      mediaRecorder.current.start();

      setRecording(true);

      setPaused(false);

      setTime(0);

    }

    catch (err) {

      console.log(err);

      setError(

        "Unable to access microphone. Please allow microphone permission."

      );

    }

  };
    const pauseRecording = () => {

    if (mediaRecorder.current) {

      mediaRecorder.current.pause();

      setPaused(true);

    }

  };



  const resumeRecording = () => {

    if (mediaRecorder.current) {

      mediaRecorder.current.resume();

      setPaused(false);

    }

  };



  const stopRecording = () => {

    if (mediaRecorder.current) {

      mediaRecorder.current.stop();

    }

    setRecording(false);

    setPaused(false);

  };



  const uploadRecording = async () => {

    if (!audioBlob) {

      setError("Please record audio first.");

      return;

    }

    try {

      setUploading(true);
      setError("");

      // Convert the WebM blob to a standard WAV blob for Python compatibility
      const wavBlob = await convertBlobToWav(audioBlob);

      const file = new File(
        [wavBlob],
        `recording_${Date.now()}.wav`,
        { type: "audio/wav" }
      );

      const data = await uploadAudio(

        file,

        formatTime(),

        true

      );

      if (data.success) {
        setSuccessMsg("Recording successfully uploaded and analyzed!");
        setTimeout(() => navigate("/result", { state: { recording: data.recording } }), 1500);
      } else {
        setError(data.message || "Upload failed.");
      }

    }

    catch (err) {

      console.log(err);

      setError("Server Error");

    }

    finally {

      setUploading(false);

    }

  };



  const deleteRecording = () => {

    if (audioURL) {

      URL.revokeObjectURL(audioURL);

    }

    setAudioURL("");

    setAudioBlob(null);

    setTime(0);

    setError("");

  };
    return (

    <div className="flex min-h-screen bg-slate-50">

      <Sidebar />

      <main className="flex-1 p-6 md:p-10">

        <Card className="max-w-3xl mx-auto text-center">

          <h1 className="text-3xl font-bold text-blue-600">

            Voice Recording

          </h1>

          <p className="text-gray-500 mt-2">

            Record your voice sample

          </p>

          {
            error &&
            <div className="mt-5 bg-red-100 border border-red-200 text-red-600 p-3 rounded-xl flex items-center">
              <span className="font-semibold">{error}</span>
            </div>
          }
          {
            successMsg &&
            <div className="mt-5 bg-green-100 border border-green-200 text-green-700 p-3 rounded-xl flex items-center">
              <CheckCircle className="mr-2" size={20} />
              <span className="font-semibold">{successMsg}</span>
            </div>
          }

          <motion.div

            animate={

              recording

                ?

                {

                  scale: [1, 1.15, 1]

                }

                :

                {}

            }

            transition={{

              repeat: Infinity,

              duration: 1

            }}

            className="mt-10 mx-auto bg-blue-100 rounded-full w-40 h-40 flex items-center justify-center"

          >

            <Mic

              size={80}

              className="text-blue-600"

            />

          </motion.div>

          <h2 className="text-4xl font-bold mt-8">

            {formatTime()}

          </h2>

          <div className="flex justify-center gap-2 mt-8 h-20 items-center">

            {

              [1,2,3,4,5,6,7].map((item)=>(

                <motion.div

                  key={item}

                  animate={

                    recording

                    ?

                    {

                      height:[20,60,30,70,20]

                    }

                    :

                    {

                      height:20

                    }

                  }

                  transition={{

                    repeat:Infinity,

                    duration:0.8,

                    delay:item*0.1

                  }}

                  className="w-2 bg-blue-500 rounded-full"

                />

              ))

            }

          </div>

          <div className="flex flex-wrap justify-center gap-4 mt-8">

            {

              !recording &&

              <Button onClick={startRecording}>

                <Mic className="inline mr-2"/>

                Start Recording

              </Button>

            }

            {

              recording && !paused &&

              <button

                onClick={pauseRecording}

                className="bg-yellow-500 text-white px-6 py-3 rounded-xl"

              >

                <Pause className="inline mr-2"/>

                Pause

              </button>

            }

            {

              paused &&

              <button

                onClick={resumeRecording}

                className="bg-green-600 text-white px-6 py-3 rounded-xl"

              >

                <Play className="inline mr-2"/>

                Resume

              </button>

            }

            {

              recording &&

              <button

                onClick={stopRecording}

                className="bg-red-600 text-white px-6 py-3 rounded-xl"

              >

                <Square className="inline mr-2"/>

                Stop

              </button>

            }

          </div>

          {

            audioURL &&

            <div className="mt-10">

              <h3 className="font-semibold">

                Recorded Audio

              </h3>

              <audio

                controls

                src={audioURL}

                className="w-full mt-3"

              />

              <div className="flex justify-center gap-4 mt-6">

                <button

                  onClick={deleteRecording}

                  className="bg-red-100 text-red-600 px-5 py-2 rounded-xl"

                >

                  <Trash2 className="inline"/>

                  Delete

                </button>

                <Button

                  onClick={uploadRecording}

                  disabled={uploading}

                >

                  <CheckCircle className="inline mr-2"/>

                  {

                    uploading

                    ?

                    "Analyzing with AI... (Wait ~10s)"

                    :

                    "Upload Recording"

                  }

                </Button>

              </div>

            </div>

          }

        </Card>

      </main>

    </div>

  );

}

export default RecordVoice;