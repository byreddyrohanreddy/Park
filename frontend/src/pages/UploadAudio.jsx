import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  UploadCloud,
  FileAudio,
  Trash2,
  CheckCircle
} from "lucide-react";

import Sidebar from "../components/Sidebar";
import Card from "../components/Card";
import Button from "../components/Button";

import { uploadAudio } from "../services/api";

function UploadAudio() {

  const navigate = useNavigate();

  const [file, setFile] = useState(null);

  const [audioURL, setAudioURL] = useState("");

  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [successMsg, setSuccessMsg] = useState("");
  const [uploading, setUploading] = useState(false);



  const handleFile = (selectedFile) => {

    setError("");

    if (!selectedFile) return;

    const allowed = [

      ".wav",

      ".mp3",

      ".flac"

    ];

    const extension =

      selectedFile.name

      .substring(

        selectedFile.name.lastIndexOf(".")

      )

      .toLowerCase();

    if (!allowed.includes(extension)) {

      setError(

        "Only WAV, MP3 and FLAC files are allowed."

      );

      return;

    }

    setFile(selectedFile);

    setAudioURL(

      URL.createObjectURL(selectedFile)

    );

    setProgress(0);

  };



  const handleUpload = async () => {

    if (!file) {

      setError("Please choose a file.");

      return;

    }

    try {

      setUploading(true);

      setProgress(20);

      const data = await uploadAudio(

        file,

        "00:00"

      );

      setProgress(100);

      if (data.success) {
        setSuccessMsg("Audio successfully uploaded and analyzed!");
        setTimeout(() => navigate("/result", { state: { recording: data.recording } }), 1500);
      } else {

        setError(

          data.message ||

          "Upload failed."

        );

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



  const removeFile = () => {

    setFile(null);

    setAudioURL("");

    setProgress(0);

    setError("");

  };
    return (

    <div className="flex min-h-screen bg-slate-50">

      <Sidebar />

      <main className="flex-1 p-6 md:p-10">

        <Card className="max-w-3xl mx-auto">

          <h1 className="text-3xl font-bold text-blue-600 text-center">

            Upload Audio

          </h1>

          <p className="text-center text-gray-500 mt-2">

            Upload your voice recording

          </p>

          <motion.div

            whileHover={{ scale: 1.02 }}

            className="mt-8 border-2 border-dashed border-blue-300 rounded-3xl p-10 text-center"

          >

            <UploadCloud

              size={60}

              className="mx-auto text-blue-600"

            />

            <h3 className="text-xl font-semibold mt-4">

              Drag & Drop Audio File

            </h3>

            <p className="text-gray-500 mt-2">

              Supported formats: .wav, .mp3, .flac

            </p>

            <input

              type="file"

              id="audioUpload"

              accept=".wav,.mp3,.flac"

              className="hidden"

              onChange={(e) =>

                handleFile(e.target.files[0])

              }

            />

            <label

              htmlFor="audioUpload"

              className="inline-block mt-5 bg-blue-600 text-white px-6 py-3 rounded-xl cursor-pointer"

            >

              Browse Files

            </label>

          </motion.div>

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

          {

            file &&

            <Card className="mt-8">

              <div className="flex items-center gap-4">

                <div className="bg-blue-100 p-4 rounded-full">

                  <FileAudio className="text-blue-600"/>

                </div>

                <div className="flex-1">

                  <h3 className="font-semibold">

                    {file.name}

                  </h3>

                  <p className="text-sm text-gray-500">

                    {(file.size / 1024).toFixed(2)} KB

                  </p>

                </div>

                <button

                  onClick={removeFile}

                  className="text-red-500"

                >

                  <Trash2/>

                </button>

              </div>

              <div className="mt-6">

                <div className="flex justify-between">

                  <span>

                    Upload Progress

                  </span>

                  <span>

                    {progress}%

                  </span>

                </div>

                <div className="w-full bg-gray-200 rounded-full h-3 mt-2">

                  <div

                    className="bg-blue-600 h-3 rounded-full transition-all"

                    style={{

                      width: `${progress}%`

                    }}

                  />

                </div>

              </div>

              <audio

                controls

                src={audioURL}

                className="w-full mt-6"

              />

              <div className="flex justify-center gap-4 mt-6">

                <Button

                  onClick={handleUpload}

                  disabled={uploading}

                >

                  <CheckCircle className="inline mr-2"/>

                  {

                    uploading

                    ?

                    "Uploading..."

                    :

                    "Upload"

                  }

                </Button>

              </div>

            </Card>

          }

        </Card>

      </main>

    </div>

  );

}

export default UploadAudio;