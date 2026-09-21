const Recording = require("../models/Recording");
const fs = require("fs");
const path = require("path");
const util = require("util");
const exec = util.promisify(require("child_process").exec);



// Upload Recording
exports.uploadRecording = async (req, res) => {

    try {
        if (!req.file) {
            return res.status(400).json({
                success: false,
                message: "No audio file uploaded"
            });
        }

        let predictionText = "Pending";
        let confidenceScore = 0;
        let durationStr = req.body.duration || "00:00";

        try {
            // Path to our robust model script packaged inside the backend
            const pythonScript = path.join(__dirname, "../ml/predict.py");
            const modelDir = path.join(__dirname, "../ml/artifacts");
            const audioAbsPath = path.resolve(req.file.path);
            
            // Read file into a Blob to send via native fetch FormData
            const fileBuffer = fs.readFileSync(audioAbsPath);
            const ext = path.extname(req.file.originalname || req.file.path) || '.wav';
            const blob = new Blob([fileBuffer], { type: req.file.mimetype || 'audio/wav' });
            
            const formData = new FormData();
            formData.append('audio', blob, req.file.originalname || `audio${ext}`);
            formData.append('isNoisyMic', req.body.isNoisyMic === 'true' ? 'true' : 'false');
            
            console.log("Sending to persistent ML API...");
            
            const response = await fetch('http://127.0.0.1:5001/predict', {
                method: 'POST',
                body: formData
            });
            
            const rawText = await response.text();
            let results;
            try {
                // Sanitize any non-standard JSON tokens (e.g. NaN or Infinity)
                const sanitizedText = rawText
                    .replace(/:\s*NaN\b/g, ': null')
                    .replace(/:\s*Infinity\b/g, ': 999999')
                    .replace(/:\s*-Infinity\b/g, ': -999999');
                results = JSON.parse(sanitizedText);
            } catch (jsonErr) {
                console.error("Failed to parse ML response:", rawText);
                throw new Error("Invalid response format from ML service");
            }
            
            if (!response.ok) {
                throw new Error((results && results.error) || "ML API returned an error");
            }
            
            console.log("ML Output:", results);
            
            if (results && results.length > 0) {
                const data = results[0];
                predictionText = data.prediction === "PD" ? "Parkinson's Disease" : "Healthy Control";
                
                // Convert decimal to percentage for frontend compatibility
                const validProb = (typeof data.probability_pd === 'number' && !isNaN(data.probability_pd))
                    ? data.probability_pd
                    : 0.5;
                const rawProb = data.prediction === "PD" ? validProb : (1.0 - validProb);
                confidenceScore = Math.round(rawProb * 100);
                
                // Get accurate duration from python script
                if (data.duration) {
                    durationStr = data.duration;
                }
            }
        } catch (mlErr) {
            console.error("ML Model Execution Error:", mlErr);
            // If the model fails, the recording is still saved as 'Pending'
        }

        const recording = await Recording.create({
            user: req.user.id,
            fileName: req.file.filename,
            filePath: req.file.path,
            duration: durationStr,
            prediction: predictionText,
            confidence: confidenceScore
        });

        res.status(201).json({

            success: true,

            message: "Recording Uploaded Successfully",

            recording

        });

    }

    catch (err) {

        res.status(500).json({

            success: false,

            message: err.message

        });

    }

};



// Get Recording History

exports.getHistory = async (req, res) => {

    try {

        const recordings = await Recording.find({

            user: req.user.id

        })

        .sort({

            createdAt: -1

        });

        res.json({

            success: true,

            totalRecordings: recordings.length,

            recordings

        });

    }

    catch (err) {

        res.status(500).json({

            success: false,

            message: err.message

        });

    }

};




// Dashboard Data

exports.getDashboard = async (req, res) => {

    try {

        const totalRecordings = await Recording.countDocuments({

            user: req.user.id

        });

        const latest = await Recording.find({

            user: req.user.id

        })

        .sort({

            createdAt: -1

        })

        .limit(5);

        res.json({

            success: true,

            totalRecordings,

            latest

        });

    }

    catch (err) {

        res.status(500).json({

            success: false,

            message: err.message

        });

    }

};




// Delete Recording

exports.deleteRecording = async (req, res) => {

    try {

        const recording = await Recording.findById(req.params.id);

        if (!recording) {

            return res.status(404).json({

                success: false,

                message: "Recording Not Found"

            });

        }

        if (recording.user.toString() !== req.user.id) {

            return res.status(401).json({

                success: false,

                message: "Unauthorized"

            });

        }

        if (fs.existsSync(recording.filePath)) {

            fs.unlinkSync(recording.filePath);

        }

        await recording.deleteOne();

        res.json({

            success: true,

            message: "Recording Deleted"

        });

    }

    catch (err) {

        res.status(500).json({

            success: false,

            message: err.message

        });

    }

};




// Play / Download Recording

exports.getRecording = async (req, res) => {

    try {

        const recording = await Recording.findById(req.params.id);

        if (!recording) {

            return res.status(404).json({

                success: false,

                message: "Recording Not Found"

            });

        }

        if (recording.user.toString() !== req.user.id) {

            return res.status(401).json({

                success: false,

                message: "Unauthorized"

            });

        }

        const absolutePath = path.resolve(recording.filePath);
        if (fs.existsSync(absolutePath)) {
            res.sendFile(absolutePath);
        } else {
            return res.status(404).json({ success: false, message: "Audio file not found on disk" });
        }

    }

    catch (err) {

        res.status(500).json({

            success: false,

            message: err.message

        });

    }

};