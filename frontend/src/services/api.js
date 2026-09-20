const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:5000";
const API_URL = `${BASE_URL}/api`;



// -------------------------
// Get JWT Token
// -------------------------

const getToken = () => {

    return localStorage.getItem("token");

};



// -------------------------
// Register
// -------------------------

export const registerUser = async (userData) => {

    const response = await fetch(

        `${API_URL}/auth/register`,

        {

            method: "POST",

            headers: {

                "Content-Type": "application/json"

            },

            body: JSON.stringify(userData)

        }

    );

    return response.json();

};



// -------------------------
// Login
// -------------------------

export const loginUser = async (userData) => {

    const response = await fetch(

        `${API_URL}/auth/login`,

        {

            method: "POST",

            headers: {

                "Content-Type": "application/json"

            },

            body: JSON.stringify(userData)

        }

    );

    const data = await response.json();

    if (data.token) {

        localStorage.setItem(

            "token",

            data.token

        );

    }

    return data;

};



// -------------------------
// Dashboard
// -------------------------

export const getDashboard = async () => {

    const response = await fetch(

        `${API_URL}/dashboard`,

        {

            headers: {

                Authorization: `Bearer ${getToken()}`

            }

        }

    );

    return response.json();

};



// -------------------------
// Recording History
// -------------------------

export const getHistory = async () => {

    const response = await fetch(

        `${API_URL}/recordings/history`,

        {

            headers: {

                Authorization: `Bearer ${getToken()}`

            }

        }

    );

    return response.json();

};

export const getRecordings = getHistory;



// -------------------------
// Upload Audio
// -------------------------

export const uploadAudio = async (
    file,
    duration = "00:00",
    isNoisyMic = false
) => {
    const formData = new FormData();
    formData.append(
        "audio",
        file
    );
    formData.append(
        "duration",
        duration
    );
    formData.append(
        "isNoisyMic",
        isNoisyMic ? "true" : "false"
    );

    const response = await fetch(

        `${API_URL}/recordings/upload`,

        {

            method: "POST",

            headers: {

                Authorization: `Bearer ${getToken()}`

            },

            body: formData

        }

    );

    return response.json();

};



// -------------------------
// Delete Recording
// -------------------------

export const deleteRecording = async (

    id

) => {

    const response = await fetch(

        `${API_URL}/recordings/${id}`,

        {

            method: "DELETE",

            headers: {

                Authorization: `Bearer ${getToken()}`

            }

        }

    );

    return response.json();

};


// -------------------------
// Get Recording URL
// -------------------------

export const getRecordingURL = (id) => {

    return `${API_URL}/recordings/${id}`;

};



// -------------------------
// Generate Clinical Report
// -------------------------

export const generateReport = async (payload, audience = "clinician") => {

    const response = await fetch(

        `${API_URL}/report/generate`,

        {

            method: "POST",

            headers: {

                "Content-Type": "application/json",

                Authorization: `Bearer ${getToken()}`

            },

            body: JSON.stringify({ payload, audience })

        }

    );

    return response.json();

};



// -------------------------
// Get Explainability Data
// POST /api/report/explainability — sends audio file, gets back SHAP + Grad-CAM + Saliency
// -------------------------

export const getExplainability = async (audioFile) => {

    const formData = new FormData();
    formData.append("audio", audioFile);

    const response = await fetch(

        `${API_URL}/report/explainability`,

        {

            method: "POST",

            headers: {

                Authorization: `Bearer ${getToken()}`

            },

            body: formData

        }

    );

    return response.json();

};



// -------------------------
// Logout
// -------------------------

export const logout = () => {

    localStorage.removeItem(

        "token"

    );

};