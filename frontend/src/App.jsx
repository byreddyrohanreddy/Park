import { BrowserRouter, Routes, Route } from "react-router-dom";

import Landing from "./pages/Landing";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Dashboard from "./pages/Dashboard";
import RecordVoice from "./pages/RecordVoice";
import UploadAudio from "./pages/UploadAudio";
import History from "./pages/History";
import Profile from "./pages/Profile";
import Result from "./pages/Result";


function App() {

  return (
    <BrowserRouter>

      <Routes>

        <Route path="/" element={<Landing />} />

        <Route 
          path="/login" 
          element={<Login />} 
        />

        <Route 
          path="/register" 
          element={<Register />} 
        />

        <Route 
          path="/dashboard" 
          element={<Dashboard />} 
        />

        <Route 
          path="/record" 
          element={<RecordVoice />} 
        />

        <Route 
          path="/upload" 
          element={<UploadAudio />} 
        />

        <Route 
          path="/history" 
          element={<History />} 
        />

        <Route 
          path="/profile" 
          element={<Profile />} 
        />

        <Route 
          path="/result" 
          element={<Result />} 
        />

      </Routes>

    </BrowserRouter>
  );
}

export default App;