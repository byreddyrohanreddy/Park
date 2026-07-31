import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { User, Mail, Lock, Eye, EyeOff } from "lucide-react";
import { registerUser } from "../services/api";
import Button from "../components/Button";

function Register() {
  const navigate = useNavigate();

  const [showPassword, setShowPassword] = useState(false);
  const [form, setForm] = useState({
    name: "",
    email: "",
    password: ""
  });
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    if (!form.name || !form.email || !form.password) {
      setError("Please fill all fields");
      return;
    }

    try {
      const data = await registerUser(form);
      if (data.message === "Registration Successful" || data.success || !data.error) {
        setSuccess("Registration Successful! Redirecting to login...");
        setTimeout(() => {
          navigate("/login");
        }, 2000);
      } else {
        setError(data.message || "Registration Failed");
      }
    } catch (err) {
      setError("Server Error");
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center px-6">
      <motion.div
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="bg-white rounded-3xl shadow-card p-8 w-full max-w-md"
      >
        <h1 className="text-3xl font-bold text-center text-blue-600">
          Create Account
        </h1>

        <p className="text-center text-gray-500 mt-2">
          Join VoiceCare PD
        </p>

        {error && (
          <p className="bg-red-100 text-red-600 p-3 rounded-lg mt-5">
            {error}
          </p>
        )}

        {success && (
          <p className="bg-green-100 text-green-600 p-3 rounded-lg mt-5">
            {success}
          </p>
        )}

        <form onSubmit={handleSubmit} className="mt-6 space-y-5">
          <div>
            <label className="text-sm font-medium text-gray-700">Full Name</label>
            <div className="flex items-center border rounded-xl px-3 mt-2">
              <User size={20} className="text-gray-400" />
              <input
                type="text"
                name="name"
                value={form.name}
                onChange={handleChange}
                placeholder="Enter your name"
                className="w-full p-3 outline-none"
              />
            </div>
          </div>

          <div>
            <label className="text-sm font-medium text-gray-700">Email</label>
            <div className="flex items-center border rounded-xl px-3 mt-2">
              <Mail size={20} className="text-gray-400" />
              <input
                type="email"
                name="email"
                value={form.email}
                onChange={handleChange}
                placeholder="Enter email"
                className="w-full p-3 outline-none"
              />
            </div>
          </div>

          <div>
            <label className="text-sm font-medium text-gray-700">Password</label>
            <div className="flex items-center border rounded-xl px-3 mt-2">
              <Lock size={20} className="text-gray-400" />
              <input
                type={showPassword ? "text" : "password"}
                name="password"
                value={form.password}
                onChange={handleChange}
                placeholder="Create password"
                className="w-full p-3 outline-none"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <EyeOff size={20} className="text-gray-400" /> : <Eye size={20} className="text-gray-400" />}
              </button>
            </div>
          </div>

          <Button type="submit" className="w-full">
            Register
          </Button>

          <p className="text-center mt-5 text-gray-600">
            Already have an account?
            <Link to="/login" className="text-blue-600 ml-2 font-semibold">
              Login
            </Link>
          </p>
        </form>
      </motion.div>
    </div>
  );
}

export default Register;