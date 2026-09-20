const express=require("express");
const cors=require("cors");
const dotenv=require("dotenv");
const path=require("path");
const fs=require("fs");

dotenv.config({ path: path.join(__dirname, ".env") });

const recordingRoutes=require("./routes/recordingRoutes");
const dashboardRoutes =
require("./routes/dashboardRoutes");

if (!fs.existsSync(path.join(__dirname, "uploads"))) {
    fs.mkdirSync(path.join(__dirname, "uploads"), { recursive: true });
}

const connectDB=require("./config/db");

const authRoutes=require("./routes/authRoutes");

const app=express();

connectDB();

app.use(cors());

app.use(express.json());

app.get("/",(req,res)=>{

res.send("VoiceCare PD Backend Running");

});

app.use("/api/auth",authRoutes);

const PORT=process.env.PORT||5000;

app.listen(PORT,()=>{

console.log(`Server Running on Port ${PORT}`);

});
app.use("/api/recordings",recordingRoutes);
app.use(
    "/api/dashboard",
    dashboardRoutes
);

const reportRoutes = require("./routes/reportRoutes");
app.use("/api/report", reportRoutes);