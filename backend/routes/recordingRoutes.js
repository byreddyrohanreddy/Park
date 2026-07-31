const express = require("express");
const router = express.Router();

const auth = require("../middleware/authMiddleware");
const upload = require("../middleware/uploadMiddleware");

const {
  uploadRecording,
  getHistory,
  getDashboard,
  deleteRecording,
  getRecording,
} = require("../controllers/recordingController");


// =========================
// Upload Recording
// =========================
router.post(
  "/upload",
  auth,
  upload.single("audio"),
  uploadRecording
);


// =========================
// Recording History
// =========================
router.get(
  "/",
  auth,
  getHistory
);


// =========================
// Dashboard
// =========================
router.get(
  "/dashboard",
  auth,
  getDashboard
);


// =========================
// Get Single Recording
// =========================
router.get(
  "/:id",
  auth,
  getRecording
);


// =========================
// Delete Recording
// =========================
router.delete(
  "/:id",
  auth,
  deleteRecording
);

module.exports = router;