const multer = require("multer");
const path = require("path");

const uploadDir = path.join(__dirname, "../uploads");

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, uploadDir);
    },

    filename: function (req, file, cb) {

        const uniqueName =
            Date.now() +
            "-" +
            Math.round(Math.random() * 1E9) +
            path.extname(file.originalname);

        cb(null, uniqueName);

    }

});

const fileFilter = (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const allowedExts = [".wav", ".mp3", ".flac", ".ogg", ".mpeg", ".webm", ".aac", ".m4a"];
    
    if (allowedExts.includes(ext) || file.mimetype.startsWith("audio/") || file.mimetype === "video/mp4" && (ext === ".m4a" || ext === ".aac")) {
        cb(null, true);
    }
    else {
        cb(new Error("Only audio files are allowed"));
    }
};

module.exports = multer({

    storage,

    fileFilter

});