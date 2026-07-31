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

    const allowed = [
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/flac",
        "audio/webm",
        "audio/ogg"
    ];

    if (allowed.includes(file.mimetype)) {

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