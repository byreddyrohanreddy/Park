const Recording = require("../models/Recording");

exports.getDashboard = async (req, res) => {

    try {

        const recordings = await Recording.find({

            user: req.user.id

        }).sort({

            createdAt: -1

        });

        res.json({

            success: true,

            totalRecordings: recordings.length,

            latest: recordings.slice(0, 5),

            recentActivities: recordings.slice(0, 5)

        });

    }

    catch (err) {

        res.status(500).json({

            message: err.message

        });

    }

};