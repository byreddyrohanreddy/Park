const mongoose = require("mongoose");

const recordingSchema = new mongoose.Schema(

{

    user:{

        type:mongoose.Schema.Types.ObjectId,

        ref:"User",

        required:true

    },

    fileName:{

        type:String,

        required:true

    },

    filePath:{

        type:String,

        required:true

    },

    duration:{

        type:String,

        default:"00:00"

    },

    prediction:{

        type:String,

        default:"Pending"

    },

    confidence:{

        type:Number,

        default:0

    }

},

{

    timestamps:true

}

);


module.exports = mongoose.model(

"Recording",

recordingSchema

);