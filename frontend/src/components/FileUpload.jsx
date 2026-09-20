import {
UploadCloud
}
from "lucide-react";


function FileUpload({
onFileSelect
}){


return(

<div

className="
border-2
border-dashed
border-blue-300
rounded-2xl
p-10
text-center
hover:bg-blue-50
transition
"

>


<UploadCloud
size={50}
className="
mx-auto
text-blue-500
"
/>


<h3
className="
mt-4
font-semibold
"
>

Drag & Drop Audio File

</h3>


<p
className="
text-gray-500
my-3
"
>

.wav .mp3 .flac .aac .m4a supported

</p>


<input

type="file"

accept=".wav,.mp3,.flac,.ogg,.mpeg,.aac,.m4a,audio/*"

onChange={(e)=>
onFileSelect(e.target.files[0])
}

className="
hidden
"

id="upload"

/>


<label

htmlFor="upload"

className="
cursor-pointer
bg-blue-600
text-white
px-5
py-2
rounded-xl
inline-block
"

>

Browse Files

</label>


</div>

)

}


export default FileUpload;