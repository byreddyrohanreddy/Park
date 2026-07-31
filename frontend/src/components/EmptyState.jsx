import {
FileAudio
}
from "lucide-react";


function EmptyState({message}){


return(

<div
className="
text-center
py-10
"
>


<FileAudio

size={60}

className="
mx-auto
text-gray-400
"

/>


<h3
className="
text-xl
font-semibold
mt-4
"
>

No Data

</h3>


<p
className="
text-gray-500
mt-2
"
>

{message}

</p>



</div>

)

}


export default EmptyState;