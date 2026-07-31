import {CheckCircle, XCircle} from "lucide-react";


function Toast({
message,
type="success"
}){


return(

<div
className={`
fixed
right-5
top-20
px-5
py-3
rounded-xl
shadow-lg
flex
gap-3
items-center

${
type==="success"
?
"bg-green-500"
:
"bg-red-500"
}

text-white

`}
>


{

type==="success"

?

<CheckCircle/>

:

<XCircle/>

}


{message}


</div>

)

}


export default Toast;