function ConfirmDialog({

open,
title,
message,
onConfirm,
onCancel

}){


if(!open)
return null;



return(

<div
className="
fixed
inset-0
bg-black/40
flex
items-center
justify-center
z-50
"
>


<div
className="
bg-white
rounded-2xl
p-6
w-96
"
>


<h2
className="
text-xl
font-bold
"
>

{title}

</h2>


<p
className="
text-gray-500
mt-3
"
>

{message}

</p>




<div
className="
flex
justify-end
gap-3
mt-6
"
>


<button

onClick={onCancel}

className="
px-4
py-2
bg-gray-200
rounded-lg
"

>

Cancel

</button>




<button

onClick={onConfirm}

className="
px-4
py-2
bg-red-600
text-white
rounded-lg
"

>

Confirm

</button>



</div>



</div>



</div>

)

}


export default ConfirmDialog;