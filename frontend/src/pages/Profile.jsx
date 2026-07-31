import {
User,
Edit,
Lock,
LogOut
}
from "lucide-react";

import Sidebar from "../components/Sidebar";
import Card from "../components/Card";



function Profile(){


const user={

name:"Sravika",
email:"sravika@example.com"

};




return(

<div
className="
flex
min-h-screen
bg-slate-50
"
>


<Sidebar/>




<main
className="
flex-1
p-6
md:p-10
"
>


<Card
className="
max-w-xl
mx-auto
text-center
"
>



<div
className="
w-32
h-32
rounded-full
bg-blue-100
mx-auto
flex
items-center
justify-center
"
>

<User
size={70}
className="
text-blue-600
"
/>


</div>





<h1
className="
text-3xl
font-bold
mt-5
"
>

{user.name}

</h1>



<p
className="
text-gray-500
"
>

{user.email}

</p>







<div
className="
space-y-4
mt-8
"
>


<button
className="
w-full
flex
items-center
justify-center
gap-3
bg-blue-600
text-white
p-3
rounded-xl
"
>

<Edit/>

Edit Profile

</button>





<button
className="
w-full
flex
items-center
justify-center
gap-3
bg-teal-600
text-white
p-3
rounded-xl
"
>


<Lock/>

Change Password

</button>





<button
className="
w-full
flex
items-center
justify-center
gap-3
bg-red-600
text-white
p-3
rounded-xl
"
>

<LogOut/>

Logout

</button>



</div>





</Card>


</main>


</div>


)

}


export default Profile;