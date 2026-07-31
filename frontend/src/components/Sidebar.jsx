import {
LayoutDashboard,
Mic,
Upload,
History,
User
}
from "lucide-react";


import {Link} from "react-router-dom";


function Sidebar(){


const menu=[

{
name:"Dashboard",
icon:<LayoutDashboard/>,
path:"/dashboard"
},

{
name:"Record Voice",
icon:<Mic/>,
path:"/record"
},

{
name:"Upload Audio",
icon:<Upload/>,
path:"/upload"
},

{
name:"History",
icon:<History/>,
path:"/history"
},

{
name:"Profile",
icon:<User/>,
path:"/profile"
}

];


return(

<aside
className="
hidden
md:flex
flex-col
w-64
bg-white
shadow-lg
min-h-screen
p-5
"
>


<h2
className="
text-xl
font-bold
text-blue-600
mb-8
"
>

VoiceCare PD

</h2>



{

menu.map((item)=>(

<Link

key={item.name}

to={item.path}

className="
flex
items-center
gap-3
p-3
rounded-xl
hover:bg-blue-50
transition
"

>

{item.icon}

{item.name}


</Link>


))

}


</aside>

)

}


export default Sidebar;