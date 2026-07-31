import { Link } from "react-router-dom";
import { Menu, Moon, Sun } from "lucide-react";
import { useState } from "react";


function Navbar(){

const [dark,setDark]=useState(false);
const [open,setOpen]=useState(false);


const toggleDark=()=>{

setDark(!dark);

document.documentElement.classList.toggle(
"dark"
);

};


return(

<nav className="
bg-white
dark:bg-slate-900
shadow-sm
fixed
top-0
w-full
z-50
">


<div className="
max-w-7xl
mx-auto
px-6
py-4
flex
justify-between
items-center
">


{/* Logo */}

<Link
to="/"
className="
text-2xl
font-bold
text-blue-600
"
>

VoiceCare
<span className="text-teal-500">
 PD
</span>

</Link>



{/* Desktop Menu */}

<div className="
hidden
md:flex
gap-8
items-center
">


<Link to="/" className="navlink">
Home
</Link>

<a href="#about" className="navlink">
About
</a>

<a href="#contact" className="navlink">
Contact
</a>


<Link to="/login" className="navlink">
Login
</Link>


<Link
to="/register"
className="
bg-blue-600
text-white
px-5
py-2
rounded-full
hover:bg-blue-700
transition
"
>
Register
</Link>



<button
onClick={toggleDark}
>

{
dark?
<Sun/>
:
<Moon/>
}

</button>


</div>



{/* Mobile */}

<button
className="md:hidden"
onClick={()=>setOpen(!open)}
>

<Menu/>

</button>


</div>


{
open &&

<div className="
md:hidden
bg-white
dark:bg-slate-900
p-5
space-y-4
">

<Link to="/">
Home
</Link>

<Link to="/login">
Login
</Link>

<Link to="/register">
Register
</Link>


</div>

}


</nav>

)

}


export default Navbar;