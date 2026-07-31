import { motion } from "framer-motion";
import {
  Mic,
  Upload,
  ShieldCheck,
  Activity,
  Brain,
  HeartPulse
} from "lucide-react";

import { Link } from "react-router-dom";

import Navbar from "../components/Navbar";
import Footer from "../components/Footer";
import Card from "../components/Card";
import Button from "../components/Button";


function Landing(){


const features=[

{
icon:<Mic size={35}/>,
title:"Voice Recording",
desc:
"Record voice samples easily using your device microphone."
},

{
icon:<Upload size={35}/>,
title:"Audio Upload",
desc:
"Upload your existing voice recordings securely."
},

{
icon:<ShieldCheck size={35}/>,
title:"Secure Platform",
desc:
"Designed with privacy and healthcare security in mind."
},

{
icon:<Activity size={35}/>,
title:"Voice History",
desc:
"Manage and review your previous recordings."
},

{
icon:<Brain size={35}/>,
title:"Future AI Ready",
desc:
"Prepared for future integration with analysis systems."
},

{
icon:<HeartPulse size={35}/>,
title:"Healthcare Focused",
desc:
"Simple interface designed for patients and clinicians."
}

];



return(

<div className="
bg-slate-50
min-h-screen
">


<Navbar/>



{/* Hero Section */}

<section
className="
pt-32
pb-20
px-6
"
>


<div
className="
max-w-7xl
mx-auto
grid
md:grid-cols-2
gap-12
items-center
"
>


<motion.div

initial={{
opacity:0,
x:-50
}}

animate={{
opacity:1,
x:0
}}

transition={{
duration:0.8
}}

>


<h1
className="
text-4xl
md:text-6xl
font-bold
leading-tight
text-slate-800
"
>


Voice Analysis Platform
for Parkinson's Care


</h1>


<p
className="
mt-6
text-lg
text-gray-600
"
>

A modern healthcare interface that allows users
to record and upload voice samples while managing
their health data efficiently.


</p>



<div
className="
mt-8
flex
gap-5
"
>


<Link to="/register">

<Button>

Get Started

</Button>

</Link>



<Link to="/login">

<button
className="
px-6
py-3
rounded-xl
border
border-blue-600
text-blue-600
hover:bg-blue-50
transition
"
>

Login

</button>


</Link>


</div>


</motion.div>





{/* Hero Card */}

<motion.div

initial={{
opacity:0,
scale:0.8
}}

animate={{
opacity:1,
scale:1
}}

transition={{
duration:0.8
}}

>


<div
className="
bg-white
rounded-3xl
shadow-card
p-8
"
>


<div
className="
bg-blue-100
rounded-2xl
p-10
text-center
"
>


<HeartPulse
size={100}
className="
mx-auto
text-blue-600
"
/>


<h2
className="
mt-5
text-2xl
font-bold
text-blue-700
"
>

VoiceCare PD

</h2>


<p
className="
text-gray-600
mt-3
"
>

Healthcare voice management system

</p>


</div>


</div>


</motion.div>


</div>


</section>





{/* Features Section */}

<section
id="about"
className="
py-20
px-6
"
>


<div
className="
max-w-7xl
mx-auto
"
>


<h2
className="
text-3xl
font-bold
text-center
text-slate-800
"
>

Application Features

</h2>


<p
className="
text-center
text-gray-600
mt-3
"
>

Simple tools for managing voice recordings

</p>



<div
className="
grid
sm:grid-cols-2
lg:grid-cols-3
gap-8
mt-12
"
>


{

features.map((feature,index)=>(


<motion.div

key={index}

initial={{
opacity:0,
y:30
}}

whileInView={{
opacity:1,
y:0
}}

transition={{
delay:index*0.1
}}

>


<Card>


<div
className="
text-blue-600
"
>

{feature.icon}

</div>


<h3
className="
text-xl
font-semibold
mt-5
"
>

{feature.title}

</h3>


<p
className="
text-gray-600
mt-3
"
>

{feature.desc}

</p>


</Card>


</motion.div>


))

}


</div>


</div>


</section>






{/* CTA */}

<section
className="
px-6
pb-20
"
>


<div
className="
max-w-5xl
mx-auto
bg-gradient-to-r
from-blue-600
to-teal-500
rounded-3xl
p-10
text-center
text-white
"
>


<h2
className="
text-3xl
font-bold
"
>

Start Managing Your Voice Records Today

</h2>


<p
className="
mt-4
"
>

Create your account and explore the platform.

</p>


<Link
to="/register"
>


<button
className="
mt-6
bg-white
text-blue-600
px-8
py-3
rounded-xl
font-semibold
hover:scale-105
transition
"
>

Create Account

</button>


</Link>


</div>


</section>



<Footer/>


</div>

)

}


export default Landing;