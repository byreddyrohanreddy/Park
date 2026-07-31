import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Eye, EyeOff, Lock, Mail } from "lucide-react";
import { loginUser } from "../services/api";
import Button from "../components/Button";


function Login(){


const navigate = useNavigate();


const [showPassword,setShowPassword]=useState(false);

const [form,setForm]=useState({
email:"",
password:""
});


const [error,setError]=useState("");



const handleChange=(e)=>{

setForm({

...form,

[e.target.name]:e.target.value

});

};



const handleSubmit = async (e) => {

    e.preventDefault();

    setError("");

    if (!form.email || !form.password) {

        setError("Please fill all fields");

        return;

    }

    try {

        const data = await loginUser(form);

        if (data.token) {

            localStorage.setItem("token", data.token);

            localStorage.setItem(
                "user",
                JSON.stringify(data.user)
            );

            navigate("/dashboard");

        }

        else {

            setError(data.message || "Login Failed");

        }

    }

    catch (err) {

        setError("Server Error");

    }

};



return(

<div
className="
min-h-screen
bg-slate-50
flex
items-center
justify-center
px-6
"
>


<motion.div

initial={{
opacity:0,
y:40
}}

animate={{
opacity:1,
y:0
}}

transition={{
duration:0.6
}}

className="
bg-white
rounded-3xl
shadow-card
p-8
w-full
max-w-md
"

>


<h1
className="
text-3xl
font-bold
text-center
text-blue-600
"
>

Welcome Back

</h1>


<p
className="
text-center
text-gray-500
mt-2
"
>

Login to VoiceCare PD

</p>



{

error &&

<p
className="
bg-red-100
text-red-600
p-3
rounded-lg
mt-5
"
>

{error}

</p>

}




<form
onSubmit={handleSubmit}
className="
mt-6
space-y-5
"
>



<div>

<label>
Email
</label>


<div
className="
flex
items-center
border
rounded-xl
px-3
mt-2
"
>


<Mail
size={20}
className="
text-gray-400
"
/>


<input

type="email"

name="email"

value={form.email}

onChange={handleChange}

placeholder="Enter email"

className="
w-full
p-3
outline-none
"

/>


</div>


</div>





<div>

<label>
Password
</label>


<div
className="
flex
items-center
border
rounded-xl
px-3
mt-2
"
>


<Lock
size={20}
className="
text-gray-400
"
/>



<input

type={
showPassword?
"text":
"password"
}

name="password"

value={form.password}

onChange={handleChange}

placeholder="Enter password"

className="
w-full
p-3
outline-none
"

/>



<button

type="button"

onClick={()=>
setShowPassword(!showPassword)
}

>

{

showPassword?

<EyeOff size={20}/>

:

<Eye size={20}/>

}


</button>


</div>


</div>





<div
className="
flex
justify-between
items-center
text-sm
"
>


<label
className="
flex
gap-2
"
>

<input
type="checkbox"
/>

Remember Me

</label>



<a
href="#"
className="
text-blue-600
"
>

Forgot Password?

</a>


</div>





<Button
type="submit"
className="
w-full
"
>

Login

</Button>



<p
className="
text-center
mt-5
text-gray-600
"
>

Don't have an account?

<Link
to="/register"
className="
text-blue-600
ml-2
font-semibold
"
>

Register

</Link>


</p>



</form>



</motion.div>


</div>

)

}

export default Login;