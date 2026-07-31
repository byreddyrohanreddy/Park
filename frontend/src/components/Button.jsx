function Button({
children,
onClick,
type="button",
className=""
}){


return(

<button

type={type}

onClick={onClick}

className={`
bg-blue-600
text-white
px-6
py-3
rounded-xl
font-semibold
hover:bg-blue-700
transition
shadow-md

${className}

`}

>

{children}

</button>

)

}


export default Button;