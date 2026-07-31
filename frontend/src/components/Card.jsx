function Card({
children,
className=""
}){


return(

<div

className={`
bg-white
rounded-2xl
shadow-card
p-6

${className}

`}

>

{children}

</div>

)

}


export default Card;