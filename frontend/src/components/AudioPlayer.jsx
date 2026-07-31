function AudioPlayer({
src
}){


return(

<audio

controls

className="
w-full
mt-3
"

>

<source src={src}/>

Your browser does not support audio.

</audio>

)

}


export default AudioPlayer;