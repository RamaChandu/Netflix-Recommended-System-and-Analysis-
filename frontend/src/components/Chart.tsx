
import Plot from "react-plotly.js";
export default function Chart({data,layout}:{data:any[];layout:any}){return <Plot data={data} layout={{paper_bgcolor:"transparent",plot_bgcolor:"transparent",font:{color:"#d4d4d8"},margin:{l:45,r:20,t:20,b:45},...layout}} config={{responsive:true,displayModeBar:false}} style={{width:"100%",height:"100%"}}/>}
