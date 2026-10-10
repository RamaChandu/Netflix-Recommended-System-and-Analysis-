
const API = import.meta.env.VITE_API_URL || "http://localhost:8000/api";
async function request(path:string, init?:RequestInit){const r=await fetch(API+path,init); if(!r.ok) throw new Error(await r.text()); return r.json();}
export const api={
 health:()=>request("/health"), summary:()=>request("/summary"),
 genres:()=>request("/analytics/genres?limit=15"), countries:()=>request("/analytics/countries?limit=15"),
 types:()=>request("/analytics/content-types"), ratings:()=>request("/analytics/ratings"), trends:()=>request("/analytics/trends"),
 quality:()=>request("/analytics/data-quality"),
 titles:(p:Record<string,string|number>)=>request("/titles?"+new URLSearchParams(Object.entries(p).map(([k,v])=>[k,String(v)]))),
 detail:(id:string)=>request("/titles/"+id),
 recommend:(body:any)=>request("/recommendations",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}),
 train:(body:any)=>request("/ml/train",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)}),
 comparison:(task:string)=>request("/ml/comparison?task="+task),
 predict:(body:any)=>request("/ml/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body)})
};
