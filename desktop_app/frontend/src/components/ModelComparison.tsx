import {useEffect,useState} from 'react';
import {useAnalysis} from '../stores/analysis';
import {api} from '../services/api';
import {Badge,ErrorBox,Loading,Section,percent} from './ui';
import {Bar,BarChart,CartesianGrid,Cell,ResponsiveContainer,Tooltip,XAxis,YAxis} from 'recharts';

interface ModelResult {model:string;name:string;retained:boolean;available?:boolean;status:string;predicted_class?:string|null;confidence?:number|null;probabilities?:Record<string,number>|null;reason:string;flows_evaluated?:number;mean_confidence?:number|null;agreement?:number|null}
interface Comparison {flow_id?:string;model_run?:string;status:string;models:ModelResult[];total_flows?:number;single_flow?:ModelResult[]|null;reason?:string}
export function ModelComparison({perFlow=false}:{perFlow?:boolean}){
 const {analysisId,selectedFlowId}=useAnalysis();
 const [data,setData]=useState<Comparison|null>(null);const [error,setError]=useState('');
 useEffect(()=>{
  let cancelled=false;let timer:ReturnType<typeof setTimeout>;setData(null);setError('');
  if(!analysisId||(perFlow&&!selectedFlowId))return;
  async function poll(){try{
   const route=perFlow?`flows/${selectedFlowId}/comparison`:'comparison';
   const result=await api<Comparison>(`/api/analysis/${analysisId}/${route}`);
   if(cancelled)return;setData(result);if(result.status==='Running')timer=setTimeout(poll,800);
  }catch(error){if(!cancelled)setError(String(error))}}
  void poll();return()=>{cancelled=true;clearTimeout(timer)};
 },[analysisId,perFlow,perFlow?selectedFlowId:null]);
 const exact=perFlow||data?.total_flows===1;
 const rows=(!perFlow&&data?.single_flow)||data?.models||[];
 const metric=(row:ModelResult)=>exact?row.confidence:row.mean_confidence;
 return <Section title="Five-model comparison" note={exact?'Exact flow predictions · LightGBM remains retained':'Analysis-wide results · LightGBM remains retained'}>
 <p className="help-text">{perFlow?selectedFlowId:`${data?.total_flows??'—'} flows in this analysis`} · Comparison predictions never replace the operational result or Phase-II query.</p>
 <ErrorBox message={error}/>{!data&&!error?<Loading label="Reading Phase-I model results…"/>:data&&<>
 {!rows.length?<p className="muted">{data.reason||'Comparison results are being prepared during Phase I.'}</p>:<>
 <div className="table-container"><table><thead><tr><th>Model</th>{exact?<><th>Predicted class</th><th>Confidence</th></>:<><th>Flows evaluated</th><th>Mean confidence</th><th>Agreement with LightGBM</th></>}<th>Role / status</th></tr></thead>
 <tbody>{rows.map(row=><tr key={row.model} className={row.retained?'retained-model':''}>
 <td><strong>{row.name}</strong></td>{exact?<><td>{row.predicted_class||'—'}</td><td>{row.confidence==null?'—':percent(row.confidence)}</td></>:<><td>{row.flows_evaluated??0} / {data.total_flows}</td><td>{row.mean_confidence==null?'—':percent(row.mean_confidence)}</td><td>{row.agreement==null?'—':percent(row.agreement)}</td></>}
 <td><Badge tone={row.status==='Completed'?'success':row.status==='Failed'?'danger':'warning'}>{row.status}</Badge><small>{row.retained?'Retained classifier':'Comparison model'}</small>{row.reason&&<small>{row.reason}</small>}</td>
 </tr>)}</tbody></table></div>
 <h3 className="comparison-chart-title">{exact?'Current-flow prediction confidence':'Mean prediction confidence by model'}</h3>
 <p className="help-text">{exact?'Maximum predicted probability (%)':'Mean maximum predicted probability across all evaluated flows (%)'}</p>
 {!exact?<div className="comparison-chart aggregate-chart" role="img" aria-label="Mean prediction confidence by model"><ResponsiveContainer width="100%" height={300}><BarChart data={rows.map(row=>({...row,meanPercent:row.mean_confidence==null?null:row.mean_confidence*100}))} margin={{top:20,right:15,left:20,bottom:10}}><CartesianGrid stroke="var(--border-color)" strokeDasharray="3 5" vertical={false}/><XAxis dataKey="name" interval={0} tick={{fill:'var(--text-body)',fontSize:13}}/><YAxis domain={[0,100]} tick={{fill:'var(--text-body)',fontSize:13}} label={{value:'Mean confidence (%)',angle:-90,position:'insideLeft',offset:-10,fill:'var(--text-body)',fontSize:13}}/><Tooltip contentStyle={{background:'var(--bg-card)',border:'1px solid var(--border-color)',fontSize:13}}/><Bar isAnimationActive={false} dataKey="meanPercent" name="Mean confidence (%)" maxBarSize={70} label={{position:'top',fill:'var(--text-body)',fontSize:13,formatter:(value:any)=>typeof value==='number'?`${value.toFixed(1)}%`:''}}>{rows.map(row=><Cell key={row.model} fill={row.retained?'var(--btn-primary)':'var(--btn-secondary)'}/>)}</Bar></BarChart></ResponsiveContainer><p className="help-text">LightGBM is highlighted as the retained classifier. Failed models have no bar.</p></div>:<div className="comparison-chart" aria-label="Current-flow model confidence">{rows.map(row=><div className={`probability ${row.retained?'retained-model':''}`} key={row.model}>
 <span>{row.name}{row.retained&&<small>Retained</small>}</span><div className="bar-track">{metric(row)!=null&&<div style={{width:`${metric(row)!*100}%`}}/>}</div>
 <strong>{metric(row)==null?row.status:`${exact?row.predicted_class+' · ':''}${percent(metric(row)!)}`}</strong>
 </div>)}</div>}
 {perFlow&&<details><summary>Class probabilities by model</summary>{rows.filter(r=>r.probabilities).map(r=><p key={r.model}>{r.name}: {Object.entries(r.probabilities!).map(([label,p])=>`${label} ${percent(p)}`).join(' · ')}</p>)}</details>}
 </>}
 </>}
 </Section>
}
