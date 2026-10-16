import React from 'react';
import {act,cleanup,fireEvent,render,screen} from '@testing-library/react';
import {afterEach,describe,expect,it,vi} from 'vitest';
import App from '../frontend/src/App';
import {useAnalysis} from '../frontend/src/stores/analysis';
import type {RecordData,Status,Summary} from '../frontend/src/types';

vi.mock('motion/react',()=>({useReducedMotion:()=>true,motion:{div:({children}:any)=><div>{children}</div>}}));

const summary={total_flows:2,mean_confidence:.98,class_count:1,class_distribution:[{label:'benign',count:2}],confidence_distribution:[{label:'high',count:2}],audit:{status:'VERIFIED'},scope:'test',model:'lightgbm',model_run:'run',mode:'review'} as Summary;
const phase2:RecordData={mode:'review',unavailable:[],blockers:{},query:{predicted_class_label:'benign',confidence:.98},retrieval:[],provenance:[],reranking:[],context:{context_status:'READY',evidence:[]},reasoning:{},fusion:{status:'COMPLETED'},decision:{final_class:'benign',evidence_status:'INSUFFICIENT'},runtime_status:{llm:'COMPLETED',fusion:'COMPLETED'},sources:{}};
const completeStatus={analysis_id:'ABCDEF123456',input:{},status:'complete',stages:['query','retrieval','provenance','reranking','context','reasoning','fusion','decision'].map(id=>({id,label:id,status:'complete',message:''})),error:null,selected_flow_id:'FLOW-1'} as Status;

function backend(status:()=>Status=()=>completeStatus,result:()=>RecordData|null=()=>phase2){
 return vi.fn(async(path:string,method?:string)=>{
  if(path==='/api/health')return {status:'ready',contracts:[],evidence:{}};
  if(path==='/api/llm/status')return {status:'READY'};
  if(path.endsWith('/status'))return status();
  if(path.endsWith('/summary'))return summary;
  if(path.endsWith('/phase2')&&method!=='POST')return result();
  if(path.endsWith('/comparison'))return {flow_id:'FLOW-1',model_run:'run',models:[]};
  if(path.includes('/flows/FLOW-1'))return {id:'FLOW-1',source_split:'test',source_row_index:0,model:'lightgbm',predicted_class:'benign',confidence:.98,probabilities:{benign:.98,malware:.02}};
  return {rows:[{id:'FLOW-1'}]};
 });
}

function completedState(autoNavigationActive=false){
 useAnalysis.setState({analysisId:'ABCDEF123456',input:{dataset:'CIC-Darknet2020'},selectedFlowId:'FLOW-1',phase1:summary,phase2,finalDecision:phase2.decision,status:completeStatus,workflow:'COMPLETE',autoNavigationActive,phase2RunOrigin:autoNavigationActive?'new_analysis_auto':'manual_browsing',phase2Starting:false});
}

afterEach(()=>{
 cleanup();vi.useRealTimers();vi.restoreAllMocks();localStorage.clear();
 useAnalysis.setState({analysisId:null,input:null,selectedFlowId:null,phase1:null,phase2:null,finalDecision:null,status:null,flow:null,error:null,workflow:'IDLE',autoNavigationActive:false,phase2RunOrigin:'manual_browsing',phase2Starting:false});
});

describe('explicit manual Phase-II execution',()=>{
 function setup(route:string){
  vi.useFakeTimers();let outcome='before';let posts=0;
  completedState(false);useAnalysis.setState({phase2:null,finalDecision:null,workflow:'PHASE1_COMPLETE'});
  const currentStatus=()=>({...completeStatus,status:outcome==='running'?'running':outcome==='failed'?'partial':outcome==='offline'?'partial':outcome==='before'?'phase1_complete':'complete',stages:completeStatus.stages.map(s=>({...s,status:outcome==='failed'&&s.id==='reasoning'?'failed':s.status}))});
  const currentResult=()=>outcome==='complete'?phase2:outcome==='offline'?{...phase2,mode:'fresh_execution',decision:{},fusion:{},runtime_status:{llm:'OFFLINE',fusion:'WAITING'},llm_runtime:{status:'OFFLINE'},unavailable:['reasoning','fusion','decision']}:outcome==='failed'?{...phase2,decision:{},unavailable:['reasoning'],runtime_status:{llm:'FAILED'}}:null;
  const base=backend(currentStatus,currentResult);
  window.desktop={request:vi.fn(async(path:string,method?:string)=>{
   if(method==='POST'&&(path.endsWith('/phase2')||path.endsWith('/resume'))){posts++;outcome='running';return {status:'running'}}
   return base(path,method);
  })} as any;
  window.location.hash=route;render(<App/>);
  return {outcome:(value:string)=>{outcome=value},posts:()=>posts};
 }
 async function clickRun(route:string){
  await act(async()=>{await vi.advanceTimersByTimeAsync(20)});
  await act(async()=>{fireEvent.click(screen.getByRole('button',{name:route==='/flow'?'Review with Provenance-Aware RAG':'Analyze with Phase II'}));await vi.advanceTimersByTimeAsync(20)});
 }
 it.each(['/phase1','/flow'])('opens Final Decision after a manual execution from %s',async route=>{
  const run=setup(route);await clickRun(route);
  expect(run.posts()).toBe(1);expect(useAnalysis.getState().phase2RunOrigin).toBe('manual_phase2_start');
  run.outcome('complete');await act(async()=>{await vi.advanceTimersByTimeAsync(2000)});
  expect(window.location.hash).toBe('#/decision');expect(useAnalysis.getState().autoNavigationActive).toBe(false);
 });
 it('cancels only the redirect after sidebar browsing during a manual execution',async()=>{
  const run=setup('/phase1');await clickRun('/phase1');
  fireEvent.click(screen.getByRole('link',{name:'Classification',exact:true}));
  run.outcome('complete');await act(async()=>{await vi.advanceTimersByTimeAsync(4000)});
  expect(window.location.hash).toBe('#/phase1');expect(run.posts()).toBe(1);
 });
 it('preserves manual origin through LLM offline and retry',async()=>{
  const run=setup('/phase1');await clickRun('/phase1');run.outcome('offline');
  await act(async()=>{await vi.advanceTimersByTimeAsync(2000)});
  expect(useAnalysis.getState().phase2RunOrigin).toBe('manual_phase2_start');
  await act(async()=>{fireEvent.click(screen.getByRole('button',{name:'Retry LLM'}));await vi.advanceTimersByTimeAsync(20)});
  run.outcome('complete');await act(async()=>{await vi.advanceTimersByTimeAsync(2000)});
  expect(window.location.hash).toBe('#/decision');expect(run.posts()).toBe(2);
 });
 it('does not open Final Decision when a required stage fails',async()=>{
  const run=setup('/phase1');await clickRun('/phase1');run.outcome('failed');
  await act(async()=>{await vi.advanceTimersByTimeAsync(4000)});
  expect(window.location.hash).toBe('#/progress');
 });
 it('does not persist manual execution origin or its in-flight guard',async()=>{
  setup('/phase1');await clickRun('/phase1');
  const saved=JSON.parse(localStorage.getItem('provenanceguard-session-v1')!).state;
  expect(saved.autoNavigationActive).toBeUndefined();expect(saved.phase2RunOrigin).toBeUndefined();expect(saved.phase2Starting).toBeUndefined();
 });
});

describe('manual navigation overrides automatic completion routing',()=>{
 it('opens Final Decision once for a newly submitted automatic run',async()=>{
  completedState(true);window.location.hash='/progress';window.desktop={request:backend()} as any;
  render(<App/>);
  await act(async()=>{await new Promise(resolve=>setTimeout(resolve,20))});
  expect(window.location.hash).toBe('#/decision');
  expect(useAnalysis.getState().autoNavigationActive).toBe(false);
 });

 it.each([
  ['/phase1','Traffic, classified.'],
  ['/evidence','A question grounded in traffic.'],
  ['/provenance','Relevance, with a source of trust.'],
  ['/reasoning','Evidence in. A bounded explanation out.'],
 ])('keeps completed manual route %s stable beyond five seconds',async(path,title)=>{
  vi.useFakeTimers();completedState(false);window.location.hash=path;window.desktop={request:backend()} as any;
  render(<App/>);expect(screen.getByText(title)).toBeTruthy();
  await act(async()=>{await vi.advanceTimersByTimeAsync(6100)});
  expect(window.location.hash).toBe(`#${path}`);
 });

 it('does not persist automatic navigation when reopening a completed analysis',async()=>{
  useAnalysis.getState().begin('ABCDEF123456',{dataset:'CIC-Darknet2020'});
  const saved=JSON.parse(localStorage.getItem('provenanceguard-session-v1')!);
  expect(saved.state.autoNavigationActive).toBeUndefined();
  useAnalysis.setState({autoNavigationActive:false});
  await useAnalysis.persist.rehydrate();
  expect(useAnalysis.getState().autoNavigationActive).toBe(false);
 });

 it('keeps Classification open when fusion finishes after a manual sidebar click',async()=>{
  vi.useFakeTimers();let completed=false;
  const running={...completeStatus,status:'running'};
  completedState(true);useAnalysis.setState({status:running,workflow:'PHASE2_RUNNING'});
  window.location.hash='/progress';window.desktop={request:backend(()=>completed?completeStatus:running)} as any;
  render(<App/>);
  fireEvent.click(screen.getByRole('link',{name:'Classification'}));
  expect(useAnalysis.getState().autoNavigationActive).toBe(false);
  completed=true;
  await act(async()=>{await vi.advanceTimersByTimeAsync(6100)});
  expect(window.location.hash).toBe('#/phase1');
  expect(screen.getByText('Traffic, classified.')).toBeTruthy();
 });

 it('starts Phase II once for a new single-flow run',async()=>{
  vi.useFakeTimers();let started=false;let posts=0;
  const one={...summary,total_flows:1};
  useAnalysis.setState({analysisId:'ABCDEF123456',input:{dataset:'CIC-Darknet2020'},selectedFlowId:null,phase1:one,phase2:null,status:completeStatus,workflow:'PHASE1_COMPLETE',autoNavigationActive:true});
  window.location.hash='/progress';
  window.desktop={request:vi.fn(async(path:string,method?:string)=>{
   if(path==='/api/health')return {status:'ready',contracts:[],evidence:{}};
   if(path==='/api/llm/status')return {status:'READY'};
   if(path.endsWith('/status'))return started?{...completeStatus,status:'running'}:completeStatus;
   if(path.endsWith('/summary'))return one;
   if(path.includes('/flows?page='))return {rows:[{id:'FLOW-1'}]};
   if(path.endsWith('/phase2')&&method==='POST'){posts++;started=true;return {status:'running'}}
   if(path.endsWith('/phase2'))return null;
   if(path.endsWith('/flows/FLOW-1'))return {id:'FLOW-1',source_split:'test'};
   return null;
  })} as any;
  render(<App/>);
  await act(async()=>{await vi.advanceTimersByTimeAsync(6100)});
  expect(posts).toBe(1);
  expect(useAnalysis.getState().autoNavigationActive).toBe(true);
 });
});
