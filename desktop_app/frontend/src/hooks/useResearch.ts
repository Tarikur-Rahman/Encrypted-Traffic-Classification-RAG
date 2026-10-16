import {useEffect,useState} from 'react';
import {useLocation,useNavigate} from 'react-router-dom';
import {api} from '../services/api';
import {useAnalysis} from '../stores/analysis';
import type {Health,RecordData,Status,Summary} from '../types';
import {terminalPhase1Failure,workflowAction,workflowState} from '../services/workflow';

const activeJobs=new Set<string>();

export function useResearch(){
 const {analysisId,selectedFlowId,autoNavigationActive,phase2RunOrigin,phase2Starting,set,select}=useAnalysis();const navigate=useNavigate();const location=useLocation();
 useEffect(()=>{let cancelled=false;let timer:ReturnType<typeof setTimeout>;
  async function poll(){try{if(phase2Starting)return;const health=await api<Health>('/api/health');if(cancelled)return;set({health});
   try{const llmStatus=await api<RecordData>('/api/llm/status');if(!cancelled)set({llmStatus})}catch{/* Backend health remains independently visible. */}
   if(analysisId){const status=await api<Status>(`/api/analysis/${analysisId}/status`);if(cancelled)return;set({status,error:status.error});
    if(status.status!=='running'){
     if(terminalPhase1Failure(status.status)){set({workflow:'PHASE1_FAILED',autoNavigationActive:false});return}
     const phase1=await api<Summary>(`/api/analysis/${analysisId}/summary`);if(cancelled)return;set({phase1});
     if(!phase1)return;
     let selected=selectedFlowId||status.selected_flow_id||null;
     if(selected&&!selectedFlowId)select(selected);
     if(autoNavigationActive&&workflowAction(status,phase1,selected,null)==='SELECT_ONLY_FLOW'){const only=await api<{rows:RecordData[]}>(`/api/analysis/${analysisId}/flows?page=1&page_size=1`);selected=only.rows[0]?.id||null;if(selected)select(selected)}
     if(autoNavigationActive&&workflowAction(status,phase1,selected,null)==='OPEN_FLOW_EXPLORER'){set({workflow:'FLOW_REQUIRED'});if(location.pathname!=='/flows')navigate('/flows');return}
     if(selected){
      const phase2=await api<RecordData|null>(`/api/analysis/${analysisId}/flows/${selected}/phase2`);if(cancelled)return;
      set({phase2,finalDecision:phase2?.decision??null,workflow:workflowState(status,phase1,selected,phase2)});
      const action=workflowAction(status,phase1,selected,phase2);
      if(autoNavigationActive&&['new_analysis_auto','manual_phase2_start'].includes(phase2RunOrigin)&&action==='OPEN_DECISION'){set({autoNavigationActive:false,phase2RunOrigin:'manual_browsing'});if(location.pathname!=='/decision')navigate('/decision');return}
      if(autoNavigationActive&&action==='SHOW_LLM_OFFLINE'){if(location.pathname!=='/reasoning')navigate('/reasoning');return}
      if(autoNavigationActive&&action==='START_PHASE2'){const key=`${analysisId}:${selected}`;if(!activeJobs.has(key)){activeJobs.add(key);try{
       const flow=await api<RecordData>(`/api/analysis/${analysisId}/flows/${selected}`);
       await api(`/api/analysis/${analysisId}/flows/${selected}/phase2`,'POST',{mode:flow.source_split==='test'?'review':'fresh'});
       set({workflow:'PHASE2_RUNNING'});if(location.pathname!=='/progress')navigate('/progress');
      }catch(error){if(!String(error).includes('already running'))set({error:String(error)})}finally{activeJobs.delete(key)}}}
     }else set({workflow:workflowState(status,phase1,null,null)});
    }else set({workflow:workflowState(status,null,selectedFlowId,null)});
   }
  }catch(error){if(!cancelled)set({error:String(error)})}finally{if(!cancelled)timer=setTimeout(poll,1800)}}
  void poll();return()=>{cancelled=true;clearTimeout(timer)};
 },[analysisId,selectedFlowId,autoNavigationActive,phase2RunOrigin,phase2Starting,set,select,navigate,location.pathname]);
}
export function useRequest<T>(path:string|null){
 const [data,setData]=useState<T|null>(null);const [error,setError]=useState('');const [loading,setLoading]=useState(false);
 useEffect(()=>{let current=true;setData(null);setError('');if(!path)return;setLoading(true);
 api<T>(path).then(value=>{if(current)setData(value)}).catch(err=>{if(current)setError(String(err))}).finally(()=>{if(current)setLoading(false)});
 return()=>{current=false};},[path]);return {data,error,loading};
}
