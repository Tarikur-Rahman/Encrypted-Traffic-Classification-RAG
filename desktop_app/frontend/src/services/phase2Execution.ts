import {api} from './api';
import {useAnalysis} from '../stores/analysis';

export async function startManualPhase2(mode?:'fresh'|'review'){
 const {analysisId,selectedFlowId,phase2Starting,set}=useAnalysis.getState();
 if(!analysisId||!selectedFlowId||phase2Starting)throw new Error('Select a flow and wait for any pending start.');
 set({autoNavigationActive:true,phase2RunOrigin:'manual_phase2_start',phase2Starting:true,phase2:null,finalDecision:null,error:null});
 try{
  const flow=mode?null:await api<{source_split:string}>(`/api/analysis/${analysisId}/flows/${selectedFlowId}`);
  await api(`/api/analysis/${analysisId}/flows/${selectedFlowId}/phase2`,'POST',{mode:mode||(flow?.source_split==='test'?'review':'fresh')});
  set({phase2Starting:false,workflow:'PHASE2_RUNNING'});
 }catch(error){set({phase2Starting:false,autoNavigationActive:false,phase2RunOrigin:'manual_browsing',error:String(error)});throw error}
}
