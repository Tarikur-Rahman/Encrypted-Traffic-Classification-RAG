import {create} from 'zustand';
import {persist} from 'zustand/middleware';
import type {Flow,Health,RecordData,Status,Summary} from '../types';
import type {WorkflowState} from '../services/workflow';
export type Phase2RunOrigin='new_analysis_auto'|'manual_phase2_start'|'manual_browsing';
interface AnalysisState {
 analysisId:string|null;input:RecordData|null;phase1:Summary|null;selectedFlowId:string|null;phase2:RecordData|null;finalDecision:RecordData|null;
 status:Status|null;flow:Flow|null;health:Health|null;error:string|null;
 workflow:WorkflowState;llmStatus:RecordData|null;autoNavigationActive:boolean;
 phase2RunOrigin:Phase2RunOrigin;phase2Starting:boolean;
 set:(patch:Partial<AnalysisState>)=>void;
 begin:(id:string,input:RecordData)=>void;
 select:(id:string)=>void;
 cancelAutoNavigation:()=>void;
}
export const useAnalysis=create<AnalysisState>()(persist((set)=>({analysisId:null,input:null,phase1:null,selectedFlowId:null,phase2:null,finalDecision:null,status:null,flow:null,health:null,error:null,workflow:'IDLE',llmStatus:null,autoNavigationActive:false,phase2RunOrigin:'manual_browsing',phase2Starting:false,
 set,begin:(id,input)=>set({analysisId:id,input,phase1:null,selectedFlowId:null,phase2:null,finalDecision:null,status:null,flow:null,error:null,workflow:'PHASE1_RUNNING',autoNavigationActive:true,phase2RunOrigin:'new_analysis_auto',phase2Starting:false}),
 select:(id)=>set(state=>state.selectedFlowId===id?{}:{selectedFlowId:id,flow:null,phase2:null,finalDecision:null}),
 cancelAutoNavigation:()=>set({autoNavigationActive:false,phase2RunOrigin:'manual_browsing'}),
}),{name:'provenanceguard-session-v1',partialize:s=>({analysisId:s.analysisId,input:s.input,selectedFlowId:s.selectedFlowId,workflow:s.workflow})}));
