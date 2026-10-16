import type {RecordData,Status,Summary} from '../types';

export type WorkflowState='IDLE'|'PHASE1_RUNNING'|'PHASE1_COMPLETE'|'FLOW_REQUIRED'|'PHASE2_RUNNING'|'TRUSTED_CONTEXT_COMPLETE'|'LLM_OFFLINE'|'VALIDATION_FAILED'|'FUSION_FAILED'|'COMPLETE'|'PHASE1_FAILED';
export type WorkflowAction='WAIT'|'STOP'|'SELECT_ONLY_FLOW'|'OPEN_FLOW_EXPLORER'|'START_PHASE2'|'SHOW_LLM_OFFLINE'|'OPEN_DECISION';

export function workflowState(status:Status|null,summary:Summary|null,selectedFlowId:string|null,phase2:RecordData|null):WorkflowState{
 if(!status)return 'IDLE';
 if(status.status==='error'||status.status==='failed'||status.status==='cancelled'||status.status==='unsupported'||status.status==='incompatible')return 'PHASE1_FAILED';
 if(status.status==='running')return status.stages.some(stage=>['query','corpus','index','retrieval','provenance','reranking','context','reasoning','fusion','decision'].includes(stage.id)&&stage.status==='running')?'PHASE2_RUNNING':'PHASE1_RUNNING';
 if(!summary)return 'IDLE';
 if(phase2?.decision?.final_class)return 'COMPLETE';
 if(phase2?.runtime_status?.llm==='OFFLINE')return 'LLM_OFFLINE';
 if(phase2?.runtime_status?.llm==='FAILED')return 'VALIDATION_FAILED';
 if(phase2?.runtime_status?.fusion==='FAILED')return 'FUSION_FAILED';
 if(phase2?.context?.context_status)return 'TRUSTED_CONTEXT_COMPLETE';
 if(summary.total_flows>1&&!selectedFlowId)return 'FLOW_REQUIRED';
 return 'PHASE1_COMPLETE';
}

export const terminalPhase1Failure=(status:string)=>['error','failed','cancelled','unsupported','incompatible'].includes(status.toLowerCase());

export function phase2Complete(status:Status,phase2:RecordData|null){
 const required=['query','retrieval','provenance','reranking','context','reasoning','fusion','decision'];
 return status.status==='complete'&&!!phase2?.decision?.final_class&&!!phase2?.decision?.evidence_status
  &&!!phase2?.fusion&&Object.keys(phase2.fusion).length>0&&!phase2?.unavailable?.length
  &&phase2?.reasoning?.response_valid!==false
  &&required.every(id=>status.stages.some(stage=>stage.id===id&&['complete','reused'].includes(stage.status)))
  &&!status.stages.some(stage=>['failed','error','unavailable'].includes(stage.status));
}

export function workflowAction(status:Status|null,summary:Summary|null,selectedFlowId:string|null,phase2:RecordData|null):WorkflowAction{
 if(!status||status.status==='running')return 'WAIT';
 if(terminalPhase1Failure(status.status))return 'STOP';
 if(!summary||summary.total_flows<1)return 'WAIT';
 if(phase2Complete(status,phase2))return 'OPEN_DECISION';
 if(phase2?.runtime_status?.llm==='OFFLINE')return 'SHOW_LLM_OFFLINE';
 if(selectedFlowId&&!phase2)return 'START_PHASE2';
 if(!selectedFlowId&&summary.total_flows===1)return 'SELECT_ONLY_FLOW';
 if(!selectedFlowId&&summary.total_flows>1)return 'OPEN_FLOW_EXPLORER';
 return 'WAIT';
}
