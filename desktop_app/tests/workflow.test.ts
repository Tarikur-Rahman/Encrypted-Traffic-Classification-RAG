import {describe,expect,it} from 'vitest';
import {workflowAction,workflowState} from '../frontend/src/services/workflow';
import type {Status,Summary} from '../frontend/src/types';

const status=(value='complete')=>({analysis_id:'A',input:{},status:value,stages:['query','retrieval','provenance','reranking','context','reasoning','fusion','decision'].map(id=>({id,label:id,status:'complete',message:''})),error:null}) as Status;
const summary=(flows=1)=>({total_flows:flows}) as Summary;

describe('automatic continuation state machine',()=>{
 it('selects the only eligible flow after successful Phase I',()=>expect(workflowAction(status(),summary(1),null,null)).toBe('SELECT_ONLY_FLOW'));
 it.each(['failed','incompatible','unsupported','cancelled'])('stops after %s Phase I',value=>expect(workflowAction(status(value),summary(),null,null)).toBe('STOP'));
 it('opens Flow Explorer when multiple flows require an explicit choice',()=>expect(workflowAction(status(),summary(4),null,null)).toBe('OPEN_FLOW_EXPLORER'));
 it('starts Phase II once an eligible flow is selected',()=>expect(workflowAction(status(),summary(4),'FLOW-2',null)).toBe('START_PHASE2'));
 it('pauses at the LLM without discarding trusted context',()=>{
  const phase2={context:{context_status:'READY'},runtime_status:{llm:'OFFLINE',fusion:'WAITING'}};
  expect(workflowAction(status(),summary(),'FLOW-1',phase2)).toBe('SHOW_LLM_OFFLINE');
  expect(workflowState(status(),summary(),'FLOW-1',phase2)).toBe('LLM_OFFLINE');
 });
 it('does not redirect on fusion alone',()=>expect(workflowAction(status(),summary(),'FLOW-1',{runtime_status:{fusion:'COMPLETED'}})).toBe('WAIT'));
 it('opens an existing final result instead of executing Phase II twice',()=>expect(workflowAction(status(),summary(),'FLOW-1',{fusion:{final_class:'benign'},decision:{final_class:'benign',evidence_status:'INSUFFICIENT'}})).toBe('OPEN_DECISION'));
 it.each([{decision:{final_class:'benign'}},{decision:{final_class:'benign',evidence_status:'INSUFFICIENT'},fusion:{}}])('waits for persisted decision and fusion data',phase2=>expect(workflowAction(status(),summary(),'FLOW-1',phase2)).toBe('WAIT'));
 it('waits while Phase I is still running',()=>expect(workflowAction(status('running'),null,null,null)).toBe('WAIT'));
});
