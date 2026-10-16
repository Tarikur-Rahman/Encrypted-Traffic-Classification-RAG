"""Selected external flow only; durable stage outputs and explicit failure states."""
from backend.external_phase2.schemas import identity
from backend.external_phase2.safety import write_stage
from backend.external_phase2.prompt_runtime import build_external_prompt
from backend.external_phase2 import llm_runtime
from backend.external_phase2.fusion_runtime import run_external_fusion
from src.rag import phase2_common as common
from uuid import uuid4
from backend.utils.paths import analysis_directory,output_path
import json

def run_external_phase2(session,query,result,stage):
    number=next(i for i,r in enumerate(session['flows']) if r['id']==query['query_id'])
    attempt=uuid4().hex
    result['runtime_attempt']=attempt
    def save(category,name,value):
        path=write_stage(session['analysis_id'],number,category,name,value,attempt=attempt)
        result['sources'][category+'_'+name]=path
    result['evaluation_scope']='external_runtime'
    result['runtime_status']=dict(trusted_context='COMPLETED',llm='WAITING',fusion='WAITING')
    current='reasoning'
    runtime=dict(status='WAITING',**identity(query))
    try:
        save('trusted_context','context',result['context'])
        from backend.adapters.fresh_phase2 import resources
        _,_,verified,_,registry,_=resources()
        prompt=build_external_prompt(query,result['context'],registry,verified)
        save('llm','prompt',prompt)
        runtime.update(llm_runtime.check_llm_runtime())
        result['llm_runtime']=runtime
        try:
            runtime['prompt_tokens']=llm_runtime.token_count(prompt)
        except (ValueError,ImportError,OSError,RuntimeError) as exc:
            runtime['token_check_error']=str(exc)
            if runtime['status']=='READY':
                raise
        save('llm','runtime',runtime)
        if runtime['status']!='READY':
            result['runtime_status']['llm']=runtime['status']
            for key in ('reasoning','fusion','decision'):
                result['unavailable'].append(key)
                result['blockers'][key]=runtime['message']
            stage('reasoning',runtime['status'].lower(),runtime['message'])
            stage('fusion','waiting','Waiting for a validated local LLM response')
            stage('decision','unavailable','No fused decision without validated reasoning')
            return
        result['runtime_status']['llm']='RUNNING'
        runtime['status']='RUNNING'
        stage('reasoning','running','Generating one external V5 contract-v2 response using the frozen local model')
        save('llm','runtime',runtime)
        errors=[]
        runtime['attempt_errors']=errors
        response=llm_runtime.run_external_constrained_llm(prompt,runtime['prompt_tokens'],registry,verified,errors.append)
        validated=llm_runtime.validate_external_llm_response(prompt,response,registry,verified)
        save('llm','response',dict(response,response_validation_status='PASS'))
        result['reasoning']=response
        runtime.update(status='COMPLETED',**{k:response[k] for k in ('prompt_tokens','completion_tokens','total_tokens','latency_seconds','retry_count')})
        runtime['attempt_errors']=errors
        result['runtime_status']['llm']='COMPLETED'
        stage('reasoning','complete','Structured response, claim linkage and V5 contract-v2 gate validated')
        current='fusion'
        result['runtime_status']['fusion']='RUNNING'
        stage('fusion','running','Applying frozen retain-classifier fusion to validated external evidence')
        # Recheck canonical authorization at the final boundary.
        decision=run_external_fusion(query,validated,result['context'],verified,registry)
        save('fusion','decision',decision)
        result['fusion']=decision
        result['decision']=dict(decision,supporting_evidence_ids=common.parse_json(decision['cited_chunk_ids']))
        result['runtime_status']['fusion']='COMPLETED'
        stage('fusion','complete','Frozen retain-classifier decision with verified context citations')
        stage('decision','complete','External runtime final classification available')
    except (ValueError,TypeError,KeyError,ImportError,OSError,RuntimeError) as exc:
        key='llm' if current=='reasoning' else 'fusion'
        result['runtime_status'][key]='FAILED'
        if current=='reasoning':
            runtime.update(status='FAILED',message=str(exc))
        for key in (('reasoning','fusion','decision') if current=='reasoning' else ('fusion','decision')):
            result['unavailable'].append(key)
            result['blockers'][key]=str(exc)
            stage(key,'failed' if key==current else 'waiting',str(exc))
    finally:
        result['llm_runtime']=runtime
        save('llm','runtime',runtime)

def resume_external_llm(session,flow_id,stage):
    """Resume after context without repeating Phase I, retrieval or context construction."""
    common.require(flow_id in session.get('phase2',{}),'No preserved Phase-II state is available')
    result=session['phase2'][flow_id]
    common.require(result.get('context',{}).get('context_status') in {'READY','INSUFFICIENT'},'Trusted context is unavailable')
    common.require(not result.get('decision'),'Fusion is already complete')
    query=result['query']
    result['reasoning']={}
    result['fusion']={}
    result['decision']={}
    result['unavailable']=[key for key in result.get('unavailable',[]) if key not in {'reasoning','fusion','decision'}]
    for key in ('reasoning','fusion','decision'):
        result.get('blockers',{}).pop(key,None)
    run_external_phase2(session,query,result,stage)
    number=next(i for i,row in enumerate(session['flows']) if row['id']==flow_id)
    path=output_path(analysis_directory(session['analysis_id'])/f'phase2-{number}.json')
    result.setdefault('sources',{})['runtime']=str(path)
    path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    session['phase2'][flow_id]=result
    session['status']='complete' if result.get('decision') else 'partial'
    return result
