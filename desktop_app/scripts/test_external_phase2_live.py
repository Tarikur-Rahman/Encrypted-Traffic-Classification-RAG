"""Actual local-server acceptance. No simulated responses or historical replay."""
import sys,json
from itertools import islice
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.config import APP,ROOT,RUNTIME
from backend.utils.write_guard import install
from backend.adapters import input_adapter,phase1_adapter,fresh_phase2
from backend.adapters.artifacts import rows,digest
from backend.session import store
from backend.external_phase2.llm_runtime import check_llm_runtime,runtime_args,tokenizer
from backend.external_phase2.safety import assert_desktop_write_path

def main():
    install(APP)
    report={'preflight':check_llm_runtime(),'inputs':[],'passed':False}
    alternatives='--ustc-alternatives' in sys.argv
    target=assert_desktop_write_path(RUNTIME/('external-phase2-alternative-inputs.json' if alternatives else 'external-phase2-live-report.json'))
    try:
        if report['preflight']['status']!='READY':
            raise RuntimeError(report['preflight']['message'])
        # Known frozen prompt: compare schema/citation/token behavior, not prose.
        from src.rag import llm_v3_runtime as frozen
        from src.rag.audit_qwen_prompt_lengths import prompt_tokens
        # A fixed three-row sample records rejected model claims as well as valid
        # outputs. A rejection is never counted as a completed classification.
        report['known_prompts']=[]
        for prompt in islice(rows(ROOT/'results/rag/test/final_4300_v5_contract_v2/stage08_llm/test_llm_prompts.csv'),3):
            count=prompt_tokens(tokenizer(),prompt['prompt_text'])
            errors=[]
            generated=frozen.generate(prompt,count,runtime_args(),errors.append)
            logical={k:generated[k] for k in frozen.FIELDS}
            frozen.validate_logical(logical,prompt)
            accepted=generated.get('runtime_fallback_applied')!='YES' and generated.get('validator_fallback_applied')!='YES'
            report['known_prompts'].append(dict(query_id=prompt['query_id'],accepted_without_fallback=accepted,errors=errors,
                runtime={k:generated.get(k) for k in ('runtime_fallback_reason','validator_fallback_reason','finish_reason','prompt_tokens','completion_tokens')},
                validated_response=logical if accepted else None))
        cases=[('USTC CSV',RUNTIME/'regression-inputs/ustc-raw.csv','USTC-TFC2016','raw_flows'),
            ('USTC PCAP',ROOT/'data/raw/USTC-TFC2016/Benign/Facetime.pcap','USTC-TFC2016','auto'),
            ('CIC CSV',RUNTIME/'regression-inputs/cic-raw.csv','CIC-Darknet2020','raw_flows')]
        if alternatives:
            cases=[('USTC model-feature CSV',RUNTIME/'regression-inputs/ustc-features.csv','USTC-TFC2016','model_features'),
                ('USTC Skype PCAP',ROOT/'data/raw/USTC-TFC2016/Benign/Skype.pcap','USTC-TFC2016','auto')]
        for label,path,dataset,input_stage in cases:
            original=digest(path)
            source=input_adapter.validate(str(path),dataset,input_stage)
            if not source['compatible']:
                raise ValueError(source['reason'])
            session=store.create(source)
            stage=lambda k,s,m:store.stage(session,k,s,m)
            phase1_adapter.execute(session,stage)
            for observation in session['flows'][:3]:
                flow=observation['id']
                result=fresh_phase2.execute(session,flow,stage)
                store.save(session)
                item=dict(input=label,analysis_id=session['analysis_id'],flow_id=flow,status=session['status'],
                    runtime_status=result.get('runtime_status'),decision=result['decision'],blockers=result['blockers'],
                    source_unchanged=digest(path)==original,reasoning=result['reasoning'],context_chunks=len(result['context'].get('evidence',[])))
                item['passed']=item['source_unchanged'] and bool(result['decision'].get('final_class')) and not result['unavailable']
                report['inputs'].append(item)
                target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
                print(label,flow,session['status'],result['blockers'],flush=True)
                if item['passed']:
                    assert result['decision']['source_dataset']=='external'
                    assert result['decision']['evaluation_scope']=='external_runtime'
                    assert set(result['reasoning']['cited_chunk_ids'])<=set(json.loads(result['context']['selected_chunk_ids']))
        report['input_support']={label:any(item['passed'] for item in report['inputs'] if item['input']==label) for label,_,_,_ in cases}
        report['passed']=any(item['accepted_without_fallback'] for item in report['known_prompts']) and all(report['input_support'].values())
        if not report['passed']:
            raise ValueError('One or more live acceptance checks failed; see per-input results')
    except Exception as exc:
        report['failure']=str(exc)
        raise
    finally:
        target.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps({'passed':report['passed'],'report':str(target),'failure':report.get('failure')}),flush=True)

if __name__=='__main__':
    main()
