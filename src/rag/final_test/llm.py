"""Frozen V5 response-contract-v2 TEST execution; reuse durable V5 engine."""
from contextlib import contextmanager
from functools import partial
from .contracts import *
from .preflight import frozen_inputs, evidence
from src.rag import run_v5_contract_v2 as v5

runtime=v5.runtime
SUBSET=rel(8,'stratified_test_subset_seed42.csv')
PROMPTS=rel(8,'test_llm_prompts.csv')
EXPORT=rel(8,'test_llm_responses_for_import.jsonl')
EXECUTION=BASE+FOLDERS[7]+'/execution/'
AUDIT=rel(8,'test_llm_token_audit.json')


def subset(root):
    require_stages(root,1)
    chosen,_=cohort(root)
    if (root/SUBSET).exists():
        require(list(rows(root,SUBSET))==chosen,'Saved cohort differs')
    else: table(root,SUBSET,chosen)
    return chosen


def prepare(root):
    require_stages(root,1,2,3,4,5,6,7)
    chosen,_=cohort(root); ids=[r['query_id'] for r in chosen]
    registry,chunks,verified=evidence(root)
    verified={k:dict(value,chunk_text=chunks[k]['chunk_text']) for k,value in verified.items()}
    contexts=unique_contexts(root,ids)
    prepared=[]; payloads={}; patterns={}
    for q in ids:
        context=contexts[q]; assert_safe(context)
        payload={k:context[k] for k in ('query_id','source_dataset','source_split','classifier_prediction','traffic_profile')}
        payload['trusted_evidence']=json.loads(context['evidence_records_json'])
        state=v5.contract.admissible(payload,registry,verified,source_split='test')
        prepared.append(dict(query_id=q,source_dataset=payload['source_dataset'],classifier_prediction=payload['classifier_prediction'],
                             prompt_text=v5.contract.prompt(payload,registry,verified,state),response_schema=v5.contract.schema(state,q)))
        payloads[q]=payload;patterns[q]=state['pattern']
    tokenizer=runtime.qwen.load_tokenizer(root)
    audit=runtime.token_audit(prepared,tokenizer)
    require(audit['gate_status']=='PASS','TEST token limit exceeded; stop, no truncation or tuning')
    inputs=frozen_inputs(root)+[rel(1),rel(7)]
    parent=json.loads(safe_path(root,V5_RUN+'configuration.json').read_text(encoding='utf-8'))
    current_tokenizer={p:sha(root,p) for p,_,_ in runtime.qwen.tokenizer_files(root)}
    require(current_tokenizer==parent['tokenizer_sha256'],'Frozen Qwen tokenizer differs')
    cfg=dict(run_id=RUN_ID,selected_query_ids=ids,selection_sha256=runtime.digest(json_text(ids)),
             frozen_configuration=FROZEN,input_sha256={p:sha(root,p) for p in sorted(set(inputs))},
             tokenizer_sha256=current_tokenizer,token_audit=audit,evidence_patterns=patterns,
             prompt_sha256={r['query_id']:runtime.digest(r['prompt_text']) for r in prepared},
             request_schema_sha256={r['query_id']:runtime.digest(json_text(r['response_schema'])) for r in prepared},
             parent_validation_configuration_sha256=parent['configuration_sha256'],
             validation_validator_sha256=VALIDATION_VALIDATOR_SHA256,
             test_runtime_validator_sha256=sha(root,'src/rag/positive_evidence_v5.py'),
             source_split='test',ground_truth_used=False,correctness_used=False,corrections=0,
             model=runtime.MODEL,endpoint=runtime.ENDPOINT,max_model_len=5120,max_completion_tokens=1024,temperature=0,seed=42)
    cfg['configuration_sha256']=runtime.digest(json_text(cfg))
    return prepared,payloads,registry,verified,cfg


def unique_contexts(root,ids):
    expected=set(ids); found={}
    for row in rows(root,rel(7)):
        require(row['query_id'] in expected and row['query_id'] not in found,'Unexpected/duplicate context')
        require(row['source_split']=='test' and row['rag_partition']=='final_test','Non-TEST context')
        found[row['query_id']]=row
    require(set(found)==expected,'Missing TEST contexts')
    return found


def summary(completed):
    return dict(run_id=RUN_ID,evaluation_scope=SCOPE,selected=TOTAL,completed=len(completed),remaining=TOTAL-len(completed),
                accepted=sum(r['validation_status']=='PASS' for r in completed.values()),
                rejected=sum(r['validation_status']!='PASS' for r in completed.values()),
                evidence_status_counts=dict(Counter(r['validated_response']['evidence_status'] for r in completed.values() if r['validation_status']=='PASS')),
                response_contract_version='v2',ground_truth_used=False)


@contextmanager
def runtime_scope():
    changes=dict(BASE=EXECUTION,NEW_BASE=EXECUTION,RUN_ID=RUN_ID,SCOPE=SCOPE,
                 request=v5.request,outcomes=partial(v5.outcomes,source_split='test'),summary=summary)
    old={k:getattr(runtime,k) for k in changes}
    try:
        for k,value in changes.items():setattr(runtime,k,value)
        yield
    finally:
        for k,value in old.items():setattr(runtime,k,value)


def audit(root,*,publish=False):
    prepared,payloads,registry,verified,cfg=prepare(root)
    if publish:
        subset(root)
        values=[dict(r,response_schema=json_text(r['response_schema']),qwen_prompt_tokens=cfg['token_audit']['prompt_tokens'][r['query_id']]) for r in prepared]
        if (root/PROMPTS).exists():
            require(list(rows(root,PROMPTS))==[{k:str(v) for k,v in r.items()} for r in values],'Saved prompts differ')
        else: table(root,PROMPTS,values)
        target=safe_path(root,AUDIT,write=True)
        if target.exists():require(json.loads(target.read_text(encoding='utf-8'))==cfg,'Token/preflight receipt differs')
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('x',encoding='utf-8') as stream:
                stream.write(json_text(cfg)+'\n');stream.flush();os.fsync(stream.fileno())
    if (root/(EXECUTION+runtime.CONFIG)).exists():
        require(json.loads(safe_path(root,EXECUTION+runtime.CONFIG).read_text(encoding='utf-8'))==cfg,'Saved execution configuration differs')
        with runtime_scope():
            completed,state=runtime.recover(root,prepared,payloads,registry,verified,cfg,inspect_only=True)
        print(json_text(dict(summary(completed),recovery=state)))
    print(json_text(dict(token_audit=cfg['token_audit']['gate_status'],prompt_count=TOTAL,ground_truth_used=0)))
    return prepared,payloads,registry,verified,cfg


def run(root,*,export=False,resume=False,repair=False,reset=False):
    require(not repair and not reset,'No automatic ledger repair or transport reset in frozen V5')
    if committed(root,8): return
    prepared,payloads,registry,verified,cfg=prepare(root)
    require(json.loads(safe_path(root,AUDIT).read_text(encoding='utf-8'))==cfg,'Publish matching token audit before execution')
    with runtime_scope():
        if not export:
            result=runtime.execute(root,prepared,payloads,registry,verified,cfg,resume=resume)
            print(json_text(result))
            return
        require(json.loads(safe_path(root,EXECUTION+runtime.CONFIG).read_text(encoding='utf-8'))==cfg,'Execution identity differs')
        completed,state=runtime.recover(root,prepared,payloads,registry,verified,cfg,inspect_only=True)
        require(len(completed)==TOTAL and all(not v for v in state.values()),'Incomplete/ambiguous execution')
    before=begin(root,8,[rel(7),SUBSET,PROMPTS,AUDIT,*frozen_inputs(root),
                         *[EXECUTION+n for n in (runtime.CONFIG,runtime.JOURNAL,runtime.RAW,runtime.VALIDATED)]])
    if before is None:return
    records=[completed[r['query_id']] for r in prepared]
    # Original raw bytes and ALL accepted/rejected validation wrappers remain
    # durable in execution/. Rejections are never converted into accepted output.
    target=safe_path(root,rel(8),write=True)
    with target.open('w',encoding='utf-8') as stream:
        for record in records:
            stream.write(json_text(record)+'\n')
        stream.flush();os.fsync(stream.fileno())
    with safe_path(root,EXPORT,write=True).open('w',encoding='utf-8') as stream:
        for record in records:
            if record['validation_status']=='PASS':stream.write(json_text(record['validated_response'])+'\n')
        stream.flush();os.fsync(stream.fileno())
    rejected=rel(8,'test_llm_rejected_responses.jsonl')
    with safe_path(root,rejected,write=True).open('w',encoding='utf-8') as stream:
        for record in records:
            if record['validation_status']!='PASS':stream.write(json_text(record)+'\n')
        stream.flush();os.fsync(stream.fileno())
    from .summaries import stage_summaries
    finish(root,8,before,[rel(8),EXPORT,rejected,PROMPTS,SUBSET,AUDIT,*stage_summaries(root,8)],TOTAL)


def completed(root):
    """Read-only exact replay for downstream consumers; no tokenization or HTTP."""
    cfg=json.loads(safe_path(root,AUDIT).read_text(encoding='utf-8'))
    unsigned=dict(cfg); signature=unsigned.pop('configuration_sha256')
    require(runtime.digest(json_text(unsigned))==signature,'TEST configuration hash differs')
    require(cfg['frozen_configuration']==FROZEN,'TEST scientific configuration differs')
    for name,digest in cfg['input_sha256'].items():
        require(sha(root,name)==digest,'TEST input changed: '+name)
    require(json.loads(safe_path(root,EXECUTION+runtime.CONFIG).read_text(encoding='utf-8'))==cfg,'Runtime configuration changed')
    ids=[r['query_id'] for r in cohort(root)[0]]
    require(ids==cfg['selected_query_ids'],'TEST selection differs')
    registry,chunks,verified=evidence(root)
    verified={k:dict(v,chunk_text=chunks[k]['chunk_text']) for k,v in verified.items()}
    contexts=unique_contexts(root,ids);prepared=[];payloads={}
    for q in ids:
        context=contexts[q];assert_safe(context)
        p={k:context[k] for k in ('query_id','source_dataset','source_split','classifier_prediction','traffic_profile')}
        p['trusted_evidence']=json.loads(context['evidence_records_json'])
        state=v5.contract.admissible(p,registry,verified,source_split='test')
        prompt=v5.contract.prompt(p,registry,verified,state);schema=v5.contract.schema(state,q)
        require(runtime.digest(prompt)==cfg['prompt_sha256'][q] and runtime.digest(json_text(schema))==cfg['request_schema_sha256'][q],'Prompt/schema differs')
        prepared.append(dict(query_id=q,prompt_text=prompt,response_schema=schema));payloads[q]=p
    with runtime_scope():
        result,state=runtime.recover(root,prepared,payloads,registry,verified,cfg,inspect_only=True)
    require(len(result)==TOTAL and all(not value for value in state.values()),'Incomplete/ambiguous TEST execution')
    saved={}
    for record in rows(root,rel(8)):
        require(record['query_id'] not in saved,'Duplicate exported TEST result')
        saved[record['query_id']]=record
    require(saved==result,'Stage8 publication differs from durable replay')
    return result
