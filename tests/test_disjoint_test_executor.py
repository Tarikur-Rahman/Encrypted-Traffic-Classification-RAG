"""Offline adapter tests; never use real requests or tokenizer/model loading."""
import json
import pytest
from src.evaluation import run_disjoint_test_4319 as adapter


@pytest.fixture
def case(tmp_path,monkeypatch):
    (tmp_path/adapter.OUTPUT).mkdir(parents=True)
    for name in ('subset_manifest.csv','prompt_token_audit.csv','preflight.csv','configuration.json'):
        (tmp_path/adapter.OUTPUT/name).write_text('immutable preparation')
    identity=adapter.c.snapshot(tmp_path,[adapter.OUTPUT+n for n in
        ('subset_manifest.csv','prompt_token_audit.csv','preflight.csv','configuration.json')])
    prompt=dict(query_id='USTC-TEST-000001',source_dataset='USTC-TFC2016',classifier_prediction='benign',
        prompt_text='frozen prompt',allowed_chunk_ids='[]')
    selected={prompt['query_id']:(prompt,100)}
    calls=[]
    durable=[]
    original_append=adapter.execution.append_response
    def append_response(stream,event):
        original_append(stream,event)  # Keep the real write/flush/fsync path.
        position=stream.tell()
        try:
            stream.seek(0)
            assert event in [json.loads(line) for line in stream]
        finally:
            stream.seek(position)
        durable.append(event)
    monkeypatch.setattr(adapter.execution,'append_response',append_response)
    def request(prompt,tokens,args,deadline,phase,persist,on_error):
        calls.append(phase)
        event=dict(query_id=prompt['query_id'],phase=phase,source_split='test',prompt_tokens=tokens)
        persist(event)
        return event
    def validate(prompt,event,args):
        # Verified through the owned handle after the real flush/fsync; a
        # second open of the locked ledger is prohibited on Windows.
        assert event in durable
        return dict(query_id=prompt['query_id'],source_split='test',evidence_status='INSUFFICIENT',
            reasoning_decision='ABSTAIN',supported_class=None,cited_chunk_ids=[],explanation='No evidence.')
    monkeypatch.setattr(adapter.runtime,'post_json',lambda *a,**k:pytest.fail('Network prohibited'))
    monkeypatch.setattr(adapter.runtime,'continuation_request',request)
    monkeypatch.setattr(adapter.runtime,'continuation_result',validate)
    return tmp_path,selected,identity,calls


def test_status_only_no_writes_or_requests(case):
    root,selected,identity,calls=case
    before={p:p.read_bytes() for p in (root/adapter.OUTPUT).iterdir()}
    adapter.run(root,selected,None,identity,adapter.runtime_args())
    assert not calls and before=={p:p.read_bytes() for p in (root/adapter.OUTPUT).iterdir()}


def test_raw_first_resume_never_reissues(case,monkeypatch):
    root,selected,identity,calls=case
    original=adapter.runtime.continuation_result
    monkeypatch.setattr(adapter.runtime,'continuation_result',lambda *a:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        adapter.run(root,selected,None,identity,adapter.runtime_args(),execute=True)
    assert calls==['initial']
    monkeypatch.setattr(adapter.runtime,'continuation_result',original)
    adapter.run(root,selected,None,identity,adapter.runtime_args(),execute=True,resume=True)
    adapter.run(root,selected,None,identity,adapter.runtime_args(),execute=True,resume=True)
    assert calls==['initial']
    rows=list(adapter.c.rows(root,adapter.OUTPUT+'validated_responses.jsonl'))
    assert len(rows)==1 and rows[0]['source_split']=='test'
    assert adapter.c.snapshot(root,[r['path'] for r in identity])==identity


def test_single_positive_correction_and_durable_recovery(monkeypatch):
    events,calls={},[]
    prompt=dict(query_id='USTC-TEST-000001',prompt_text='frozen')
    def persist(event):events[(event['query_id'],event['phase'])]=event
    def request(p,t,a,d,phase,persist,on_error):
        calls.append(phase)
        event=dict(query_id=p['query_id'],phase=phase,prompt_tokens=t)
        persist(event)
        return event
    def validate(p,e,a):
        assert (p['query_id'],e['phase']) in events
        return dict(query_id=p['query_id'],validator_fallback_reason='POSITIVE_EVIDENCE',
            grounding_gate_error_message='frozen rejection',evidence_status='INSUFFICIENT')
    monkeypatch.setattr(adapter.runtime,'continuation_request',request)
    monkeypatch.setattr(adapter.runtime,'continuation_result',validate)
    monkeypatch.setattr(adapter.runtime,'correction_prompt',lambda p,e:dict(p,prompt_text='unchanged correction'))
    monkeypatch.setattr(adapter.runtime.audit,'prompt_tokens',lambda *a:200)
    monkeypatch.setattr(adapter.execution,'continuation_retry_record',lambda *a:dict(query_id=prompt['query_id']))
    adapter.resolve(prompt,100,adapter.runtime_args(),None,events,persist,lambda e:None,True)
    adapter.resolve(prompt,100,adapter.runtime_args(),None,events,persist,lambda e:None,False)
    assert calls==['initial','correction']


def test_unknown_or_validation_execution_id_rejected(tmp_path):
    path=tmp_path/adapter.OUTPUT/'raw_responses.jsonl'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(query_id='USTC-VAL-000001',phase='initial',source_split='validation'))+'\n')
    with pytest.raises(ValueError,match='Unknown'):
        adapter.ledger(tmp_path,'raw_responses.jsonl',{'USTC-TEST-000001'},raw=True)


def test_fixed_envelope():
    args=adapter.runtime_args()
    assert (args.version,args.max_model_len,args.max_completion_tokens)==('v4',5120,1024)
    assert args.final_frozen_test and args.runtime_configuration==adapter.runtime.SEMANTIC_STRATIFIED_1200
