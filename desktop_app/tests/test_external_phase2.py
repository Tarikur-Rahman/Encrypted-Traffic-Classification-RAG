"""Frozen-artifact parity and adversarial boundary tests, not fake production output."""
import sys,json,copy
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.config import ROOT,APP,RUNTIME
import pytest
from backend.adapters.artifacts import rows,select
from backend.adapters.fresh_phase2 import resources
from backend.external_phase2 import trusted_context_runtime as contexts,prompt_runtime as prompts,llm_runtime as llm,fusion_runtime as fusion
from backend.external_phase2.schemas import identity
from src.rag import phase2_common as common,materialize_constrained_llm_experiment as frozen_prompts,llm_v3_runtime as frozen_runtime

BASE='results/rag/test/final_4300_v5_contract_v2/stage08_llm/'

@pytest.fixture(scope='module',params=['USTC-TFC2016','CIC-Darknet2020'])
def known(request):
    query=next(r for r in rows(ROOT/'results/rag/test/final_4300_v5_contract_v2/stage01_structured_queries/test_query_records.csv')
               if r['source_dataset']==request.param)
    qid=query['query_id']
    raw=select(BASE+'test_llm_responses.jsonl','query_id',qid)[0]
    original=select('results/rag/v5f_validation/stage04_retrieval/query_records.csv','query_id',qid)[0]
    external={k:v for k,v in original.items() if k not in {'true_class_id','true_class_label','classifier_correct'}}
    external.update(query_id='EXT-REGRESSION-'+qid.replace('-VAL-','-'),source_dataset='external',analysis_contract=request.param,
        source_split='external_inference',rag_partition='external_inference',evaluation_scope='external_runtime')
    ranked=select('results/rag/v5f_validation/stage06_reranking/reranked_evidence.csv','query_id',qid)
    ranked=[dict(r,query_id=external['query_id'],source_dataset='external') for r in ranked]
    frozen_context=select('results/rag/v5f_validation/stage07_trusted_contexts/trusted_context_records.csv','query_id',qid)[0]
    frozen_fusion=select(BASE+'fusion_decisions.csv','query_id',qid)[0]
    policy,evidence,verified,_=resources()
    chunks={r['chunk_id']:r for r in evidence.chunks}
    context=contexts.build_external_trusted_context(external,ranked,chunks,verified,policy)
    prompt=prompts.build_external_prompt(external,context)
    response=dict({k:raw[k] for k in frozen_runtime.FIELDS},**identity(external))
    return SimpleNamespace(query=external,ranked=ranked,context=context,prompt=prompt,response=response,
        frozen_context=frozen_context,frozen_fusion=frozen_fusion,policy=policy,chunks=chunks,verified=verified)

def test_context_exact_frozen_parity(known):
    for key in ('selected_chunk_ids','selected_document_ids','rerank_ranks','evidence_records_json','context_text','token_counts','source_types'):
        assert known.context[key]==known.frozen_context[key],key
    assert known.context['context_token_count']==int(known.frozen_context['context_token_count'])
    assert known.context['chunk_count']==len(json.loads(known.frozen_context['selected_chunk_ids']))
    assert known.context['distinct_document_count']==len(set(json.loads(known.frozen_context['selected_document_ids'])))
    assert known.context['token_budget']==2048 and known.context['top_n']==5 and known.context['max_chunks_per_document']==2

def test_prompt_frozen_instructions_and_payload_parity(known):
    original=frozen_prompts.prompt_record(known.frozen_context,'v4')
    prefix=lambda p:p['prompt_text'].partition('INPUT_JSON:\n')[0]
    assert prefix(original)==prefix(known.prompt)
    original_payload=json.loads(original['prompt_text'].partition('INPUT_JSON:\n')[2])
    external_payload=json.loads(known.prompt['prompt_text'].partition('INPUT_JSON:\n')[2])
    for key in ('classifier_prediction','classifier_confidence','traffic_profile','trusted_evidence','allowed_chunk_ids','allowed_classes'):
        assert external_payload[key]==original_payload[key],key
    assert external_payload['source_dataset']=='external' and external_payload['source_split']=='external_inference'
    assert llm.token_count(known.prompt)<=4096

def test_fusion_exact_frozen_parity(known):
    validated=llm.validate_external_llm_response(known.prompt,known.response)
    result=fusion.run_external_fusion(known.query,validated,known.context,known.verified)
    for key in ('final_class','evidence_status','fusion_reason','cited_chunk_ids','cited_document_ids','provenance_summary','evidence_explanation'):
        assert result[key]==known.frozen_fusion[key],key
    assert result['evaluation_scope']=='external_runtime' and result['source_dataset']=='external'

def test_context_rejects_unverified_and_altered_rank(known):
    for change in ({'provenance_status':'UNVERIFIED'},{'authorized_for_retrieval':'NO'},{'final_rerank_score':'999'}):
        ranked=copy.deepcopy(known.ranked);ranked[0].update(change)
        with pytest.raises(ValueError):
            contexts.build_external_trusted_context(known.query,ranked,known.chunks,known.verified,known.policy)

def test_response_citations_classes_and_identity_fail_closed(known):
    for change in ({'cited_chunk_ids':['NOT-SUPPLIED']},{'query_id':'EXT-WRONG'},{'evidence_status':'INVENTED'},
        {'source_dataset':'CIC-Darknet2020'}, {'supported_class':'invalid','evidence_status':'SUPPORTED','reasoning_decision':'SUPPORT_CLASS'}):
        with pytest.raises(ValueError):
            llm.validate_external_llm_response(known.prompt,dict(known.response,**change))
    with pytest.raises(ValueError,match='Unvalidated'):
        fusion.run_external_fusion(known.query,dict(known.response,response_validation_status='FAIL'),known.context,known.verified)

def test_exact_ustc_positive_gate_failure_and_vocabulary(known):
    if known.query['analysis_contract']!='USTC-TFC2016':
        pytest.skip('USTC vocabulary regression')
    assert common.CLASSES['USTC-TFC2016']==('benign','malware')
    chunk=known.context['evidence'][0]['chunk_id']
    invalid=dict(known.response,reasoning_decision='SUPPORT_CLASS',evidence_status='SUPPORTED',
        supported_class=known.query['predicted_class_label'],cited_chunk_ids=[chunk],
        explanation='A broad cited passage is evidence for the predicted class.')
    with pytest.raises(ValueError,match='V4 positive response requires nonempty EVIDENCE_QUOTE and EVIDENCE_LINK'):
        llm.validate_external_llm_response(known.prompt,invalid)
    valid=dict(known.response,reasoning_decision='ABSTAIN',evidence_status='INSUFFICIENT',
        supported_class=None,cited_chunk_ids=[],explanation='No qualifying positive evidence line is supplied.')
    assert llm.validate_external_llm_response(known.prompt,valid)['response_validation_status']=='PASS'
    for spelling in ('Benign','Malware','malicious','BENIGN','MALWARE'):
        with pytest.raises(ValueError,match='Invalid supported class/decision/citations'):
            llm.validate_external_llm_response(known.prompt,dict(invalid,supported_class=spelling))
    retried=llm.semantic_retry_prompt(known.prompt,'V4 positive response requires nonempty EVIDENCE_QUOTE and EVIDENCE_LINK')
    assert retried['prompt_text']!=known.prompt['prompt_text']
    assert retried['prompt_text'].partition('INPUT_JSON:\n')[2]==known.prompt['prompt_text'].partition('INPUT_JSON:\n')[2]
    assert 'Apply mandatory rules 32 and 33 exactly.' in retried['prompt_text']

def test_fusion_rechecks_cited_authorization(known):
    # Synthetic response only for this negative unit test; uses an actual context ID.
    chunk=known.context['evidence'][0]['chunk_id']
    logical=dict(known.response,reasoning_decision='ABSTAIN',evidence_status='CONFLICTING',supported_class=None,
        explanation='Boundary test fixture, not generated reasoning.',cited_chunk_ids=[chunk])
    validated=llm.validate_external_llm_response(known.prompt,logical)
    changed=copy.deepcopy(known.verified);changed[chunk]['authorized_for_downstream']='NO'
    with pytest.raises(ValueError,match='Unverified/unauthorized'):
        fusion.run_external_fusion(known.query,validated,known.context,changed)

def test_original_function_code_and_globals_are_unchanged(known):
    engine=llm.external_engine(known.query['analysis_contract'])
    assert engine.generate.__code__ is frozen_runtime.generate.__code__
    assert engine.validate_response.__code__ is frozen_runtime.importer.validate_response.__code__
    assert 'external' not in common.CLASSES
    assert frozen_runtime.common is common
    with pytest.raises(ValueError):
        common.validate_query(known.query,source_split='external_inference')
    schema=engine.response_schema(known.prompt)
    assert schema['properties']['supported_class']['enum']==[*common.CLASSES[known.query['analysis_contract']],None]

def test_original_transport_runtime_structural_parity(known,monkeypatch):
    # A recorded response tests unchanged generation/validation code. It is NOT
    # a live LLM test and is never used by production to manufacture an answer.
    engine=llm.external_engine(known.query['analysis_contract'])
    count=llm.token_count(known.prompt)
    logical={k:known.response[k] for k in frozen_runtime.FIELDS}
    data={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(logical)}}],
        'usage':{'prompt_tokens':count,'completion_tokens':30,'total_tokens':count+30}}
    def post_json(endpoint,payload,timeout):
        from src.rag.audit_qwen_prompt_lengths import prompt_tokens
        actual=prompt_tokens(llm.tokenizer(),payload['messages'][0]['content'])
        data['usage']={'prompt_tokens':actual,'completion_tokens':30,'total_tokens':actual+30}
        return data
    monkeypatch.setitem(engine.generate.__globals__,'post_json',post_json)
    result=llm.run_external_constrained_llm(known.prompt,count)
    assert result['query_id']==known.query['query_id'] and result['completion_tokens']==30
    assert 'raw_model_response' not in result
    invalid=copy.deepcopy(logical);invalid['cited_chunk_ids']=['NOT-SUPPLIED']
    data['choices'][0]['message']['content']=json.dumps(invalid)
    with pytest.raises(ValueError,match='fusion blocked'):
        llm.run_external_constrained_llm(known.prompt,count)

def test_loopback_only_and_frozen_runtime_values():
    with pytest.raises(ValueError):
        frozen_runtime.endpoint_url('https://example.com/v1/chat/completions')
    args=llm.runtime_args()
    # V5 contract-v2 explicitly removed automatic retries and uses the
    # diagnostic runtime's 180-second transport envelope.
    assert (args.max_model_len,args.max_completion_tokens,args.retry_limit,args.timeout)==(5120,1024,0,180)

def test_artifact_write_containment():
    from backend.external_phase2.safety import assert_desktop_write_path,write_stage
    for path in (ROOT/'results/no.json',APP/'../metadata/no.json',APP/'../src/no.json'):
        with pytest.raises(ValueError):
            assert_desktop_write_path(path)
    with pytest.raises(ValueError):
        write_stage('SAFE',0,'llm','response',{},attempt='../escape')

def test_offline_keeps_context_and_never_calls_llm_or_fusion(known,monkeypatch):
    from backend.external_phase2 import orchestrator
    from backend.session import store
    monkeypatch.setattr(llm,'check_llm_runtime',lambda:dict(status='OFFLINE',message='Local constrained-LLM runtime is offline.',model=frozen_runtime.MODEL))
    def forbidden(*args,**kwargs):
        raise AssertionError('Downstream stage must not run while offline')
    monkeypatch.setattr(llm,'run_external_constrained_llm',forbidden)
    monkeypatch.setattr(orchestrator,'run_external_fusion',forbidden)
    session=store.create({'mode':'external_inference','dataset':known.query['analysis_contract']})
    session['flows']=[{'id':known.query['query_id']}]
    result=dict(context=known.context,sources={},unavailable=[],blockers={},reasoning={},fusion={},decision={})
    events=[]
    orchestrator.run_external_phase2(session,known.query,result,lambda *args:events.append(args))
    assert result['context']['evidence'] and result['decision']=={} and result['reasoning']=={}
    assert result['runtime_status']=={'trusted_context':'COMPLETED','llm':'OFFLINE','fusion':'WAITING'}
    assert ('reasoning','offline','Local constrained-LLM runtime is offline.') in events
    assert all(Path(path).resolve().is_relative_to(APP) for path in result['sources'].values())
    assert json.loads(Path(result['sources']['trusted_context_context']).read_text(encoding='utf-8'))['context_text']==known.context['context_text']

def test_preflight_health_and_model_routes_without_inference(monkeypatch):
    import io
    class Endpoint:
        def __init__(self,model):self.model=model;self.calls=[]
        def open(self,request,timeout):
            self.calls.append(request)
            if isinstance(request,str):
                value = {'data':[{'id':self.model,'max_model_len':5120}]} if request.endswith('/v1/models') else {}
                response = io.BytesIO(json.dumps(value).encode())
                response.status = 200
                return response
            raise AssertionError('Readiness must not POST to the chat-completions endpoint.')
    endpoint=Endpoint(frozen_runtime.MODEL)
    monkeypatch.setattr(llm.urllib.request,'build_opener',lambda *args:endpoint)
    result=llm.check_llm_runtime()
    assert result['status']=='READY' and result['readiness_verified'] and len(endpoint.calls)==2
    assert endpoint.calls==['http://127.0.0.1:8000/health','http://127.0.0.1:8000/v1/models']
    endpoint.model='wrong-model'
    assert llm.check_llm_runtime()['status']=='FAILED'
