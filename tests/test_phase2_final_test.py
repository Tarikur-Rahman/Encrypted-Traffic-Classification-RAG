"""Synthetic V5 TEST pipeline tests. No real cohorts, labels, models or network."""
import base64
import copy
import json
from collections import Counter
from pathlib import Path
import pytest
from test_rag_v5_diagnostic import population, packet
from src.rag.final_test import contracts as c, preflight, upstream, llm, downstream, summaries
from src.rag import run_phase2_final_test as cli
from src.rag import positive_evidence_v5 as gate
from src.evaluation import finalize_v5_contract_v2_stage9 as fusion_projection
from src.evaluation import finalize_validation_phase2 as final_projection


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*a,**k):pytest.fail('Unexpected network')
    monkeypatch.setattr(llm.runtime,'post',forbidden)
    monkeypatch.setattr(upstream.interface,'infer',forbidden)
    monkeypatch.setattr(upstream.retrieval,'load_query_encoder',forbidden)


def historical():
    rows=[]
    for d,labels in c.CLASSES.items():
        offset=0
        for label in labels:
            for i in range(69 if label=='Tor' else 850):
                prefix='CIC' if d.startswith('CIC') else 'USTC'
                rows.append(dict(query_id=f'{prefix}-TEST-{offset:06d}',source_dataset=d,
                                 classifier_prediction=label,source_split='test',stratum=d+'/'+label,sampling_seed='42'))
                offset+=1
    rows.sort(key=lambda r:r['query_id'])
    for i,row in enumerate(rows,1):row['selection_ordinal']=str(i)
    return rows


def test_exact_outcome_blind_historical_selection():
    source=historical(); selected,excluded=c.select_cohort(source)
    assert len(selected)==len({r['query_id'] for r in selected})==4300
    assert Counter((r['source_dataset'],r['classifier_prediction']) for r in selected)==Counter(c.TARGETS)
    tor=[r for r in source if r['classifier_prediction']=='Tor']
    assert [r['query_id'] for r in excluded]==[r['query_id'] for r in tor[50:]]
    changed=[dict(r,true_class_label='ignored',classifier_correct=False,evidence_status='CONTRADICTED') for r in source]
    assert c.select_cohort(changed)==(selected,excluded)
    assert c.SCOPE=='FIXED_HISTORICAL_TEST_COMPARISON'


@pytest.mark.parametrize('mutation',['duplicate','order','split','seed','ordinal','short'])
def test_historical_selection_fail_closed(mutation):
    source=historical()
    if mutation=='duplicate':source[-1]=dict(source[0])
    elif mutation=='order':source.reverse()
    elif mutation=='split':source[0]['source_split']='validation'
    elif mutation=='seed':source[0]['sampling_seed']='43'
    elif mutation=='ordinal':source[0]['selection_ordinal']='2'
    else:source.pop()
    with pytest.raises(ValueError):c.select_cohort(source)


def test_namespace_and_frozen_evidence_reuse():
    assert c.BASE=='results/rag/test/final_4300_v5_contract_v2/'
    assert '/v5f_validation/' in c.CHUNKS and '/v5f_validation/' in c.DOCUMENTS
    assert all('/v5f_validation/' in p for p in preflight.INDEX_FILES)
    assert c.GT_PATH.startswith(c.LEGACY_BASE) and not c.GT_PATH.startswith(c.BASE)
    assert c.FROZEN['response_contract_version']=='v2'
    assert c.FROZEN['context_profile']=='evidence_diverse'
    assert c.FROZEN['max_chunks_per_document']==0
    assert c.FROZEN['fusion_policy']=='retain_classifier'


def test_test_runtime_records_current_validator_identity(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    current=c.sha(root,'src/rag/positive_evidence_v5.py')
    assert current=='8851dc8308e86043931c9cc18502c10e0e51f13d9eaec2d7b9e2f6601ae139a8'
    assert c.VALIDATION_VALIDATOR_SHA256==llm.v5.contract.FROZEN_VALIDATOR_SHA256
    assert current!=c.VALIDATION_VALIDATOR_SHA256
    recorded={}
    sha=c.sha
    monkeypatch.setattr(c,'committed',lambda *a:False)
    monkeypatch.setattr(c,'snapshot',lambda *a:[])
    monkeypatch.setattr(c,'sha',lambda unused,name:sha(root,name))
    monkeypatch.setattr(c,'configuration',lambda unused,name,values:recorded.update(values))
    c.begin(tmp_path,8,[])
    assert recorded['test_runtime_validator_sha256']==current


def test_preflight_reads_frozen_v5_completion_budget(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    saved=json.loads((root/c.V5_RUN/'configuration.json').read_text(encoding='utf-8'))
    identity={'path':c.LEGACY_QUERIES}
    monkeypatch.setattr(preflight,'rows',lambda *a:iter([dict(status='PASS',output_metadata_json=json.dumps([identity]))]))
    monkeypatch.setattr(preflight,'snapshot',lambda *a:[identity])
    hashes={**saved['input_sha256'],**saved['tokenizer_sha256'],c.BUNDLE:c.BUNDLE_SHA256}
    monkeypatch.setattr(preflight,'sha',lambda unused,name:hashes[name])
    settings=dict(input_metadata_json='[]',retrieval_profile='hypothesis_balanced',profile='evidence_diverse',
                  top_n='5',token_budget='2048',max_chunks_per_document='0',diversity_threshold='1.0',development_id=c.DEVELOPMENT_ID)
    monkeypatch.setattr(preflight,'config',lambda unused,name:dict(settings,profile='task_relevance') if name==c.RERANK_CONFIG else settings)
    monkeypatch.setattr(preflight,'evidence',lambda *a:None)
    checked=preflight.check_settings(root)
    assert 'max_completion_tokens' not in checked
    request=llm.v5.request(dict(prompt_text='synthetic',response_schema={}),{})
    assert checked['max_tokens']==request['max_tokens']==c.FROZEN['max_completion_tokens']==1024
    assert checked['max_model_len']==c.FROZEN['max_model_len']==5120
    monkeypatch.setattr(llm.runtime.qwen,'prompt_tokens',lambda unused,text:int(text))
    for tokens,status in ((4096,'PASS'),(4097,'FAIL')):
        audit=llm.runtime.token_audit([dict(query_id='synthetic',prompt_text=str(tokens))],None)
        assert audit['reserved_completion_tokens']==checked['max_tokens']
        assert audit['max_model_len']==checked['max_model_len']
        assert audit['gate_status']==status
    # A correctly signed but incompatible parent must still fail closed.
    for field in ('max_tokens','max_model_len'):
        altered=copy.deepcopy(saved);altered[field]+=1
        altered.pop('configuration_sha256')
        altered['configuration_sha256']=llm.runtime.digest(c.json_text(altered))
        class SavedConfiguration:
            def read_text(self,**kwargs):return json.dumps(altered)
        monkeypatch.setattr(preflight,'safe_path',lambda *a:SavedConfiguration())
        with pytest.raises(ValueError,match='Frozen V5 runtime differs'):
            preflight.check_settings(root)


def test_stage4_passes_explicit_test_scope(tmp_path,monkeypatch,phase2_case):
    from contextlib import nullcontext
    from types import SimpleNamespace
    query=dict(phase2_case.query,query_id='USTC-TEST-000001',source_split='test',rag_partition='final_test')
    monkeypatch.setattr(upstream,'require_stages',lambda *a:None)
    monkeypatch.setattr(upstream,'begin',lambda *a:[])
    monkeypatch.setattr(upstream,'frozen_inputs',lambda *a:[])
    monkeypatch.setattr(upstream.index.v1,'load_token_policy',lambda *a:None)
    monkeypatch.setattr(upstream.retrieval_v3,'load_index',lambda *a:SimpleNamespace(chunks=[]))
    monkeypatch.setattr(upstream.retrieval,'load_query_encoder',lambda *a:SimpleNamespace(device='cpu'))
    monkeypatch.setattr(upstream,'config',lambda *a:dict(dense_device='cpu'))
    monkeypatch.setattr(upstream,'queries',lambda *a:iter([query]))
    monkeypatch.setattr(upstream,'writer',lambda *a:nullcontext())
    class ReachedExplicitTestScope(Exception):pass
    def balanced(evidence,batch,encoder,policy,*,source_split):
        assert source_split=='test'
        assert batch==[query]
        upstream.retrieval_v3.hypotheses(query,source_split=source_split)
        raise ReachedExplicitTestScope()
    monkeypatch.setattr(upstream.retrieval_v3,'balanced_batch',balanced)
    with pytest.raises(ReachedExplicitTestScope):upstream.stage4(tmp_path)
    assert not list(tmp_path.iterdir())


def test_explicit_test_domain_preserves_validation_semantics(population):
    records,payloads,registry,verified,values,_=population
    q=records[0]['query_id']; original=payloads[q]
    gate.validate(values[q],original,registry,verified,method=llm.runtime.method)
    p=copy.deepcopy(original);p.update(query_id=q.replace('-VAL-','-TEST-'),source_split='test')
    v=dict(values[q],query_id=p['query_id'])
    with pytest.raises(ValueError):gate.validate(v,p,registry,verified,method=llm.runtime.method)
    gate.validate(v,p,registry,verified,method=llm.runtime.method,source_split='test')
    with pytest.raises(ValueError):gate.validate(values[q],original,registry,verified,method=llm.runtime.method,source_split='test')
    before=llm.v5.contract.admissible(original,registry,verified)
    assert llm.v5.contract.admissible(p,registry,verified,source_split='test')==before


@pytest.mark.parametrize('field',['ground_truth','true_class_label','classifier_correct','confidence'])
def test_no_evaluation_or_confidence_evidence(population,field):
    records,payloads,registry,verified,_,_=population
    p=copy.deepcopy(payloads[records[0]['query_id']]);p['query_id']=p['query_id'].replace('-VAL-','-TEST-');p['source_split']='test'
    p[field]='forbidden'
    with pytest.raises(ValueError):gate.qualifying(p,registry,verified,method=llm.runtime.method,source_split='test')


@pytest.mark.parametrize('status',['SUPPORTED','CONTRADICTED','INSUFFICIENT','CONFLICTING'])
def test_test_projection_never_overrides(status):
    q='USTC-TEST-000001'
    payload=dict(query_id=q,reasoning_decision='ABSTAIN',evidence_status=status,supported_class=None,cited_chunk_ids=[],explanation='linkage')
    record=dict(query_id=q,validation_status='PASS',response_contract_version='v2',validated_response=payload,
                derived_validator_payload=payload,raw_v2_model_response={'claim_ids':[]},validated_claim_linkage=[],
                model_explanation='Narrative — retained',narrative_validation_status='NOT_SEMANTICALLY_VALIDATED')
    context=dict(query_id=q,source_dataset='USTC-TFC2016',source_split='test',rag_partition='final_test',
                 classifier_prediction='benign',classifier_confidence='.1',classifier_probabilities='{}',evidence_records_json='[]')
    result=fusion_projection.fuse_accepted(context,record,source_split='test',response_reference=llm.EXECUTION+'validated_responses.jsonl')
    assert result['final_class']=='benign'
    final=final_projection.finalize(result,source_split='test',response_reference=c.rel(9))
    assert final['final_class']=='benign' and final['model_explanation']==record['model_explanation']
    record['validation_status']='REJECTED'
    with pytest.raises(ValueError):fusion_projection.fuse_accepted(context,record,source_split='test',response_reference='test')


def test_evaluation_cannot_read_truth_before_stage10(tmp_path,monkeypatch):
    def incomplete(*a):raise ValueError('Stage10 incomplete')
    def forbidden(*a,**k):pytest.fail('Ground truth opened before gate')
    monkeypatch.setattr(downstream,'require_stages',incomplete)
    monkeypatch.setattr(downstream,'rows',forbidden)
    with pytest.raises(ValueError,match='incomplete'):downstream.evaluate(tmp_path)


def test_cli_preflight_never_writes(tmp_path,monkeypatch):
    monkeypatch.setattr(preflight,'check',lambda *a,**k:dict(selected=4300,unique=4300))
    assert cli.main(['--project-root',str(tmp_path),'--stage','preflight'])==0
    assert not list(tmp_path.iterdir())


def test_raw_first_resume_never_reissues(tmp_path,monkeypatch,population):
    rows,payloads,registry,verified,_,_=population
    p=copy.deepcopy(payloads[rows[0]['query_id']]);p.update(query_id=p['query_id'].replace('-VAL-','-TEST-'),source_split='test')
    state=llm.v5.contract.admissible(p,registry,verified,source_split='test')
    value=dict(query_id=p['query_id'],evidence_status=state['evidence_status'],supported_class=state['supported_class'],
               reasoning_decision=state['reasoning_decision'],claim_ids=state['claim_ids'],
               cited_chunk_ids=sorted({k for v in state['claim_chunks'].values() for k in v}),explanation='Original narrative.')
    row=dict(query_id=p['query_id'],prompt_text='synthetic',response_schema=llm.v5.contract.schema(state,p['query_id']))
    cfg=dict(configuration_sha256='fixture',prompt_sha256={p['query_id']:llm.runtime.digest('synthetic')},
             token_audit=dict(prompt_tokens={p['query_id']:200}))
    body=packet(value);calls=[]
    monkeypatch.setattr(llm.runtime,'post',lambda b:(calls.append(b) or body))
    original=llm.v5.LEGACY_OUTCOMES
    def interrupt(*a,**k):raise KeyboardInterrupt()
    monkeypatch.setattr(llm.v5,'LEGACY_OUTCOMES',interrupt)
    with llm.runtime_scope():
        with pytest.raises(KeyboardInterrupt):llm.runtime.execute(tmp_path,[row],{p['query_id']:p},registry,verified,cfg)
        raw=list(llm.runtime.jsonl(tmp_path,llm.EXECUTION+llm.runtime.RAW))
        assert base64.b64decode(raw[0]['response_base64'])==body
        monkeypatch.setattr(llm.v5,'LEGACY_OUTCOMES',original)
        result=llm.runtime.execute(tmp_path,[row],{p['query_id']:p},registry,verified,cfg,resume=True)
        assert result['accepted']==1 and len(calls)==1
        saved=list(llm.runtime.jsonl(tmp_path,llm.EXECUTION+llm.runtime.VALIDATED))
        assert saved[0]['raw_v2_model_response']==value


def test_duplicate_durable_id_rejected(tmp_path):
    with llm.runtime_scope():
        path=tmp_path/(llm.EXECUTION+llm.runtime.RAW);path.parent.mkdir(parents=True)
        path.write_text('{"query_id":"q"}\n{"query_id":"q"}\n',encoding='utf-8')
        with pytest.raises(ValueError):llm.runtime.ledger(tmp_path,llm.runtime.RAW,{'q'})


def test_rejected_summary_never_counts_as_insufficient():
    result=llm.summary({'q':dict(validation_status='REJECTED',validated_response=None)})
    assert result['rejected']==1 and result['evidence_status_counts']=={}


def test_overall_metrics_are_posthoc_dataset_qualified(tmp_path,monkeypatch):
    # Synthetic perfect predictions across all six classes; no real TEST input.
    final=[];truth=[]
    for dataset,labels in c.CLASSES.items():
        for index,label in enumerate(labels):
            q=('USTC' if dataset.startswith('USTC') else 'CIC')+f'-TEST-{index:06d}'
            final.append(dict(query_id=q,source_dataset=dataset,final_class=label,classifier_confidence='0.9',
                              classifier_probabilities=json.dumps({k:.9 if k==label else .1/(len(labels)-1) for k in labels}),
                              evidence_status='INSUFFICIENT'))
            truth.append(dict(query_id=q,source_dataset=dataset,source_split='test',true_class_id=str(index),true_class_label=label))
    gates=[];outputs={}
    monkeypatch.setattr(downstream,'TOTAL',6)
    monkeypatch.setattr(downstream,'SOURCE_COUNTS',{d:len(labels) for d,labels in c.CLASSES.items()})
    monkeypatch.setattr(downstream,'require_stages',lambda root,*stages:gates.append(stages))
    monkeypatch.setattr(downstream,'snapshot',lambda *a:[])
    def read(root,name,*,evaluation=False):
        if name==c.GT_PATH:
            assert gates and 10 in gates[0] and evaluation
            return iter(truth)
        assert name==c.rel(10)
        return iter(final)
    monkeypatch.setattr(downstream,'rows',read)
    monkeypatch.setattr(downstream,'table',lambda root,name,data,*a:outputs.update({name:list(data)}))
    downstream.evaluate(tmp_path)
    metrics=outputs[c.rel(10,'test_evaluation_metrics.csv')]
    overall={r['metric']:r['value'] for r in metrics if r['source_dataset']=='OVERALL'}
    assert overall['total_support']==6
    assert all(v==1 for k,v in overall.items() if k!='total_support')
    assert {r['source_dataset'] for r in metrics}=={'OVERALL',*c.CLASSES}
    assert len(outputs[c.rel(10,'test_evaluation_confusion_matrix.csv')])==20
    assert not list(tmp_path.iterdir())
