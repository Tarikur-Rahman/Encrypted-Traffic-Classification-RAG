"""Real frozen-artifact regressions. Fixtures and outputs stay inside desktop_app."""
import csv
import json
from itertools import islice
from pathlib import Path
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.config import APP, ROOT, RUNTIME
import pandas as pd
from backend.adapters import artifacts, input_adapter, external_runtime, fresh_phase2
from backend.adapters.contracts import contract, transform
from backend.session import store
from backend.utils.paths import output_path

@pytest.mark.parametrize('dataset',['USTC-TFC2016','CIC-Darknet2020'])
def test_external_csv_regression(dataset):
    value = contract(dataset)
    original = list(islice(artifacts.rows(ROOT/f'data/processed/{dataset}/validation.csv'),3))
    saved = list(islice(artifacts.rows(ROOT/f'results/rag/interface/{value.spec.prefix}_validation_interface.csv'),3))
    with tempfile.TemporaryDirectory(dir=RUNTIME) as directory:
        source = Path(directory)/'new-features.csv'
        frame = pd.DataFrame(original)[list(value.predictor_names)]
        frame.to_csv(source,index=False)
        assert input_adapter.validate(str(source),dataset)['status']=='STAGE REQUIRED'
        inspected = input_adapter.validate(str(source),dataset,'model_features')
        session = store.create(inspected)
        before = artifacts.digest(source)
        external_runtime.classify(session,lambda *args:None)
        assert artifacts.digest(source)==before
        for observed,expected in zip(session['flows'],saved):
            record = external_runtime.record(session,observed['id'])
            assert observed['predicted_class']==expected['predicted_class_label']
            assert abs(observed['confidence']-float(expected['confidence']))<1e-8
            for label,probability in json.loads(expected['class_probabilities_json']).items():
                assert abs(observed['probabilities'][label]-probability)<1e-8
            assert json.loads(record['traffic_representation_json'])==json.loads(expected['traffic_representation_json'])
            assert record['source_dataset']=='external' and record['source_split']=='external_inference'
            assert record['analysis_contract']==dataset
            assert not {'true_class_id','true_class_label','classifier_correct'}.intersection(record)
            assert 'VAL-' not in record['query_text'] and 'development_validation' not in record['traffic_provenance_json']
        assert Path(session['external_records']).is_relative_to(APP)

@pytest.mark.parametrize('dataset',['USTC-TFC2016','CIC-Darknet2020'])
def test_real_raw_transform_regression(dataset):
    value = contract(dataset)
    reference = next(artifacts.rows(ROOT/f'data/processed/{dataset}/validation.csv'))
    raw_path = ROOT/('data/interim/USTC-TFC2016/flows.csv' if value.spec.prefix=='ustc' else 'data/raw/CIC-Darknet2020/Darknet.CSV')
    identity = 'source_pcap' if value.spec.prefix=='ustc' else 'Flow ID'
    if value.spec.prefix=='cic':
        with pd.read_csv(raw_path,chunksize=5000,float_precision='round_trip') as reader:
            raw = next(chunk[chunk[identity]==reference[identity]].iloc[0].to_dict() for chunk in reader if (chunk[identity]==reference[identity]).any())
    else:
        raw = next(row for row in artifacts.rows(raw_path) if row[identity]==reference[identity])
    matrix = transform(pd.DataFrame([raw]),value,'raw_flows')
    expected = np.array([[float(reference[f]) for f in value.predictor_names]])
    np.testing.assert_allclose(matrix,expected,atol=1e-10,rtol=1e-10)
    from dataclasses import replace
    from src.rag.materialize_phase1_interface import infer
    probability = infer(ROOT,replace(value.spec,count=1),list(value.predictor_names),matrix)[0]
    saved = next(artifacts.rows(ROOT/f'results/rag/interface/{value.spec.prefix}_validation_interface.csv'))
    np.testing.assert_allclose(probability,list(json.loads(saved['class_probabilities_json']).values()),atol=1e-8,rtol=0)
    with tempfile.TemporaryDirectory(dir=RUNTIME) as directory:
        source=Path(directory)/'new-raw-flow.csv'
        pd.DataFrame([raw])[list(value.raw_names)].to_csv(source,index=False)
        session=store.create(input_adapter.validate(str(source),dataset,'raw_flows'))
        external_runtime.classify(session,lambda *args:None)
        observed=session['flows'][0]
        assert observed['predicted_class']==saved['predicted_class_label']
        assert abs(observed['confidence']-float(saved['confidence']))<1e-8

def test_invalid_features_and_output_containment():
    value=contract('USTC-TFC2016')
    with pytest.raises(ValueError,match='Missing required'):
        transform(pd.DataFrame({'unknown':[1]}),value,'raw_flows')
    with pytest.raises(ValueError,match='finite'):
        transform(pd.DataFrame([{f:float('nan') for f in value.predictor_names}]),value,'model_features')
    for path in [ROOT/'results/forbidden.csv',APP/'../src/forbidden.py',APP]:
        with pytest.raises((ValueError,RuntimeError)):
            output_path(path)
    assert output_path(RUNTIME/'exports/result.csv').is_relative_to(APP)
    # Original CIC's duplicated Label metadata is explicitly established by
    # its pandas reader; duplicates in model predictors remain forbidden.
    source=input_adapter.validate(str(ROOT/'data/raw/CIC-Darknet2020/Darknet.CSV'),'CIC-Darknet2020','raw_flows')
    assert source['compatible'],source['reason']

def test_python_write_guard_and_caches():
    import subprocess,os
    code="import backend.config; from backend.utils.write_guard import install; install(backend.config.APP); (backend.config.ROOT/'forbidden-test-output').write_text('must be blocked')"
    process=subprocess.run([sys.executable,'-B','-c',code],cwd=APP,capture_output=True,text=True)
    assert process.returncode!=0 and 'blocked a write outside desktop_app' in process.stderr
    assert not (ROOT/'forbidden-test-output').exists()
    for variable in ('HF_HOME','TRANSFORMERS_CACHE','TORCH_HOME','TEMP','TMP'):
        assert Path(os.environ[variable]).resolve().is_relative_to(APP)

def test_fresh_deterministic_phase2_regression():
    from src.rag.run_rag_retrieval import semantic_profile
    for prefix in ['ustc','cic']:
        row = next(artifacts.rows(ROOT/f'results/rag/interface/{prefix}_validation_interface.csv'))
        row.update(semantic_profile(row['source_dataset'],row['predicted_class_label'],row['traffic_representation_json']))
        result={}
        fresh_phase2.deterministic(row,lambda *args:None,result)
        query_id=row['query_id']
        expected_query=artifacts.select('results/rag/v5f_validation/stage04_retrieval/query_records.csv','query_id',query_id)[0]
        assert row['query_text']==expected_query['query_text']
        assert row['retrieval_query_text']==expected_query['retrieval_query_text']
        expected=artifacts.select('results/rag/v5f_validation/stage04_retrieval/retrieval_results.csv','query_id',query_id)
        order=lambda data: [(r['retrieval_method'],int(r['rank']),r['chunk_id']) for r in data]
        assert order(result['retrieval'])==order(expected)
        verified={r['chunk_id']:r for r in artifacts.rows(ROOT/'results/rag/v5f_validation/stage05_provenance/provenance_chunk_records.csv')}
        assert all(r==verified[r['chunk_id']] for r in result['provenance'])
        expected=artifacts.select('results/rag/v5f_validation/stage06_reranking/reranked_evidence.csv','query_id',query_id)
        assert [r['chunk_id'] for r in result['reranking']]==[r['chunk_id'] for r in expected]
        expected=artifacts.select('results/rag/v5f_validation/stage07_trusted_contexts/trusted_context_records.csv','query_id',query_id)[0]
        assert result['context']['selected_chunk_ids']==expected['selected_chunk_ids']
        assert result['context']['context_text']==expected['context_text']
        from src.rag.materialize_constrained_llm_experiment import prompt_record
        prompt=prompt_record(result['context'],'v4')
        assert json.loads(prompt['allowed_chunk_ids'])==json.loads(expected['selected_chunk_ids'])

def test_external_phase2_preserves_scope():
    from src.rag.materialize_phase1_interface import infer
    from dataclasses import replace
    value=contract('CIC-Darknet2020')
    row=next(artifacts.rows(ROOT/'data/processed/CIC-Darknet2020/validation.csv'))
    x=np.array([[float(row[f]) for f in value.predictor_names]])
    p=infer(ROOT,replace(value.spec,count=1),list(value.predictor_names),x)
    query=external_runtime.build_observation(value,'regression',0,x[0],p[0],{'source_dataset':'external'})
    result={}
    fresh_phase2.deterministic(query,lambda *args:None,result)
    assert len(result['retrieval'])==30 and len(result['reranking'])==10
    assert all(r['source_dataset']=='external' for r in result['reranking'])
    assert result['context']['source_split']=='external_inference'
    assert result['context']['evaluation_scope']=='external_runtime'
    assert result['context']['evidence']

def test_validation_orchestration_and_unavailable_llm(monkeypatch):
    source=input_adapter.validate(str(ROOT/'results/rag/interface/cic_validation_interface.csv'),'CIC-Darknet2020')
    saved=next(artifacts.rows(source['path']))
    session=store.create(source)
    session['flows']=[dict(id=saved['query_id'],predicted_class=saved['predicted_class_label'],
        confidence=float(saved['confidence']),source_split='validation')]
    def unavailable(*args,**kwargs):
        raise ConnectionRefusedError('No local server')
    monkeypatch.setattr(fresh_phase2.socket,'create_connection',unavailable)
    result=fresh_phase2.execute(session,saved['query_id'],lambda *args:None)
    assert result['query']['query_id']==saved['query_id']
    assert result['context']['evidence'] and len(result['retrieval'])==30
    assert result['unavailable']==['reasoning','fusion','decision']
    assert 'Local constrained LLM server unavailable' in result['blockers']['reasoning']
    assert result['reasoning']=={} and result['decision']=={} and session['status']=='partial'

def test_real_pcap_inference():
    path=ROOT/'data/raw/USTC-TFC2016/Benign/Facetime.pcap'
    before=artifacts.digest(path)
    source=input_adapter.validate(str(path),'USTC-TFC2016')
    assert source['compatible'],source['reason']
    assert not input_adapter.validate(str(path),'CIC-Darknet2020')['compatible']
    session=store.create(source)
    external_runtime.classify(session,lambda *args:None)
    assert session['flows'] and session['status']=='phase1_complete'
    record=external_runtime.record(session,session['flows'][0]['id'])
    provenance=json.loads(record['traffic_provenance_json'])
    assert provenance['extractor']['version']=='6.6.0'
    assert provenance['source_dataset']=='external'
    assert artifacts.digest(path)==before
