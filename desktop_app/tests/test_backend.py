import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
from fastapi.testclient import TestClient
from backend.main import app, TOKEN
from backend.config import ROOT, RUNTIME
from backend.adapters import input_adapter, phase1_adapter, phase2_adapter, artifacts
from backend.adapters.export_adapter import export_content
from backend.session import store

client = TestClient(app, headers={'Authorization':'Bearer '+TOKEN})

def test_health_and_authentication():
    assert TestClient(app).get('/api/health').status_code == 401
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ready'
    assert len(response.json()['contracts']) == 2
    assert response.json()['evidence']['index']

def test_csv_contract_and_invalid_api_request():
    value = input_adapter.validate(str(ROOT / input_adapter.TEST_QUERIES), 'CIC-Darknet2020')
    assert value['compatible'] and value['mode'] == 'saved_test'
    assert client.post('/api/analysis/create', json={'path':'missing','dataset':'incorrect'}).status_code == 422
    with tempfile.TemporaryDirectory(dir=RUNTIME) as directory:
        path = Path(directory) / 'wrong.csv'
        path.write_text('unknown\n1\n')
        result = input_adapter.validate(str(path), 'CIC-Darknet2020')
        assert not result['compatible'] and result['missing_features']

def test_pcap_contract_fails_closed_without_touching_input():
    from backend.adapters.pcap_adapter import inspect_capture
    with tempfile.TemporaryDirectory(dir=RUNTIME) as directory:
        for extension in ['pcap','pcapng']:
            path = Path(directory) / ('capture.'+extension)
            path.write_bytes(b'capture-test')
            original = artifacts.digest(path)
            for dataset in ('USTC-TFC2016','CIC-Darknet2020'):
                result = inspect_capture(str(path), dataset)
                assert not result['compatible'] and result['reason']
            assert artifacts.digest(path) == original

def test_multiline_csv_seek_index_and_invalidation(monkeypatch):
    with tempfile.TemporaryDirectory(dir=RUNTIME) as directory:
        root = Path(directory)
        monkeypatch.setattr(artifacts, 'ROOT', root)
        path = root/'sample.csv'
        path.write_text('query_id,text\na,"first\nsecond"\nb,last\na,next\n', newline='')
        assert [r['text'] for r in artifacts.select('sample.csv','query_id','a')] == ['first\nsecond','next']
        path.write_text('query_id,text\na,changed\n')
        assert artifacts.select('sample.csv','query_id','a')[0]['text'] == 'changed'

def test_protected_file_comparison():
    spec = importlib.util.spec_from_file_location('verifier',APP/'scripts/verify_main_project_unchanged.py')
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.compare({'a':1,'b':2},{'a':3,'c':4}) == {'added':['c'],'removed':['b'],'changed':['a']}
    baseline = json.loads((APP/'protected-baseline.json').read_text())
    assert len(baseline['files']) > 27000
    assert all(not name.startswith('desktop_app/') for name in baseline['files'])

def test_real_saved_phase1_phase2_and_export():
    # Integration against real immutable research artifacts, not invented results.
    source = input_adapter.validate(str(ROOT/input_adapter.TEST_QUERIES),'CIC-Darknet2020')
    session = store.create(source)
    update = lambda k,s,m: store.stage(session,k,s,m)
    phase1_adapter.execute(session, update)
    assert session['summary']['total_flows'] == 2600
    assert abs(sum(session['flows'][0]['probabilities'].values())-1) < 1e-8
    flow_id = session['flows'][0]['id']
    result = phase2_adapter.execute(session, flow_id, update)
    assert result['decision']['final_class']
    assert len(result['retrieval']) == 30
    assert len(result['reranking']) == 10
    assert result['context']['evidence']
    assert 'raw_model_response' not in result['reasoning']
    assert result['decision']['evidence_status'] in {'SUPPORTED','INSUFFICIENT','CONTRADICTED','CONFLICTING'}
    store.save(session)
    identifier = session['analysis_id']
    store.CACHE.pop(identifier)
    assert store.get(identifier)['phase2'][flow_id]['decision'] == result['decision']
    response = client.get(f'/api/analysis/{identifier}/flows', params={'page_size':10,'min_confidence':.9,'sort':'confidence','descending':True})
    assert response.status_code == 200 and len(response.json()['rows']) == 10
    assert all(r['confidence']>=.9 for r in response.json()['rows'])
    assert client.get(f'/api/analysis/{identifier}/flows',params={'sort':'unknown'}).status_code == 422
    assert client.get('/api/analysis/INVALID/status').status_code == 404
    assert 'analysis_id' in json.loads(export_content(session,'json'))
    csv_text = export_content(session,'csv')
    assert csv_text.startswith('analysis_id,flow_id,dataset') and flow_id in csv_text
    assert client.post(f'/api/analysis/{identifier}/export',json={'format':'json'}).status_code == 200
    assert client.post(f'/api/analysis/{identifier}/export',json={'format':'exe'}).status_code == 422

def test_existing_phase1_inference_function():
    # The exact existing inference function is called on canonical validation
    # features. No GUI classifier/probability calculation is implemented.
    from dataclasses import replace
    from src.rag import materialize_phase1_interface as interface
    import numpy as np
    for original in interface.DATASETS:
        features = interface.predictor_order(ROOT, original)
        row = next(artifacts.rows(ROOT/f'data/processed/{original.name}/validation.csv'))
        x = np.array([[float(row[f]) for f in features]])
        p = interface.infer(ROOT, replace(original,count=1),features,x)
        observed = interface.observation(original,features,0,x[0],p[0])
        prefix = original.prefix
        saved = next(artifacts.rows(ROOT/f'results/rag/interface/{prefix}_validation_interface.csv'))
        assert observed['predicted_class_label'] == saved['predicted_class_label']
        assert abs(observed['confidence']-float(saved['confidence'])) < 1e-8

def test_csv_formula_escaping():
    value = dict(analysis_id='id',input={'dataset':'dataset'},flows=[dict(id='=SUM(1)',predicted_class='safe',confidence=.5)],phase2={})
    assert "'=SUM(1)" in export_content(value,'csv')
