import sys,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from backend.adapters import analysis_comparison as comparison,input_adapter,phase1_adapter
from backend.session import store
from backend.config import RUNTIME

def session():
    source=input_adapter.validate(str(RUNTIME/'regression-inputs/cic-raw.csv'),'CIC-Darknet2020','raw_flows')
    s=store.create(source);phase1_adapter.retained(s,lambda *a:None)
    return s

def test_all_flow_summary_cache_and_retained_isolation(monkeypatch):
    s=session();before=copy.deepcopy(s['flows']);calls=[]
    def predict(value,session,flows,matrix,key):
        calls.append((key,len(flows)))
        return np.array([[.7,.1,.1,.1],[.1,.7,.1,.1],[.1,.1,.7,.1]])[:len(flows)]
    monkeypatch.setattr(comparison,'predict',predict)
    result=comparison.run(s,lambda *a:None)
    assert len(calls)==4 and all(n==len(s['flows']) for _,n in calls)
    assert s['flows']==before
    aggregate=comparison.view(s)
    assert aggregate['single_flow'] is None
    assert all(m['flows_evaluated']==3 for m in aggregate['models'])
    assert aggregate['models'][0]['mean_confidence']==pytest.approx(.7)
    expected=sum(f['predicted_class']==p for f,p in zip(before,['Non-Tor','NonVPN','Tor']))/3
    assert aggregate['models'][0]['agreement']==expected
    store.save(s);store.CACHE.pop(s['analysis_id']);restored=store.get(s['analysis_id'])
    monkeypatch.setattr(comparison,'predict',lambda *a:pytest.fail('Duplicate inference'))
    comparison.run(restored,lambda *a:None)
    assert comparison.view(restored,before[2]['id'])['models'][0]['predicted_class']=='Tor'
    assert len(calls)==4

def test_failure_has_no_fabricated_summary(monkeypatch):
    s=session()
    monkeypatch.setattr(comparison,'predict',lambda *a:(_ for _ in ()).throw(RuntimeError('GPU unavailable')))
    result=comparison.run(s,lambda *a:None)
    assert all(m['status']=='Failed' and m['mean_confidence'] is None and m['flows_evaluated']==0 for m in result['models'] if not m['retained'])
    assert next(m for m in result['models'] if m['retained'])['status']=='Completed'

def test_single_flow_and_invalid_cache(monkeypatch):
    s=session();s['flows']=s['flows'][:1]
    monkeypatch.setattr(comparison,'predict',lambda *a:np.array([[.1,.2,.3,.4]]))
    comparison.run(s,lambda *a:None)
    assert comparison.view(s)['single_flow'][0]['predicted_class']=='VPN'
    path=comparison.cache_path(s);data=json.loads(path.read_text());data['identity']['version']=-1;path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='identity differs'):comparison.view(s)

def test_phase1_waits_for_comparison(monkeypatch):
    s={};events=[]
    monkeypatch.setattr(phase1_adapter,'retained',lambda s,stage:s.update(status='phase1_complete'))
    def run(s,stage):assert s['status']=='running';events.append('comparison')
    monkeypatch.setattr(comparison,'run',run)
    phase1_adapter.execute(s,lambda *a:None)
    assert events==['comparison'] and s['status']=='phase1_complete'

def test_comparison_uses_exact_retained_predictor_bytes(monkeypatch):
    from src.rag import materialize_phase1_interface as interface
    original=interface.infer;captured=[]
    def infer(root,spec,names,matrix):
        captured.append(matrix.tobytes())
        return original(root,spec,names,matrix)
    monkeypatch.setattr(interface,'infer',infer)
    s=session()
    matrix=comparison.matrix_for(s,comparison.contract(s['input']['dataset']))
    assert matrix.tobytes()==captured[0]
