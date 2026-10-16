"""Real frozen GPU artifacts plus identity/failure boundaries. No training."""
import sys,json,copy,subprocess,time
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.config import APP,RUNTIME
from backend.adapters import input_adapter,external_runtime,model_comparison,gpu_bridge
from backend.adapters.contracts import contract,transform
from backend.session import store
import pandas as pd

@pytest.fixture(scope='module',params=['USTC-TFC2016','CIC-Darknet2020'])
def case(request):
    dataset=request.param;prefix='ustc' if dataset.startswith('USTC') else 'cic'
    path=RUNTIME/f'regression-inputs/{prefix}-raw.csv'
    session=store.create(input_adapter.validate(str(path),dataset,'raw_flows'))
    external_runtime.classify(session,lambda *a:None)
    value=contract(dataset);matrix=transform(pd.read_csv(path,nrows=1,float_precision='round_trip'),value,'raw_flows')
    return session,value,matrix

@pytest.mark.parametrize('model',['random_forest','xgboost'])
def test_gpu_batch_matches_exact_single_flow(case,model):
    import numpy as np
    session,value,matrix=case;flow=session['flows'][0]
    single=gpu_bridge.infer(value,session,flow,matrix,model)
    batch=gpu_bridge.infer(value,session,{'id':'EXT-'+session['analysis_id']+'-BATCH'},np.repeat(matrix,2,axis=0),model,[flow['id'],flow['id']])
    np.testing.assert_allclose(batch,np.array([[single['probabilities'][label] for label in value.class_names]]*2),atol=1e-7,rtol=0)

@pytest.mark.parametrize('model',['random_forest','xgboost'])
def test_real_gpu_artifact_exact_row_and_probabilities(case,model):
    session,value,matrix=case;flow=session['flows'][0]
    result=gpu_bridge.infer(value,session,flow,matrix,model)
    assert result['status']=='PASS' and result['runtime']=='validated_gpu'
    assert result['python']==gpu_bridge.PYTHON
    assert result['flow_id']==flow['id']
    assert result['predicted_class'] in value.class_names
    assert set(result['probabilities'])==set(value.class_names)
    assert sum(result['probabilities'].values())==pytest.approx(1,abs=1e-5)
    assert result['confidence']==max(result['probabilities'].values())

def test_full_comparison_keeps_retained_flow_and_query(case):
    session,value,matrix=case;flow=session['flows'][0]
    before=copy.deepcopy(flow);query=external_runtime.record(session,flow['id'])
    changes=[];result=model_comparison.compare(session,flow['id'],changes.append)
    assert len(result['models'])==5
    assert all(row['available'] for row in result['models']),result
    assert [r['status'] for r in changes].count('Running')==5
    retained=next(r for r in result['models'] if r['retained'])
    assert retained['model']=='lightgbm' and retained['probabilities']==before['probabilities']
    assert flow==before and external_runtime.record(session,flow['id'])==query

def test_gpu_failure_propagates_without_substitution(case,monkeypatch):
    session,value,matrix=case
    def fail(*a,**k):raise subprocess.TimeoutExpired('validated_gpu',120)
    monkeypatch.setattr(gpu_bridge.subprocess,'run',fail)
    with pytest.raises(subprocess.TimeoutExpired):gpu_bridge.infer(value,session,session['flows'][0],matrix,'random_forest')

@pytest.mark.parametrize('field',['predictor_names','values','flow_id','artifact_sha256','preprocessing_sha256','model_run'])
def test_worker_rejects_identity_mismatch(field):
    records=sorted((RUNTIME/'gpu_manual_audit/bridge').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    sample=next(json.loads(p.read_text()) for p in records if json.loads(p.read_text()).get('returncode')==0)
    payload=copy.deepcopy(sample['request'])
    if field=='predictor_names':payload[field]=list(reversed(payload[field]))
    elif field=='values':payload[field][0][0]+=1
    else:payload[field]='MISMATCH'
    result=subprocess.run(sample['command'],input=json.dumps(payload),capture_output=True,text=True,timeout=120,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    assert result.returncode!=0
    response=json.loads(result.stdout)
    assert response['status']=='FAIL' and 'probabilities' not in response

def test_job_returns_queued_rows_without_blocking(case,monkeypatch):
    import threading
    session,_,_=case;session=copy.deepcopy(session);session['analysis_id']='EEEEEEEEEEEE'
    started=threading.Event();release=threading.Event()
    def work(session,flow,update):
        started.set();release.wait(5)
        return {'models':[]}
    monkeypatch.setattr(model_comparison,'compare',work)
    try:
        start=time.monotonic();result=model_comparison.comparison_job(session,session['flows'][0]['id'])
        assert time.monotonic()-start<1
        assert result['status']=='Running' and len(result['models'])==5
        assert all(r['status']=='Queued' for r in result['models'])
    finally:release.set()
