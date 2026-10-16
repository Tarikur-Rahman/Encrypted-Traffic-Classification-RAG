"""Resume/idempotency tests for the external Phase-II boundary."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from backend.main import TOKEN,app
from backend.session import store
from backend.external_phase2 import orchestrator

client=TestClient(app,headers={'Authorization':'Bearer '+TOKEN})

def external_session():
    session=store.create({'mode':'external_inference','dataset':'USTC-TFC2016'})
    flow_id='EXT-RESUME-FLOW'
    session['status']='partial'
    session['flows']=[{'id':flow_id}]
    session['phase2'][flow_id]={
        'query':{'query_id':flow_id},
        'retrieval':[{'chunk_id':'KEEP-RETRIEVAL'}],
        'provenance':[{'chunk_id':'KEEP-PROVENANCE'}],
        'reranking':[{'chunk_id':'KEEP-RERANKING'}],
        'context':{'context_status':'READY','context_text':'preserved'},
        'reasoning':{},'fusion':{},'decision':{},
        'runtime_status':{'trusted_context':'COMPLETED','llm':'OFFLINE','fusion':'WAITING'},
        'unavailable':['reasoning','fusion','decision'],
        'blockers':{'reasoning':'offline','fusion':'offline','decision':'offline'},
        'sources':{'retrieval':'original-retrieval','context':'original-context'},
    }
    return session,flow_id

def test_llm_resume_preserves_all_deterministic_phase2_results(monkeypatch):
    session,flow_id=external_session()
    before={key:session['phase2'][flow_id][key] for key in ('retrieval','provenance','reranking','context')}
    called=[]
    def llm_boundary(actual_session,query,result,stage):
        called.append((actual_session,query,result))
        result['runtime_status'].update(llm='COMPLETED',fusion='COMPLETED')
        result['decision']={'final_class':'benign'}
    monkeypatch.setattr(orchestrator,'run_external_phase2',llm_boundary)
    result=orchestrator.resume_external_llm(session,flow_id,lambda *args:None)
    assert len(called)==1
    assert all(result[key] is before[key] for key in before)
    assert result['sources']['retrieval']=='original-retrieval'
    assert result['decision']['final_class']=='benign'

def test_completed_phase2_resume_is_idempotent_and_does_not_submit(monkeypatch):
    session,flow_id=external_session()
    session['phase2'][flow_id]['decision']={'final_class':'benign'}
    def forbidden(*args,**kwargs):
        raise AssertionError('A completed Phase II must not be submitted again')
    monkeypatch.setattr('backend.main.submit',forbidden)
    response=client.post(f'/api/analysis/{session["analysis_id"]}/flows/{flow_id}/phase2/resume')
    assert response.status_code==200
    assert response.json()=={'analysis_id':session['analysis_id'],'status':'complete','existing':True}
