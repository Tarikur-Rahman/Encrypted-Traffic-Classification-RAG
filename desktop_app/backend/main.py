import os
import sys
sys.dont_write_bytecode = True
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
import secrets
import socket
import json
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from backend.config import ROOT, RUNTIME, DATASETS
from backend.adapters import input_adapter, phase1_adapter, phase2_adapter
from backend.adapters.artifacts import checklist
from backend.adapters.export_adapter import export_content
from backend.session import store

TOKEN = os.environ.get('PG_API_TOKEN') or secrets.token_urlsafe(32)
app = FastAPI(title='ProvenanceGuard ETI', docs_url=None, redoc_url=None, openapi_url=None)
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='research-adapter')
logging.basicConfig(filename=RUNTIME / 'backend.log', level=logging.INFO)

@app.middleware('http')
async def authenticate(request: Request, call_next):
    if not secrets.compare_digest(request.headers.get('authorization',''), 'Bearer '+TOKEN):
        return JSONResponse({'detail': 'Unauthorized local client.'}, status_code=401)
    return await call_next(request)

@app.exception_handler(ValueError)
async def bad_input(request, exception):
    return JSONResponse({'detail': str(exception)}, status_code=422)

@app.exception_handler(KeyError)
async def missing(request, exception):
    return JSONResponse({'detail': str(exception)}, status_code=404)

class InputRequest(BaseModel):
    path: str = Field(min_length=1)
    dataset: Literal['USTC-TFC2016','CIC-Darknet2020']
    completed_only: bool = True
    input_stage: Literal['auto','raw_flows','model_features'] = 'auto'

class InspectRequest(BaseModel):
    path: str = Field(min_length=1)
    input_stage: Literal['auto','raw_flows','model_features'] = 'auto'

class Phase2Request(BaseModel):
    mode: Literal['fresh','review'] = 'fresh'

class ExportRequest(BaseModel):
    format: Literal['json','csv']

class Health(BaseModel):
    status: str
    project_root: str
    contracts: list[dict]
    evidence: dict
    execution: str

@app.get('/api/health', response_model=Health)
def health():
    import importlib.util
    from src.rag.materialize_phase1_interface import DATASETS as specs
    return dict(status='ready', project_root=str(ROOT), execution='Read-only research adapters',
        contracts=[dict(dataset=s.name, model='LightGBM', model_run=s.run,
            model_present=(ROOT / f'models/{s.name}/{s.run}/lightgbm.txt').is_file(),
            inference_dependency=bool(importlib.util.find_spec('lightgbm')), sources=input_adapter.sources(s.name),
            pcap_inference=s.prefix=='ustc',
            external_csv=True, external_phase2='external trusted context, V5 constrained reasoning and retain-classifier fusion; requires expected local LLM server') for s in specs],
        evidence=dict(corpus=(ROOT/'data/processed/rag/evidence_chunks_v2.csv').is_file(),
                      index=(ROOT/'artifacts/rag/index_v2/dense_embeddings.npy').is_file(),
                      audit=checklist('metadata/rag/evidence_index_v2_validation_checklist.csv')))

@app.get('/api/llm/status')
def llm_status():
    from backend.external_phase2.llm_runtime import check_llm_runtime
    return check_llm_runtime()

@app.post('/api/input/validate')
def validate(value: InputRequest):
    return input_adapter.validate(value.path, value.dataset,value.input_stage)

@app.post('/api/input/inspect')
def inspect(value: InspectRequest):
    return input_adapter.inspect(value.path,value.input_stage)

@app.post('/api/analysis/create')
def create(value: InputRequest):
    source = input_adapter.validate(value.path, value.dataset,value.input_stage)
    if not source['compatible']:
        raise ValueError(source['reason'])
    session = store.create(source, value.completed_only)
    return {'analysis_id': session['analysis_id'], 'input': source}

def run_job(session, function, *args):
    try:
        function(session, *args, lambda key,status,message: store.stage(session,key,status,message))
    except Exception as error:
        logging.exception('Analysis %s failed', session['analysis_id'])
        with store.LOCK:
            session.update(status='error', error=('A required research dependency is unavailable: '+str(error)) if isinstance(error, ImportError) else str(error))
            for stage in session['stages']:
                if stage['status'] == 'running':
                    stage.update(status='error', message=session['error'])
    finally:
        store.save(session)

def submit(session, function, *args):
    with store.LOCK:
        if session['status'] == 'running':
            raise HTTPException(409, 'This analysis is already running.')
        session.update(status='running', error=None)
        store.save(session)
        EXECUTOR.submit(run_job, session, function, *args)
    return {'analysis_id': session['analysis_id'], 'status': 'running'}

@app.post('/api/analysis/{identifier}/phase1')
def phase1(identifier: str):
    session = store.get(identifier)
    if session.get('flows'):
        raise HTTPException(409, 'Phase I is already available. Create a new session to run it again.')
    return submit(session, phase1_adapter.execute)

@app.get('/api/analysis/{identifier}/status')
def status(identifier: str):
    session = store.get(identifier)
    with store.LOCK:
        return {**{k:session[k] for k in ('analysis_id','input','status','stages','error')},
            'selected_flow_id':session.get('selected_flow_id')}

@app.get('/api/analysis/{identifier}/summary')
def summary(identifier: str):
    return store.get(identifier).get('summary')

@app.get('/api/analysis/{identifier}/comparison')
def comparison_summary(identifier: str):
    from backend.adapters.analysis_comparison import view
    return view(store.get(identifier))

@app.get('/api/analysis/{identifier}/flows')
def flows(identifier: str, page: int=1, page_size: int=25, search: str='', predicted_class: str='',
          min_confidence: float=0, sort: str='id', descending: bool=False):
    if page < 1 or not 1 <= page_size <= 200 or not 0 <= min_confidence <= 1 or sort not in {'id','predicted_class','confidence'}:
        raise ValueError('Invalid flow table options.')
    session=store.get(identifier)
    data = [dict(r,phase2_completed=bool(session.get('phase2',{}).get(r['id'],{}).get('decision'))) for r in session.get('flows',[]) if search.lower() in (r['id']+' '+r['predicted_class']).lower()
            and (not predicted_class or r['predicted_class'] == predicted_class) and r['confidence'] >= min_confidence]
    data.sort(key=lambda r:r[sort], reverse=descending)
    return dict(total=len(data), page=page, page_size=page_size, rows=data[(page-1)*page_size:page*page_size])

@app.get('/api/analysis/{identifier}/flows/{flow_id}')
def flow(identifier: str, flow_id: str):
    return phase1_adapter.detail(store.get(identifier), flow_id)

@app.post('/api/analysis/{identifier}/flows/{flow_id}/phase2')
def phase2(identifier: str, flow_id: str, value: Phase2Request = Phase2Request()):
    session = store.get(identifier)
    if not any(r['id'] == flow_id for r in session.get('flows', [])):
        raise KeyError('Unknown selected flow.')
    with store.LOCK:
        if session['status'] == 'running':
            raise HTTPException(409, 'Wait for the current analysis to finish.')
        existing=session.get('phase2',{}).get(flow_id)
        if existing and existing.get('decision'):
            return {'analysis_id':identifier,'status':'complete','existing':True}
        if existing and existing.get('context'):
            raise HTTPException(409, 'Phase II already has preserved context. Resume the LLM stage instead.')
        session['selected_flow_id'] = flow_id
        for stage in session['stages']:
            if stage['id'] in {'query','corpus','index','retrieval','provenance','reranking','context','reasoning','fusion','decision'}:
                stage.update(status='pending', message='')
    if value.mode == 'review':
        if session['input']['mode']=='external_inference':
            raise ValueError('External flows have no historical results. Choose fresh analysis.')
        return submit(session, phase2_adapter.execute, flow_id)
    from backend.adapters.fresh_phase2 import execute
    return submit(session, execute, flow_id)

@app.post('/api/analysis/{identifier}/flows/{flow_id}/phase2/resume')
def resume_phase2(identifier: str, flow_id: str):
    session=store.get(identifier)
    if not any(row['id']==flow_id for row in session.get('flows',[])):
        raise KeyError('Unknown selected flow.')
    with store.LOCK:
        if session['status']=='running':
            raise HTTPException(409,'Wait for the current analysis to finish.')
        existing=session.get('phase2',{}).get(flow_id)
        if existing and existing.get('decision'):
            return {'analysis_id':identifier,'status':'complete','existing':True}
        if session['input']['mode']!='external_inference':
            raise ValueError('LLM-only resume is available for external runtime analyses.')
        session['selected_flow_id']=flow_id
    from backend.external_phase2.orchestrator import resume_external_llm
    return submit(session,resume_external_llm,flow_id)

@app.get('/api/analysis/{identifier}/flows/{flow_id}/{view}')
def phase2_view(identifier: str, flow_id: str, view: str):
    if view == 'comparison':
        from backend.adapters.analysis_comparison import view as comparison_view
        return comparison_view(store.get(identifier), flow_id)
    if view not in {'query','evidence','provenance','reranking','context','reasoning','decision','phase2'}:
        raise HTTPException(404, 'Unknown research view.')
    result = store.get(identifier).get('phase2', {}).get(flow_id)
    if not result:
        return None
    return result if view == 'phase2' else result.get({'evidence':'retrieval'}.get(view,view))

@app.get('/api/analysis/{identifier}/audit')
def audit(identifier: str):
    session = store.get(identifier)
    return {k:v for k,v in session.items() if k != 'flows'}

@app.post('/api/analysis/{identifier}/export')
def export(identifier: str, value: ExportRequest):
    session = store.get(identifier)
    if session['status'] == 'running':
        raise HTTPException(409, 'Wait for the analysis to finish before exporting.')
    # Backend returns data; only the native save dialog may select a new file.
    return dict(content=export_content(session, value.format), filename=f'ProvenanceGuard-{identifier}.{value.format}')

def main():
    import uvicorn
    from backend.config import APP
    from backend.utils.write_guard import install
    install(APP)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(('127.0.0.1', 0))
    sock.listen(128)
    print(json.dumps({'port':sock.getsockname()[1], 'ready':True}), flush=True)
    server = uvicorn.Server(uvicorn.Config(app, log_level='warning', access_log=False))
    server.run(sockets=[sock])

if __name__ == '__main__':
    main()
