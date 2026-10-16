"""Windows-to-WSL bridge for the unchanged, authoritative CUDA models."""
import hashlib,json,subprocess,uuid
from pathlib import Path
from backend.config import APP,ROOT
from backend.adapters.artifacts import digest

PYTHON='/home/dark_bird/miniforge3/envs/ustc-gpu/bin/python'
DISTRIBUTION='Ubuntu-24.04'

def linux_path(path):
    path=Path(path).resolve()
    return '/mnt/'+path.drive[0].lower()+'/'+path.as_posix()[3:]

def infer(value,session,flow,matrix,key,flow_ids=None):
    import numpy as np
    artifact=value.model_path.parent/('random_forest.pkl' if key=='random_forest' else 'xgboost.json')
    values=np.asarray(matrix,dtype='<f8')
    payload=dict(model=key,dataset=value.dataset_id,model_run=value.spec.run,analysis_id=session['analysis_id'],
        flow_id=flow['id'],predictor_names=list(value.predictor_names),values=values.tolist(),
        row_sha256=hashlib.sha256(values.tobytes()).hexdigest(),artifact_sha256=digest(artifact),
        preprocessing_sha256=digest(value.preprocessing_path))
    if flow_ids is not None:payload['flow_ids']=flow_ids
    command=['wsl.exe','-d',DISTRIBUTION,'--',PYTHON,'-B',linux_path(APP/'backend/adapters/gpu_worker.py')]
    directory=APP/'runtime/gpu_manual_audit/bridge';directory.mkdir(parents=True,exist_ok=True)
    record=dict(request=payload,command=command)
    try:
        process=subprocess.run(command,input=json.dumps(payload,allow_nan=False),capture_output=True,text=True,timeout=120,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        record.update(returncode=process.returncode,stdout=process.stdout,stderr=process.stderr)
        result=json.loads(process.stdout)
        if process.returncode or result.get('status')!='PASS':raise RuntimeError(result.get('error') or process.stderr)
        for field in ['model','flow_id','row_sha256','artifact_sha256']:
            if result.get(field)!=payload[field]:raise ValueError('GPU response identity differs: '+field)
        if flow_ids is not None:
            p=np.asarray(result['batch_probabilities'],dtype=float)
            if result.get('flow_ids')!=flow_ids or p.shape!=(len(flow_ids),len(value.class_names)) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=1),1,atol=1e-5):raise ValueError('Invalid GPU batch response')
            return p
        p=result['probabilities']
        if set(p)!=set(value.class_names):raise ValueError('GPU class mapping differs')
        probabilities=np.array([p[label] for label in value.class_names],dtype=float)
        if not np.isfinite(probabilities).all() or (probabilities<0).any() or (probabilities>1).any() or not np.isclose(probabilities.sum(),1,atol=1e-5):raise ValueError('Invalid GPU response probabilities')
        label=value.class_names[int(np.argmax(probabilities))]
        if result['predicted_class']!=label or abs(result['confidence']-p[label])>1e-8:raise ValueError('GPU result winner differs')
        return result
    except Exception as error:
        record['error']=str(error);raise
    finally:
        (directory/(uuid.uuid4().hex+'.json')).write_text(json.dumps(record,indent=2),encoding='utf-8')
