"""Read-only, per-flow baseline comparison. Never feeds retained classification."""
from backend.config import ROOT
from backend.adapters.contracts import contract, transform
from backend.adapters.artifacts import digest
from backend.utils.paths import analysis_directory

MODELS = [('random_forest','Random Forest','random_forest.pkl'),
          ('1d_cnn','1D CNN','1d_cnn.pt'), ('xgboost','XGBoost','xgboost.json'),
          ('lightgbm','LightGBM','lightgbm.txt'), ('cnn_lstm','CNN-LSTM','cnn_lstm.pt')]

def compare(session, flow_id, update=None):
    import numpy as np
    import pandas as pd
    flow=next((r for r in session.get('flows',[]) if r['id']==flow_id),None)
    if flow is None:
        raise KeyError('Unknown flow.')
    source=session['input']; value=contract(source['dataset'])
    matrix=None; reason='Comparison requires the exact external predictor row; historical outputs are not new model inference.'
    if source.get('mode')=='external_inference':
        path=source['path'] if source['extension']=='.csv' else analysis_directory(session['analysis_id'])/'native-flows.csv'
        if digest(source['path']) != session['input_hash']:
            raise ValueError('Source changed since retained inference; comparison blocked.')
        index=flow['source_row_index']
        frame=pd.read_csv(path,skiprows=range(1,index+1),nrows=1,float_precision='round_trip')
        matrix=transform(frame,value,source['input_stage'])
    result=[]
    for key,label,filename in MODELS:
        artifact=value.model_path.parent/filename
        row=dict(model=key,name=label,retained=key=='lightgbm',artifact=str(artifact.relative_to(ROOT)),
                 available=False,status='Unavailable',predicted_class=None,confidence=None,probabilities=None)
        if update:update(dict(row,status='Running',reason='Loading the frozen artifact for this predictor row.'))
        if key=='lightgbm':
            row.update(available=True,status='Retained result',predicted_class=flow['predicted_class'],
                       confidence=flow['confidence'],probabilities=flow['probabilities'],reason='Operational Phase-I result; unchanged.')
        elif not artifact.is_file():
            row['reason']='Current authoritative model artifact is missing.'
        elif matrix is None:
            row['reason']=reason
        elif key in {'random_forest','xgboost'}:
            try:
                from backend.adapters.gpu_bridge import infer
                prediction=infer(value,session,flow,matrix,key)
                row.update(available=True,status='Live inference',predicted_class=prediction['predicted_class'],
                    confidence=prediction['confidence'],probabilities=prediction['probabilities'],
                    runtime=prediction['runtime'],reason='Frozen artifact; identical predictor row; WSL ustc-gpu CUDA inference.')
            except Exception as error:
                row.update(status='Failed',reason=str(error))
        else:
            try:
                import torch
                if not torch.cuda.is_available():
                    raise ValueError('Validated neural inference requires CUDA; no CPU fallback.')
                from importlib import import_module
                module=import_module('src.models.'+value.spec.prefix+'_train_models')
                checkpoint=torch.load(artifact,map_location='cpu',weights_only=True)
                if checkpoint['model']!=key or tuple(checkpoint['feature_names'])!=value.predictor_names:
                    raise ValueError('Checkpoint model/predictor order differs from current contract.')
                if value.spec.prefix=='cic' and tuple(checkpoint['class_labels'])!=value.class_names:
                    raise ValueError('Checkpoint class mapping differs from current contract.')
                model=module.build_neural_model(key,torch).to('cuda')
                model.load_state_dict(checkpoint['model_state_dict'],strict=True); model.eval()
                x=np.asarray(matrix,dtype=np.float32)
                if not np.isfinite(x).all():
                    raise ValueError('Predictors overflow float32 input.')
                with torch.inference_mode():
                    logits=model(torch.from_numpy(x).unsqueeze(1).to('cuda'))
                    if value.spec.prefix=='ustc':
                        positive=float(torch.sigmoid(logits).reshape(-1)[0].cpu())
                        probabilities=np.array([1-positive,positive])
                    else:
                        probabilities=torch.softmax(logits,dim=1)[0].cpu().numpy().astype(float)
                if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(),1,atol=1e-5):
                    raise ValueError('Invalid model probability vector.')
                winner=int(np.argmax(probabilities))
                row.update(available=True,status='Live inference',predicted_class=value.class_names[winner],
                    confidence=float(probabilities[winner]),probabilities=dict(zip(value.class_names,map(float,probabilities))),
                    reason='Frozen checkpoint; identical predictor row; CUDA inference.',device='cuda')
                del model
            except Exception as error:
                row['reason']=str(error)
        result.append(row)
        if update:update(row)
    return dict(flow_id=flow_id,dataset=value.dataset_id,model_run=value.spec.run,models=result,
                retained_classifier='lightgbm',scope='Selected flow only; comparison never changes Phase II')


from concurrent.futures import ThreadPoolExecutor
from threading import RLock
from copy import deepcopy
_executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='model-comparison')
_lock=RLock()
_jobs={}

def comparison_job(session,flow_id):
    flow=next((r for r in session.get('flows',[]) if r['id']==flow_id),None)
    if flow is None:raise KeyError('Unknown flow.')
    key=(session['analysis_id'],flow_id,session.get('input_hash'))
    with _lock:
        if key not in _jobs:
            job=dict(flow_id=flow_id,model_run=flow['model_run'],status='Running',models=[
                dict(model=model,name=name,retained=model=='lightgbm',available=False,status='Queued',
                     predicted_class=None,confidence=None,probabilities=None,reason='Waiting for model inference.')
                for model,name,_ in MODELS])
            _jobs[key]=job
            def update(row):
                with _lock:
                    index=next(i for i,r in enumerate(job['models']) if r['model']==row['model'])
                    job['models'][index]=deepcopy(row)
            def work():
                try:
                    result=compare(session,flow_id,update)
                    with _lock:job.update(result,status='Completed')
                except Exception as error:
                    with _lock:
                        for row in job['models']:
                            if row['status'] in {'Queued','Running'}:row.update(status='Failed',reason=str(error))
                        job['status']='Failed'
            _executor.submit(work)
        return deepcopy(_jobs[key])
