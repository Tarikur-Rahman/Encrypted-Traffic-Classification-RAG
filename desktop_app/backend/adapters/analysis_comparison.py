"""Desktop-owned all-flow inference and durable, identity-bound comparison results."""
import json
from collections import Counter
from backend.config import ROOT
from backend.adapters.artifacts import digest
from backend.adapters.contracts import contract,transform
from backend.adapters.model_comparison import MODELS
from backend.utils.paths import analysis_directory

VERSION=1

def identity(session):
    from importlib.metadata import version
    value=contract(session['input']['dataset'])
    source=session['input']
    predictor_path=(source['path'] if source['extension']=='.csv' else analysis_directory(session['analysis_id'])/'native-flows.csv') if source.get('mode')=='external_inference' else ROOT/'data/processed'/value.dataset_id/(session['flows'][0]['source_split']+'.csv')
    saved_matrix=analysis_directory(session['analysis_id'])/'comparison-predictors.npy'
    if source.get('mode')=='external_inference' and saved_matrix.exists():predictor_path=saved_matrix
    return dict(version=VERSION,analysis_id=session['analysis_id'],input_hash=digest(session['input']['path']),
        predictor_file_hash=digest(predictor_path) if __import__('pathlib').Path(predictor_path).exists() else None,
        environment={name:version(name) for name in ['pandas','numpy','torch','lightgbm']},
        gpu_runtime='Ubuntu-24.04/ustc-gpu:python3.11.16/cuml26.8.0/cupy14.2.0/xgboost3.2.0',
        stage=session['input']['input_stage'] if 'input_stage' in session['input'] else 'model_features',
        model_run=value.spec.run,predictors=list(value.predictor_names),preprocessing=digest(value.preprocessing_path),
        models={key:digest(value.model_path.parent/file) if (value.model_path.parent/file).exists() else None for key,_,file in MODELS},
        implementation={file:digest(__import__('pathlib').Path(__file__).parent/file) for file in ['analysis_comparison.py','gpu_worker.py','gpu_bridge.py','contracts.py']},
        flows=[(f['id'],f['source_row_index'],f['predicted_class'],f['probabilities']) for f in session['flows']])

def cache_path(session):return analysis_directory(session['analysis_id'])/'five-model-comparison.json'

def read(session):
    path=cache_path(session)
    if not path.exists():return None
    result=json.loads(path.read_text(encoding='utf-8'))
    # Normalize tuples through JSON so reopened and in-memory identities agree.
    if result['identity']!=json.loads(json.dumps(identity(session))):
        raise ValueError('Comparison cache identity differs; create a new analysis to recompute safely.')
    return result

def matrix_for(session,value):
    import pandas as pd
    source=session['input'];flows=session['flows']
    if source.get('mode')=='external_inference':
        import numpy as np
        saved=analysis_directory(session['analysis_id'])/'comparison-predictors.npy'
        if saved.exists():
            matrix=np.load(saved,allow_pickle=False)
            if matrix.ndim!=2 or matrix.shape[1]!=value.predictor_count or not np.isfinite(matrix).all():raise ValueError('Invalid retained predictor cache')
            return matrix[[f['source_row_index'] for f in flows]]
        path=source['path'] if source['extension']=='.csv' else analysis_directory(session['analysis_id'])/'native-flows.csv'
        frame=pd.read_csv(path,low_memory=False,**({'float_precision':'round_trip'} if source['extension']=='.csv' else {}))
        return transform(frame.iloc[[f['source_row_index'] for f in flows]],value,source['input_stage'])
    splits={f['source_split'] for f in flows}
    if len(splits)!=1 or next(iter(splits)) not in {'validation','test'}:raise ValueError('Exact historical predictor rows are unavailable')
    frame=pd.read_csv(ROOT/'data/processed'/value.dataset_id/(next(iter(splits))+'.csv'),float_precision='round_trip')
    return transform(frame.iloc[[f['source_row_index'] for f in flows]],value,'model_features')

def predict(value,session,flows,matrix,key):
    import numpy as np
    if key in {'random_forest','xgboost'}:
        from backend.adapters.gpu_bridge import infer
        return infer(value,session,{'id':'EXT-'+session['analysis_id']+'-BATCH'},matrix,key,[f['id'] for f in flows])
    import torch
    from importlib import import_module
    if not torch.cuda.is_available():raise ValueError('Validated neural inference requires CUDA')
    module=import_module('src.models.'+value.spec.prefix+'_train_models')
    checkpoint=torch.load(value.model_path.parent/(key+'.pt'),map_location='cpu',weights_only=True)
    if checkpoint['model']!=key or tuple(checkpoint['feature_names'])!=value.predictor_names:raise ValueError('Neural model/predictor identity differs')
    if value.spec.prefix=='cic' and tuple(checkpoint['class_labels'])!=value.class_names:raise ValueError('Neural class mapping differs')
    model=module.build_neural_model(key,torch).to('cuda')
    try:
        model.load_state_dict(checkpoint['model_state_dict'],strict=True);model.eval()
        with torch.inference_mode():
            x=np.asarray(matrix,dtype=np.float32)
            if not np.isfinite(x).all():raise ValueError('Predictor overflow')
            logits=model(torch.from_numpy(x).unsqueeze(1).to('cuda'))
            if value.spec.prefix=='ustc':
                positive=torch.sigmoid(logits).reshape(-1).cpu().numpy()
                return np.column_stack([1-positive,positive]).astype(float)
            return torch.softmax(logits,dim=1).cpu().numpy().astype(float)
    finally:
        del model
        torch.cuda.empty_cache()

def run(session,stage):
    import numpy as np
    existing=read(session)
    if existing:
        stage('comparison','complete','Reused identity-verified comparison cache')
        return existing
    value=contract(session['input']['dataset']);flows=session['flows'];before=identity(session)
    if before['input_hash']!=session['input_hash']:raise ValueError('Input changed since retained inference')
    result=dict(identity=before,status='Running',total_flows=len(flows),model_run=value.spec.run,models=[],flows={f['id']:[] for f in flows})
    session['comparison_progress']=dict(status='Running',models=[])
    try:matrix=matrix_for(session,value);matrix_error=None
    except Exception as error:matrix=None;matrix_error=str(error)
    for key,name,filename in MODELS:
        summary=dict(model=key,name=name,retained=key=='lightgbm',status='Running',flows_evaluated=0,mean_confidence=None,agreement=None,class_distribution={},reason='')
        session['comparison_progress']['models'].append(summary)
        stage('comparison','running',f'{name}: running; {len(flows):,} flows')
        outputs=[]
        try:
            if key=='lightgbm':
                outputs=[dict(predicted_class=f['predicted_class'],confidence=f['confidence'],probabilities=f['probabilities']) for f in flows]
            else:
                if matrix_error:raise ValueError(matrix_error)
                for start in range(0,len(flows),2048):
                    batch=flows[start:start+2048]
                    p=np.asarray(predict(value,session,batch,matrix[start:start+2048],key))
                    if p.shape!=(len(batch),len(value.class_names)) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=1),1,atol=1e-5):raise ValueError('Invalid comparison probabilities')
                    for probabilities in p:
                        winner=int(np.argmax(probabilities))
                        outputs.append(dict(predicted_class=value.class_names[winner],confidence=float(probabilities[winner]),probabilities=dict(zip(value.class_names,map(float,probabilities)))))
                    stage('comparison','running',f'{name}: {len(outputs):,}/{len(flows):,} flows evaluated')
            summary.update(status='Completed',flows_evaluated=len(outputs),mean_confidence=sum(o['confidence'] for o in outputs)/len(outputs),
                agreement=sum(o['predicted_class']==f['predicted_class'] for o,f in zip(outputs,flows))/len(outputs),class_distribution=dict(Counter(o['predicted_class'] for o in outputs)))
        except Exception as error:
            # A failed model has no misleading partial aggregate or fabricated bars.
            outputs=[];summary.update(status='Failed',reason=str(error))
        for i,flow in enumerate(flows):
            result['flows'][flow['id']].append(dict(model=key,name=name,retained=key=='lightgbm',available=bool(outputs),status=summary['status'],reason=summary['reason'],
                **(outputs[i] if outputs else dict(predicted_class=None,confidence=None,probabilities=None))))
        result['models'].append(dict(summary))
    if identity(session)!=before:raise ValueError('Input/runtime identity changed during comparison')
    result['status']='Completed';path=cache_path(session);temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(result,allow_nan=False),encoding='utf-8');temp.replace(path)
    session['comparison_progress']=dict(status='Completed',models=result['models'])
    failed=[r['name'] for r in result['models'] if r['status']=='Failed']
    stage('comparison','complete','Comparison finished; '+('failed models: '+', '.join(failed) if failed else 'all five models available'))
    return result

def view(session,flow_id=None):
    result=read(session)
    if result is None:
        return dict(session.get('comparison_progress',dict(status='Unavailable',models=[])),total_flows=len(session.get('flows',[])),reason='No persisted Phase-I comparison is available for this older analysis.')
    if flow_id is not None:
        if flow_id not in result['flows']:raise KeyError('Unknown flow')
        return dict(status='Completed',flow_id=flow_id,model_run=result['model_run'],models=result['flows'][flow_id])
    return dict(status=result['status'],total_flows=result['total_flows'],model_run=result['model_run'],models=result['models'],
        single_flow=result['flows'][session['flows'][0]['id']] if result['total_flows']==1 else None)
