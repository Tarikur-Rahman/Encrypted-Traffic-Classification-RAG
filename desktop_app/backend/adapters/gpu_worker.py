"""Single-row GPU inference in the existing ustc-gpu environment; no training."""
import os,sys,json,hashlib,contextlib
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[3]
APP=ROOT/'desktop_app'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(APP))
for key in ['CUPY_CACHE_DIR','NUMBA_CACHE_DIR','XDG_CACHE_HOME','TMPDIR']:
    directory=APP/'runtime/gpu-cache'/key.lower();directory.mkdir(parents=True,exist_ok=True);os.environ[key]=str(directory)

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def run(request):
    import numpy as np,pandas as pd,cupy as cp,xgboost as xgb,pickle
    from importlib.metadata import version
    from src.rag import materialize_phase1_interface as interface
    from importlib import import_module
    for package,expected in [('cuml-cu13','26.8.0'),('cupy-cuda13x','14.2.0'),('xgboost','3.2.0'),('pandas','3.0.3')]:
        if version(package)!=expected:raise ValueError(f'{package} version differs: {version(package)} != {expected}')
    if cp.cuda.runtime.getDeviceCount()<1:raise ValueError('No CUDA device available')
    spec=next(s for s in interface.DATASETS if s.name==request['dataset'])
    names=interface.predictor_order(ROOT,spec)
    if request['model'] not in {'random_forest','xgboost'}:raise ValueError('Unsupported GPU bridge model')
    if request['predictor_names']!=list(names) or request['model_run']!=spec.run:raise ValueError('Predictor order or run identity differs')
    if not request['flow_id'].startswith('EXT-'+request['analysis_id']+'-'):raise ValueError('Flow identity differs')
    matrix=np.asarray(request['values'],dtype='<f8')
    count=len(request.get('flow_ids',[request['flow_id']]))
    if not 1<=count<=2048 or matrix.shape!=(count,len(names)) or not np.isfinite(matrix).all():raise ValueError('Invalid predictor row')
    row_hash=hashlib.sha256(matrix.tobytes()).hexdigest()
    if row_hash!=request['row_sha256']:raise ValueError('Preprocessed predictor bytes differ')
    artifact=interface.model_path(ROOT,spec).parent/('random_forest.pkl' if request['model']=='random_forest' else 'xgboost.json')
    if sha(artifact)!=request['artifact_sha256'] or sha(ROOT/spec.parameters)!=request['preprocessing_sha256']:raise ValueError('Artifact or preprocessing identity differs')
    module=import_module('src.models.'+spec.prefix+'_train_models')
    frame=pd.DataFrame(matrix,columns=names)
    if request['model']=='random_forest':
        with artifact.open('rb') as stream:model=pickle.load(stream)
        if not type(model).__module__.startswith('cuml.'):raise ValueError('Artifact is not a cuML model')
        if int(model.n_features_in_)!=len(names):raise ValueError('Random Forest predictor count differs')
        if spec.prefix=='ustc':
            positive=module.cuml_probability(cp,model,module.gpu_features(cp,frame))
            p=np.column_stack([1-positive,positive]).astype(float)
        else:
            raw=model.predict_proba(cp.asarray(module.float32_values(frame)))
            if not isinstance(raw,cp.ndarray):raise ValueError('Random Forest returned host probabilities')
            p=module.ordered_probabilities(cp.asnumpy(raw),cp.asnumpy(cp.asarray(model.classes_)).tolist())
    elif spec.prefix=='ustc':
        model=xgb.XGBClassifier();model.load_model(artifact);model.set_params(device='cuda')
        if model.get_booster().num_features()!=len(names):raise ValueError('XGBoost predictor count differs')
        positive=module.xgboost_probability(cp,model,frame,int(model.best_iteration)+1)
        p=np.column_stack([1-positive,positive]).astype(float)
    else:
        model=xgb.Booster();model.load_model(artifact);model.set_param({'device':'cuda'})
        if model.num_features()!=len(names):raise ValueError('XGBoost predictor count differs')
        module.verify_xgboost_cuda(model)
        raw=module.xgboost_call(lambda:model.inplace_predict(cp.asarray(module.float32_values(frame))))
        if not isinstance(raw,cp.ndarray):raise ValueError('XGBoost returned host probabilities')
        p=module.ordered_probabilities(cp.asnumpy(raw),list(range(len(spec.labels))))
    if p.shape!=(count,len(spec.labels)) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(axis=1),1,atol=1e-5):raise ValueError('Invalid GPU probabilities')
    if sha(artifact)!=request['artifact_sha256']:raise ValueError('Artifact changed during inference')
    if 'flow_ids' in request:
        return dict(status='PASS',model=request['model'],flow_id=request['flow_id'],flow_ids=request['flow_ids'],row_sha256=row_hash,artifact_sha256=request['artifact_sha256'],batch_probabilities=p.tolist(),runtime='validated_gpu')
    p=p[0]
    winner=int(np.argmax(p))
    return dict(status='PASS',model=request['model'],flow_id=request['flow_id'],row_sha256=row_hash,
        artifact_sha256=request['artifact_sha256'],predicted_class=spec.labels[winner],confidence=float(p[winner]),
        probabilities=dict(zip(spec.labels,map(float,p))),runtime='validated_gpu',python=sys.executable,
        versions={name:version(name) for name in ['cuml-cu13','cupy-cuda13x','xgboost']},cuda_runtime=cp.cuda.runtime.runtimeGetVersion())

if __name__=='__main__':
    try:
        from backend.utils.write_guard import install
        install(APP)
        request=json.load(sys.stdin)
        with contextlib.redirect_stdout(sys.stderr):result=run(request)
        print(json.dumps(result,allow_nan=False))
    except Exception as error:
        print(json.dumps({'status':'FAIL','error':str(error),'type':type(error).__name__}));sys.exit(1)
