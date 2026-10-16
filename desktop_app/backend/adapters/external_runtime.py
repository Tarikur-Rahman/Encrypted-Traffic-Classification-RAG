"""External input orchestration; original computations, explicit external identity."""
from dataclasses import replace
from datetime import datetime, timezone
import csv
import json
from backend.config import ROOT, APP
from backend.adapters.artifacts import digest
from backend.adapters.contracts import contract, transform, extraction_options
from backend.utils.paths import analysis_directory, output_path

def build_observation(value, analysis_id, index, features, probabilities, provenance):
    from src.rag import materialize_phase1_interface as interface
    from src.rag.run_rag_retrieval import semantic_profile
    # Reuse the computation/template kernel. Its temporary schema identity is
    # replaced before persistence or downstream calls. It does not read a split.
    row = interface.observation(value.spec,list(value.predictor_names),index,features,probabilities)
    identifier = f'EXT-{analysis_id}-{index:06d}'
    query = json.loads(row['query_text'])
    query.update(source_dataset='external',analysis_contract=value.dataset_id,evaluation_scope='external_runtime',
                 traffic_representation_reference=identifier+':traffic_representation_json')
    row.update(query_id=identifier,source_dataset='external',analysis_contract=value.dataset_id,
               source_split='external_inference',rag_partition='external_inference',evaluation_scope='external_runtime',
               traffic_provenance_json=interface.encode(dict(provenance,source_row_index=index)),
               query_text=interface.encode(query))
    row.update(semantic_profile(value.dataset_id,row['predicted_class_label'],row['traffic_representation_json']))
    return row

def classify(session, stage):
    import pandas as pd
    from src.rag import materialize_phase1_interface as interface
    from backend.adapters.input_adapter import validate
    source = session['input']
    stage('input','running','Inspecting external source and computing SHA-256')
    checked = validate(source['path'],source['dataset'],source.get('input_stage','auto'))
    if not checked['compatible']:
        raise ValueError(checked['reason'])
    source.update(checked)
    source_hash = digest(source['path'])
    session['input_hash'] = source_hash
    stage('input','complete','Read-only external input; source hash recorded')
    value = contract(source['dataset'])
    directory = analysis_directory(session['analysis_id'])
    extractor = None
    if source['extension'] != '.csv':
        stage('extraction','running','Extracting native USTC flows with frozen NFStream 6.6.0 options')
        import subprocess, sys, os
        options = extraction_options()
        extracted = output_path(directory/'native-flows.csv')
        # A module host ensures Npcap registration is repeated by Windows spawn.
        # Native extraction is isolated so a failed worker cannot hang the API.
        log = output_path(directory/'extraction.log')
        with log.open('x',encoding='utf-8') as output:
            worker = subprocess.Popen([sys.executable,'-B','-m','backend.adapters.capture_worker',source['path'],str(extracted)],
                cwd=APP,env=os.environ.copy(),stdout=output,stderr=output,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
            try:
                code = worker.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                if os.name=='nt':
                    subprocess.run(['taskkill','/PID',str(worker.pid),'/T','/F'],capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    worker.kill()
                worker.wait()
                raise ValueError('NFStream extraction exceeded 30 minutes; see '+str(log))
        if code:
            raise ValueError('NFStream extraction failed; see '+str(log))
        frame = pd.read_csv(extracted,low_memory=False) if extracted.stat().st_size else pd.DataFrame()
        extractor = dict(name='NFStreamer',version='6.6.0',configuration=options)
        stage('extraction','complete',f'{len(frame):,} native flows extracted')
    else:
        stage('extraction','not_applicable','Input is a flow CSV')
        frame = pd.read_csv(source['path'],low_memory=False,float_precision='round_trip')
    if frame.empty:
        raise ValueError('The selected input contains no flows.')
    stage('schema','running','Validating extracted or selected feature columns')
    required = value.raw_names if source['input_stage']=='raw_flows' else value.predictor_names
    missing = [f for f in required if f not in frame]
    if missing:
        raise ValueError('Missing contract columns: '+', '.join(missing))
    stage('schema','complete',f'{value.predictor_count} required model predictors present')
    stage('preprocessing','running','Applying saved TRAIN statistics; no fit')
    matrix = transform(frame,value,source['input_stage'])
    # Persist exactly the bytes supplied to retained inference for comparison.
    # This is a desktop-owned runtime output, never a preprocessing artifact.
    import numpy as np
    np.save(output_path(directory/'comparison-predictors.npy'),matrix,allow_pickle=False)
    stage('preprocessing','complete','Frozen transform applied' if source['input_stage']=='raw_flows' else 'Explicitly declared frozen model features validated')
    stage('classification','running',f'Calling existing frozen LightGBM inference for {len(matrix):,} flows')
    probabilities = interface.infer(ROOT,replace(value.spec,count=len(matrix)),list(value.predictor_names),matrix)
    provenance = dict(source_dataset='external',source_split='external_inference',analysis_contract=value.dataset_id,
        source_file_name=source['name'],source_sha256=source_hash,input_type=source['extension'],analysis_id=session['analysis_id'],
        classifier_model='lightgbm',classifier_run_id=value.spec.run,predictor_count=value.predictor_count,
        preprocessing_contract=str(value.preprocessing_path.relative_to(ROOT)),preprocessing_sha256=digest(value.preprocessing_path),
        model_sha256=digest(value.model_path),extractor=extractor,processing_timestamp=datetime.now(timezone.utc).isoformat())
    record_path = output_path(directory/'interface.jsonl')
    result = []
    offsets = {}
    with record_path.open('x',encoding='utf-8',newline='\n') as stream:
        for i,p in enumerate(probabilities):
            record = build_observation(value,session['analysis_id'],i,matrix[i],p,provenance)
            offsets[record['query_id']] = stream.tell()
            stream.write(json.dumps(record,allow_nan=False)+'\n')
            result.append(dict(id=record['query_id'],predicted_class=record['predicted_class_label'],confidence=record['confidence'],
                probabilities=json.loads(record['class_probabilities_json']),model='lightgbm',model_run=value.spec.run,
                source_split='external_inference',source_dataset='external',analysis_contract=value.dataset_id,source_row_index=i))
    if digest(source['path']) != source_hash:
        raise ValueError('The source input changed during processing. These results cannot be accepted.')
    session.update(flows=result,external_records=str(record_path),external_offsets=offsets,model_artifact={'path':str(value.model_path),'sha256':provenance['model_sha256']})
    from collections import Counter
    counts = Counter(r['predicted_class'] for r in result)
    histogram = [dict(label=f'{i*10}–{(i+1)*10}%',count=0) for i in range(10)]
    for row in result:
        histogram[min(9,int(row['confidence']*10))]['count'] += 1
    session['summary'] = dict(total_flows=len(result),mean_confidence=sum(r['confidence'] for r in result)/len(result),
        class_count=len(counts),class_distribution=[dict(label=k,count=v) for k,v in counts.items()],confidence_distribution=histogram,
        audit={'status':'EXTERNAL_INFERENCE','note':'Frozen model/transform regression-tested; no benchmark integrity or correctness claim for external traffic.'},
        scope='External input · '+value.dataset_id+' analysis contract',model='LightGBM',model_run=value.spec.run,mode='external_inference')
    stage('classification','complete',f'{len(result):,} fresh predictions; no ground-truth labels assigned')
    stage('representation','complete','Existing observation and semantic-profile builders; external identity recorded')
    stage('traffic_provenance','complete','Source/model/preprocessing hashes and extraction configuration recorded')
    session['status'] = 'phase1_complete'

def record(session, flow_id):
    if flow_id not in session.get('external_offsets',{}):
        raise KeyError('Unknown external flow.')
    path = output_path(session['external_records'])
    with path.open(encoding='utf-8',newline='') as stream:
        stream.seek(session['external_offsets'][flow_id])
        return json.loads(stream.readline())
