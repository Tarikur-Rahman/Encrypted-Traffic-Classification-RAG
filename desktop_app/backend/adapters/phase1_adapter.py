from pathlib import Path
from collections import Counter
from backend.config import ROOT
from backend.adapters.artifacts import rows, decode, digest, checklist

from src.rag.final_test.contracts import BASE as TEST_BASE

FINAL = TEST_BASE + 'stage10_final/test_final_classification_records.csv'

def execute(session, stage):
    retained(session,stage)
    session['status']='running'
    from backend.adapters.analysis_comparison import run
    run(session,stage)
    session['status']='phase1_complete'

def retained(session, stage):
    if session['input'].get('mode') == 'external_inference':
        from backend.adapters.external_runtime import classify
        return classify(session,stage)
    source = session['input']
    stage('input', 'running', 'Checking the selected file against its frozen contract')
    from backend.adapters.input_adapter import validate
    checked = validate(source['path'], source['dataset'])
    if not checked['compatible']:
        raise ValueError(checked['reason'])
    session['input_hash'] = digest(source['path'])
    stage('input', 'complete', 'Exact authoritative file match; SHA-256 recorded')
    stage('extraction', 'not_applicable', 'Already materialized CSV')
    stage('schema', 'complete', 'Exact frozen artifact schema')
    stage('preprocessing', 'reused', 'Frozen preprocessing; no fitting or new dataset split')
    stage('classification', 'running', 'Reading saved predictions' if source['mode'].startswith('saved') else 'Calling existing LightGBM inference')
    wanted = None
    if source['mode'] == 'saved_test' and session['completed_only']:
        wanted = {r['query_id'] for r in rows(ROOT / FINAL) if r['source_dataset'] == source['dataset']}
    records = []
    if source['mode'] == 'validation_inference':
        from src.rag import materialize_phase1_interface as interface
        spec = next(s for s in interface.DATASETS if s.name == source['dataset'])
        features = interface.predictor_order(ROOT, spec)
        x, _ = interface.load_validation(ROOT, spec, features)
        p = interface.infer(ROOT, spec, features, x)
        stream = (interface.observation(spec, features, i, x[i], p[i]) for i in range(len(x)))
    else:
        stream = rows(source['path'])
    for row in stream:
        if row['source_dataset'] != source['dataset'] or (wanted is not None and row['query_id'] not in wanted):
            continue
        records.append(dict(id=row['query_id'], predicted_class=row['predicted_class_label'],
                            confidence=float(row['confidence']), probabilities=decode(row['class_probabilities_json'], {}),
                            model=row['classifier_model'], model_run=row['classifier_run_id'],
                            source_split=row['source_split'], source_row_index=int(row['source_row_index'])))
    if not records:
        raise ValueError('No predictions matched the selected dataset and result scope.')
    if digest(source['path']) != session['input_hash']:
        raise ValueError('The input file changed during analysis. Results were discarded.')
    counts = Counter(r['predicted_class'] for r in records)
    histogram = [dict(label=f'{i*10}–{(i+1)*10}%', count=0) for i in range(10)]
    for row in records:
        histogram[min(9, int(row['confidence'] * 10))]['count'] += 1
    audit_path = (TEST_BASE + 'stage01_structured_queries/test_query_validation_checklist.csv'
                  if source['mode'] == 'saved_test' else
                  'results/rag/interface/interface_validation_checklist.csv')
    session['flows'] = records
    session['summary'] = dict(total_flows=len(records), mean_confidence=sum(r['confidence'] for r in records)/len(records),
        class_count=len(counts), class_distribution=[dict(label=k, count=v) for k,v in counts.items()],
        confidence_distribution=histogram, audit=checklist(audit_path),
        scope='Completed LLM research subset' if wanted is not None else 'Full selected dataset artifact',
        model='LightGBM', model_run=records[0]['model_run'], mode=source['mode'])
    model_path = ROOT / 'models' / source['dataset'] / records[0]['model_run'] / 'lightgbm.txt'
    session['model_artifact'] = dict(path=str(model_path), sha256=digest(model_path) if model_path.exists() else None)
    stage('classification', 'reused' if source['mode'].startswith('saved') else 'complete', f'{len(records):,} predictions available')
    stage('representation', 'reused', 'Traffic representations and provenance remain in the authoritative artifact')
    stage('traffic_provenance', 'reused', 'Authoritative provenance retained; input hash verified')
    session['status'] = 'phase1_complete'

def detail(session, flow_id):
    if session['input'].get('mode') == 'external_inference':
        from backend.adapters.external_runtime import record
        saved = record(session,flow_id)
        row = next(r for r in session['flows'] if r['id']==flow_id)
        return dict(row,representation=decode(saved['traffic_representation_json'],{}),
            provenance=decode(saved['traffic_provenance_json'],{}),query=decode(saved['query_text'],{}),
            metadata={'note':'External traffic; no benchmark membership or ground truth claimed.'})
    from backend.adapters.artifacts import select
    row = next((r for r in session.get('flows', []) if r['id'] == flow_id), None)
    if row is None:
        raise KeyError('Unknown flow in this analysis.')
    source = session['input']
    path = Path(source['canonical'])
    if source['mode'] == 'validation_inference':
        prefix = 'ustc' if source['dataset'] == 'USTC-TFC2016' else 'cic'
        path = ROOT / f'results/rag/interface/{prefix}_validation_interface.csv'
    record = select(str(path), 'query_id', flow_id)
    if not record:
        raise ValueError('The authoritative flow representation is unavailable.')
    saved = record[0]
    if saved['predicted_class_label'] != row['predicted_class'] or abs(float(saved['confidence'])-row['confidence']) > 1e-8:
        raise ValueError('Fresh inference differs from the saved Phase-II interface; downstream reuse is blocked.')
    return dict(row, representation=decode(saved['traffic_representation_json'], {}),
                provenance=decode(saved['traffic_provenance_json'], {}), query=decode(saved.get('query_text'), {}),
                network_endpoints=None, metadata={'note': 'Endpoint identifiers are excluded by the research interface.'})
