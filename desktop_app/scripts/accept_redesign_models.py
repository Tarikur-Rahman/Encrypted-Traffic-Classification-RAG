"""Six-class retained regression plus real comparison; desktop-only outputs."""
import sys,json,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.config import ROOT,RUNTIME,APP
from backend.adapters import contracts,input_adapter,external_runtime,artifacts
from backend.adapters.model_comparison import compare
from backend.session import store
from backend.utils.write_guard import install
import pandas as pd
import numpy as np

install(APP)
report=[]
for dataset in ['USTC-TFC2016','CIC-Darknet2020']:
    value=contracts.contract(dataset)
    references={}
    for r in artifacts.rows(ROOT/f'results/rag/interface/{value.spec.prefix}_validation_interface.csv'):
        references.setdefault(r['predicted_class_label'],r)
        if len(references)==len(value.class_names):break
    indices={int(r['source_row_index']):r for r in references.values()}
    frame=pd.read_csv(ROOT/f'data/processed/{dataset}/validation.csv',float_precision='round_trip')
    with tempfile.TemporaryDirectory(dir=RUNTIME) as directory:
        path=Path(directory)/'comparison.csv'
        frame.iloc[list(indices)][list(value.predictor_names)].to_csv(path,index=False)
        session=store.create(input_adapter.validate(str(path),dataset,'model_features'))
        external_runtime.classify(session,lambda *a:None)
        for flow,ref in zip(session['flows'],indices.values()):
            expected=json.loads(ref['class_probabilities_json'])
            assert flow['predicted_class']==ref['predicted_class_label']
            np.testing.assert_allclose(list(flow['probabilities'].values()),list(expected.values()),atol=1e-8,rtol=0)
            before=json.dumps(flow,sort_keys=True)
            comparison=compare(session,flow['id'])
            assert json.dumps(flow,sort_keys=True)==before
            assert len(comparison['models'])==5
            report.append(dict(dataset=dataset,label=flow['predicted_class'],phase1_pass=True,comparison=comparison))
            print(dataset,flow['predicted_class'],[(r['name'],r['status']) for r in comparison['models']],flush=True)
(RUNTIME/'redesign_audit/model-acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
