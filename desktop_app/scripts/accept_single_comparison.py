import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.config import APP,RUNTIME
from backend.utils.write_guard import install
from backend.adapters import input_adapter,phase1_adapter,analysis_comparison
from backend.session import store
import pandas as pd
install(APP)
path=RUNTIME/'phase1_auto_audit/single-flow.csv'
pd.read_csv(RUNTIME/'regression-inputs/cic-raw.csv',nrows=1,float_precision='round_trip').to_csv(path,index=False)
session=store.create(input_adapter.validate(str(path),'CIC-Darknet2020','raw_flows'))
phase1_adapter.execute(session,lambda *args:store.stage(session,*args));store.save(session)
result=analysis_comparison.view(session)
assert result['total_flows']==1 and len(result['single_flow'])==5
assert all(row['available'] and row['probabilities'] for row in result['single_flow'])
report=dict(passed=True,analysis_id=session['analysis_id'],comparison=result)
(RUNTIME/'phase1_auto_audit/single-flow-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
