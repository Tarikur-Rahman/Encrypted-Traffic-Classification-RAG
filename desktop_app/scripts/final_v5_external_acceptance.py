"""Run the current V5 external Phase-II contract against all acceptance inputs."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import APP, ROOT, RUNTIME
from backend.utils.write_guard import install
from backend.adapters import input_adapter, phase1_adapter, fresh_phase2
from backend.adapters.artifacts import digest
from backend.external_phase2.llm_runtime import check_llm_runtime
from backend.session import store


def run_case(label, path, dataset, input_stage):
    original = digest(path)
    source = input_adapter.validate(str(path), dataset, input_stage)
    if not source['compatible']:
        raise ValueError(f'{label}: {source["reason"]}')
    session = store.create(source)
    phase1_adapter.execute(session, lambda *args: store.stage(session, *args))
    from backend.adapters.analysis_comparison import view
    comparison = view(session)
    if not all(row['status']=='Completed' and row['flows_evaluated']==len(session['flows']) for row in comparison['models']):
        raise ValueError(f'{label}: all-flow comparison failed: {comparison}')
    rows = []
    # Each fixture is the current three-row acceptance sample. One complete
    # flow per input exercises the full contract while keeping the live-model
    # acceptance bounded and reproducible.
    for flow in session['flows'][:1]:
        result = fresh_phase2.execute(session, flow['id'], lambda *args: store.stage(session, *args))
        passed = bool(result.get('decision', {}).get('final_class')) and not result.get('unavailable')
        rows.append(dict(flow_id=flow['id'], passed=passed,
                         stages={key: bool(result.get(key)) for key in
                                 ('query', 'retrieval', 'provenance', 'reranking', 'context',
                                  'reasoning', 'fusion', 'decision')},
                         runtime_status=result.get('runtime_status'),
                         blockers=result.get('blockers', {}),
                         unavailable=result.get('unavailable', []),
                         source_unchanged=digest(path) == original))
        if not rows[-1]['passed'] or not rows[-1]['source_unchanged']:
            break
    store.save(session)
    return dict(label=label, analysis_id=session['analysis_id'], flow_count=len(rows), flows=rows,comparison=comparison)


def main():
    install(APP)
    preflight = check_llm_runtime()
    report = dict(preflight=preflight, inputs=[], passed=False)
    if preflight['status'] != 'READY':
        raise RuntimeError(preflight['message'])
    cases = [
        ('USTC CSV', RUNTIME / 'regression-inputs/ustc-raw.csv', 'USTC-TFC2016', 'raw_flows'),
        ('USTC PCAP', ROOT / 'data/raw/USTC-TFC2016/Benign/Facetime.pcap', 'USTC-TFC2016', 'auto'),
        ('CIC CSV', RUNTIME / 'regression-inputs/cic-raw.csv', 'CIC-Darknet2020', 'raw_flows'),
    ]
    for case in cases:
        report['inputs'].append(run_case(*case))
    report['passed'] = all(all(flow['passed'] for flow in item['flows'])
                           for item in report['inputs'])
    target = RUNTIME / 'final-v5-external-acceptance.json'
    target.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'], report=str(target)), indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
