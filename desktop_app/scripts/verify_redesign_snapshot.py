"""Compare current research bytes with the protected pre-task snapshot."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
directory=ROOT/'desktop_app/runtime'/(sys.argv[1] if len(sys.argv)>1 else 'redesign_audit')
if not directory.resolve().is_relative_to((ROOT/'desktop_app/runtime').resolve()):
    raise ValueError('Audit output must remain under desktop_app/runtime')
baseline=json.loads((directory/'pre_task_snapshot.json').read_text(encoding='utf-8'))
current={}
for parent,names,files in os.walk(ROOT):
    names[:]=[n for n in names if n not in baseline['excluded']]
    for name in files:
        path=Path(parent)/name
        with path.open('rb') as stream:
            current[path.relative_to(ROOT).as_posix()]=hashlib.file_digest(stream,'sha256').hexdigest()
changed=[name for name,record in baseline['files'].items() if current.get(name)!=record['sha256']]
added=sorted(set(current)-set(baseline['files']))
report=dict(compared_files=len(baseline['files']),changed_or_missing=changed,added=added,passed=not changed and not added)
(directory/'protection-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report),flush=True)
raise SystemExit(0 if report['passed'] else 1)
