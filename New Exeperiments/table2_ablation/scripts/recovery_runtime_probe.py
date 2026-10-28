"""Read-only inspection of the registered WSL runtime; logs stay in Table 2."""
import json
import os
import subprocess
import urllib.request
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]
commands=[['id'],['uname','-a'],['/home/dark_bird/miniforge3/envs/rag-llm/bin/python','-B','-c',
    "import sys,importlib.metadata as m; print(sys.executable); print(sys.version); print({k:m.version(k) for k in ['vllm','torch','transformers']})"],
    ['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv'],['ps','-eo','pid,args']]
records=[]
for cmd in commands:
    try:
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=45,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        stdout=r.stdout
        if cmd[0]=='ps':stdout='\n'.join(line for line in stdout.splitlines() if 'vllm' in line or 'api_server' in line)
        records.append(dict(command=cmd,returncode=r.returncode,stdout=stdout,stderr=r.stderr))
    except Exception as exc:records.append(dict(command=cmd,error=repr(exc)))
for path in ('/home/dark_bird/miniforge3/envs/rag-llm/bin/vllm','/home/dark_bird/miniforge3/etc/profile.d/conda.sh'):
    records.append(dict(path=path,exists=Path(path).exists()))
try:
    with urllib.request.urlopen('http://127.0.0.1:8000/v1/models',timeout=8) as f:
        records.append(dict(endpoint='http://127.0.0.1:8000/v1/models',http_status=f.status,body=json.loads(f.read())))
except Exception as exc:records.append(dict(endpoint='http://127.0.0.1:8000/v1/models',error=repr(exc)))
result=json.dumps(records,indent=2)
(OUT/'logs/recovery_wsl_probe.json').write_text(result,encoding='utf-8')
print(result)
