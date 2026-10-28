import json,subprocess,sys,importlib.metadata as metadata,os,platform
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]
records=[]
for command in [['uname','-a'],['lscpu'],['free','-b'],['nvidia-smi'],['nvidia-smi','--query-gpu=name,memory.total,memory.used,driver_version','--format=csv'],['ps','-eo','pid,args']]:
    p=subprocess.run(command,capture_output=True,text=True);records.append(dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
records.append(dict(environments=[str(p) for p in Path('/home/dark_bird/miniforge3/envs').iterdir()]))
records.append(dict(python=sys.executable,version=sys.version,packages={name:(metadata.version(name) if metadata.packages_distributions().get(name.replace('-','_')) else 'check separately') for name in ['torch','transformers','sentence-transformers','scikit-learn','numpy','vllm']}))
(OUT/'logs/runtime_probe.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print(json.dumps(records,indent=2))
