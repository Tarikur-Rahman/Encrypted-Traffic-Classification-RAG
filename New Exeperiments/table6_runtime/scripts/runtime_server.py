"""Launch the unchanged registered vLLM command using its existing environment."""
import json
import os
import shlex
import subprocess
import sys
import urllib.request
from pathlib import Path
from datetime import datetime,timezone
OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
LOG=OUT/'logs'
if '--start' in sys.argv:
    source=(ROOT/'LLM Server Run.txt').read_text(encoding='utf-8')
    command=source[source.index('CUDA_VISIBLE_DEVICES=0'):].replace('\\\n',' ')
    parts=shlex.split(command)
    index=parts.index('vllm')
    env=dict(os.environ)
    for item in parts[:index]:
        k,v=item.split('=',1);env[k]=v
    env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',VLLM_NO_USAGE_STATS='1',DO_NOT_TRACK='1')
    cache=LOG/'runtime_cache'
    for key,sub in [('XDG_CACHE_HOME','xdg'),('HF_HOME','hf'),('TORCH_HOME','torch'),('TORCHINDUCTOR_CACHE_DIR','inductor'),
        ('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda'),('VLLM_CACHE_ROOT','vllm'),('FLASHINFER_WORKSPACE_BASE','flashinfer'),('TMPDIR','tmp')]:
        directory=cache/sub;directory.mkdir(parents=True,exist_ok=True);env[key]=str(directory)
    env['PATH']='/home/dark_bird/miniforge3/envs/rag-llm/bin:'+env.get('PATH','')
    # Linux abstract IPC sockets have no filesystem entry; DrvFS cannot host
    # pathname Unix sockets. This changes transport only, not model settings.
    env['VLLM_RPC_BASE_PATH']='@table6_runtime'
    cmd=['/home/dark_bird/miniforge3/envs/rag-llm/bin/vllm',*parts[index+1:]]
    with (LOG/'runtime_vllm_server.log').open('ab') as f:
        process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(timestamp=datetime.now(timezone.utc).isoformat(),pid=process.pid,command=cmd,cwd=str(ROOT),
        source='LLM Server Run.txt',cache_environment={k:v for k,v in env.items() if str(cache) in v},
        registered_environment={k:env[k] for k in ['CUDA_VISIBLE_DEVICES','VLLM_USE_DEEP_GEMM','VLLM_USE_FLASHINFER_SAMPLER','VLLM_RPC_BASE_PATH']})
    (LOG/'runtime_vllm_launch.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))
else:
    records=[]
    for suffix in ['/health','/v1/models','/server_info']:
        url='http://127.0.0.1:8000'+suffix
        try:
            with urllib.request.urlopen(url,timeout=8) as f:
                body=f.read().decode();records.append(dict(url=url,status=f.status,body=body))
        except Exception as exc:records.append(dict(url=url,error=repr(exc)))
    (LOG/'runtime_vllm_endpoints.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(records,indent=2))
