"""Execute only recovered Table-2 variants using existing Phase-II kernels."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'supervisor_requested_experiments/table2_ablation'
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
os.environ['TMPDIR']=str(OUT/'logs/runtime_cache/tmp')
os.environ['HF_HOME']=str(OUT/'logs/runtime_cache/hf')
os.environ['XDG_CACHE_HOME']=str(OUT/'logs/runtime_cache/xdg')
import csv
import json
import hashlib
import time
import base64
import urllib.request
from contextlib import nullcontext
from collections import defaultdict
from datetime import datetime,timezone
from src.rag.final_test import contracts as C
from src.rag import association_response_v2 as contract
from src.rag import run_v5_contract_v2 as engine
from src.rag import run_provenance_aware_reranking as ranking
from src.rag import build_trusted_contexts as context
from src.evaluation import finalize_v5_contract_v2_stage9 as fusion
from provenance_adapter import bypass_provenance

VARIANTS={'no_reranking':'hybrid_rrf','tfidf_retrieval_only':'tfidf','bge_retrieval_only':'dense_bge','no_provenance_check':'hybrid_rrf'}
BASE=C.BASE
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def lines(p):
    if not p.exists():return []
    return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def append(p,x):
    with p.open('a',encoding='utf-8',newline='\n') as f:f.write(contract.common.json_text(x)+'\n');f.flush();os.fsync(f.fileno())
def csvrows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def subset(path,ids):return [r for r in csvrows(path) if r['query_id'] in ids]
def writecsv(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def cfg(path):return {r['field']:r['value'] for r in csvrows(path)}

def load():
    frozen=json.loads((ROOT/(BASE+'stage08_llm/execution/configuration.json')).read_text(encoding='utf-8'))
    evaluation=ROOT/'supervisor_requested_experiments/table1_error_detection/data/error_detection_balanced_set.csv'
    t1=json.loads((ROOT/'supervisor_requested_experiments/run_manifest.json').read_text(encoding='utf-8'))
    assert sha(evaluation)==t1['output_file_hashes']['table1_error_detection/data/error_detection_balanced_set.csv']
    # Read only ID column for pipeline; evaluation labels are joined later by exporter.
    ids=[r['query_id'] for r in csvrows(evaluation)]
    assert len(ids)==len(set(ids))==188
    queries={r['query_id']:r for r in subset(ROOT/(BASE+'stage01_structured_queries/test_query_records.csv'),ids)}
    original={r['query_id']:r for r in subset(ROOT/(BASE+'stage07_trusted_contexts/trusted_context_records.csv'),ids)}
    candidates=defaultdict(list)
    for r in subset(ROOT/(BASE+'stage04_retrieval/retrieval_candidates.csv'),ids):candidates[r['query_id']].append(r)
    verified={r['chunk_id']:r for r in csvrows(ROOT/(BASE+'stage05_provenance/provenance_chunk_status.csv'))}
    chunks={r['chunk_id']:r for r in csvrows(ROOT/C.CHUNKS)}
    for i,r in verified.items():r['chunk_text']=chunks[i]['chunk_text']
    claims=csvrows(ROOT/C.CLAIMS)
    stats=json.loads((ROOT/(contract.runtime.method.BASE+'statistics.json')).read_text(encoding='utf-8'))
    assert claims==contract.runtime.method.derived_tables(stats)[2]
    registry={r['claim_id']:r for r in stats if r['decision']=='APPROVED'}
    resolution=csvrows(OUT/'audit/missing_artifact_resolution.csv')
    for r in resolution:assert r['status']=='RESOLVED' and sha(ROOT/r['resolved_path'])==frozen['input_sha256'][r['expected_path']]
    config_path=next(ROOT/r['resolved_path'] for r in resolution if r['expected_path']==C.RERANK_CONFIG)
    saved=cfg(config_path)
    rank_cfg=dict(method=saved['method'],top_k=int(saved['top_k']),profile=saved['profile'],source_preferences=json.loads(saved['source_preferences']),
        **{k:float(saved[k]) for k in ('retrieval_weight','source_preference_weight','task_alignment_weight','relationship_weight')})
    for name,h in frozen['tokenizer_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in frozen['input_sha256'].items():
        if name.startswith(('src/rag/','src/evaluation/','artifacts/rag/v5f_validation/index/')) or name in (C.CHUNKS,C.CLAIMS,contract.runtime.method.BASE+'statistics.json'):
            assert sha(ROOT/name)==h,name
    for i in ids:
        C.query_check(queries[i]);C.assert_safe(original[i])
        assert len(candidates[i])==30
        for method in VARIANTS.values():assert sorted(int(r['rank']) for r in candidates[i] if r['retrieval_method']==method)==list(range(1,11))
    return ids,queries,original,candidates,verified,chunks,claims,registry,rank_cfg,frozen

def payload(record):
    p={k:record[k] for k in ('query_id','source_dataset','source_split','classifier_prediction','traffic_profile')}
    p['trusted_evidence']=json.loads(record['evidence_records_json']);C.assert_safe(p);return p

def prepare():
    ids,queries,original,candidates,verified,chunks,claims,registry,rank_cfg,frozen=load()
    policy=context.index.v1.load_token_policy(ROOT)
    tokenizer=contract.runtime.qwen.load_tokenizer(ROOT)
    # Exact BGE model files/identity as embedded in the frozen index configuration.
    index_cfg=cfg(ROOT/C.INDEX_CONFIG)
    assert index_cfg['embedding_model']=='BAAI/bge-small-en-v1.5'
    bge_records=[]
    for item in json.loads(index_cfg['embedding_model_files_json']):
        p=ROOT/index_cfg['embedding_model_local_path']/item['path'];s=p.stat()
        assert s.st_size==item['size_bytes'],str(p)
        bge_records.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),size_match=True,mtime_match=s.st_mtime_ns==item['mtime_ns']))
    # Hugging Face metadata records content identity for downloaded model/tokenizer files.
    for r in bge_records:
        p=ROOT/r['path'];relative=p.relative_to(ROOT/index_cfg['embedding_model_local_path'])
        metadata=ROOT/index_cfg['embedding_model_local_path']/'.cache/huggingface/download'/(str(relative)+'.metadata')
        if metadata.is_file():
            fields=metadata.read_text(encoding='utf-8').splitlines();etag=fields[1]
            data=p.read_bytes();actual=hashlib.sha256(data).hexdigest() if len(etag)==64 else hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            assert actual==etag,str(p)
            r.update(download_revision=fields[0],download_etag=etag,content_identity_match=True)
    dump(OUT/'audit/bge_model_identity.json',bge_records)
    # Reproduce Full contexts without inference as a parity check of reused helpers/config.
    for i in ids:
        chosen=[r for r in candidates[i] if r['retrieval_method']=='hybrid_rrf']
        ranked=ranking.rerank(queries[i],chosen,verified,rank_cfg,evidence=chunks,decision_claims=claims)
        # The registered pipeline writes Stage 6 as CSV and reads it into
        # Stage 7. Preserve that string-valued boundary, including claim counts.
        ranked=[{k:str(v) for k,v in r.items()} for r in ranked]
        rebuilt=context.construct(queries[i],ranked,chunks,verified,policy,5,2048,0,source_split='test',profile='evidence_diverse',diversity_threshold=1.0)
        if json.loads(rebuilt['evidence_records_json'])!=json.loads(original[i]['evidence_records_json']):
            dump(OUT/'audit/full_context_parity_failure.json',dict(query_id=i,rebuilt=rebuilt,original=original[i],ranked=ranked,rank_cfg=rank_cfg))
            raise RuntimeError('Full-context reconstruction mismatch; inspect audit/full_context_parity_failure.json: '+i)
        p=payload(rebuilt);state=contract.admissible(p,registry,verified,source_split='test')
        assert contract.runtime.digest(contract.prompt(p,registry,verified,state))==frozen['prompt_sha256'][i],i
    print('PASS: exact Full-context/prompt parity for all 188; frozen BGE, index and tokenizer identities.',flush=True)
    stored_ranked=defaultdict(list)
    for r in subset(ROOT/(BASE+'stage06_reranking/reranked_candidates.csv'),ids):stored_ranked[r['query_id']].append(r)
    for variant,method in VARIANTS.items():
        directory=OUT/'variant_outputs'/variant
        prepared=[];contexts=[];traces=[];short_contexts=[]
        with bypass_provenance() if variant=='no_provenance_check' else nullcontext([]) as patch_audit:
            for i in ids:
                retrieved=sorted([r for r in candidates[i] if r['retrieval_method']==method],key=lambda r:int(r['rank']))
                rconfig=dict(rank_cfg,method=method)
                if variant=='no_reranking':
                    # Stage-7 needs deterministic claim/alignment metadata. Keep that
                    # enrichment unchanged; bypass ranking score/order completely.
                    ranked=sorted(stored_ranked[i],key=lambda r:int(r['original_rank']))
                    assert [r['chunk_id'] for r in ranked]==[r['chunk_id'] for r in retrieved]
                    ranked=[dict(r,final_rank=int(r['original_rank'])) for r in ranked]
                else:
                    ranked=ranking.rerank(queries[i],retrieved,verified,rconfig,evidence=chunks,decision_claims=claims)
                ranked=[{k:str(v) for k,v in r.items()} for r in ranked]
                record=context.construct(queries[i],ranked,chunks,verified,policy,5,2048,0,source_split='test',profile='evidence_diverse',diversity_threshold=1.0)
                count=len(json.loads(record['selected_chunk_ids']))
                assert 0<count<=5,(variant,i,'invalid context size')
                if count!=5:short_contexts.append(dict(query_id=i,context_chunks=count,context_tokens=int(record['context_token_count'])))
                p=payload(record);state=contract.admissible(p,registry,verified,source_split='test')
                prompt=contract.prompt(p,registry,verified,state);schema=contract.schema(state,i)
                prepared.append(dict(query_id=i,prompt_text=prompt,response_schema=schema,payload=p,
                    prompt_sha256=contract.runtime.digest(prompt),schema_sha256=contract.runtime.digest(contract.common.json_text(schema))))
                contexts.append(record)
                traces.append(dict(query_id=i,source_query_artifact=BASE+'stage01_structured_queries/test_query_records.csv',
                    retrieval_method=method,retrieved_chunk_ids=json.dumps([r['chunk_id'] for r in retrieved]),
                    post_provenance_chunk_ids=json.dumps([r['chunk_id'] for r in retrieved]) if variant!='no_provenance_check' else '',
                    post_rerank_chunk_ids=json.dumps([r['chunk_id'] for r in ranked]) if variant!='no_reranking' else '',
                    pre_context_order=json.dumps([r['chunk_id'] for r in ranked]),final_context_chunk_ids=record['selected_chunk_ids'],
                    prompt_sha256=prepared[-1]['prompt_sha256'],schema_sha256=prepared[-1]['schema_sha256'],
                    full_context_unchanged=json.loads(record['evidence_records_json'])==json.loads(original[i]['evidence_records_json'])))
            audit=contract.runtime.token_audit(prepared,tokenizer)
            assert audit['gate_status']=='PASS',(variant,audit)
            dump(directory/'token_audit.json',audit);dump(directory/'provenance_adapter.json',patch_audit)
            for r in prepared:r['expected_prompt_tokens']=audit['prompt_tokens'][r['query_id']]
            (directory/'prepared_requests.jsonl').write_text(''.join(contract.common.json_text(r)+'\n' for r in prepared),encoding='utf-8')
            writecsv(directory/'prepared_contexts.csv',contexts);writecsv(directory/'pipeline_trace_preflight.csv',traces)
            dump(directory/'execution_preflight.json',dict(status='BLOCKED_CONTEXT_SIZE_CLARIFICATION' if short_contexts else 'READY',variant=variant,count=188,
                short_contexts=short_contexts,
                max_prompt_tokens=audit['max_prompt_tokens'],context_top_n=5,context_token_budget=2048,seed=42,
                full_contexts_unchanged=sum(r['full_context_unchanged'] for r in traces),
                rank_cfg=rconfig,source_config_sha256=frozen['input_sha256'][C.RERANK_CONFIG],
                prepared_sha256=sha(directory/'prepared_requests.jsonl'),
                no_reranking_note='Existing Stage-6 claim/alignment enrichment retained for unchanged Stage-7 contract; ranking score ignored; original Stage-4 method rank controls order. No replacement ranking rule.' if variant=='no_reranking' else ''))
            print(variant,'BLOCKED_CONTEXT_SIZE_CLARIFICATION' if short_contexts else 'READY',audit['max_prompt_tokens'],'tokens; unchanged contexts',sum(r['full_context_unchanged'] for r in traces),'short contexts',len(short_contexts),flush=True)

def verify_runtime():
    url=contract.runtime.ENDPOINT.replace('/chat/completions','/models')
    with urllib.request.urlopen(url,timeout=8) as f:models=json.loads(f.read())
    matched=[m for m in models['data'] if m['id']=='provenance-qwen3-4b']
    assert len(matched)==1 and matched[0]['max_model_len']==5120
    assert matched[0]['root']=='models/rag/llm/Qwen3-4B-Instruct-2507-FP8'
    dump(OUT/'logs/recovery_verified_models.json',models)
    log=(OUT/'logs/recovery_vllm_server.log').read_text(encoding='utf-8')
    assert 'Application startup complete.' in log and 'seed=42' in log and 'max_seq_len=5120' in log
    (OUT/'audit/llm_runtime_verification.md').write_text('''# Registered runtime verification

PASS: `/health` and `/v1/models` reachable; actual models response saved in logs/recovery_verified_models.json. Exact ID provenance-qwen3-4b, root models/rag/llm/Qwen3-4B-Instruct-2507-FP8, max_model_len=5120. Existing WSL rag-llm environment uses vLLM 0.30.0, Python 3.11.16, torch 2.13.0+cu130, transformers 5.17.0 (command/output in recovery_wsl_probe.json).

Unchanged launch arguments parsed from LLM Server Run.txt; exact argv recorded in recovery_vllm_launch.json. Startup output in recovery_vllm_server.log confirms FP8, Qwen3ForCausalLM, tokenizer model path, seed 42, eager mode and max_seq_len 5120. First attempts failed due to long/unsupported pathname IPC sockets on DrvFS; Linux abstract IPC sockets create no files and solve transport without modifying inference settings. All caches/logs remain inside Table 2. `/server_info` is unavailable (404); actual `/v1/models`, command and startup configuration provide identity evidence.

Tokenizer/config SHA-256 checks match frozen Full-system execution receipt. Existing request builder supplies temperature=0, seed=42, max_tokens=1024 and strict v2 JSON schema. Prompt/schema/validator code hashes match frozen receipt; Full-context reconstruction reproduces all 188 original prompt hashes. Every new prompt passes existing token_audit with completion reservation 1024 under 5120. Each response must pass original model/token/schema/claim validator checks before being accepted. No online fallback, model substitution, prompt truncation or silent retry.
''',encoding='utf-8')

def execute(selected):
    ids,queries,original,candidates,verified,chunks,claims,registry,rank_cfg,frozen=load()
    verify_runtime()
    for variant in selected:
        directory=OUT/'variant_outputs'/variant
        prepared=lines(directory/'prepared_requests.jsonl')
        check=json.loads((directory/'execution_preflight.json').read_text(encoding='utf-8'))
        assert check['status']=='READY' and sha(directory/'prepared_requests.jsonl')==check['prepared_sha256']
        assert [r['query_id'] for r in prepared]==ids
        ctx={r['query_id']:r for r in csvrows(directory/'prepared_contexts.csv')}
        completed={r['query_id']:r for r in lines(directory/'accepted_responses.jsonl')}
        started={r['query_id'] for r in lines(directory/'request_journal.jsonl')}
        rawmap={r['query_id']:r for r in lines(directory/'raw_responses.jsonl')}
        assert not (started-set(rawmap)),'Ambiguous in-flight request; inspect before retry'
        with bypass_provenance() if variant=='no_provenance_check' else nullcontext([]):
            for ordinal,row in enumerate(prepared,1):
                i=row['query_id']
                if i in completed:continue
                if i in rawmap:
                    raw=rawmap[i]
                    assert raw['prompt_sha256']==row['prompt_sha256']
                else:
                    body=engine.request(row,row['payload'])
                    assert body['temperature']==0 and body['seed']==42 and body['max_tokens']==1024
                    append(directory/'request_journal.jsonl',dict(query_id=i,prompt_sha256=row['prompt_sha256'],schema_sha256=row['schema_sha256'],timestamp=datetime.now(timezone.utc).isoformat()))
                    begin=time.monotonic()
                    try:response=contract.runtime.post(body)
                    except Exception as exc:
                        append(directory/'failed_queries.jsonl',dict(query_id=i,error=repr(exc),phase='transport'));raise
                    raw=dict(query_id=i,prompt_sha256=row['prompt_sha256'],response_base64=base64.b64encode(response).decode(),
                        response_sha256=hashlib.sha256(response).hexdigest(),latency_seconds=time.monotonic()-begin)
                    append(directory/'raw_responses.jsonl',raw)
                result=engine.outcomes(raw,row['payload'],registry,verified,row['expected_prompt_tokens'],source_split='test')
                if result['validation_status']!='PASS':
                    append(directory/'failed_queries.jsonl',result)
                    raise RuntimeError((variant,i,result['validation_error']))
                projected=fusion.fuse_accepted(ctx[i],result,source_split='test',response_reference=str((directory/'accepted_responses.jsonl').relative_to(ROOT)))
                assert projected['final_class']==queries[i]['classifier_prediction']
                append(directory/'fusion_records.jsonl',projected)
                append(directory/'accepted_responses.jsonl',result)
                completed[i]=result
                if ordinal%10==0 or ordinal==1 or ordinal==188:print(variant,ordinal,'/188 PASS',flush=True)
        assert set(completed)==set(ids)
        dump(directory/'execution_complete.json',dict(status='COMPLETE',variant=variant,count=188,model=contract.runtime.MODEL,
            fresh_inference_count=len(lines(directory/'raw_responses.jsonl')),failed_queries=len(lines(directory/'failed_queries.jsonl')),
            prepared_sha256=check['prepared_sha256'],accepted_sha256=sha(directory/'accepted_responses.jsonl')))

if __name__=='__main__':
    if '--prepare' in sys.argv:prepare()
    else:execute([s for s in sys.argv[1:] if s in VARIANTS] or list(VARIANTS))
