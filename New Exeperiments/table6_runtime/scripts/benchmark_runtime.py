"""Fresh sequential Full-system benchmark using unchanged registered kernels."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
cache=OUT/'logs/runtime_cache'
for key,sub in [('HF_HOME','hf'),('XDG_CACHE_HOME','xdg'),('TORCH_HOME','torch'),('TMPDIR','tmp'),('CUDA_CACHE_PATH','cuda')]:
    (cache/sub).mkdir(parents=True,exist_ok=True);os.environ[key]=str(cache/sub)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
import csv,json,time,hashlib,base64,urllib.request,platform,subprocess,importlib.metadata as metadata
from collections import defaultdict
from src.rag.final_test import contracts as C
from src.rag import run_rag_retrieval as retrieval,run_rag_retrieval_v3 as retriever
from src.rag import build_trusted_contexts as contexts,run_provenance_aware_reranking as ranking
from src.rag import association_response_v2 as contract,run_v5_contract_v2 as engine
from src.rag import verify_retrieval_provenance_v3 as provenance
from src.evaluation import finalize_v5_contract_v2_stage9 as fusion

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(p,value):p.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
def append(p,value):
    with p.open('a',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
def csv_append(p,value):
    exists=p.exists()
    with p.open('a',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(value))
        if not exists:w.writeheader()
        w.writerow(value);f.flush();os.fsync(f.fileno())
def config(p):return {r['field']:r['value'] for r in rows(p)}

def initialize():
    frozen=json.loads((ROOT/(C.BASE+'stage08_llm/execution/configuration.json')).read_text(encoding='utf-8'))
    sample=ROOT/'supervisor_requested_experiments/table3_retrieval_quality/data/query_sample_seed42.csv'
    assert sha(sample)=='56bc840be3a68cf21577498325824ba94437c138b52333bfc177552c44c7f649'
    ids=[r['query_id'] for r in rows(sample)];assert len(ids)==len(set(ids))==150
    queries={r['query_id']:r for r in rows(ROOT/C.rel(1)) if r['query_id'] in ids}
    original={r['query_id']:r for r in rows(ROOT/C.rel(7)) if r['query_id'] in ids}
    resolutions=rows(ROOT/'supervisor_requested_experiments/table2_ablation/audit/missing_artifact_resolution.csv')
    for r in resolutions:assert sha(ROOT/r['resolved_path'])==frozen['input_sha256'][r['expected_path']]
    paths={r['expected_path']:ROOT/r['resolved_path'] for r in resolutions}
    saved=config(paths[C.RERANK_CONFIG]);retrieve_cfg=config(paths[C.RETRIEVAL_CONFIG]);context_cfg=config(paths[C.CONTEXT_CONFIG])
    assert saved['method']=='hybrid_rrf' and saved['top_k']=='10' and retrieve_cfg['dense_device']=='cuda'
    assert context_cfg['top_n']=='5' and context_cfg['token_budget']=='2048' and context_cfg['profile']=='evidence_diverse'
    rank_cfg=dict(method=saved['method'],top_k=10,profile=saved['profile'],source_preferences=json.loads(saved['source_preferences']),**{k:float(saved[k]) for k in ('retrieval_weight','source_preference_weight','task_alignment_weight','relationship_weight')})
    identities={}
    for path,h in frozen['input_sha256'].items():
        if path.startswith(('src/rag/','src/evaluation/','artifacts/rag/v5f_validation/index/')) or path in (C.CHUNKS,C.CLAIMS):
            assert sha(ROOT/path)==h,path;identities[path]=h
    for path,h in frozen['tokenizer_sha256'].items():assert sha(ROOT/path)==h;identities[path]=h
    assert metadata.version('torch')=='2.14.0+cu130' and metadata.version('sentence-transformers')=='6.1.0' and metadata.version('scikit-learn')=='1.9.1'
    with urllib.request.urlopen(contract.runtime.ENDPOINT.replace('/chat/completions','/models'),timeout=10) as f:models=json.loads(f.read())
    model=next(m for m in models['data'] if m['id']==contract.runtime.MODEL)
    assert model['root']=='models/rag/llm/Qwen3-4B-Instruct-2507-FP8' and model['max_model_len']==5120
    dump(OUT/'logs/verified_models.json',models)
    print('Loading registered retrieval index, provenance and tokenizers (excluded from timing).',flush=True)
    policy=contexts.index.v1.load_token_policy(ROOT)
    evidence=retriever.load_index(ROOT,policy,'v5f_validation')
    chunks={r['chunk_id']:r for r in evidence.chunks}
    # Intrinsic corpus verification is the registered initialization step. Stage 5
    # applies its per-query candidate linkage/authorization checks below.
    # Historical development CSV is absent; use the authoritative final-run
    # record only after the same intrinsic verification function reproduces it.
    intrinsic=provenance.verify_chunks(evidence.chunks,provenance.empirical_documents(ROOT,'v5f_validation'))
    verified={r['chunk_id']:r for r in rows(ROOT/(C.BASE+'stage05_provenance/provenance_chunk_status.csv'))}
    assert {r['chunk_id']:r for r in intrinsic}=={k:{h:v[h] for h in provenance.HEADER} for k,v in verified.items()}
    for k,v in verified.items():v['chunk_text']=chunks[k]['chunk_text']
    claims=rows(ROOT/C.CLAIMS)
    stats=json.loads((ROOT/(contract.runtime.method.BASE+'statistics.json')).read_text(encoding='utf-8'))
    assert claims==contract.runtime.method.derived_tables(stats)[2]
    registry={r['claim_id']:r for r in stats if r['decision']=='APPROVED'}
    tokenizer=contract.runtime.qwen.load_tokenizer(ROOT)
    encoder=retrieval.load_query_encoder(ROOT,policy)
    assert str(encoder.device).startswith('cuda')
    retrieval.prepare_dense_passages(evidence,'cuda')
    # Content identity of the BGE model; registered load_index also checks all
    # local model identity fields, including exact file sizes/mtimes.
    model_files=[]
    for item in json.loads(config(ROOT/C.INDEX_CONFIG)['embedding_model_files_json']):
        p=ROOT/'models/rag/embedding_model'/item['path'];assert p.stat().st_size==item['size_bytes']
        model_files.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
    dump(OUT/'audit/model_index_identity.json',dict(input_hashes=identities,bge_model_files=model_files,recovered_configs=resolutions))
    env=dict(python_executable=sys.executable,python_version=sys.version,os=platform.platform(),packages={n:metadata.version(n) for n in ['torch','transformers','sentence-transformers','scikit-learn','numpy','scipy','tokenizers']},retrieval_device=str(encoder.device),torch_cuda=__import__('torch').version.cuda,registered_retrieval_config=retrieve_cfg,rank_cfg=rank_cfg,context_top_n=5,context_token_budget=2048,fusion_policy='retain_classifier',model=contract.runtime.MODEL,endpoint=contract.runtime.ENDPOINT,temperature=0,seed=42,max_model_len=5120,max_completion_tokens=1024)
    dump(OUT/'audit/runtime_environment_identity.json',env)
    return ids,queries,original,frozen,policy,evidence,chunks,verified,claims,registry,tokenizer,encoder,rank_cfg

def main():
    if (OUT/'raw_timings/request_journal.jsonl').exists():raise RuntimeError('Existing benchmark requests; inspect rather than silently retry/restart')
    data=initialize()
    ids,queries,original,frozen,policy,evidence,chunks,verified,claims,registry,tokenizer,encoder,rank_cfg=data
    import torch
    def one(identity,warmup,ordinal):
        query=queries[identity];C.query_check(query)
        append(OUT/'raw_timings/request_journal.jsonl',dict(query_id=identity,warmup=warmup,ordinal=ordinal,attempt=1))
        fields=dict(query_id=identity,retrieval_seconds='',rerank_context_seconds='',llm_reasoning_seconds='',component_sum_seconds='',direct_total_seconds='',interstage_overhead_seconds='',context_chunk_count='',context_token_count='',llm_response_status='NOT_SENT',validator_status='NOT_RUN',success_failure='FAILURE',retry_count=0)
        try:
            torch.cuda.synchronize()
            start=time.perf_counter()
            retrieved=list(retriever.balanced_batch(evidence,[query],encoder,policy,source_split='test'))
            assert len(retrieved)==1 and retrieved[0][0]['query_id']==identity
            candidates=retrieved[0][1]
            hybrid=[r for r in candidates if r['retrieval_method']=='hybrid_rrf']
            torch.cuda.synchronize();end_retrieval=time.perf_counter()
            assert len(candidates)==30 and len(hybrid)==10
            # Registered Full-system Stage-5 per-candidate boundary, identical
            # to upstream.stage5; all methods generated by the registered kernel.
            for r in candidates:
                v=verified[r['chunk_id']]
                assert r['document_id']==v['document_id'] and r['source_dataset']==query['source_dataset']
                r.update(source_type=chunks[r['chunk_id']]['source_type'],source_split='test',rag_partition='final_test',verification_status=v['provenance_status'],authorization_status=v['authorized_for_downstream'],verification_reason=v['verification_reason'])
                C.assert_safe(r)
            ranked=ranking.rerank(query,hybrid,verified,rank_cfg,evidence=chunks,decision_claims=claims)
            ranked=[{k:str(v) for k,v in r.items()} for r in ranked]
            context=contexts.construct(query,ranked,chunks,verified,policy,5,2048,0,source_split='test',profile='evidence_diverse',diversity_threshold=1.0)
            end_context=time.perf_counter()
            payload={k:context[k] for k in ('query_id','source_dataset','source_split','classifier_prediction','traffic_profile')}
            payload['trusted_evidence']=json.loads(context['evidence_records_json']);C.assert_safe(payload)
            state=contract.admissible(payload,registry,verified,source_split='test')
            prepared=dict(query_id=identity,prompt_text=contract.prompt(payload,registry,verified,state),response_schema=contract.schema(state,identity))
            assert contract.runtime.digest(prepared['prompt_text'])==frozen['prompt_sha256'][identity],('Full prompt mismatch',identity)
            assert contract.runtime.digest(contract.common.json_text(prepared['response_schema']))==frozen['request_schema_sha256'][identity]
            assert context['evidence_records_json']==original[identity]['evidence_records_json']
            expected=contract.runtime.qwen.prompt_tokens(tokenizer,prepared['prompt_text']);assert expected+1024<=5120
            body=engine.request(prepared,payload)
            before_llm=time.perf_counter()
            response=contract.runtime.post(body)
            end=time.perf_counter()
            fields.update(retrieval_seconds=end_retrieval-start,rerank_context_seconds=end_context-end_retrieval,llm_reasoning_seconds=end-before_llm,component_sum_seconds=(end_retrieval-start)+(end_context-end_retrieval)+(end-before_llm),direct_total_seconds=end-start,interstage_overhead_seconds=before_llm-end_context,context_chunk_count=len(payload['trusted_evidence']),context_token_count=context['context_token_count'],llm_response_status='COMPLETE')
            raw=dict(query_id=identity,response_base64=base64.b64encode(response).decode(),response_sha256=hashlib.sha256(response).hexdigest(),latency_seconds=fields['llm_reasoning_seconds'],prompt_sha256=contract.runtime.digest(prepared['prompt_text']))
            if not warmup:append(OUT/'raw_timings/raw_responses.jsonl',raw)
            result=engine.outcomes(raw,payload,registry,verified,expected,source_split='test')
            fields['validator_status']=result['validation_status']
            assert result['validation_status']=='PASS' and result['validator_fallback_applied']=='NO',result.get('validation_error')
            projected=fusion.fuse_accepted(context,result,source_split='test',response_reference='table6_runtime/raw_timings/raw_responses.jsonl#'+identity)
            assert projected['final_class']==query['classifier_prediction']
            fields['success_failure']='SUCCESS'
            if warmup:
                append(OUT/'raw_timings/warmup_records.jsonl',dict(query_id=identity,warmup=True,ordinal=ordinal,validator_status='PASS',response_id=result['server_response_id'],timings_discarded=True))
            else:
                csv_append(OUT/'raw_timings/per_query_runtime.csv',fields)
                append(OUT/'raw_timings/accepted_responses.jsonl',result)
                append(OUT/'raw_timings/pipeline_trace.jsonl',dict(query_id=identity,retrieval_method='hybrid_rrf',retrieved_chunk_ids=[r['chunk_id'] for r in hybrid],reranked_chunk_ids=[r['chunk_id'] for r in ranked],context_chunk_ids=json.loads(context['selected_chunk_ids']),prompt_sha256=raw['prompt_sha256'],response_sha256=raw['response_sha256'],validator_status='PASS',fusion_final_class=projected['final_class']))
            print(('WARMUP' if warmup else 'MEASURED'),ordinal,identity,'PASS',flush=True)
        except Exception as exc:
            append(OUT/'raw_timings/failures.jsonl',dict(query_id=identity,warmup=warmup,ordinal=ordinal,error=repr(exc),partial_timings=fields,retries=0))
            if not warmup:csv_append(OUT/'raw_timings/per_query_runtime.csv',fields)
            raise
    for ordinal,identity in enumerate(ids[:5],1):one(identity,True,ordinal)
    for ordinal,identity in enumerate(ids,1):one(identity,False,ordinal)
    dump(OUT/'raw_timings/execution_complete.json',dict(status='COMPLETE',query_count=150,warmup_count=5,failed_queries=0,retries=0,full_prompt_parity_count=150,source='FRESH 150-QUERY BENCHMARK',total_for_table='component_sum_seconds',direct_total_includes='prompt/schema construction, token audit and parity checks between context and HTTP call; excludes post-response validation/fusion and writing',per_query_sha256=sha(OUT/'raw_timings/per_query_runtime.csv')))
if __name__=='__main__':main()
