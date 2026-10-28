"""Audit and summarize authentic runtime records, never estimate missing durations."""
import csv,json,hashlib,sys,base64
from pathlib import Path
import numpy as np
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1]
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def write(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def docs():
    identity=load(OUT/'audit/runtime_environment_identity.json')
    probe=load(OUT/'logs/runtime_probe.json')
    cpu=load(OUT/'logs/windows_cpu.json');hardware=load(OUT/'logs/windows_hardware.json');windows=load(OUT/'logs/windows_os.json')
    values=[]
    def add(key,value,source):values.append(dict(field=key,value=str(value),source=source))
    add('CPU',cpu['Name'],'Win32_Processor');add('physical_cores',cpu['NumberOfCores'],'Win32_Processor');add('logical_cores',cpu['NumberOfLogicalProcessors'],'Win32_Processor')
    add('host_RAM_bytes',hardware['TotalPhysicalMemory'],'Win32_ComputerSystem')
    free=next(r['stdout'] for r in probe if r.get('command')==['free','-b'])
    add('WSL_RAM_bytes',next(s.split()[1] for s in free.splitlines() if s.startswith('Mem:')),'free -b')
    gpu=next(r['stdout'] for r in probe if r.get('command',[''])[0]=='nvidia-smi' and len(r['command'])>1)
    parts=[s.strip() for s in gpu.splitlines()[1].split(',')]
    for k,v in zip(['GPU','GPU_VRAM','GPU_initial_used_memory','GPU_driver'],parts):add(k,v,'nvidia-smi')
    add('Windows',windows['Caption']+' '+windows['Version'],'Win32_OperatingSystem');add('benchmark_OS',identity['os'],'platform.platform')
    for k in ['python_executable','python_version','torch_cuda','retrieval_device']:add(k,identity[k],'benchmark process')
    for k,v in identity['packages'].items():add('benchmark_package_'+k,v,'importlib.metadata')
    llm=next(r for r in probe if 'packages' in r)
    add('LLM_python',llm['python'],'registered runtime probe');add('LLM_python_version',llm['version'],'registered runtime probe')
    add('vLLM_version',llm['packages']['vllm'],'registered runtime probe');add('LLM_torch',llm['packages']['torch'],'registered runtime probe')
    for k in ['model','endpoint','temperature','seed','max_model_len','max_completion_tokens','context_top_n','context_token_budget','fusion_policy']:add(k,identity[k],'registered verified configuration')
    add('benchmark_concurrency',1,'sequential runner');add('query_batch_size_used',1,'registered batch-capable kernel called with one query');add('registered_max_retrieval_batch',256,'run_rag_retrieval.py / upstream.stage4')
    write(OUT/'results/runtime_environment.csv',values)
    boundaries='''# Timing boundaries

All per-query durations are measured with time.perf_counter(), in seconds. No file timestamps or estimated latencies are used. Five real warm-up queries (the first five existing sample IDs) run the same Full pipeline before measured queries; warm-up timing values are discarded. Models, indexes, tokenizers, intrinsic provenance validation and GPU passage tensors are initialized before warm-up. Initialization is not divided across queries.

Retrieval starts immediately before consuming run_rag_retrieval_v3.balanced_batch for one actual query. It executes registered hypothesis construction, TF-IDF, BGE encoding/similarity, RRF and within-method hypothesis merge, then selects the ten hybrid_rrf candidates. All three candidate branches are computed by the unchanged kernel. GPU synchronization ensures completed device work at the timing boundary. No saved retrieval result is substituted for measured work.

Reranking and context building starts at the retrieval-end timestamp. It includes the registered Stage-5 per-candidate linkage checks and verification metadata attachment (all 30 generated candidate occurrences), run_provenance_aware_reranking.rerank for ten hybrid candidates, the registered CSV string-value boundary in memory, and build_trusted_contexts.construct with evidence_diverse, top_n=5, token_budget=2048, max_per_document=0, diversity_threshold=1.0. Intrinsic corpus provenance validation is a one-time initialization step, as in the registered pipeline, rather than a corpus reload per query. The missing historical development CSV is not bypassed: original verify_chunks reproduced every final-run provenance field before timing. No provenance gate is disabled.

LLM reasoning starts immediately before the registered runtime.post(body) call and ends when the complete HTTP response bytes return. This includes its JSON serialization, local HTTP transport, server request processing and complete generation. It excludes model loading, prompt construction, one-time tokenizer initialization, response-file writes, deterministic post-response validator and retain-classifier fusion. No automatic retries are configured. Post-response validation/fusion still run for every measured query and must pass; their cost is outside these three Table-6 stages.

Direct total is independently measured from retrieval start through complete HTTP response receipt. It includes interstage prompt/schema generation, prompt token counting, and frozen-context/prompt/schema parity assertions. Those operations are outside the specified three component boundaries. They are saved as interstage_overhead_seconds, and direct_total = component_sum + interstage_overhead is verified per query. Because direct total includes this additional preparation/audit overhead, it is not a comparable sum of the exact Table-6 stages. The supervisor total therefore uses the permitted COMPONENT SUM: sum of the three reported stage arithmetic means. Both measured direct total and component sum are retained as supporting statistics, without calling the sum a direct measurement.

Sequential serving-latency protocol: the registered retrieval kernel supports batches up to 256 and the original population publisher uses batch 256. Here it is called with a single real query to obtain individual latency measurements; model.encode retains its registered maximum batch setting and the computation/weights/ranking algorithms are unchanged. This is not batched throughput or a replication of original offline wall-clock speed. No artificial concurrency or ablation is introduced. The local Qwen process retains registered defaults, including any prefix/schema caching; the five warm-up IDs are also included normally in the fixed measured sample. No server cache flush or performance tuning was introduced. Same GPU hosts BGE and Qwen. Per-query file writes and response validation occur after the timed interval.
'''
    (OUT/'audit/timing_boundaries.md').write_text(boundaries,encoding='utf-8')
    environment='# Registered runtime environment\n\nVerified local endpoint /v1/models returns provenance-qwen3-4b, root models/rag/llm/Qwen3-4B-Instruct-2507-FP8 and max_model_len=5120. Actual endpoint response: logs/verified_models.json. Exact launch argv from unchanged LLM Server Run.txt: logs/runtime_vllm_launch.json; startup completion: logs/runtime_vllm_server.log. Temperature0, seed42, max_tokens1024, strict contract-v2 schema; validator source hashes match the frozen final run.\n\nRetrieval runs in existing WSL ustc-gpu, with torch2.14.0+cu130, sentence-transformers6.1.0, scikit-learn1.9.1 and registered local BGE/index. Qwen runs in existing rag-llm with vLLM0.30.0 and torch2.13.0+cu130. All recorded model/input hashes are verified; prebenchmark snapshot matches registered Qwen weights and every BGE model file. Recovered configuration files are referenced read-only from Table2, with exact frozen hash checks. No package installation or main-source changes. Offline models only; local loopback HTTP only. Caches/logs for this experiment are confined to Table6.\n\n'+''.join(f"- {r['field']}: {r['value']} ({r['source']})\n" for r in values)
    (OUT/'audit/runtime_environment.md').write_text(environment,encoding='utf-8')
    identities=load(OUT/'audit/model_index_identity.json')
    source='# Source artifacts\n\n'+''.join(f'- `{p}`: `{h}`\n' for p,h in identities['input_hashes'].items())
    source+='\nExact recovered configuration references:\n'+json.dumps(identities['recovered_configs'],indent=2)+'\n\nBenchmark wrapper SHA-256: '+sha(OUT/'scripts/benchmark_runtime.py')+'\n'
    (OUT/'audit/source_artifacts.md').write_text(source,encoding='utf-8')
    return values

def summarize():
    environment=docs()
    receipt=load(OUT/'raw_timings/execution_complete.json')
    assert receipt['status']=='COMPLETE' and receipt['query_count']==150
    timings=read(OUT/'raw_timings/per_query_runtime.csv');ids=[r['query_id'] for r in read(OUT/'data/runtime_query_sample.csv')]
    assert [r['query_id'] for r in timings]==ids and len(set(ids))==150
    assert sha(OUT/'raw_timings/per_query_runtime.csv')==receipt['per_query_sha256']
    warmup=lines(OUT/'raw_timings/warmup_records.jsonl');assert len(warmup)==5 and all(r['timings_discarded'] for r in warmup)
    assert not lines(OUT/'raw_timings/failures.jsonl')
    raw=lines(OUT/'raw_timings/raw_responses.jsonl');accepted=lines(OUT/'raw_timings/accepted_responses.jsonl')
    assert [r['query_id'] for r in raw]==[r['query_id'] for r in accepted]==ids
    for r,a,t in zip(raw,accepted,timings):
        assert hashlib.sha256(base64.b64decode(r['response_base64'])).hexdigest()==r['response_sha256']
        assert a['validation_status']==t['validator_status']=='PASS' and t['success_failure']=='SUCCESS'
        assert a['validator_fallback_applied']=='NO' and int(t['retry_count'])==0
        assert all(float(t[k])>0 for k in ['retrieval_seconds','rerank_context_seconds','llm_reasoning_seconds','direct_total_seconds'])
        components=sum(float(t[k]) for k in ['retrieval_seconds','rerank_context_seconds','llm_reasoning_seconds'])
        assert abs(components-float(t['component_sum_seconds']))<1e-9
        assert abs(float(t['direct_total_seconds'])-components-float(t['interstage_overhead_seconds']))<1e-9
    stages=[('Retrieval','retrieval_seconds'),('Reranking and context building','rerank_context_seconds'),('LLM reasoning','llm_reasoning_seconds'),('total','component_sum_seconds'),('direct_total_audit','direct_total_seconds'),('interstage_overhead_audit','interstage_overhead_seconds')]
    stats=[]
    for name,key in stages:
        values=np.array([float(r[key]) for r in timings]);stats.append(dict(stage=name,query_count=150,mean_seconds=float(values.mean()),median_seconds=float(np.median(values)),std_seconds=float(values.std(ddof=1)),min_seconds=float(values.min()),max_seconds=float(values.max()),p95_seconds=float(np.quantile(values,.95))))
    write(OUT/'results/runtime_statistics.csv',stats)
    precision=3
    while any(round(r['mean_seconds'],precision)==0 for r in stats[:4]):precision+=1
    full=[dict(stage=r['stage'],mean_seconds_full_precision=format(r['mean_seconds'],'.17g'),display_seconds=f"{r['mean_seconds']:.{precision}f}") for r in stats[:4]]
    write(OUT/'results/table6_runtime_full_precision.csv',full)
    table=[{'Stage':r['stage'],'Time per query (s)':r['display_seconds']} for r in full]
    write(OUT/'results/table6_runtime.csv',table)
    md='Table 6: Runtime\n\n| Stage | Time per query (s) |\n|---|---|\n'+''.join(f"| {r['Stage']} | {r['Time per query (s)']} |\n" for r in table)
    for ext in ['md','txt']:(OUT/f'results/table6_runtime.{ext}').write_text(md,encoding='utf-8')
    tex='\\begin{table}[t]\n\\centering\n\\caption{Runtime}\n\\label{tab:runtime}\n\\begin{tabular}{lc}\n\\hline\nStage & Time per query (s) \\\\\n\\hline\n'+''.join(r['Stage']+' & '+r['Time per query (s)']+' \\\\\n' for r in table)+'\\hline\n\\end{tabular}\n\\end{table}\n'
    (OUT/'results/table6_runtime.tex').write_text(tex,encoding='utf-8')
    env={r['field']:r['value'] for r in environment}
    report='# Table 6 runtime: COMPLETE\n\nRuntime source: FRESH 150-QUERY BENCHMARK. Same exact Table3 sample; 150 measured queries, five warm-ups excluded, 155 actual Qwen requests. Failures: 0. Retries: 0. All 150 fresh Full-system prompts/schemas/contexts match the frozen pipeline and all responses passed the registered validator/fusion.\n\n'+md+'\n`total` is the sum of the three stage means, not the direct elapsed total. Direct mean: '+str(stats[4]['mean_seconds'])+' seconds; component-sum mean: '+str(stats[3]['mean_seconds'])+' seconds. Direct timing includes '+str(stats[5]['mean_seconds'])+' seconds/query of interstage prompt preparation/token counting/parity checks excluded from the three specified components. Full boundaries: audit/timing_boundaries.md.\n\nHardware: '+env['CPU']+' ('+env['physical_cores']+' physical / '+env['logical_cores']+' logical cores), '+env['GPU']+' / '+env['GPU_VRAM']+' VRAM, host RAM '+env['host_RAM_bytes']+' bytes, WSL RAM '+env['WSL_RAM_bytes']+' bytes. OS: '+env['Windows']+' with '+env['benchmark_OS']+'. Local model: '+env['model']+' / Qwen3-4B-Instruct-2507-FP8; vLLM '+env['vLLM_version']+', temperature0, seed42, context budget2048, max model length5120, completion cap1024. Complete software/hardware: results/runtime_environment.csv.\n\nProtocol: sequential single-query use of the registered batch-capable retrieval kernel (original offline publisher batch256); this is serving latency, not batch throughput. CPU/GPU models, index, tokenizers and provenance initialized before the five warm-ups. Qwen startup excluded. No cached retrieval outputs used for timing. Registered hypothesis-balanced hybrid branch, provenance, reranking, context builder, Qwen contract, validator and retain-classifier fusion are unchanged. Standard deviation uses ddof=1; p95 uses numpy linear quantile.\n\nThe missing historical development provenance file caused one initialization attempt to stop before any query/request. The unchanged intrinsic verifier then reproduced all final-run provenance fields; no query retry or missing duration resulted. See logs/setup_attempts.json.\n\nThese measurements depend on this hardware, runtime, sequential workload, context lengths, OS/GPU contention and registered cache behavior. They are not universal or a comparison with other papers. No manuscript or Tables1–3 edits. Main-project working-file integrity: audit/main_project_change_report.md.\n\nSupporting median/p95:\n\n'+''.join(f"- {r['stage']}: median={r['median_seconds']:.9f}s; p95={r['p95_seconds']:.9f}s.\n" for r in stats)
    (OUT/'README.md').write_text(report,encoding='utf-8')
    (OUT/'audit/runtime_integrity.md').write_text('# Runtime integrity\n\nPASS: actual perf_counter measurements, 150 exact ordered sample IDs, five warm-ups excluded, complete positive stage durations, raw response SHA-256 checks, 150 PASS validators, no fallback, no missing/excluded queries, zero failures/retries. Per-query arithmetic verifies component_sum and direct_total=component_sum+interstage_overhead. Mean is arithmetic. Full configuration hashes/prompt parity verified; no online API or model substitution.\n',encoding='utf-8')
    dump(OUT/'run_manifest.json',dict(status='COMPLETE',source=receipt['source'],execution_receipt=receipt,query_count=150,warmup_count=5,failed_queries=0,retries=0,total_definition='sum of three reported stage arithmetic means',sample_sha256=sha(OUT/'data/runtime_query_sample.csv'),statistics=stats))
    (OUT/'logs/final_terminal_report.txt').write_text(report,encoding='utf-8')
    print(report)

def seal():
    report=(OUT/'audit/main_project_change_report.md').read_text(encoding='utf-8')
    assert 'working files modified/deleted: 0' in report and 'Working files added outside Table 6: 0' in report
    p=OUT/'logs/final_terminal_report.txt'
    p.write_text(p.read_text(encoding='utf-8')+'\nExisting main-project working files modified: ZERO\n\n'+report+'\nAll result paths:\n'+'\n'.join(str(f.relative_to(ROOT)) for f in OUT.rglob('*') if f.is_file() and '/runtime_cache/' not in f.as_posix()),encoding='utf-8')
    manifest=load(OUT/'run_manifest.json');manifest['existing_main_project_working_files_modified']=0
    manifest['output_sha256']={f.relative_to(OUT).as_posix():sha(f) for f in OUT.rglob('*') if f.is_file() and f!=OUT/'run_manifest.json' and '/runtime_cache/' not in f.as_posix() and f.name!='runtime_vllm_server.log'}
    dump(OUT/'run_manifest.json',manifest);print('Final artifact hashes sealed. Existing main-project working files modified: ZERO.')
if __name__=='__main__':
    if '--docs' in sys.argv:docs()
    elif '--seal' in sys.argv:seal()
    else:summarize()
