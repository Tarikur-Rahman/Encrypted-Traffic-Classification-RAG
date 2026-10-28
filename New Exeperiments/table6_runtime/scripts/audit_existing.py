import csv,json,re,hashlib
from pathlib import Path
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1]
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
records=[];scanned=0
for folder in ['results','metadata','supervisor_requested_experiments']:
    for p in (ROOT/folder).rglob('*'):
        if not p.is_file() or OUT in p.parents or 'runtime_cache' in p.parts:continue
        if p.suffix not in ('.csv','.jsonl','.json','.log','.txt','.md'):continue
        scanned+=1;keys=[]
        if p.suffix=='.csv':
            with p.open(encoding='utf-8-sig',newline='') as f:keys=next(csv.reader(f),[])
        elif p.suffix=='.jsonl':
            with p.open(encoding='utf-8-sig') as f:
                try:keys=list(json.loads(next(f)))
                except (StopIteration,ValueError):pass
        candidate=bool(re.search('latency|timing|benchmark|profiler|runtime',p.name,re.I)) or any(re.search(r'(latency|duration|elapsed|start_time|end_time|retrieval_seconds|rerank.*seconds|total.*seconds)',k,re.I) for k in keys)
        if not candidate:continue
        rel=p.relative_to(ROOT).as_posix();final='results/rag/test/final_4300_v5_contract_v2/' in rel
        count='not established'
        if p.suffix=='.csv' and 'query_id' in keys:count=str(len({r['query_id'] for r in read(p)}))
        elif p.suffix=='.jsonl':
            found=set()
            with p.open(encoding='utf-8-sig') as f:
                for line in f:
                    r=json.loads(line)
                    if 'query_id' in r:found.add(r['query_id'])
            count=str(len(found))
        latency='latency_seconds' in keys or 'latency' in p.name
        records.append(dict(path=rel,stage='LLM HTTP request' if latency else 'candidate log/config/aggregate; no complete per-stage timer',unit='seconds' if latency else 'not a usable comparable duration',query_coverage=count,initialization_included='NO for registered raw/accepted HTTP latency; unknown for other logs' if latency else 'unknown/not applicable',registered_final_pipeline='YES' if final else 'NO or not established',usable_for_all_table6_stages='NO',notes='LLM-only telemetry; no corresponding retrieval/rerank-context/direct-total fields' if latency else 'Not complete authoritative per-query stage telemetry'))
with (OUT/'audit/existing_runtime_candidates.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
latency=ROOT/'results/rag/test/final_4300_v5_contract_v2/stage08_llm/test_llm_latency.csv'
llm=read(latency);assert len(llm)==4300 and len({r['query_id'] for r in llm})==4300
report='# Existing runtime artifact audit\n\nRead-only discovery inspected '+str(scanned)+' candidate text/CSV/JSON/log files under results, metadata and supervisor_requested_experiments (excluding this run and cache internals). Full source-by-source inventory with stage, unit, coverage, initialization and comparability is in existing_runtime_candidates.csv. File mtimes were not used to infer any timing.\n\nThe final Stage-8 test_llm_latency.csv contains 4300 unique query latency_seconds values, retry_count and fallback_category. The authoritative runtime measures time.monotonic immediately around post(body); model startup, subsequent validation and writes are excluded. Related test_llm_responses.jsonl and execution/validated_responses.jsonl duplicate accepted response latency, not missing upstream stage telemetry. Histograms summarize this same LLM-only latency.\n\nRegistered final src/rag/final_test/upstream.py stage4/stage6/stage7 emit artifacts/receipts without per-query perf_counter/elapsed timers. Stage receipt timestamps are filesystem metadata, not valid duration telemetry. Legacy runtime experiments and Table-2 ablation latencies do not provide comparable timings for all final Full-system stages. There is no complete registered 4300-query retrieval + rerank/context + LLM telemetry set.\n\nDecision: PRIORITY B, FRESH 150-QUERY BENCHMARK using the exact Table-3 sample. Existing 4300 LLM latencies will not be mixed with fresh upstream timings.\n'
(OUT/'audit/existing_runtime_artifacts.md').write_text(report,encoding='utf-8')
sample=ROOT/'supervisor_requested_experiments/table3_retrieval_quality/data/query_sample_seed42.csv'
assert sha(sample)=='56bc840be3a68cf21577498325824ba94437c138b52333bfc177552c44c7f649'
selected=read(sample);cohort=read(ROOT/'results/rag/test/final_4300_v5_contract_v2/stage01_structured_queries/cohort_manifest.csv')
assert len(selected)==len({r['query_id'] for r in selected})==150 and {r['query_id'] for r in selected}<={r['query_id'] for r in cohort}
import random
assert [r['query_id'] for r in selected]==random.Random(42).sample(sorted(r['query_id'] for r in cohort),150)
sample_report='# Runtime sample verification\n\nPASS: exact existing Table-3 sample, 150 unique IDs from authoritative 4300-query cohort; seed42 sampling independently reproduced; no resampling. SHA-256: '+sha(sample)+'. Source: '+sample.relative_to(ROOT).as_posix()+'.\n'
for name in ['runtime_sample_verification.md','sample_integrity.md']:(OUT/'audit'/name).write_text(sample_report,encoding='utf-8')
(OUT/'data/runtime_query_sample.csv').write_bytes(sample.read_bytes())
print('Existing LLM timings: 4300; complete comparable stage set: absent. Fixed sample verified:',len(selected),'candidate artifacts',len(records))
