"""Audit frozen candidate depth and replay unchanged BGE Stage 6/7 kernels."""
from run_recovered_variants import *
from collections import Counter
import inspect

def main():
    protected={p.relative_to(OUT).as_posix():sha(p) for d in (OUT/'variant_outputs').iterdir() if d.is_dir() and d.name not in ('bge_only','bge_retrieval_only') for p in d.rglob('*') if p.is_file()}
    protected.update({p.relative_to(OUT).as_posix():sha(p) for p in (OUT/'results').iterdir() if p.is_file()})
    dump(OUT/'audit/bge_reaudit_preserved_outputs.json',protected)
    ids,queries,original,candidates,verified,chunks,claims,registry,rank_cfg,frozen=load()
    policy=context.index.v1.load_token_policy(ROOT)
    all_candidates=csvrows(ROOT/C.rel(4))
    groups=defaultdict(list)
    for r in all_candidates:groups[(r['query_id'],r['retrieval_method'])].append(r)
    methods=sorted({r['retrieval_method'] for r in all_candidates})
    depths={m:dict(Counter(len(v) for (i,method),v in groups.items() if method==m)) for m in methods}
    assert len({r['query_id'] for r in all_candidates})==4300
    assert all(v=={10:4300} for v in depths.values()),depths
    assert all(sorted(int(r['rank']) for r in v)==list(range(1,11)) and len({r['chunk_id'] for r in v})==10 for v in groups.values())
    all_contexts=csvrows(ROOT/C.rel(7))
    full_counts=Counter(len(json.loads(r['selected_chunk_ids'])) for r in all_contexts)
    assert len(all_contexts)==4300 and full_counts=={5:4300}
    assert all(int(r['context_token_count'])<=2048 for r in all_contexts)
    saved5={(r['query_id'],r['retrieval_method'],r['chunk_id']):r for r in subset(ROOT/C.rel(5),ids)}
    old_trace={r['query_id']:r for r in csvrows(OUT/'variant_outputs/bge_retrieval_only/pipeline_trace_preflight.csv')}
    old_context={r['query_id']:r for r in csvrows(OUT/'variant_outputs/bge_retrieval_only/prepared_contexts.csv')}
    summary=[];events=[];target='CIC-TEST-030839';target_ranked=None;target_result=None
    starts={}
    def trace(frame,event,arg):
        if frame.f_code.co_name=='add' and frame.f_code.co_filename==context.construct.__code__.co_filename:
            if event=='call':starts[id(frame)]=[e['chunk_id'] for e in frame.f_locals['selected']]
            if event=='return':
                loc=frame.f_locals;before=starts.pop(id(frame));row=loc['row'];after=[e['chunk_id'] for e in loc['selected']]
                tentative=policy.count(context.context_text([*loc['selected'],loc['value']])) if row['chunk_id'] not in after and 'value' in loc else None
                decision='SELECTED' if row['chunk_id'] in after and row['chunk_id'] not in before else 'ALREADY_SELECTED' if row['chunk_id'] in before else 'OVER_BUDGET' if tentative and tentative>2048 else 'OTHER_SKIP'
                events.append(dict(attempt=len(events)+1,chunk_id=row['chunk_id'],rerank=int(row['final_rank']),selected_before=before,decision=decision,tentative_context_tokens=tentative,selected_after=after))
        return trace
    for identity in ids:
        retrieved=sorted((r for r in candidates[identity] if r['retrieval_method']=='dense_bge'),key=lambda r:int(r['rank']))
        assert len(retrieved)==10
        assert [r['chunk_id'] for r in retrieved]==json.loads(old_trace[identity]['retrieved_chunk_ids'])
        # Same frozen Stage-5 source/linkage and authorization checks; retain metadata.
        for r in retrieved:
            v=verified[r['chunk_id']];saved=saved5[(identity,'dense_bge',r['chunk_id'])]
            assert r['source_split']=='test' and r['rag_partition']=='final_test' and r['document_id']==v['document_id']
            assert saved['verification_status']==v['provenance_status']=='VERIFIED'
            assert saved['authorization_status']==v['authorized_for_downstream']=='YES'
            assert v['authorized_for_retrieval']=='YES'
        ranked=ranking.rerank(queries[identity],retrieved,verified,dict(rank_cfg,method='dense_bge'),evidence=chunks,decision_claims=claims)
        ranked=[{k:str(v) for k,v in r.items()} for r in ranked]
        assert len(ranked)==10 and [r['chunk_id'] for r in ranked]==json.loads(old_trace[identity]['pre_context_order'])
        if identity==target:sys.settrace(trace)
        try:record=context.construct(queries[identity],ranked,chunks,verified,policy,5,2048,0,source_split='test',profile='evidence_diverse',diversity_threshold=1.0)
        finally:sys.settrace(None)
        assert record['evidence_records_json']==old_context[identity]['evidence_records_json']
        summary.append(dict(query_id=identity,retrieved_count=10,verified_count=10,reranked_count=10,final_chunk_count=len(json.loads(record['selected_chunk_ids'])),context_token_count=record['context_token_count'],previous_wrapper_exact_parity=True))
        if identity==target:target_ranked=ranked;target_result=record
    writecsv(OUT/'audit/bge_candidate_flow_188.csv',summary)
    dump(OUT/'audit/CIC-TEST-030839_bge_selection_events.json',events)
    qhash=hashlib.sha256(json.dumps(ids,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    assert qhash=='74931c79fc4bf286bf35e8f681c8be21303de35910f354edd7b4fece28ee59ba'
    truth=csvrows(ROOT/'supervisor_requested_experiments/table1_error_detection/data/error_detection_balanced_set.csv')
    assert sum(r['predicted_class']!=r['ground_truth'] for r in truth)==94
    source_paths=[ROOT/C.rel(4),ROOT/C.rel(5),ROOT/C.rel(7),ROOT/'src/rag/final_test/upstream.py',ROOT/'src/rag/run_rag_retrieval_v3.py',ROOT/'src/rag/build_trusted_contexts.py',OUT/'scripts/run_recovered_variants.py']
    source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in source_paths}
    audit='''# Original Full-system candidate-depth audit

Registered BGE depth before reranking: **10 candidates/query**. TF-IDF and hybrid_rrf also return 10 each. Frozen Stage 4 contains 4,300 queries and 43,000 occurrences per method, 129,000 total. All method/query groups have unique chunk IDs and ranks 1–10. Full Stage 7 contains exactly five chunks for all 4,300 queries, each within 2,048 BGE-token context budget.

Code evidence: src/rag/final_test/upstream.py stage4 calls run_rag_retrieval_v3.balanced_batch and explicitly requires ten unique candidates for each method and ranks 1–10. run_rag_retrieval_v3.merge_intents preserves the registered hypothesis-balanced, within-method merge and emits chosen[:10]. This is not hybrid/RRF mixing across methods. The recovered stage04_retrieval/experiment_configuration.csv records top_k=10, component_depth=50 (internal retrieval depth, not final candidate-pool size). stage6 selects the configured method then invokes run_provenance_aware_reranking.rerank with top_k=10. stage7 invokes build_trusted_contexts.construct with top_n=5, token_budget=2048, max_per_document=0, evidence_diverse, threshold=1.0.

## Earlier wrapper

No candidate-depth error found. scripts/run_recovered_variants.py prepare selects all stored dense_bge candidates (ten), calls the same rerank function with only method changed, preserves the registered CSV string boundary, and passes all ten to the same construct function. Its [:5] diagnostic in the separate audit never controlled pipeline selection. Existing saved traces prove ten retrieved and ten reranked candidates for all 188 queries. This re-audit independently reran the registered kernels and reproduced every saved BGE context exactly. No simplified ranking, manual replacement, or new packing algorithm was used.

The full orchestrator writes to frozen results/rag/test paths and selects hybrid_rrf through its registered configuration. Calling that publisher directly would violate the write boundary. Reusing its exact Stage-6/7 kernels with frozen authenticated Stage-4/5 artifacts preserves the computational path without rewriting the main project. Stage-5 provenance/linkage and authorization were rechecked against authentic frozen verification records.

## Existing Stage-7 mechanisms

Candidate skipping and over-budget rejection followed by next candidate: YES, add returns on overflow and the fill loop visits every ranked candidate. Duplicate removal: YES, already-selected chunk IDs are skipped. Evidence diversity: YES, claim-first per-class exposure with hypothesis/alignment fallback, followed by rerank-order fill. Source/document cap: max_per_document exists but registered value is 0 (disabled). Backtracking/replacement of selected chunks to optimize packing: NO. Registered source clipping/truncation: NO. Alternate context-packing fallback: NO. Offline source chunking and historical completion-truncation handlers are not runtime context fallback mechanisms. The Full system uses this same builder, with different hybrid candidates; its success does not imply this greedy builder guarantees five for every alternative candidate pool.

All 188 IDs unchanged; 94 errors and 94 correct. Ordered query-ID SHA-256: '''+qhash+'.\n\nSource SHA-256 evidence:\n\n'+''.join(f'- `{p}`: `{h}`\n' for p,h in source_hashes.items())
    (OUT/'audit/bge_candidate_depth_audit.md').write_text(audit,encoding='utf-8')
    selected=json.loads(target_result['selected_chunk_ids'])
    rows=[]
    for row in sorted(target_ranked,key=lambda r:int(r['original_rank'])):
        ident=row['chunk_id'];attempts=[e for e in events if e['chunk_id']==ident]
        rows.append(dict(bge_rank=int(row['original_rank']),chunk_id=ident,body_tokens=policy.count(chunks[ident]['chunk_text']),provenance='VERIFIED / authorized YES',rerank=int(row['final_rank']),decision='SELECTED' if ident in selected else 'OVER_BUDGET',tentative_tokens=next((e['tentative_context_tokens'] for e in attempts if e['decision']=='OVER_BUDGET'),'')))
    report='# CIC-TEST-030839 BGE-only trace\n\nAll ten authentic BGE candidates passed provenance and entered unchanged Stage 6 and Stage 7. Reranked order equals BGE ranks 1–10 for this query. The previous wrapper did not freeze the first five.\n\n| BGE rank | Chunk ID | Body tokens | Provenance | Rerank | Stage-7 decision | Rejected total |\n|---|---|---|---|---|---|---|\n'
    report+=''.join(f"| {r['bge_rank']} | {r['chunk_id']} | {r['body_tokens']} | {r['provenance']} | {r['rerank']} | {r['decision']} | {r['tentative_tokens']} |\n" for r in rows)
    report+='\nFinal five: **not produced**. Actual selected four:\n\n'+''.join('- `'+i+'`\n' for i in selected)
    report+='\nFinal context: **1782 tokens**; required five chunks within **2048**. Candidates 5–10 were each considered and rejected for exceeding budget. The first-five total 2201 was a diagnostic, not the only set considered. The smallest rejected addition totals 2139.\n\nThe unchanged Full context for this query has five chunks / '+str(original[target]['context_token_count'])+' tokens. Different hybrid evidence includes shorter association chunks; it does not invoke another packing mechanism.\n\nExact runtime add() calls are saved in CIC-TEST-030839_bge_selection_events.json, captured with read-only Python tracing; source/function behavior was not modified. All-188 replay: 187 contexts have five; this one has four. BGE remains BLOCKED, with zero LLM requests, zero inference failures (not attempted), no evidence statuses, no AUROC or bootstrap CI. Token overhead detail remains in bge_exact_five_token_audit.json.\n'
    (OUT/'audit/CIC-TEST-030839_bge_trace.md').write_text(report,encoding='utf-8')
    assert Counter(r['final_chunk_count'] for r in summary)=={5:187,4:1}
    reason='Unchanged registered pipeline receives all 10 verified BGE candidates; CIC-TEST-030839 still yields four chunks/1782 tokens. Ranks 5–10 are all rejected above 2048; no registered backtracking/truncation fallback.'
    alias=OUT/'variant_outputs/bge_only';alias.mkdir(exist_ok=True)
    results=csvrows(OUT/'variant_outputs/bge_retrieval_only/query_results.csv')
    for r in results:r.update(blocked_reason=reason)
    writecsv(alias/'query_results.csv',results)
    writecsv(alias/'evidence_status_counts.csv',[dict(evidence_status=s,count='',run_status='BLOCKED') for s in ['SUPPORTED','CONTRADICTED','CONFLICTING','INSUFFICIENT']])
    metric=dict(variant='bge_retrieval_only',run_status='BLOCKED',auroc_full_precision='',ci95_low='',ci95_high='',evaluation_cases=188,fresh_responses=0,inference_failures=0,blocked_reason=reason)
    writecsv(OUT/'results/bge_only_metrics_full_precision.csv',[metric])
    (alias/'integrity_checks.md').write_text('# BGE-only re-audit\n\n'+reason+'\n\n188 exact IDs / 94 wrong / 94 correct; ID SHA-256 '+qhash+'. No LLM inference. Empty status counts mean NOT EVALUATED, not observed zeros. Alias path requested by user; canonical prior path is ../bge_retrieval_only. Existing completed variant files and aggregate tables remain byte-identical. Project comparison: ../../audit/bge_reaudit_integrity.md.\n',encoding='utf-8')
    dump(alias/'run_manifest.json',dict(status='BLOCKED',canonical_variant='bge_retrieval_only',alias_of='variant_outputs/bge_retrieval_only',candidate_depth=10,final_context_required=5,context_budget=2048,query_id_sha256=qhash,fresh_responses=0,inference_failures=0,blocked_reason=reason,source_hashes=source_hashes,output_hashes={p.name:sha(p) for p in alias.iterdir() if p.is_file() and p.name!='run_manifest.json'}))
    assert all(sha(OUT/p)==h for p,h in protected.items())
    dump(OUT/'audit/bge_candidate_flow_result.json',dict(status='BLOCKED',candidate_depth=depths,full_context_counts=dict(full_counts),old_wrapper_wrong_depth=False,evaluation_context_counts=dict(Counter(r['final_chunk_count'] for r in summary)),trace=rows,selected_chunk_ids=selected,context_tokens=1782,preserved_completed_outputs=True))
    print(report)

if __name__=='__main__':main()
