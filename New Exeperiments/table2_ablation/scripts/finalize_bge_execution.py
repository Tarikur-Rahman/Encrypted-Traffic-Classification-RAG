"""Publish only authentic BGE results; preserve all other measured variants."""
from finalize_recovered_variants import *
import base64
import shutil

SLUG='bge_retrieval_only'
NOTE=('The registered BGE-only retrieval branch was evaluated with the same '
      'Stage-7 builder and 2,048-token trusted-context budget as the full system. '
      'For 187/188 queries the builder selected five chunks. For '
      'CIC-TEST-030839, only four qualifying chunks could be packed before the '
      'budget was reached; every remaining registered BGE candidate would exceed '
      'the budget. No truncation, replacement, or budget modification was introduced.')

def verify_preserved():
    preserved=json.loads((OUT/'audit/bge_execution_preserved_outputs.json').read_text(encoding='utf-8'))
    assert all(sha(OUT/p)==h for p,h in preserved.items()),'Completed variant artifact changed'
    saved=json.loads((OUT/'audit/bge_execution_preserved_metrics.json').read_text(encoding='utf-8'))
    for name,expected in saved.items():assert [r for r in read(OUT/'results'/name) if r['variant']!=SLUG]==expected
    return len(preserved)

def publish():
    verify_preserved()
    directory=OUT/'variant_outputs'/SLUG
    receipt=json.loads((directory/'execution_complete.json').read_text(encoding='utf-8'))
    assert receipt['status']=='COMPLETE' and receipt['count']==188 and receipt['fresh_inference_count']==188 and receipt['failed_queries']==0
    assert sha(directory/'accepted_responses.jsonl')==receipt['accepted_sha256']
    truth=read(ROOT/'supervisor_requested_experiments/table1_error_detection/data/error_detection_balanced_set.csv')
    ids=[r['query_id'] for r in truth]
    id_hash=hashlib.sha256(json.dumps(ids,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    assert id_hash=='74931c79fc4bf286bf35e8f681c8be21303de35910f354edd7b4fece28ee59ba'
    y=np.array([int(t['predicted_class']!=t['ground_truth']) for t in truth]);assert len(set(ids))==188 and y.sum()==94
    def indexed(name):
        values=lines(directory/name) if name.endswith('.jsonl') else read(directory/name)
        assert [r['query_id'] for r in values]==ids,name
        return {r['query_id']:r for r in values}
    raw=indexed('raw_responses.jsonl');accepted=indexed('accepted_responses.jsonl');contexts=indexed('prepared_contexts.csv')
    traces=indexed('pipeline_trace_preflight.csv');fused=indexed('fusion_records.jsonl');prepared=indexed('prepared_requests.jsonl')
    results=[];trace_rows=[];cardinalities=Counter()
    for t in truth:
        i=t['query_id'];r=accepted[i];c=contexts[i];tr=traces[i]
        response=base64.b64decode(raw[i]['response_base64']);assert hashlib.sha256(response).hexdigest()==raw[i]['response_sha256']
        http=json.loads(response);value=json.loads(http['choices'][0]['message']['content'])
        assert value==r['raw_v2_model_response'] and value['evidence_status']==r['validated_response']['evidence_status']
        assert r['validation_status']=='PASS' and r['validator_fallback_applied']=='NO'
        assert r['server_model']=='provenance-qwen3-4b' and r['usage']['prompt_tokens']==prepared[i]['expected_prompt_tokens']
        assert fused[i]['final_class']==t['predicted_class']==c['classifier_prediction']
        count=len(json.loads(c['selected_chunk_ids']));cardinalities[count]+=1
        assert count==(4 if i=='CIC-TEST-030839' else 5) and int(c['context_token_count'])<=2048
        assert all(len(json.loads(tr[k]))==10 for k in ['retrieved_chunk_ids','post_provenance_chunk_ids','post_rerank_chunk_ids'])
        status=value['evidence_status'];assert status in STATUSES
        results.append(dict(query_id=i,dataset=t['dataset'],classifier_error=int(t['predicted_class']!=t['ground_truth']),phase1_predicted_class=t['predicted_class'],ground_truth=t['ground_truth'],phase2_evidence_status=status,phase2_disagree=int(status!='SUPPORTED'),variant=SLUG,status_source=(directory/'accepted_responses.jsonl').relative_to(ROOT).as_posix()+'#'+i,context_chunk_ids=c['selected_chunk_ids'],cited_chunk_ids=json.dumps(value['cited_chunk_ids']),validator_result='PASS',run_status='COMPLETE',blocked_reason=''))
        trace_rows.append(dict(**tr,final_context_chunk_count=count,final_context_token_count=int(c['context_token_count']),LLM_response_id=r['server_response_id'],LLM_response_hash=raw[i]['response_sha256'],validator_result='PASS',evidence_status=status,phase2_disagree=int(status!='SUPPORTED'),classifier_error=results[-1]['classifier_error']))
    assert cardinalities=={5:187,4:1}
    write(directory/'query_results.csv',results);write(directory/'query_execution_trace.csv',trace_rows)
    counts=Counter(r['phase2_evidence_status'] for r in results)
    write(directory/'evidence_status_counts.csv',[dict(evidence_status=s,count=counts[s],run_status='COMPLETE') for s in STATUSES])
    score=np.array([r['phase2_disagree'] for r in results]);auc=float(roc_auc_score(y,score))
    detection=float(score[y==1].mean());false_alarm=float(score[y==0].mean())
    assert abs(auc-.5*(1+detection-false_alarm))<1e-15
    rng=np.random.default_rng(42)
    wb=rng.choice(np.flatnonzero(y==1),size=(10000,94),replace=True)
    cb=rng.choice(np.flatnonzero(y==0),size=(10000,94),replace=True)
    low,high=np.quantile(.5*(score[wb].mean(1)+1-score[cb].mean(1)),[.025,.975])
    metric=dict(variant=SLUG,auroc_full_precision=number(auc),auroc_display=f'{auc:.4f}',correct_cases=94,wrong_cases=94,disagree_wrong=int(score[y==1].sum()),disagree_correct=int(score[y==0].sum()),detection_rate=number(detection),false_alarm_rate=number(false_alarm))
    ci=dict(variant=SLUG,auroc=number(auc),ci95_low=number(low),ci95_high=number(high))
    write(OUT/'results/bge_only_metrics_full_precision.csv',[dict(**metric,ci95_low=number(low),ci95_high=number(high),run_status='COMPLETE',fresh_responses=188,failed_responses=0,five_chunk_contexts=187,four_chunk_contexts=1)])
    for name,replacement in [('table2_metrics_full_precision.csv',metric),('table2_auroc_bootstrap_ci.csv',ci)]:
        values=read(OUT/'results'/name)
        write(OUT/'results'/name,[replacement if r['variant']==SLUG else r for r in values])
    values=read(OUT/'results/status_counts_by_variant.csv')
    for r in values:
        if r['variant']==SLUG:r.update(count=counts[r['evidence_status']],run_status='COMPLETE')
    write(OUT/'results/status_counts_by_variant.csv',values)
    resultmap={r['query_id']:r for r in results};transitions=read(OUT/'results/status_transition_from_full.csv');changes=0
    for r in transitions:
        if r['variant']==SLUG:
            v=resultmap[r['query_id']];changed=v['phase2_evidence_status']!=r['full_status'];changes+=changed
            r.update(variant_status=v['phase2_evidence_status'],changed_yes_no='YES' if changed else 'NO',variant_disagree=v['phase2_disagree'],run_status='COMPLETE')
    write(OUT/'results/status_transition_from_full.csv',transitions)
    pairs=read(OUT/'results/variant_pairwise_summary.csv')
    for r in pairs:
        if r['variant']==SLUG:r.update(status_changes=changes,compared_cases=188,run_status='COMPLETE')
    write(OUT/'results/variant_pairwise_summary.csv',pairs)
    table=read(OUT/'results/table2_which_parts_of_phase2_matter.csv')
    for r in table:
        if r['Setup']=='BGE retrieval only':r['AUROC']=f'{auc:.4f}'
    write(OUT/'results/table2_which_parts_of_phase2_matter.csv',table)
    md='| Setup | AUROC |\n|---|---|\n'+''.join(f"| {r['Setup']} | {r['AUROC']} |\n" for r in table)
    for ext in ['md','txt']:(OUT/f'results/table2_which_parts_of_phase2_matter.{ext}').write_text(md,encoding='utf-8')
    tex='\\begin{table}[t]\n\\centering\n\\caption{Which parts of Phase II matter?}\n\\label{tab:phase2-ablation}\n\\begin{tabular}{lc}\n\\hline\nSetup & AUROC \\\\\n\\hline\n'+''.join(r['Setup']+' & '+r['AUROC']+' \\\\\n' for r in table)+'\\hline\n\\end{tabular}\n\\end{table}\n'
    (OUT/'results/table2_which_parts_of_phase2_matter.tex').write_text(tex,encoding='utf-8')
    special=next(r for r in trace_rows if r['query_id']=='CIC-TEST-030839')
    dump(OUT/'audit/CIC-TEST-030839_bge_completed_trace.json',special)
    dump(OUT/'audit/bge_execution_validation.json',dict(status='PASS',query_id_sha256=id_hash,same_188_queries=True,raw_response_hashes_verified=188,validator_passes=188,failures=0,context_counts=dict(cardinalities),special_query=special,auroc=auc,bootstrap_ci95=[float(low),float(high)],status_counts=dict(counts),status_changes_from_full=changes))
    (directory/'integrity_checks.md').write_text('# BGE-only completed execution\n\n188 exact ordered IDs; 94 wrong and 94 correct. All 188 fresh HTTP response hashes and raw statuses match accepted outputs. Registered model/token/schema/validator checks passed with zero fallback. Retain-classifier fusion preserved all predictions. Ground truth joined only after inference.\n\n'+NOTE+'\n\nProject integrity: ../../audit/bge_execution_integrity.md.\n',encoding='utf-8')
    manifest=json.loads((directory/'run_manifest.json').read_text(encoding='utf-8'))
    manifest.update(run_status='COMPLETE',blocked_reason='',fresh_inference=True,fresh_inference_count=188,failed_queries=0,execution_receipt=receipt,context_cardinality_policy=NOTE,query_id_sha256=id_hash)
    manifest['change']='Authentic stored dense_bge candidates only, depth 10; unchanged provenance, Stage-6 reranking, Stage-7 context selection, registered Qwen/validator and retain-classifier fusion. Top-n cap 5 and budget 2048 unchanged; one natural four-chunk context.'
    source_paths=[ROOT/'results/rag/test/final_4300_v5_contract_v2/stage04_retrieval/retrieval_candidates.csv',OUT/'scripts/run_recovered_variants.py',OUT/'scripts/finalize_bge_execution.py']
    source_paths += [ROOT/r['resolved_path'] for r in read(OUT/'audit/missing_artifact_resolution.csv')]
    manifest['recovery_source_hashes']={p.relative_to(ROOT).as_posix():sha(p) for p in source_paths}
    source_file=directory/'source_artifacts.md'
    source_file.write_text(source_file.read_text(encoding='utf-8')+'\n## Completed BGE execution\n\n'+NOTE+'\n\n'+''.join(f'- `{p.relative_to(ROOT).as_posix()}`: SHA-256 `{sha(p)}`.\n' for p in source_paths),encoding='utf-8')
    manifest['output_hashes']={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='run_manifest.json'}
    dump(directory/'run_manifest.json',manifest)
    alias=OUT/'variant_outputs/bge_only'
    for name in ['query_results.csv','evidence_status_counts.csv','query_execution_trace.csv','integrity_checks.md']:shutil.copyfile(directory/name,alias/name)
    dump(alias/'run_manifest.json',dict(run_status='COMPLETE',alias_of='variant_outputs/bge_retrieval_only',fresh_inference_count=188,failed_queries=0,context_cardinality_policy=NOTE,execution_receipt=receipt,output_hashes={p.name:sha(p) for p in alias.iterdir() if p.is_file() and p.name!='run_manifest.json'}))
    current='# Table 2: COMPLETE\n\n'+md+'\n'+NOTE+'\n\nAll five previously measured rows and their per-variant artifacts are preserved. Same 188 real TEST-derived queries (94 wrong/94 correct), no resampling. BGE: 188 fresh registered Qwen responses, zero failures; CIC-TEST-030839 validator PASS. Registered model provenance-qwen3-4b / Qwen3-4B-Instruct-2507-FP8, temperature 0, seed 42, model limit 5120, completion cap 1024. Prompt/schema/validator and downstream code unchanged.\n\nBGE detection rate: '+number(detection)+'; false alarm rate: '+number(false_alarm)+'; AUROC: '+number(auc)+'; 95% bootstrap CI: ['+number(low)+', '+number(high)+']. Bootstrap matches Table 2: 10,000 paired stratified resamples, numpy default_rng(42), 94 errors and 94 correct per draw.\n\nBGE evidence-status counts: '+json.dumps(dict(counts))+'. Status changes from Full: '+str(changes)+'. The earlier exact-five block was superseded by explicit user authorization to preserve the registered top-k cap and natural context output; no pipeline repair or new packing method was introduced. Earlier blocked audits are historical, not current run status.\n\nSee audit/bge_candidate_depth_audit.md, audit/CIC-TEST-030839_bge_completed_trace.json, audit/bge_execution_validation.json, and audit/bge_execution_integrity.md. Complete response/trace artifacts are in variant_outputs/bge_retrieval_only; bge_only is a documented output alias. All four configurations were previously recovered by exact frozen hash; audit/missing_artifact_resolution.csv records paths and identity.\n\nInterpretation: same top-k cap and budget, not perfectly fixed-five cardinality. AUROC measures classifier-error detection, not overall RAG quality. The small balanced cohort and conditional bootstrap limit generalization; differences are descriptive. The registered schema fixes the admissible evidence status and the LLM provides narrative/structured linkage. The provenance ablation uses a clean trusted corpus and does not establish performance against adversarial evidence. No Table 3 or manuscript changes.\n'
    (OUT/'README.md').write_text(current,encoding='utf-8')
    (OUT/'audit/bge_completed_implementation_note.md').write_text('# BGE implementation note\n\n'+NOTE+'\n\nSame top-k cap = 5; same budget = 2048; unchanged context builder; one natural four-chunk output. This supersedes the prior exact-five constraint only, as explicitly authorized by attachment 2ead36d7-0020-4f83-bfea-605d98f76356.\n',encoding='utf-8')
    (OUT/'logs/bge_execution_final_report.md').write_text(current+'\nDetailed BGE metric:\n\n```json\n'+json.dumps(dict(**metric,ci=ci,status_counts=dict(counts),special_query=special),indent=2)+'\n```\n',encoding='utf-8')
    main_manifest=json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
    main_manifest.update(status='COMPLETE',completed_variants=SLUGS,blocked_variants={})
    main_manifest['fresh_inference_counts'][SLUG]=188;main_manifest['failed_query_counts'][SLUG]=0
    main_manifest['bge_completed_execution']=dict(receipt=receipt,context_note=NOTE,query_id_sha256=id_hash,report='logs/bge_execution_final_report.md')
    dump(OUT/'run_manifest.json',main_manifest)
    verify_preserved()
    print(current)

def integrity():
    preserved_count=verify_preserved()
    before={r['path']:r for r in read(OUT/'audit/main_project_file_hashes_bge_execution_before.csv')}
    after={r['path']:r for r in read(OUT/'audit/main_project_file_hashes_bge_execution_after.csv')}
    changed=[p for p in before if before[p]!=after.get(p)];added=sorted(set(after)-set(before))
    assert not [p for p in changed+added if not p.startswith('.git/')],(changed,added)
    report='# BGE execution integrity\n\nExisting main-project working files modified: **ZERO**. Working files added outside Table 2: **ZERO**.\n\n'+str(len(before))+' before files / '+str(len(after))+' after files hashed. '+str(preserved_count)+' completed-variant artifacts unchanged; all five other metric/CI rows preserved exactly.\n\nGit metadata changes reported separately:\n\n```json\n'+json.dumps(dict(changed=changed,added=added),indent=2)+'\n```\n'
    (OUT/'audit/bge_execution_integrity.md').write_text(report,encoding='utf-8')
    final=OUT/'logs/bge_execution_final_report.md';final.write_text(final.read_text(encoding='utf-8')+'\n'+report,encoding='utf-8')
    p=OUT/'run_manifest.json';manifest=json.loads(p.read_text(encoding='utf-8'))
    manifest['bge_execution_integrity']=dict(existing_main_project_working_files_modified=0,before_files=len(before),after_files=len(after),git_metadata_changed=changed,git_metadata_added=added)
    manifest['output_hashes']={f.relative_to(OUT).as_posix():sha(f) for f in OUT.rglob('*') if f.is_file() and f!=p and '/runtime_cache/' not in f.as_posix() and f.name!='recovery_vllm_server.log'}
    dump(p,manifest);print(report)

if __name__=='__main__':
    if '--integrity' in sys.argv:integrity()
    else:publish()
