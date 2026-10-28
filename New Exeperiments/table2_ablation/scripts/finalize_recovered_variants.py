"""Publish only complete authentic runs; preserve Full and rules-only outputs."""
import csv
import json
import hashlib
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'supervisor_requested_experiments/table2_ablation'
NAMES=['Full system','LLM replaced by simple rules','No reranking','TF-IDF retrieval only','BGE retrieval only','No provenance check']
SLUGS=['full_system','llm_replaced_by_simple_rules','no_reranking','tfidf_retrieval_only','bge_retrieval_only','no_provenance_check']
STATUSES=['SUPPORTED','CONTRADICTED','CONFLICTING','INSUFFICIENT']
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []
def number(v):return format(float(v),'.17g')

def main():
    previous=json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
    for name,h in previous['output_hashes'].items():
        if name.startswith(('variant_outputs/full_system/','variant_outputs/llm_replaced_by_simple_rules/')):
            assert sha(OUT/name)==h,('Completed row changed',name)
    evaluation=ROOT/'supervisor_requested_experiments/table1_error_detection/data/error_detection_balanced_set.csv'
    truth=read(evaluation);ids=[r['query_id'] for r in truth]
    assert sha(evaluation)==previous['evaluation_set_sha256']
    y=np.array([int(r['predicted_class']!=r['ground_truth']) for r in truth]);assert y.sum()==94 and len(y)==188
    preserved_metrics=read(OUT/'results/table2_metrics_full_precision.csv')[:2]
    preserved_ci=read(OUT/'results/table2_auroc_bootstrap_ci.csv')[:2]
    rng=np.random.default_rng(42)
    wb=rng.choice(np.flatnonzero(y==1),size=(10000,94),replace=True)
    cb=rng.choice(np.flatnonzero(y==0),size=(10000,94),replace=True)
    metrics=list(preserved_metrics);cis=list(preserved_ci);counts=[];transitions=[];pairs=[];table=[]
    full=read(OUT/'variant_outputs/full_system/query_results.csv')
    all_results={s:read(OUT/'variant_outputs'/s/'query_results.csv') for s in SLUGS[:2]}
    completed=SLUGS[:2].copy();blocked={};inference={s:0 for s in SLUGS};failures={s:0 for s in SLUGS}
    for slug in SLUGS[2:]:
        directory=OUT/'variant_outputs'/slug
        inference[slug]=len(lines(directory/'raw_responses.jsonl'));failures[slug]=len(lines(directory/'failed_queries.jsonl'))
        if not (directory/'execution_complete.json').exists():
            preflight=json.loads((directory/'execution_preflight.json').read_text(encoding='utf-8')) if (directory/'execution_preflight.json').exists() else {}
            reason=preflight.get('status','NOT_EXECUTED')
            if preflight.get('blocked_reason'):reason+=': '+preflight['blocked_reason']
            if preflight.get('short_contexts'):reason+=': '+json.dumps(preflight['short_contexts'])
            if failures[slug]:reason+='; execution failed, see failed_queries.jsonl'
            blocked[slug]=reason
            blocked_rows=read(directory/'query_results.csv')
            for row in blocked_rows:
                assert not row['phase2_evidence_status'] and not row['phase2_disagree']
                row['blocked_reason']=reason
            write(directory/'query_results.csv',blocked_rows)
            (directory/'integrity_checks.md').write_text('# BGE retrieval only\n\nBLOCKED: '+reason+'\n\nAll 188 evaluation IDs preserved; zero inference, no statuses or AUROC assigned. Model/config prerequisites were resolved. See ../../audit/bge_context_budget_report.md for chunk IDs, per-chunk counts, prompt overhead and source-mechanism audit.\n',encoding='utf-8')
            blocked_manifest=json.loads((directory/'run_manifest.json').read_text(encoding='utf-8'))
            blocked_manifest.update(run_status='BLOCKED',blocked_reason=reason,fresh_inference_count=0,
                output_hashes={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='run_manifest.json'})
            dump(directory/'run_manifest.json',blocked_manifest)
            metrics.append(dict(variant=slug,auroc_full_precision='',auroc_display='BLOCKED',correct_cases=94,wrong_cases=94,disagree_wrong='',disagree_correct='',detection_rate='',false_alarm_rate=''))
            cis.append(dict(variant=slug,auroc='',ci95_low='',ci95_high=''))
            continue
        receipt=json.loads((directory/'execution_complete.json').read_text(encoding='utf-8'))
        assert receipt['status']=='COMPLETE' and receipt['count']==188
        assert sha(directory/'accepted_responses.jsonl')==receipt['accepted_sha256']
        accepted={r['query_id']:r for r in lines(directory/'accepted_responses.jsonl')}
        raw={r['query_id']:r for r in lines(directory/'raw_responses.jsonl')}
        trace={r['query_id']:r for r in read(directory/'pipeline_trace_preflight.csv')}
        contexts={r['query_id']:r for r in read(directory/'prepared_contexts.csv')}
        fused={r['query_id']:r for r in lines(directory/'fusion_records.jsonl')}
        assert set(accepted)==set(raw)==set(trace)==set(contexts)==set(fused)==set(ids)
        result=[];trace_rows=[]
        for t in truth:
            i=t['query_id'];r=accepted[i];tr=trace[i];ctx=contexts[i]
            assert r['validation_status']=='PASS' and r['validator_fallback_applied']=='NO'
            assert fused[i]['final_class']==t['predicted_class']==ctx['classifier_prediction']
            status=r['validated_response']['evidence_status'];assert status in STATUSES
            value=dict(query_id=i,dataset=t['dataset'],classifier_error=int(t['predicted_class']!=t['ground_truth']),
                phase1_predicted_class=t['predicted_class'],ground_truth=t['ground_truth'],phase2_evidence_status=status,phase2_disagree=int(status!='SUPPORTED'),
                variant=slug,status_source=(directory/'accepted_responses.jsonl').relative_to(ROOT).as_posix()+'#'+i,
                context_chunk_ids=ctx['selected_chunk_ids'],cited_chunk_ids=json.dumps(r['validated_response']['cited_chunk_ids']),validator_result='PASS',run_status='COMPLETE',blocked_reason='')
            result.append(value)
            trace_rows.append(dict(**tr,LLM_response_id=r['server_response_id'],LLM_response_hash=raw[i]['response_sha256'],
                validator_result='PASS',evidence_status=status,phase2_disagree=value['phase2_disagree'],classifier_error=value['classifier_error']))
        write(directory/'query_results.csv',result);write(directory/'query_execution_trace.csv',trace_rows)
        status_counts=Counter(r['phase2_evidence_status'] for r in result)
        write(directory/'evidence_status_counts.csv',[dict(evidence_status=s,count=status_counts[s],run_status='COMPLETE') for s in STATUSES])
        score=np.array([r['phase2_disagree'] for r in result]);auc=float(roc_auc_score(y,score))
        assert abs(auc-.5*(score[y==1].mean()+1-score[y==0].mean()))<1e-15
        boot=.5*(score[wb].mean(1)+1-score[cb].mean(1));low,high=np.quantile(boot,[.025,.975])
        metrics.append(dict(variant=slug,auroc_full_precision=number(auc),auroc_display=f'{auc:.4f}',correct_cases=94,wrong_cases=94,
            disagree_wrong=int(score[y==1].sum()),disagree_correct=int(score[y==0].sum()),detection_rate=number(score[y==1].mean()),false_alarm_rate=number(score[y==0].mean())))
        cis.append(dict(variant=slug,auroc=number(auc),ci95_low=number(low),ci95_high=number(high)))
        all_results[slug]=result;completed.append(slug)
        (directory/'integrity_checks.md').write_text(f'# {slug}\n\nCOMPLETE: 188 exact Table-1 IDs, 94 errors/94 correct, no resampling. Actual local LLM responses: {len(raw)}. All passed exact registered model, token, schema and claim-linkage validation; all retained Phase-I classes. No fallback statuses. Ground truth joined only after accepted outputs. Full trace in query_execution_trace.csv.\n',encoding='utf-8')
        manifest=json.loads((directory/'run_manifest.json').read_text(encoding='utf-8'))
        manifest.update(run_status='COMPLETE',blocked_reason='',fresh_inference=True,fresh_inference_count=len(raw),failed_queries=failures[slug],
            execution_receipt=receipt,recovered_config_resolution=read(OUT/'audit/missing_artifact_resolution.csv'),
            process_adapter='scripts/provenance_adapter.py' if slug=='no_provenance_check' else None)
        manifest['change']={
            'no_reranking':'Use original Stage-4 hybrid rank order; retain unchanged saved claim/alignment metadata needed by Stage 7. No rerank scoring or sorting used for context order.',
            'tfidf_retrieval_only':'Use saved authentic tfidf candidates only; disable dense/hybrid method selection; retain within-method hypothesis balancing and registered downstream code.',
            'bge_retrieval_only':'Use saved authentic dense_bge candidates only; disable lexical/hybrid method selection; retain within-method hypothesis balancing and registered downstream code.',
            'no_provenance_check':'Skip Stage-5 gate and process-scope remove only downstream provenance/authorization predicates; retain metadata, quality, IDs, text and claim/schema constraints.'}[slug]
        source_paths=[ROOT/( 'results/rag/test/final_4300_v5_contract_v2/'+name) for name in
            ['stage01_structured_queries/test_query_records.csv','stage04_retrieval/retrieval_candidates.csv','stage05_provenance/provenance_chunk_status.csv','stage06_reranking/reranked_candidates.csv','stage08_llm/execution/configuration.json']]
        source_paths += [ROOT/r['resolved_path'] for r in read(OUT/'audit/missing_artifact_resolution.csv')]
        source_paths += [OUT/'scripts/run_recovered_variants.py',OUT/'scripts/provenance_adapter.py']
        source_text='# Actual recovery execution sources\n\n'+''.join(f'- `{p.relative_to(ROOT).as_posix()}`: SHA-256 `{sha(p)}`.\n' for p in source_paths)
        source_text+='\nRetained original corpus, chunks, claim registry and model/index identities are documented in audit/model_index_identity.json and audit/bge_model_identity.json. Query-specific source and pipeline trace: query_execution_trace.csv.\n'
        (directory/'source_artifacts.md').write_text(source_text,encoding='utf-8')
        manifest['recovery_source_hashes']={p.relative_to(ROOT).as_posix():sha(p) for p in source_paths}
        manifest['output_hashes']={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='run_manifest.json'}
        dump(directory/'run_manifest.json',manifest)
    for name,slug in zip(NAMES,SLUGS):
        metric=next(r for r in metrics if r['variant']==slug);table.append({'Setup':name,'AUROC':metric['auroc_display']})
        done=slug in completed;records=all_results.get(slug,[])
        changes=0
        for status in STATUSES:counts.append(dict(variant=slug,evidence_status=status,count=sum(r['phase2_evidence_status']==status for r in records) if done else '',run_status='COMPLETE' if done else 'BLOCKED'))
        if slug!='full_system':
            for j,f in enumerate(full):
                v=records[j] if done else None
                changed=v['phase2_evidence_status']!=f['phase2_evidence_status'] if done else False
                changes+=int(changed)
                transitions.append(dict(variant=slug,query_id=f['query_id'],full_status=f['phase2_evidence_status'],variant_status=v['phase2_evidence_status'] if done else '',
                    changed_yes_no=('YES' if changed else 'NO') if done else 'NOT_EVALUATED',full_disagree=f['phase2_disagree'],variant_disagree=v['phase2_disagree'] if done else '',run_status='COMPLETE' if done else 'BLOCKED'))
        pairs.append(dict(variant=slug,status_changes=changes if done else '',compared_cases=188 if done else 0,run_status='COMPLETE' if done else 'BLOCKED'))
    for name,rows in [('table2_metrics_full_precision.csv',metrics),('table2_auroc_bootstrap_ci.csv',cis),('status_counts_by_variant.csv',counts),
        ('status_transition_from_full.csv',transitions),('variant_pairwise_summary.csv',pairs),('table2_which_parts_of_phase2_matter.csv',table)]:write(OUT/'results'/name,rows)
    assert read(OUT/'results/table2_metrics_full_precision.csv')[:2]==preserved_metrics
    assert read(OUT/'results/table2_auroc_bootstrap_ci.csv')[:2]==preserved_ci
    md='Table 2: Which parts of Phase II matter?\n\n| Setup | AUROC |\n|---|---|\n'+''.join(f"| {r['Setup']} | {r['AUROC']} |\n" for r in table)
    state='PARTIAL' if blocked else 'COMPLETE'
    for ext in ['md','txt']:(OUT/f'results/table2_which_parts_of_phase2_matter.{ext}').write_text(md+'\nStatus: '+state+'.\n',encoding='utf-8')
    tex='\\begin{table}[t]\n\\centering\n\\caption{Which parts of Phase II matter?}\n\\label{tab:phase2-ablation}\n\\begin{tabular}{lc}\n\\hline\nSetup & AUROC \\\\\n\\hline\n'+''.join(r['Setup']+' & '+r['AUROC']+' \\\\\n' for r in table)+'\\hline\n\\end{tabular}\n\\par\\footnotesize Status: '+state+'.\n\\end{table}\n'
    (OUT/'results/table2_which_parts_of_phase2_matter.tex').write_text(tex,encoding='utf-8')
    largest=max((r for r in metrics[2:] if r['variant'] in completed),key=lambda r:abs(float(r['auroc_full_precision'])-float(metrics[0]['auroc_full_precision'])),default=None)
    notes=f'''# Table 2 recovery and ablation results

Status: {state}. Full/rules-only rows, CIs and per-variant artifacts are preserved unchanged. Same 188 real TEST-derived IDs, no resampling. Evaluation file SHA-256: {sha(evaluation)}.

{md}

All four missing configurations were recovered from Git commit f5732687bb68bad29d388d28828b4e0773684961, with exact Full-system-recorded SHA-256 matches (CSV CRLF restoration and raw JSON). Originals were not restored into the main project. Paths: audit/missing_artifact_resolution.csv.

Registered WSL rag-llm environment and local Qwen3-4B-Instruct-2507-FP8 server were used; model ID provenance-qwen3-4b, max length 5120, completion cap 1024, temperature 0, seed 42. No online fallback. Logs/caches are experiment-local; abstract IPC sockets have no filesystem path. See audit/llm_runtime_verification.md.

No reranking: original hybrid retrieval order, unchanged saved claim/alignment metadata required by Stage 7; rerank ordering ignored. TF-IDF and BGE: select authentic stored method-specific candidates, unchanged registered downstream ranking/context/LLM/validator/fusion. No provenance: scoped in-memory removal of provenance/authorization predicates only, metadata retained, schema/IDs/text/claim linkage/quality still checked; main source unchanged. Full context reconstruction reproduced all 188 frozen prompt hashes before any inference. The initial parity issue was a CSV-boundary count string versus integer; preserving the registered CSV boundary resolved it without semantic change.

Evidence statuses are obtained only from accepted authentic responses. Error=predicted_class != ground truth, joined after inference. Disagree=0 for SUPPORTED, 1 for CONTRADICTED/CONFLICTING/INSUFFICIENT. AUROC uses this binary score; no confidence or logistic score. Bootstrap: 10,000 paired stratified draws, numpy default_rng(42), 94 errors/94 correct each, percentile 95% interval. Existing completed rows and CIs retained exactly.

Fresh inference counts: {json.dumps(inference)}. Failed query counts: {json.dumps(failures)}. Blocked: {json.dumps(blocked)}.

BGE exact-five audit: audit/bge_context_budget_report.md and audit/bge_exact_five_token_audit.json. CIC-TEST-030839 has top-five body counts 351, 320, 511, 396, 368 BGE tokens; net context-header/separator overhead 255, total 2201 against 2048. The unchanged builder selects four (1782); every remaining fifth chunk exceeds budget. No registered exact-five truncation/packing mechanism exists. Full uses the same builder with its different retrieved chunks. Qwen prompt accounting is separate: five-chunk diagnostic prompt 3223 plus 1024 completion reservation = 4247 under 5120, so the failure is the trusted-context budget, not the model window. No diagnostic prompt was sent for BGE.

Largest measured absolute AUROC change among completed ablations: {largest['variant'] if largest else 'none'}, delta {float(largest['auroc_full_precision'])-float(metrics[0]['auroc_full_precision']) if largest else 0:.8f}. This is descriptive, not a significance claim. Full AUROC below 0.5 is not successful classifier-error detection. Classifier-error AUROC is not overall RAG quality. Clean trusted-corpus provenance ablation does not test adversarial evidence. Small balanced prediction-stratified cohort and conditional bootstrap limit generalization. Original schema fixes an admissible evidence status; the LLM supplies narrative/structured linkage under that restriction.

Before/after integrity: audit/recovery_main_project_change_report.md. Zero project working-file changes outside Table 2; strict all-files equality is not met because Codex checkpoint refs/objects in .git changed. These differences are preserved and disclosed. No Table 3 or manuscript work performed.
'''
    (OUT/'README.md').write_text(notes,encoding='utf-8')
    previous.update(status='PARTIAL_BLOCKED' if blocked else 'COMPLETE',completed_variants=completed,blocked_variants=blocked,
        fresh_inference_counts=inference,failed_query_counts=failures,recovery_resolution=read(OUT/'audit/missing_artifact_resolution.csv'))
    dump(OUT/'run_manifest.json',previous)
    report=notes+'\nAUROC 95% CI:\n'+json.dumps(cis,indent=2)+'\n\nStatus changes:\n'+json.dumps(pairs,indent=2)
    report+='\n\nOriginal missing artifacts and exact recovered paths:\n'+json.dumps(read(OUT/'audit/missing_artifact_resolution.csv'),indent=2)
    report+='\n\nAll output paths:\n'+'\n'.join(sorted(p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file()))+'\n'
    (OUT/'logs/recovery_final_terminal_report.txt').write_text(report,encoding='utf-8')
    print(report[:12000])

def verify_integrity():
    before={r['path']:r for r in read(OUT/'audit/main_project_file_hashes_recovery_before.csv')}
    after={r['path']:r for r in read(OUT/'audit/main_project_file_hashes_recovery_after.csv')}
    changed=[p for p in before if before[p]!=after.get(p)];added=sorted(set(after)-set(before))
    working_changed=[p for p in changed if not p.startswith('.git/')]
    working_added=[p for p in added if not p.startswith('.git/')]
    assert all(p.startswith('.git/refs/codex/turn-diffs/') for p in changed if p.startswith('.git/'))
    assert all(p.startswith(('.git/objects/','.git/refs/codex/turn-diffs/')) for p in added if p.startswith('.git/'))
    report=f'# Recovery file-integrity report\n\nExisting main-project working files modified/deleted: {len(working_changed)}\n\nWorking files added outside Table 2: {len(working_added)}\n\nFull SHA-256 comparison: {len(before)} before files and {len(after)} after files, excluding only table2_ablation.\n\nStrict all-files equality: {"PASS" if not changed and not added else "NOT MET"}. Existing .git metadata changed: {len(changed)-len(working_changed)}; .git additions: {len(added)-len(working_added)}. Changes are confined to refs/codex/turn-diffs checkpoint refs and Git objects, consistent with automatic Codex checkpointing. No Git metadata was restored/deleted to conceal these differences. Thus ZERO applies to project working files, not all filesystem entries.\n\nChanged: {changed}\n\nAdded: {added}\n'
    (OUT/'audit/recovery_main_project_change_report.md').write_text(report,encoding='utf-8')
    assert not working_changed and not working_added,report
    manifest=json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
    manifest.update(existing_main_project_files_modified=0,existing_files_hash_verified=len(before),
        integrity_scope='Project working files; strict all-files equality not met because Codex Git checkpoint metadata changed',
        strict_all_files_equal=not changed and not added,git_metadata_changed=changed,git_metadata_added=added)
    report_path=OUT/'logs/recovery_final_terminal_report.txt'
    terminal=report_path.read_text(encoding='utf-8').split('\n\nAll output paths:')[0]
    terminal='Existing main-project working files modified: ZERO\nGit checkpoint metadata differences: see integrity report; strict all-files equality NOT MET\nSame 188 query IDs verified: YES\n\n'+terminal
    terminal+='\n\n'+report+'\n\nAll output paths:\n'+'\n'.join(sorted(p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file()))+'\n'
    report_path.write_text(terminal,encoding='utf-8')
    manifest['output_hashes']={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p!=OUT/'run_manifest.json' and '/runtime_cache/' not in p.as_posix() and p.name!='recovery_vllm_server.log'}
    manifest['volatile_hash_exclusions']='runtime_cache and live recovery_vllm_server.log; all outside-Table2 files independently hashed'
    dump(OUT/'run_manifest.json',manifest)
    print(terminal)

if __name__=='__main__':
    if '--verify-integrity' in sys.argv:verify_integrity()
    else:main()
