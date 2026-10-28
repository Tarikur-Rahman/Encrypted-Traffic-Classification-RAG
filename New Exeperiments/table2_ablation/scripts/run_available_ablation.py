"""Frozen-artifact Table 2 evaluation; never substitutes unavailable inference.

Rules call the existing admissible-state kernel on hash-verified frozen inputs.
Other ablations stop at preflight; blocked rows never receive inferred statuses.
"""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'supervisor_requested_experiments/table2_ablation'
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path.insert(0,str(ROOT))
import csv
import json
import hashlib
import urllib.request
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.rag import association_response_v2 as contract
from src.rag.final_test import contracts as C

BASE=C.BASE
T1='supervisor_requested_experiments/table1_error_detection/'
NAMES=['Full system','LLM replaced by simple rules','No reranking','TF-IDF retrieval only','BGE retrieval only','No provenance check']
SLUGS=['full_system','llm_replaced_by_simple_rules','no_reranking','tfidf_retrieval_only','bge_retrieval_only','no_provenance_check']
STATUSES=['SUPPORTED','CONTRADICTED','CONFLICTING','INSUFFICIENT']
SOURCES={}

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def use(name,purpose):
    p=ROOT/name
    if not p.is_file():raise RuntimeError('Missing required artifact: '+name)
    SOURCES[name]={'purpose':purpose,'sha256':sha(p)}
    return p

def text(path,value):
    path.write_text(value,encoding='utf-8')

def dump(path,value):
    text(path,json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))

def rows(name,purpose):
    with use(name,purpose).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def by_id(items,key='query_id'):
    mapped={r[key]:r for r in items}
    assert len(mapped)==len(items),'Duplicate IDs: '+key
    return mapped

def save(items,path,columns=None):
    pd.DataFrame(items,columns=columns).to_csv(path,index=False,float_format='%.17g')

def main():
    assert (OUT/'audit/main_project_file_hashes_before.csv').is_file(),'Run full before-hash inventory first'
    t1manifest=json.loads(use('supervisor_requested_experiments/run_manifest.json','Table 1 frozen output hashes').read_text())
    eval_path=T1+'data/error_detection_balanced_set.csv'
    balanced=rows(eval_path,'Exact Table 1 evaluation IDs and post-hoc targets; never passed into Phase II')
    eval_hash=SOURCES[eval_path]['sha256']
    assert eval_hash==t1manifest['output_file_hashes']['table1_error_detection/data/error_detection_balanced_set.csv']
    ids=[r['query_id'] for r in balanced]
    truth=by_id(balanced)
    assert len(ids)==188 and len(set(ids))==188
    y=np.array([int(r['predicted_class']!=r['ground_truth']) for r in balanced])
    assert sum(y)==94 and sum(1-y)==94
    assert all(int(r['classifier_correct'])==1-int(v) for r,v in zip(balanced,y))
    t1metrics=rows(T1+'results/table1_metrics_full_precision.csv','Exact Phase-II baseline metric')
    assert SOURCES[T1+'results/table1_metrics_full_precision.csv']['sha256']==t1manifest['output_file_hashes']['table1_error_detection/results/table1_metrics_full_precision.csv']
    expected=float(next(r['AUROC'] for r in t1metrics if r['Method']=='Phase II evidence status'))
    frozen_config_path=BASE+'stage08_llm/execution/configuration.json'
    cfg=json.loads(use(frozen_config_path,'Registered model, tokenizer, code and input hashes; token/schema identity').read_text())
    config={k:cfg[k] for k in ('model','endpoint','max_model_len','max_completion_tokens','temperature','seed','test_runtime_validator_sha256')}
    assert config==dict(model=contract.runtime.MODEL,endpoint=contract.runtime.ENDPOINT,max_model_len=5120,max_completion_tokens=1024,
        temperature=0,seed=42,test_runtime_validator_sha256=C.sha(ROOT,'src/rag/positive_evidence_v5.py'))
    assert cfg['ground_truth_used'] is False and cfg['correctness_used'] is False
    evidence_paths=[C.BUNDLE,contract.runtime.method.BASE+'statistics.json',C.CHUNKS,C.CLAIMS,
        'src/rag/association_response_v2.py','src/rag/positive_evidence_v5.py','src/rag/build_semantic_association_v5.py',
        'src/rag/run_v5_contract_v2.py','src/rag/run_v5_diagnostic.py','src/rag/final_test/llm.py',
        'src/rag/final_test/upstream.py','src/rag/build_trusted_contexts.py','src/rag/run_provenance_aware_reranking.py',
        'src/rag/final_test/contracts.py','src/rag/final_test/preflight.py']
    for name in evidence_paths:
        p=use(name,'Frozen evidence or existing implementation; no source changes')
        if name in cfg['input_sha256']:assert sha(p)==cfg['input_sha256'][name],name
    for name,h in cfg['tokenizer_sha256'].items():
        p=use(name,'Frozen tokenizer identity, read only')
        assert sha(p)==h,name
    query_path=BASE+'stage01_structured_queries/test_query_records.csv'
    context_path=BASE+'stage07_trusted_contexts/trusted_context_records.csv'
    query=by_id(rows(query_path,'Frozen safe Phase-I observations'))
    contexts=by_id(rows(context_path,'Reuse unchanged full-system contexts for Full and rules-only'))
    assert sha(ROOT/context_path)==cfg['input_sha256'][context_path]
    assert sha(ROOT/query_path)==cfg['input_sha256'][query_path]
    response_path=BASE+'stage08_llm/test_llm_responses.jsonl'
    responses=by_id([json.loads(s) for s in use(response_path,'Authoritative Full-system accepted outputs').read_text().splitlines()])
    assert SOURCES[response_path]['sha256']==t1manifest['source_artifacts'][response_path]['sha256']
    chunks=by_id(rows(C.CHUNKS,'Frozen evidence chunk text and identities'),'chunk_id')
    verified=by_id(rows(BASE+'stage05_provenance/provenance_chunk_status.csv','Reuse saved Stage-5 verified graph'),'chunk_id')
    for identity,r in verified.items():
        r['chunk_text']=chunks[identity]['chunk_text']
        assert r['document_id']==chunks[identity]['document_id']
    stats=json.loads((ROOT/(contract.runtime.method.BASE+'statistics.json')).read_text())
    bundle=json.loads((ROOT/C.BUNDLE).read_text())
    assert sha(ROOT/(contract.runtime.method.BASE+'statistics.json'))==bundle['files']['statistics.json']
    claims=rows(C.CLAIMS,'Existing registered claims checked against existing deterministic derived_tables helper')
    assert claims==contract.runtime.method.derived_tables(stats)[2]
    registry={r['claim_id']:r for r in stats if r['decision']=='APPROVED'}
    # Build processing payloads only from frozen safe contexts, never evaluation rows.
    payloads={}
    for identity in ids:
        q,c,r=query[identity],contexts[identity],responses[identity]
        C.query_check(q);C.assert_safe(c)
        assert q['classifier_prediction']==c['classifier_prediction']==truth[identity]['predicted_class']
        assert r['validation_status']=='PASS' and r['prompt_sha256']==cfg['prompt_sha256'][identity]
        assert r['validated_response']['evidence_status']==truth[identity]['phase2_evidence_status']
        assert len(json.loads(c['selected_chunk_ids']))==5
        payload={k:c[k] for k in ('query_id','source_dataset','source_split','classifier_prediction','traffic_profile')}
        payload['trusted_evidence']=json.loads(c['evidence_records_json'])
        C.assert_safe(payload)
        payloads[identity]=payload
    text(OUT/'data/phase2_inputs_without_ground_truth.jsonl',''.join(json.dumps(payloads[q],ensure_ascii=False)+'\n' for q in ids))
    save([dict(query_id=q,selection_ordinal=i) for i,q in enumerate(ids)],OUT/'data/evaluation_ids.csv')
    # Read-only local endpoint check: never start a model or send inference here.
    model_check={'url':cfg['endpoint'].rsplit('/',1)[0]+'/models','timestamp':datetime.now(timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen(model_check['url'],timeout=8) as f:
            model_check.update(http_status=f.status,response=json.loads(f.read()))
        model_check['registered_model_listed']=any(r['id']==cfg['model'] for r in model_check['response'].get('data',[]))
        model_check['status']='AVAILABLE_IDENTITY_REQUIRES_FULL_PREFLIGHT'
    except Exception as exc:
        model_check.update(status='UNAVAILABLE',error=repr(exc))
    dump(OUT/'logs/model_endpoint_preflight.json',model_check)
    required=[C.RETRIEVAL_CONFIG,C.RERANK_CONFIG,C.CONTEXT_CONFIG,C.V5_RUN+'configuration.json',
        *['artifacts/rag/v5f_validation/index/'+n for n in ('tfidf_vectorizer.joblib','tfidf_matrix.npz','dense_embeddings.npy','dense_chunk_ids.csv')]]
    missing=[p for p in required if not (ROOT/p).is_file()]
    if not missing and model_check['status']!='UNAVAILABLE':
        raise RuntimeError('Prerequisites changed: fresh-inference variants require a new complete execution preflight; no automatic run')
    blocked='Registered local model unavailable: '+model_check.get('error','identity/tokenizer/server configuration not fully verified')+'. Missing frozen dependencies: '+', '.join(missing)
    provenance_block='No registered gate-disable option: rerank(), construct(), and qualifying() each enforce provenance/authorization. Skipping Stage 5 alone leaves these gates active. A blanket require bypass or relabeling evidence VERIFIED would not be a faithful one-factor ablation. '+blocked
    changes=[
        'None. Reuse exact frozen Phase-II statuses used by Table 1.',
        'Remove LLM decision mechanism; call association_response_v2.admissible on unchanged contexts and verified claim graph. All retrieval, provenance and context artifacts reused.',
        'Planned: bypass Stage-6 ranking and retain Stage-5 hybrid retrieval rank; keep Stage-7 diversity policy. Not executed. No replacement ranking invented.',
        'Planned: select existing tfidf candidates only, then unchanged downstream code/config. Not executed.',
        'Planned: select existing dense BGE candidates only, then unchanged downstream code/config. Not executed.',
        'Planned: bypass provenance acceptance/rejection without changing corpus or metadata. Not executed; no existing scoped bypass API found.']
    complete={}; variant_records={}; reasons={}; states=[]
    for idx,(name,slug) in enumerate(zip(NAMES,SLUGS)):
        directory=OUT/'variant_outputs'/slug;directory.mkdir(exist_ok=True)
        result=[]
        reason='' if idx<2 else provenance_block if idx==5 else blocked
        reasons[slug]=reason
        for identity in ids:
            status='';citations=[];validator='NOT_RUN';source='';chunk_ids=[]
            if idx==0:
                r=responses[identity]['validated_response'];status=r['evidence_status'];citations=r['cited_chunk_ids']
                source=response_path;validator='FROZEN_PASS';chunk_ids=json.loads(contexts[identity]['selected_chunk_ids'])
            elif idx==1:
                p=payloads[identity]
                state=contract.admissible(p,registry,verified,source_split='test')
                status=state['evidence_status'];citations=sorted({c for cs in state['claim_chunks'].values() for c in cs})
                legacy=dict(query_id=identity,evidence_status=status,reasoning_decision=state['reasoning_decision'],supported_class=state['supported_class'],
                    explanation='CLAIM_IDS: '+contract.common.json_text(state['claim_ids']) if state['claim_ids'] else 'No qualifying approved claim.',cited_chunk_ids=citations)
                contract.runtime.validator.validate(legacy,p,registry,verified,method=contract.runtime.method,source_split='test')
                prompt=contract.prompt(p,registry,verified,state)
                schema=contract.schema(state,identity)
                assert contract.runtime.digest(prompt)==cfg['prompt_sha256'][identity],identity
                assert contract.runtime.digest(contract.common.json_text(schema))==cfg['request_schema_sha256'][identity],identity
                states.append(dict(query_id=identity,**state))
                source='src/rag/association_response_v2.py:admissible';validator='PASS_EXISTING_DETERMINISTIC_VALIDATOR'
                chunk_ids=json.loads(contexts[identity]['selected_chunk_ids'])
            # The evaluation-only join occurs after component execution.
            t=truth[identity]
            result.append(dict(query_id=identity,dataset=t['dataset'],classifier_error=int(t['predicted_class']!=t['ground_truth']),
                phase1_predicted_class=t['predicted_class'],ground_truth=t['ground_truth'],phase2_evidence_status=status,
                phase2_disagree=int(status!='SUPPORTED') if status else None,variant=slug,status_source=source,
                context_chunk_ids=json.dumps(chunk_ids) if idx<2 else '',cited_chunk_ids=json.dumps(citations) if idx<2 else '',
                validator_result=validator,run_status='COMPLETE' if idx<2 else 'BLOCKED',blocked_reason=reason))
        variant_records[slug]=result
        if idx<2:complete[slug]=np.array([r['phase2_disagree'] for r in result])
        save(result,directory/'query_results.csv')
        save([dict(evidence_status=s,count=sum(r['phase2_evidence_status']==s for r in result) if idx<2 else None,run_status='COMPLETE' if idx<2 else 'BLOCKED') for s in STATUSES],directory/'evidence_status_counts.csv')
        text(directory/'integrity_checks.md',f'# {name}\n\n188 unique IDs, exact Table 1 order; targets joined post-hoc.\n\nRun status: {"COMPLETE" if idx<2 else "BLOCKED"}.\n\nMissing final statuses: {0 if idx<2 else 188}. No failed or blocked inference is converted into INSUFFICIENT.\n\n{changes[idx]}\n\n{reason}\n')
    assert float(roc_auc_score(y,complete['full_system']))==expected,'Full-system metric mismatch: STOP'
    dump(OUT/'variant_outputs/llm_replaced_by_simple_rules/admissible_states.json',states)
    rng=np.random.default_rng(42)
    # Paired, stratified bootstrap: same sampled indices across completed variants.
    wi=np.flatnonzero(y==1);ci=np.flatnonzero(y==0)
    wb=rng.choice(wi,size=(10000,len(wi)),replace=True);cb=rng.choice(ci,size=(10000,len(ci)),replace=True)
    metrics=[];cis=[];counts=[];transitions=[];pairs=[];table_rows=[]
    for name,slug in zip(NAMES,SLUGS):
        done=slug in complete
        scores=complete.get(slug)
        auc=float(roc_auc_score(y,scores)) if done else None
        low=high=None
        if done:
            boot=.5*(scores[wb].mean(axis=1)+1-scores[cb].mean(axis=1))
            low,high=map(float,np.quantile(boot,[.025,.975]))
        metrics.append(dict(variant=slug,auroc_full_precision=auc,auroc_display=f'{auc:.4f}' if done else 'BLOCKED',
            correct_cases=94,wrong_cases=94,disagree_wrong=int(scores[y==1].sum()) if done else None,
            disagree_correct=int(scores[y==0].sum()) if done else None,detection_rate=float(scores[y==1].mean()) if done else None,
            false_alarm_rate=float(scores[y==0].mean()) if done else None))
        cis.append(dict(variant=slug,auroc=auc,ci95_low=low,ci95_high=high))
        table_rows.append({'Setup':name,'AUROC':f'{auc:.4f}' if done else 'BLOCKED'})
        for s in STATUSES:
            counts.append(dict(variant=slug,evidence_status=s,count=sum(r['phase2_evidence_status']==s for r in variant_records[slug]) if done else None,run_status='COMPLETE' if done else 'BLOCKED'))
        changed=0
        if slug!='full_system':
            for full,variant in zip(variant_records['full_system'],variant_records[slug]):
                change=full['phase2_evidence_status']!=variant['phase2_evidence_status'] if done else None
                changed+=int(change) if done else 0
                transitions.append(dict(variant=slug,query_id=full['query_id'],full_status=full['phase2_evidence_status'],
                    variant_status=variant['phase2_evidence_status'],changed_yes_no=('YES' if change else 'NO') if done else 'NOT_EVALUATED',
                    full_disagree=full['phase2_disagree'],variant_disagree=variant['phase2_disagree'],run_status='COMPLETE' if done else 'BLOCKED'))
        pairs.append(dict(variant=slug,status_changes=changed if done else None,compared_cases=188 if done else 0,run_status='COMPLETE' if done else 'BLOCKED'))
    for items,filename in [(metrics,'table2_metrics_full_precision.csv'),(cis,'table2_auroc_bootstrap_ci.csv'),(counts,'status_counts_by_variant.csv'),
        (transitions,'status_transition_from_full.csv'),(pairs,'variant_pairwise_summary.csv'),(table_rows,'table2_which_parts_of_phase2_matter.csv')]:save(items,OUT/'results'/filename)
    table='| Setup | AUROC |\n|---|---|\n'+''.join(f"| {r['Setup']} | {r['AUROC']} |\n" for r in table_rows)
    title='Table 2: Which parts of Phase II matter?\n\n'
    warning='INCOMPLETE: blocked variants were not run and have no AUROC. Their status counts and outcomes are blank (not evaluated).\n'
    for ext in ('md','txt'):text(OUT/f'results/table2_which_parts_of_phase2_matter.{ext}',title+table+'\n'+warning)
    tex='\\begin{table}[t]\n\\centering\n\\caption{Which parts of Phase II matter?}\n\\label{tab:phase2-ablation}\n\\begin{tabular}{lc}\n\\hline\nSetup & AUROC \\\\\n\\hline\n'
    tex+=''.join(r['Setup']+' & '+r['AUROC']+' \\\\\n' for r in table_rows)
    tex+='\\hline\n\\end{tabular}\n\\par\\smallskip\\footnotesize Incomplete: blocked variants were not run.\n\\end{table}\n'
    text(OUT/'results/table2_which_parts_of_phase2_matter.tex',tex)
    source_text='# Source artifacts\n\n'+''.join(f"- `{p}`: {v['purpose']}. SHA-256 `{v['sha256']}`.\n" for p,v in SOURCES.items())
    text(OUT/'audit/source_artifacts.md',source_text)
    preflight=f'''# Preflight

Evaluation-set SHA-256: {eval_hash}. Matches Table 1 manifest. Exact 188 unique IDs in original order; 94 wrong and 94 correct. No evaluation-set resampling.

Full-system Phase-II AUROC: {expected!r}; exact binary float equality with Table 1 verified. Ground truth and classifier confidence are absent from saved Phase-II runtime payloads. No combined score is used.

Registered LLM configuration: {json.dumps(config)}

Endpoint check: {json.dumps(model_check)}

Missing frozen dependencies:
{chr(10).join('- `'+p+'`' for p in missing)}

Frozen validator, contract, evidence, tokenizer, context, prompt and schema identities checked. All 188 original prompt/schema hashes were reproduced by existing helpers without inference. Rules use the frozen statistics/claim registry and Stage-5 verification artifacts directly; no source statistics or classifiers were regenerated. Original development loaders depend on the absent results/rag/v5f_validation tree; the rules row reuses final-test verification artifacts and checks input hashes recorded in the frozen execution configuration.

No fresh inference was attempted. No model was started or substituted. No reranking/TF-IDF/BGE variants executed. No provenance variant executed: no registered bypass exists and provenance is enforced in downstream helpers too.
'''
    text(OUT/'audit/preflight.md',preflight)
    text(OUT/'audit/integrity_checks.md',preflight+'\nCompleted rows: no duplicate/missing statuses, all IDs resolved, rules passed existing validator, Full frozen validation PASS. Blocked rows: 188 missing statuses each, explicitly BLOCKED, not converted to fallback predictions. One intended change only in executed rules row.\n\nProject-wide before/after SHA-256 comparison is recorded in main_project_change_report.md.\n')
    readme=f'''# Experiment 2: Which parts of Phase II matter?

{warning}

The exact Table 1 balanced set is retained (188 IDs; 94 Phase-I errors plus the original 94 correct cases, seed 42). Its hash is {eval_hash}. There was no new selection. Classifier error is defined by predicted class != post-hoc ground truth.

Full system reuses the authoritative Table 1 statuses; full-precision AUROC is {expected!r}. Rules-only calls `src/rag/association_response_v2.py:admissible`, which calls the existing qualifying-evidence logic. It uses unchanged frozen retrieval/provenance/reranking/contexts and removes the LLM decision mechanism. Structured output passes `positive_evidence_v5.validate`. Rule citations contain all qualifying claims; no LLM narrative is generated or claimed. The original prompt/schema already constrain evidence status to the deterministic admissible state, so identical statuses are expected by construction. This measures the status decision, not explanation quality.

{title}{table}
Changes/plans:
{chr(10).join('- '+n+': '+c for n,c in zip(NAMES,changes))}

Blocked variants: {blocked}

No provenance check additionally requires a defensible scoped bypass across downstream gates; skipping only Stage 5 would not remove those checks. No guessed thresholds, rankings, fake VERIFIED labels or blanket validation bypass were used.

LLM configuration: {json.dumps(config)}. No fresh LLM inference occurred. Full and rules-only need none. The other variants require fresh inference whenever their context/prompt changes, with the registered local server and completed preflight; no frozen downstream responses were reused for changed inputs.

AUROC is ROC-AUC(classifier_error, phase2_disagree), with SUPPORTED=0 and CONTRADICTED/CONFLICTING/INSUFFICIENT=1. No ordinal status score, classifier confidence or logistic score is used. Supporting 95% CIs use 10,000 paired stratified bootstrap draws (94 errors and 94 correct each), numpy default_rng(42), percentile interval. For a binary score, bootstrap AUROC is exactly (detection rate + 1 - false alarm rate)/2. The same bootstrap indices are shared between completed variants; they do not change the evaluation cohort.

Largest observed change among completed ablations: rules-only changes {pairs[1]['status_changes']} statuses and AUROC by {metrics[1]['auroc_full_precision']-expected:.17g}. No comparison across all components is possible while four rows are blocked. Baseline AUROC below 0.5 is not successful classifier-error detection. Error-detection AUROC is not overall provenance-aware RAG quality. The small balanced, prediction-stratified cohort limits generalization; intervals condition on these cases and do not account for flow-group dependence.

See audit/preflight.md for exact missing dependencies, audit/main_project_change_report.md for file safety, and logs/final_terminal_report.txt for outputs. All writes are confined to this directory. Table 1 and manuscript remain untouched; Table 3 was not started.

Reproduction: run project_hashes.py before, run_available_ablation.py, then project_hashes.py after using Python -B. The wrapper deliberately stops if dependencies change instead of silently enabling new inference.
'''
    text(OUT/'README.md',readme)
    for idx,slug in enumerate(SLUGS):
        directory=OUT/'variant_outputs'/slug
        text(directory/'source_artifacts.md',source_text)
        dump(directory/'run_manifest.json',dict(variant=slug,setup=NAMES[idx],run_status='COMPLETE' if idx<2 else 'BLOCKED',
            blocked_reason=reasons[slug],change=changes[idx],evaluation_set_sha256=eval_hash,query_ids=ids,random_seed=42,
            fresh_inference=False,registered_llm_configuration=config,source_artifacts=SOURCES,
            output_hashes={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='run_manifest.json'}))
    report=f'Existing main-project files modified: see mandatory SHA-256 after-comparison\nEvaluation-set hash: {eval_hash}\nTotal 188; wrong 94; correct 94\nFull-system AUROC: {expected!r}\n\n{title}{table}\nAUROC 95% CI:\n'+pd.DataFrame(cis).to_string(index=False)+'\n\nStatus changes:\n'+pd.DataFrame(pairs).to_string(index=False)+f'\n\nFresh inference performed: NONE. Changed-context variants blocked.\nRegistered model: {json.dumps(config)}\n\nBlocked variants: {", ".join(NAMES[2:])}\n{blocked}\n\n{provenance_block}\n\nMethodological warning: incomplete table; classifier-error detection is not overall RAG quality; deterministic schema fixes full-system evidence status; balanced small-cohort bootstrap is descriptive.\n'
    paths={p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file()}|{(OUT.relative_to(ROOT)/name).as_posix() for name in ('logs/final_terminal_report.txt','run_manifest.json','audit/main_project_file_hashes_after.csv','audit/main_project_change_report.md')}
    report+='\nEvery generated/planned audit output path:\n'+'\n'.join(sorted(paths))+'\n'
    text(OUT/'logs/final_terminal_report.txt',report)
    dump(OUT/'run_manifest.json',dict(timestamp=datetime.now(timezone.utc).isoformat(),status='PARTIAL_BLOCKED',
        evaluation_set_sha256=eval_hash,python=sys.executable,python_version=sys.version,random_seed=42,bootstrap_resamples=10000,
        source_artifacts=SOURCES,completed_variants=SLUGS[:2],blocked_variants={s:reasons[s] for s in SLUGS[2:]},
        output_hashes={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p!=OUT/'run_manifest.json'}))
    print(report)

if __name__=='__main__':main()
