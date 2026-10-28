"""Read-only frozen-artifact analysis. Run with Python -B; writes only task folder."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'supervisor_requested_experiments'
TASK = OUT / 'table1_error_detection'
os.environ['MPLCONFIGDIR'] = str(TASK / 'logs' / 'matplotlib')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
import csv
import json
import hashlib
import re
import subprocess
from datetime import datetime, timezone
from collections import Counter
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.rag.final_test import contracts as C

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def inventory():
    result = {}
    for directory, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if not (Path(directory) / d == OUT) and d != '.git']
        for name in files:
            p = Path(directory) / name
            stat = p.stat()
            result[p.relative_to(ROOT).as_posix()] = [stat.st_size, stat.st_mtime_ns]
    return result

def write(path, text):
    path.write_text(text, encoding='utf-8')

def save(frame, folder, name):
    frame.to_csv(TASK / folder / name, index=False, float_format='%.17g')

sources = {}
def use(name, reason):
    p = ROOT / name
    if not p.is_file():
        raise RuntimeError('Missing prerequisite: ' + name)
    sources[name] = {'purpose': reason, 'sha256': digest(p)}
    return p

def rows(name, reason):
    with use(name, reason).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def indexed(records):
    result = {r['query_id']: r for r in records}
    assert len(result) == len(records), 'Duplicate query IDs'
    return result

def main():
    baseline_path = OUT / 'audit/existing_file_inventory_before.json'
    if not baseline_path.exists():
        write(baseline_path, json.dumps(inventory(), indent=2))
    before = json.loads(baseline_path.read_text())
    git_status = subprocess.check_output(['git','--no-optional-locks','status','--porcelain'], cwd=ROOT, text=True)
    git_hash = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    source_code = ['README.md','RESEARCH_PROTOCOL.md','configs/project_protocol.json',
        'prepare_final_paper_figures.py','src/rag/final_test/contracts.py',
        'src/rag/final_test/downstream.py','src/rag/final_test/llm.py',
        'src/rag/run_evidence_supported_fusion.py','src/rag/materialize_phase1_interface.py',
        'src/rag/phase2_common.py','src/rag/phase2_foundation.py',
        'final_paper_figures/project_final_test_figures/Preparation_Report.json']
    for name in source_code:
        use(name, 'Read-only audit of authority, label isolation, frozen prediction or operating rules')
    # Preserve the complete rule-search evidence, including ignored metadata/config artifacts.
    search = subprocess.run(['rg','-n','-i','--no-ignore','-g','*.py','-g','*.json','-g','*.md','-g','*configuration.csv',
        '-g','*config*.csv','-g','*contract*.csv','-g','*protocol*',
        'threshold|combiner|logistic|error.detect','configs','src','metadata','results/rag','RESEARCH_PROTOCOL.md'],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    write(OUT / 'audit/operating_rule_search.txt', search.stdout + search.stderr)
    assert search.returncode in (0,1), 'Operating-rule search failed'
    for name in sorted(set(re.findall(r'^(.+?):\d+:', search.stdout, flags=re.MULTILINE))):
        use(name.replace('\\','/'), 'Existing operating-rule search match; audited for applicability')
    base = C.BASE
    pred_path = base+'stage01_structured_queries/test_query_records.csv'
    manifest_path = base+'stage01_structured_queries/cohort_manifest.csv'
    final_path = base+'stage10_final/test_final_classification_records.csv'
    eval_path = base+'stage10_final/test_evaluation_records.csv'
    phase_path = base+'stage08_llm/test_llm_responses.jsonl'
    context_path = base+'stage07_trusted_contexts/trusted_context_records.csv'
    config_path = base+'stage08_llm/execution/configuration.json'
    queries = indexed(rows(pred_path, 'Authoritative retained Phase-I predictions and saved labeled probabilities'))
    cohort = indexed(rows(manifest_path, 'Exact final comparison-set membership'))
    final = indexed(rows(final_path, 'Verify retained predictions and final evidence status'))
    evaluation = indexed(rows(eval_path, 'Cross-check independently reconstructed post-hoc correctness'))
    truth = indexed(rows(C.GT_PATH, 'Evaluation-only ground truth joined by exact query ID'))
    contexts = indexed(rows(context_path, 'Inspect actual Phase-II context fields for ground-truth isolation'))
    config = json.loads(use(config_path, 'Frozen execution receipt: no ground truth or correctness used; prompt hashes').read_text())
    responses = indexed([json.loads(line) for line in use(phase_path, 'Frozen accepted Phase-II verdicts; no inference').read_text().splitlines()])
    assert len(queries) == C.TOTAL
    assert all(set(x) == set(queries) for x in (cohort, final, evaluation, contexts, responses))
    assert set(queries) <= set(truth)
    assert config['ground_truth_used'] is False and config['correctness_used'] is False
    assert set(config['selected_query_ids']) == set(queries)
    records = []
    mismatches = []
    max_sum_error = 0.0
    for identity in sorted(queries):
        q, t, f, e, r = queries[identity], truth[identity], final[identity], evaluation[identity], responses[identity]
        C.query_check(q)
        C.assert_safe(contexts[identity])
        C.assert_safe(r)
        assert all(v['source_dataset'] == q['source_dataset'] for v in (t,f,e,cohort[identity],contexts[identity]))
        p = json.loads(q['classifier_probability_vector'])
        assert p == json.loads(q['class_probabilities_json']) == json.loads(f['classifier_probabilities']) == json.loads(e['classifier_probabilities'])
        assert all(np.isfinite(v) and 0 <= v <= 1 for v in p.values())
        max_sum_error = max(max_sum_error, abs(sum(p.values())-1))
        assert abs(sum(p.values())-1) < 1e-10
        confidence = float(q['classifier_confidence'])
        predicted = q['classifier_prediction']
        assert abs(confidence-p[predicted]) < 1e-10 and abs(confidence-max(p.values())) < 1e-10
        if predicted != max(p, key=p.get):
            mismatches.append(identity)
        assert predicted == f['classifier_prediction'] == f['final_class'] == e['final_class']
        assert t['true_class_label'] == e['true_class_label'] and t['source_split'] == 'test'
        correct = int(predicted == t['true_class_label'])
        assert correct == int(e['correct'])
        assert r['validation_status'] == 'PASS'
        status = r['validated_response']['evidence_status']
        assert status == r['effective_evidence_status'] == f['evidence_status'] == e['evidence_status']
        assert status in {'SUPPORTED','CONTRADICTED','CONFLICTING','INSUFFICIENT'}
        assert r['prompt_sha256'] == config['prompt_sha256'][identity]
        records.append(dict(query_id=identity,dataset=q['source_dataset'],ground_truth=t['true_class_label'],
            predicted_class=predicted,classifier_correct=correct,classifier_confidence=confidence,
            probability_vector=q['classifier_probability_vector'],phase2_evidence_status=status,
            phase2_disagree=int(status!='SUPPORTED'),confidence_error_score=1-confidence,
            source_prediction_artifact=pred_path,source_phase2_artifact=phase_path,source_ground_truth_artifact=C.GT_PATH,
            source_row_index=q['source_row_index'],classifier_run_id=q['classifier_run_id']))
    all_data = pd.DataFrame(records)
    wrong = all_data.loc[all_data.classifier_correct.eq(0)].copy()
    correct = all_data.loc[all_data.classifier_correct.eq(1)].copy()
    assert len(wrong) >= 5 and len(correct) >= len(wrong)
    counts = all_data.groupby(['dataset','classifier_correct']).size().to_string()
    audit = f'''# Integrity checks

Reconstructed directly from retained Phase-I predictions versus evaluation-only labels before sampling or metrics.

- Unique final comparison IDs: {len(all_data)}; duplicate IDs: 0.
- Correct: {len(correct)}; wrong: {len(wrong)}. No expected error count was hard-coded.
- Exact IDs agree across cohort, queries, final records, evaluation, contexts and responses.
- Missing Phase-II statuses: 0; missing confidence: 0; missing ground truth: 0.
- All probability vectors finite, in [0,1], normalized within 1e-10; maximum sum deviation: {max_sum_error:.17g}.
- Saved confidence equals predicted-class/max probability within 1e-10.
- Predicted-class versus dictionary argmax mismatches: {len(mismatches)}; IDs: {mismatches}.
- Frozen Stage-8 config records ground_truth_used=false and correctness_used=false.
- Existing query_check/assert_safe helpers passed on all queries, contexts and responses. Response prompt hashes match frozen config.
- src/rag/final_test/llm.py:prepare projects only query ID, dataset, split, prediction, traffic profile and trusted evidence. downstream.py:evaluate joins labels only after Stage 10.
- Thus stored processing inputs and code support ground-truth isolation. Original prompt CSV is absent; exact prompt text was not replayed or regenerated. This audit does not independently attest historical server execution.

Dataset breakdown (classifier_correct 0=wrong, 1=correct):
```
{counts}
```
'''
    write(OUT/'audit/integrity_checks.md',audit)
    print(f'AUDIT CONFIRMED before analysis: {len(all_data)} total, {len(correct)} correct, {len(wrong)} wrong.', flush=True)
    if '--audit-only' in sys.argv:
        return
    sample = correct.sample(n=len(wrong),random_state=42,replace=False)
    balanced = pd.concat([wrong,sample],ignore_index=True)
    assert len(sample)==len(wrong) and not set(sample.query_id)&set(wrong.query_id)
    y = 1-balanced.classifier_correct.to_numpy()
    X = balanced[['confidence_error_score','phase2_disagree']].to_numpy()
    scores = np.full(len(y),np.nan)
    folds = np.full(len(y),-1)
    model_details=[]
    for fold,(train,test) in enumerate(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(X,y)):
        assert not set(train)&set(test)
        model=LogisticRegression(random_state=42)
        model.fit(X[train],y[train])
        scores[test]=model.predict_proba(X[test])[:,1]
        folds[test]=fold
        model_details.append(dict(fold=fold,training_size=len(train),test_size=len(test),
            coefficients=model.coef_.tolist(),intercept=model.intercept_.tolist(),parameters=model.get_params()))
    assert np.isfinite(scores).all() and (folds>=0).all()
    balanced['phase2_error_score']=balanced.phase2_disagree
    balanced['combined_oof_error_score']=scores
    balanced['fold']=folds
    write(TASK/'logs/combiner_models.json',json.dumps(model_details,indent=2))
    target_far=float(balanced.loc[y==0,'phase2_disagree'].mean())
    methods=['Classifier confidence','Phase II evidence status','Both combined']
    score_arrays=[X[:,0],X[:,1],scores]
    threshold_records=[]
    metrics=[]
    sensitivities=[]
    flags=[]
    for method,s in zip(methods,score_arrays):
        candidates=np.r_[np.inf,np.unique(s)[::-1]]
        sensitivity=[]
        for threshold in candidates:
            flagged=s>=threshold
            sensitivity.append(dict(Method=method,threshold=float(threshold),
                detection_rate=float(flagged[y==1].mean()),false_alarm_rate=float(flagged[y==0].mean()),
                flagged_wrong=int(flagged[y==1].sum()),flagged_correct=int(flagged[y==0].sum())))
        if method == methods[1]:
            threshold=0.5
            rule='Supervisor binary Disagree definition (score >= 0.5)'
        else:
            # Tie break never consults error detection: minimize FAR gap, then lower FAR,
            # then highest threshold (conservative within an identical-FAR interval).
            chosen=min(sensitivity,key=lambda r:(abs(r['false_alarm_rate']-target_far),r['false_alarm_rate'],-r['threshold']))
            threshold=chosen['threshold']
            rule='matched-FAR operating point; minimum absolute FAR gap, then lower FAR, then highest threshold'
        flag=s>=threshold
        flags.append(flag)
        metrics.append({'Method':method,'Detection Rate':float(flag[y==1].mean()),
            'False Alarm Rate':float(flag[y==0].mean()),'AUROC':float(roc_auc_score(y,s))})
        threshold_records.append(dict(Method=method,threshold=threshold,comparison='score >= threshold',
            selection_method=rule,target_phase2_far=target_far,selection_scope='balanced analysis set',
            classifier_confidence_cutoff=(1-threshold if method==methods[0] else None)))
        for r in sensitivity:
            r.update(selected=r['threshold']==threshold,target_phase2_far=target_far)
        sensitivities.extend(sensitivity)
    for name,flag in zip(['confidence_flag','phase2_flag','combined_flag'],flags):
        balanced[name]=flag.astype(int)
    for frame,name in [(all_data,'all_4300_posthoc_correctness.csv'),(wrong,'all_wrong_cases.csv'),
        (sample,'random_correct_cases_seed42.csv'),(balanced,'error_detection_balanced_set.csv'),
        (balanced[['query_id','dataset','classifier_correct','fold','combined_oof_error_score']],'fold_assignments.csv')]:
        save(frame,'data',name)
    metric_frame=pd.DataFrame(metrics)
    save(metric_frame,'results','table1_metrics_full_precision.csv')
    save(metric_frame,'results','table1_can_phase2_detect_classifier_errors.csv')
    save(pd.DataFrame(threshold_records),'results','table1_thresholds.csv')
    sensitivity_frame=pd.DataFrame(sensitivities)
    save(sensitivity_frame.loc[sensitivity_frame.Method.eq(methods[0])],'results','confidence_threshold_sensitivity.csv')
    save(sensitivity_frame.loc[sensitivity_frame.Method.eq(methods[2])],'results','combined_threshold_sensitivity.csv')
    save(balanced.loc[y==1],'results','wrong_cases_phase2_status.csv')
    save(balanced.loc[y==0],'results','correct_sample_phase2_status.csv')
    status_rows=[]
    for scope,frame in [('full_4300',all_data),('balanced',balanced)]:
        for status in ['SUPPORTED','CONTRADICTED','CONFLICTING','INSUFFICIENT']:
            status_rows.append(dict(scope=scope,phase2_evidence_status=status,
                classifier_correct=int(((frame.phase2_evidence_status==status)&(frame.classifier_correct==1)).sum()),
                classifier_wrong=int(((frame.phase2_evidence_status==status)&(frame.classifier_correct==0)).sum())))
    save(pd.DataFrame(status_rows),'results','status_by_correctness.csv')
    categories=np.select([flags[0]&~flags[1],~flags[0]&flags[1],flags[0]&flags[1]],
        ['low confidence only','Phase II only','both'],default='neither')
    overlap=balanced.loc[y==1,['query_id','dataset','classifier_confidence','phase2_evidence_status','confidence_flag','phase2_flag']].copy()
    overlap['caught_by']=categories[y==1]
    save(overlap,'results','overlap_analysis.csv')
    overlap_counts=Counter(overlap.caught_by)
    order=['low confidence only','Phase II only','both','neither']
    save(pd.DataFrame([dict(caught_by=k,count=overlap_counts[k]) for k in order]),'results','overlap_summary.csv')
    table='| Method | Detection Rate | False Alarm Rate | AUROC |\n|---|---|---|---|\n'
    for r in metrics:
        table+=f"| {r['Method']} | {r['Detection Rate']:.2%} | {r['False Alarm Rate']:.2%} | {r['AUROC']:.4f} |\n"
    heading='Table 1: Can Phase II detect classifier errors?\n\nData: every test flow the classifier got wrong, plus the same number of randomly chosen flows it got right.\n\nFlag: Phase II output is CONTRADICTED, CONFLICTING or INSUFFICIENT.\n\n'
    definitions='\nDetection rate = flagged wrong flows ÷ total wrong flows\n\nFalse alarm rate = flagged correct flows ÷ total correct flows\n'
    note='\nClassifier confidence and Both combined use a matched-FAR operating point selected on this analysis set. Rates are descriptive; these are not independently validated deployment thresholds. CSV values are fractions.\n'
    for extension in ['md','txt']:
        write(TASK/f'results/table1_can_phase2_detect_classifier_errors.{extension}',heading+table+definitions+note)
    tex='\\begin{table}[t]\n\\centering\n\\caption{Can Phase II detect classifier errors?}\n\\label{tab:phase2-error-detection}\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lccc}\n\\hline\nMethod & Detection Rate & False Alarm Rate & AUROC \\\\\n\\hline\n'
    for r in metrics:
        tex+=f"{r['Method']} & {100*r['Detection Rate']:.2f}\\% & {100*r['False Alarm Rate']:.2f}\\% & {r['AUROC']:.4f} "+'\\\\\n'
    tex+='\\hline\n\\end{tabular}\n\\par\\smallskip\\footnotesize\nData: all wrong flows plus an equally sized random correct sample.\nFlag: CONTRADICTED, CONFLICTING or INSUFFICIENT.\nDetection rate = flagged wrong flows $\\div$ total wrong flows.\nFalse alarm rate = flagged correct flows $\\div$ total correct flows.\nConfidence and combined rows: matched-FAR operating point.\n\\end{table}\n'
    write(TASK/'results/table1_can_phase2_detect_classifier_errors.tex',tex)
    fig,ax=plt.subplots(figsize=(6,5))
    for method,s,r in zip(methods,score_arrays,metrics):
        fpr,tpr,_=roc_curve(y,s)
        ax.plot(fpr,tpr,label=f"{method} (AUC {r['AUROC']:.4f})",marker='o' if method==methods[1] else None)
    ax.plot([0,1],[0,1],'k--',alpha=.4)
    ax.set(xlabel='False Alarm Rate',ylabel='Detection Rate',title=f'Classifier error detection ({len(y)} balanced queries)',xlim=(0,1),ylim=(0,1.02))
    ax.legend(fontsize=8,loc='lower right');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(TASK/f'figures/error_detection_roc.{ext}',dpi=300)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    bars=ax.bar(order,[overlap_counts[k] for k in order]);ax.bar_label(bars)
    ax.set(ylabel='Wrong Phase-I cases',title='Error-detection overlap at matched FAR');fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(TASK/f'figures/error_detection_overlap.{ext}',dpi=300)
    plt.close(fig)
    rules='''No registered classifier-error detection threshold or confidence-plus-status error-score combiner was found in the audited protocol/config/source scope (search evidence saved). The USTC 0.5 threshold determines its benign/malware class; it is not an error-detection threshold. The legacy evidence_override confidence <=0.5 rule changes a class only under an optional policy. The authoritative final cohort freezes retain_classifier and forbids overrides. That fusion rule remains unchanged and does not define the requested error score.

The project normally prohibits TEST threshold tuning. This explicitly requested supervisor analysis uses matched FAR on the analysis set and a new out-of-fold error combiner as descriptive post-hoc evaluation only, not a change to frozen Phase-I/II methods or a deployment operating rule.
'''
    write(OUT/'audit/operating_rule_audit.md',rules)
    readme=f'''# Supervisor-requested Experiment 1

Tested whether frozen Phase-II disagreement detects retained Phase-I classifier mistakes. Ground truth was used only in this post-hoc evaluation. Independently reconstructed {len(wrong)} wrong and {len(correct)} correct out of {len(all_data)} queries; these errors compare Phase-I predicted class against ground truth, not LLM accuracy.

All {len(wrong)} wrong cases were retained. Correct cases were sorted by query_id then sampled without replacement with pandas sample(n={len(wrong)}, random_state=42); balanced set size {len(balanced)}. Wrong rows (ID order) precede the correct sample (sample-return order).

Disagree = 1 for CONTRADICTED, CONFLICTING or INSUFFICIENT; SUPPORTED = 0. Confidence error score = 1 minus the saved maximum/predicted-class probability. No probabilities were recomputed.

Both combined uses sklearn LogisticRegression defaults (C=1, L2, lbfgs, intercept, no class weighting or feature scaling; full parameters saved) on confidence_error_score and phase2_disagree. StratifiedKFold(5, shuffle=True, random_state=42) predicts each row using only the other four folds. Target 1 means Phase-I wrong. No Phase-I classifier was retrained. Fold assignments and OOF probabilities are saved.

AUROC uses sklearn roc_auc_score with error as the positive class. Phase-II AUROC uses binary Disagree, with no ordinal evidence ranking. Combined AUROC uses only OOF probabilities.

{rules}
For continuous scores, flag score >= threshold. Among all distinct score thresholds plus infinity, choose closest FAR to Phase-II FAR; ties use lower FAR, then highest threshold. This does not optimize detection rate. Full threshold sensitivities and actual thresholds are saved. Phase-II flags Disagree directly.

{heading}{table}{definitions}{note}
Overlap among wrong cases: confidence only {overlap_counts['low confidence only']}; Phase II only {overlap_counts['Phase II only']}; both {overlap_counts['both']}; neither {overlap_counts['neither']}.

Phase II catches {overlap_counts['Phase II only']} errors missed by confidence at this operating point, with false alarms on {int(flags[1][y==0].sum())}/{len(sample)} sampled correct cases. Combined AUROC {'improves' if metrics[2]['AUROC']>metrics[0]['AUROC'] else 'does not improve'} over confidence in this sample. This is a point estimate, not a significance claim.

Reused the final_4300_v5_contract_v2 cohort, Stage-1 probabilities, Stage-8 accepted responses, Stage-10 records and separate evaluation-only labels. No fresh classifier or LLM inference occurred. See ../audit/source_artifacts.md for paths and run_manifest.json for hashes.

Limitations: original cohort is prediction-stratified; the balanced case-control sample is not natural error prevalence. Matched-FAR thresholds are selected on the analysis set. OOF combiner is evaluated on a small sample without uncertainty intervals and may reflect dataset composition. Original prompt CSV is absent; stored contexts, code, frozen config and response hashes support label isolation but do not independently replay historical prompts. Existing prepare_final_paper_figures.py was already git-modified before work; its content was preserved.

Reproduce from repository root: `.venv/Scripts/python.exe -B supervisor_requested_experiments/table1_error_detection/scripts/run_experiment.py`. All writes, including Matplotlib cache, stay under supervisor_requested_experiments. Table 2 and other experiments were not started.
'''
    write(OUT/'README.md',readme)
    write(TASK/'README.md',readme)
    write(OUT/'audit/source_artifacts.md','# Source artifacts\n\n'+''.join(f"- `{n}`: {v['purpose']}. SHA-256: `{v['sha256']}`.\n" for n,v in sources.items())+'\nAdditional rule-search inputs are identified by filename/line in operating_rule_search.txt; they were not used as numerical data.\n')
    after=inventory()
    changed=[n for n,s in before.items() if after.get(n)!=s]
    added=sorted(set(after)-set(before))
    source_changed=[n for n,v in sources.items() if digest(ROOT/n)!=v['sha256']]
    assert not changed and not added and not source_changed, (changed,added,source_changed)
    audit+=f'\n- Correct sample size: {len(sample)} = wrong count; no overlapping IDs; random seed 42.\n- OOF coverage: every selected ID exactly once in held-out fold, five folds.\n- Existing project files modified/deleted: 0; files generated outside task folder: 0.\n- Verification: full non-.git file size/mtime inventory ({len(before)} files) unchanged; all source SHA-256 hashes unchanged. .git inspected read-only with --no-optional-locks.\n- Pre-existing git status: `{git_status.strip()}`. This prior modification was preserved.\n'
    write(OUT/'audit/integrity_checks.md',audit)
    report=f'Existing project files modified: ZERO\nFull comparison set: {len(all_data)}\nWrong: {len(wrong)}\nCorrect: {len(correct)}\nRandom correct sample: {len(sample)}\nBalanced set: {len(balanced)}\n\n{table}\nOverlap: {dict((k,overlap_counts[k]) for k in order)}\n\nSource artifacts:\n'+'\n'.join(sources)+'\n\nLimitations: descriptive matched-FAR thresholds; small balanced, prediction-stratified cohort; missing original prompt CSV limits retrospective prompt verification.\n'
    report+='\nEvery output path (including manifest and this report):\n'+'\n'.join(sorted({p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file()}|{'supervisor_requested_experiments/run_manifest.json','supervisor_requested_experiments/table1_error_detection/logs/final_terminal_report.txt'}))+'\n'
    write(TASK/'logs/final_terminal_report.txt',report)
    manifest=dict(timestamp=datetime.now(timezone.utc).isoformat(),git_commit=git_hash,git_status_before=git_status,
        python_executable=sys.executable,python_version=sys.version,package_versions={k:v.__version__ for k,v in [('numpy',np),('pandas',pd),('sklearn',sklearn),('matplotlib',matplotlib)]},
        source_artifacts=sources,random_seed=42,wrong_case_count=len(wrong),correct_count=len(correct),sampled_correct_count=len(sample),
        full_comparison_size=len(all_data),total_evaluation_size=len(balanced),thresholds=threshold_records,
        existing_files_modified=0,inventory_verification='All existing non-.git sizes/mtime unchanged; source SHA-256 unchanged',
        output_file_hashes={p.relative_to(OUT).as_posix():digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='run_manifest.json'},
        manifest_hash_note='Manifest excludes its own hash to avoid self-reference')
    write(OUT/'run_manifest.json',json.dumps(manifest,indent=2,allow_nan=False))
    print(report)

if __name__=='__main__':
    main()
