"""Recover only artifacts whose exact frozen SHA-256 is proved."""
import csv
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'supervisor_requested_experiments/table2_ablation'
def sha(b):return hashlib.sha256(b).hexdigest()
def write(p,s):p.write_text(s,encoding='utf-8')
def git(*args):return subprocess.check_output(['git','--no-optional-locks',*args],cwd=ROOT)
def main():
    preflight=(OUT/'audit/preflight.md').read_text(encoding='utf-8')
    expected=re.findall(r'^- `(results/rag/v5f_validation/[^`]+)`',preflight,re.M)
    assert len(expected)==4
    cfgpath='results/rag/test/final_4300_v5_contract_v2/stage08_llm/execution/configuration.json'
    cfg=json.loads((ROOT/cfgpath).read_text(encoding='utf-8'))
    with (OUT/'audit/main_project_file_hashes_recovery_before.csv').open(encoding='utf-8',newline='') as f:inventory=list(csv.DictReader(f))
    purposes=['Registered hypothesis-balanced retrieval, BGE/index identity and candidate limits',
        'Registered task-relevance reranking method, weights, preferences and top-k',
        'Registered evidence-diverse context policy, final top-5 and 2048-token budget',
        'Parent contract-v2 LLM configuration, tokenizer/schema/validator provenance']
    plan='# Recovery plan\n\nSearch all current project files (including hidden paths/environments/.git) by SHA-256 and basename, then local Git history for the exact path. Configuration IDs and contents are corroborative only; no similarly named substitute is accepted. No original path is recreated.\n\n'
    records=[];search=[]
    for path,purpose in zip(expected,purposes):
        target=cfg['input_sha256'][path]
        plan+=f'## {Path(path).name}\n\nExpected path: `{path}`\n\nPurpose: {purpose}.\n\nExpected SHA-256: `{target}`.\n\nReferences: `{cfgpath}` input_sha256 and `src/rag/final_test/contracts.py` / `preflight.py`. Required to establish unchanged components for all four variants, including No reranking (registered baseline even where ranking itself is bypassed).\n\n'
        candidates=[r for r in inventory if Path(r['path']).name==Path(path).name or r['sha256']==target]
        search.extend(dict(expected_path=path,candidate_path=r['path'],candidate_sha256=r['sha256'],exact_hash=r['sha256']==target) for r in candidates)
        matches=[r for r in candidates if r['sha256']==target]
        if matches:
            resolved=matches[0]['path'];note='Current file exact hash; referenced in place';identity='exact SHA-256'
        else:
            commits=git('log','--all','--format=%H','--',path).decode().splitlines()
            found=None
            for commit in commits:
                probe=subprocess.run(['git','--no-optional-locks','show',commit+':'+path],cwd=ROOT,capture_output=True)
                if probe.returncode:continue
                raw=probe.stdout
                for mode,data in [('raw_git_blob',raw),('restore_CRLF',raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))]:
                    search.append(dict(expected_path=path,candidate_path=commit+':'+path+'#'+mode,candidate_sha256=sha(data),exact_hash=sha(data)==target))
                    if sha(data)==target:found=(commit,mode,data);break
                if found:break
            if not found:
                records.append(dict(expected_path=path,resolved_path='',identity_check='NO_EXACT_MATCH',hash_match=False,status='UNRESOLVED',notes='No hash-equivalent current or historical artifact'));continue
            commit,mode,data=found
            destination=OUT/'data/recovered_configs'/Path(path).parent.name/Path(path).name
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(data)
            resolved=destination.relative_to(ROOT).as_posix()
            identity='exact frozen SHA-256 from Git blob ('+mode+')'
            note=f'Git {commit}:{path}; {mode}. Materialization needed because existing filesystem readers cannot consume a Git object directly. Original repository paths remain untouched.'
        records.append(dict(expected_path=path,resolved_path=resolved,identity_check=identity,hash_match=True,status='RESOLVED',notes=note))
    write(OUT/'audit/recovery_plan.md',plan)
    for filename,data in [('missing_artifact_resolution.csv',records),('artifact_search_candidates.csv',search)]:
        with (OUT/'audit'/filename).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    # Verify the parent signed configuration and the exact shared model/index identity.
    resolved={r['expected_path']:ROOT/r['resolved_path'] for r in records if r['status']=='RESOLVED'}
    assert len(resolved)==4,records
    parent=json.loads(resolved[expected[3]].read_text(encoding='utf-8'))
    assert parent['configuration_sha256']==cfg['parent_validation_configuration_sha256']
    identity=[]
    for path,h in cfg['tokenizer_sha256'].items():
        actual=sha((ROOT/path).read_bytes());assert actual==h
        identity.append(dict(path=path,expected_sha256=h,actual_sha256=actual,match=True))
    for path,h in cfg['input_sha256'].items():
        if path.startswith('artifacts/rag/v5f_validation/index/'):
            actual=sha((ROOT/path).read_bytes());assert actual==h
            identity.append(dict(path=path,expected_sha256=h,actual_sha256=actual,match=True))
    write(OUT/'audit/model_index_identity.json',json.dumps(identity,indent=2))
    manifest=json.loads((ROOT/'supervisor_requested_experiments/run_manifest.json').read_text())
    evaluation=ROOT/'supervisor_requested_experiments/table1_error_detection/data/error_detection_balanced_set.csv'
    assert sha(evaluation.read_bytes())==manifest['output_file_hashes']['table1_error_detection/data/error_detection_balanced_set.csv']
    rows=list(csv.DictReader(io.StringIO(evaluation.read_text(encoding='utf-8'))))
    ids=[r['query_id'] for r in rows];assert len(ids)==len(set(ids))==188
    wrong=sum(r['predicted_class']!=r['ground_truth'] for r in rows);assert wrong==94
    previous=json.loads((OUT/'variant_outputs/full_system/run_manifest.json').read_text())['query_ids']
    assert ids==previous
    canonical=json.dumps(ids,separators=(',',':'),ensure_ascii=False).encode()
    idhash=sha(canonical);assert idhash==sha(json.dumps(previous,separators=(',',':'),ensure_ascii=False).encode())
    write(OUT/'audit/evaluation_set_reverification.md',f'# Evaluation-set reverification\n\n188 unique real TEST-derived IDs; 94 wrong + 94 correct. No resampling, replacement or reordering.\n\nFile SHA-256: `{sha(evaluation.read_bytes())}` (matches Table 1 manifest).\n\nOrdered query-ID SHA-256: `{idhash}` (compact UTF-8 JSON array; exact equality with preserved Full-system IDs). Table 1 recorded its file hash rather than a separate ordered-ID hash; this audit computes the ID hash transparently from the unchanged file.\n')
    print(json.dumps(records,indent=2));print('Evaluation set and model/index hashes verified.')
if __name__=='__main__':main()
