"""Publish BGE re-audit integrity without rewriting completed research results."""
from pathlib import Path
import csv,json,hashlib
OUT=Path(__file__).resolve().parents[1]
def rows(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
before={r['path']:r for r in rows(OUT/'audit/main_project_file_hashes_bge_reaudit_before.csv')}
after={r['path']:r for r in rows(OUT/'audit/main_project_file_hashes_bge_reaudit_after.csv')}
changed=[p for p in before if before[p]!=after.get(p)]
added=sorted(set(after)-set(before))
working=[p for p in changed+added if not p.startswith('.git/')]
preserved=json.loads((OUT/'audit/bge_reaudit_preserved_outputs.json').read_text(encoding='utf-8'))
assert all(sha(OUT/p)==h for p,h in preserved.items()),'Completed outputs changed'
assert not working,working
integrity=dict(before_files=len(before),after_files=len(after),existing_main_project_working_files_modified=0,working_files_added_outside_table2=0,git_metadata_changed=changed,git_metadata_added=added,preserved_output_files=len(preserved),completed_outputs_hash_check='PASS')
(OUT/'audit/bge_reaudit_integrity.json').write_text(json.dumps(integrity,indent=2),encoding='utf-8')
report='# BGE re-audit integrity\n\nExisting main-project working files modified: **ZERO**. Files added outside Table 2, excluding Git metadata: **ZERO**.\n\n'+str(len(before))+' before files / '+str(len(after))+' after files hashed, excluding only table2_ablation. All '+str(len(preserved))+' preserved completed-variant and existing aggregate result files match their pre-audit SHA-256 hashes.\n\nGit metadata differences (reported separately):\n\n```json\n'+json.dumps(dict(changed=changed,added=added),indent=2)+'\n```\n'
(OUT/'audit/bge_reaudit_integrity.md').write_text(report,encoding='utf-8')
result=json.loads((OUT/'audit/bge_candidate_flow_result.json').read_text(encoding='utf-8'))
final='# BGE-only candidate-flow re-audit: BLOCKED\n\nRegistered depth: 10 BGE candidates per query, independently confirmed in code/config and all 43,000 frozen occurrences. Old wrapper wrong depth: NO. Reranked order for CIC-TEST-030839: raw BGE ranks 1–10. All ten passed provenance. Same 188 IDs/94 wrong/94 correct and ordered-ID hash verified.\n\n'
final+=(OUT/'audit/CIC-TEST-030839_bge_trace.md').read_text(encoding='utf-8')
final+='\n## Execution/metrics\n\nAll 188 BGE queries completed LLM inference: NO. Preflight replay completed 188; 187 fit exactly five and one does not. Fresh LLM responses: 0. Inference failures: 0 (not attempted); context-contract failures: 1. Evidence-status counts: NOT EVALUATED for all four statuses. AUROC and bootstrap 95% CI: unavailable; no values fabricated.\n\n'
final+=(OUT/'results/table2_which_parts_of_phase2_matter.md').read_text(encoding='utf-8')
final+='\n## Integrity and interpretation\n\n'+report+'\nThis is a limitation of the registered greedy context selection under this alternative retrieval pool. It does not prove no arbitrary five-chunk combination could fit, and no unregistered combinatorial replacement was attempted. Full-system success uses different candidate evidence. Table 3 was not started.\n'
(OUT/'logs/bge_reaudit_final_report.md').write_text(final,encoding='utf-8')
readme=OUT/'README.md'
readme.write_text(readme.read_text(encoding='utf-8')+'\nBGE candidate-flow re-audit: [depth audit](audit/bge_candidate_depth_audit.md), [query trace](audit/CIC-TEST-030839_bge_trace.md), [complete report](logs/bge_reaudit_final_report.md). All ten candidates were already used; unchanged replay remains blocked. Completed result files are byte-identical. Requested bge_only directory is a documented alias containing blocked records, not a separate executed variant.\n',encoding='utf-8')
manifest_path=OUT/'run_manifest.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
manifest['bge_candidate_flow_reaudit']=dict(status='BLOCKED',old_wrapper_wrong_depth=False,candidate_depth=10,fresh_responses=0,context_failures=1,report='logs/bge_reaudit_final_report.md',integrity=integrity)
manifest['output_hashes']={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p!=manifest_path and '/runtime_cache/' not in p.as_posix() and p.name!='recovery_vllm_server.log'}
manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print(final)
