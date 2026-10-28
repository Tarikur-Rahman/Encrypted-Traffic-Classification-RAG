"""Finalize documentation and hashes; never calculate experiment metrics."""
import csv,json
from pathlib import Path
from project_integrity import sha
from workbook_io import read_workbook
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1]
manifest_path=OUT/'run_manifest.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
assert manifest['status']=='WAITING FOR HUMAN ANNOTATION'
assert not list((OUT/'results').iterdir()),'Final metric files must not exist before human annotation'
assert all(sha(ROOT/p)==h for p,h in manifest['source_artifact_sha256'].items())
for name in ['retrieval_relevance_annotation.xlsx','retrieval_relevance_annotation_blinded.xlsx']:
    _,rows=read_workbook(OUT/'annotation'/name)
    assert len(rows)==2250 and all(r['relevance']=='' for r in rows)
with (OUT/'annotation/retrieval_relevance_annotation.csv').open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f));assert len(rows)==2250 and all(r['relevance']=='' for r in rows)
integrity=(OUT/'audit/main_project_change_report.md').read_text(encoding='utf-8')
assert 'working files modified/deleted: 0' in integrity and 'Working files added outside Table 3: 0' in integrity
readme='''# Table 3: Is retrieval finding relevant evidence?

**WAITING FOR HUMAN ANNOTATION**

Prepared 150 unique real queries and 2,250 unlabeled retrieval occurrences: 750 each for TF-IDF, BGE, and Hybrid (current). Every method uses the same 150 IDs. All relevance cells remain blank; no final metrics have been calculated. The `results/` directory is intentionally empty.

The sample is a uniform draw without replacement using Python `random.Random(42).sample` from lexicographically sorted IDs in the authoritative frozen 4,300-query manifest. Sampling depends only on IDs. No ground truth, correctness, evidence status, confidence, or retrieval result was used to select queries. This is not the balanced Table-1/Table-2 sample. The parent 4,300-query cohort is a fixed historical comparison rather than a naturally weighted traffic population.

Raw retrieval is reused from the authentic registered Stage-4 artifact. Saved ranks 1–5 are preserved for each method, including the registered within-method hypothesis balancing. No Stage-6 reranking, Stage-7 context packing, token-budget filtering, LLM execution, or evidence-status reasoning was performed. The BGE four-chunk context issue from Table 2 does not apply here.

## Files for the human annotator

- Preferred: [blinded workbook](annotation/retrieval_relevance_annotation_blinded.xlsx).
- Instructions and illustrative examples: [annotation guide](annotation/ANNOTATION_GUIDE.md).
- Alternative unblinded templates: [workbook](annotation/retrieval_relevance_annotation.xlsx), [CSV](annotation/retrieval_relevance_annotation.csv).

Give the annotator only the blinded workbook and guide while labeling. The actual method mapping is stored only in `audit/blinded_method_mapping.json`. Method groups are randomly ordered within each query, while original ranks remain intact. Full query/chunk text is retained; expand rows or use the formula bar to read long cells. No labels may be supplied by Codex, Qwen, ChatGPT, or another LLM.

## After human labels are returned

Use [calculate_table3_metrics.py](scripts/calculate_table3_metrics.py) with the completed CSV/XLSX and `--human-annotated`. First add `--validate-only`; omit it only after validation succeeds. See the guide for the command. The script verifies exact source fields and sampled identities as well as 2,250 binary labels, 150 queries, three methods/query, and ranks 1–5. Blank labels, duplicate rows, formulas, changed source text, or foreign queries are rejected before any output is written.

Precision@5 is the mean query-level relevant count divided by five. nDCG@5 uses each method/query's own five binary labels to construct IDCG; if none is relevant, nDCG is zero. This specified within-five normalization is not corpus-wide recall. Supporting percentile bootstrap intervals use 10,000 paired query-level resamples with seed 42 and remain outside the exact supervisor table. The script writes CSV, Markdown, TXT, IEEE-compatible LaTeX, full-precision/per-query metrics, relevance counts, and bootstrap CIs only after completed human labels pass validation.

## Traceability and safety

Source: `results/rag/test/final_4300_v5_contract_v2/stage01_structured_queries/cohort_manifest.csv` and its `test_query_records.csv`; retrieval: `stage04_retrieval/retrieval_candidates.csv`; evidence: `data/processed/rag/v5f_validation/evidence_chunks.csv`.

The annotation `structured_query` is the actual `retrieval_query_text`. The separate source `query_text` field contains confidence and was not exported. Annotation columns are explicitly whitelisted; no retrieval scores or outcome metadata are exposed. Raw outputs retain retrieval scores only for auditing.

'''
readme+='Sample seed: **42**. Sample file SHA-256: `'+manifest['sample_file_sha256']+'`. Ordered query-ID SHA-256: `'+manifest['ordered_query_ids_sha256']+'`. Source manifest SHA-256: `'+manifest['source_manifest_sha256']+'`.\n\n'
readme+='Existing main-project working files modified: **ZERO**, including Table 1 and Table 2. See [integrity report](audit/main_project_change_report.md), [sample audit](audit/sample_integrity.md), [retrieval audit](audit/retrieval_integrity.md), and [annotation audit](audit/annotation_integrity.md). No Table 4/runtime or manuscript work was started.\n'
(OUT/'README.md').write_text(readme,encoding='utf-8')
checks=dict(synthetic_unit_tests='5 passed: rank-weight arithmetic, zero-IDCG convention, blinded/unblinded identities, missing/invalid labels, tampering/duplicates/ranks',real_blinded_template_validation='Correctly rejected at first blank relevance cell; no metrics written',xlsx_round_trip='Both 2250-row workbooks retain exact complete strings and blank labels',all_final_results_absent=True,source_hashes_reverified=True)
(OUT/'logs/preannotation_validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
report='Existing main-project working files modified: ZERO\nFinal comparison source: '+manifest['source_manifest']+'\n150 unique real queries selected: YES\nSample seed: 42\nSample SHA-256: '+manifest['sample_file_sha256']+'\nTF-IDF retrieval: COMPLETE (authentic saved ranks reused), 750 rows\nBGE retrieval: COMPLETE (authentic saved ranks reused), 750 rows\nHybrid retrieval: COMPLETE (authentic saved ranks reused), 750 rows\nTotal annotation rows: 2250\n'
for name in ['annotation/retrieval_relevance_annotation.xlsx','annotation/retrieval_relevance_annotation_blinded.xlsx','annotation/retrieval_relevance_annotation.csv','annotation/ANNOTATION_GUIDE.md','scripts/calculate_table3_metrics.py']:report+=str((OUT/name).resolve())+'\n'
report+='\nAll relevance labels BLANK. Final metrics not calculated.\n\nWAITING FOR HUMAN ANNOTATION\n'
(OUT/'logs/preannotation_final_report.txt').write_text(report,encoding='utf-8')
manifest.update(existing_main_project_working_files_modified=0,validation=checks)
manifest['output_sha256']={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p!=manifest_path}
manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
print(report)
