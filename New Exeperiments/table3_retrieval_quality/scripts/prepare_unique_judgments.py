"""Prepare blank unique judgments from the existing blinded occurrence workbook."""
import csv,json
from zipfile import ZipFile
from propagate_human_labels import OUT,SOURCE,MAP,UNIQUE_HEADERS,MAP_HEADERS,sha,derive_unique
from workbook_io import read_workbook,write_workbook

def main():
    output=OUT/'annotation/retrieval_relevance_unique_human_judgments.xlsx'
    if output.exists() or MAP.exists():raise ValueError('Helper already exists; refusing to overwrite human work')
    protected={p.relative_to(OUT).as_posix():sha(p) for folder in ['data','retrieval_outputs','annotation','results'] for p in (OUT/folder).rglob('*') if p.is_file()}
    _,original=read_workbook(SOURCE)
    assert len(original)==2250 and len({r['query_id'] for r in original})==150
    assert {r['retrieval_method'] for r in original}=={'Method A','Method B','Method C'}
    unique,mapping=derive_unique(original)
    assert len(mapping)==2250 and sum(r['occurrence_count'] for r in unique)==2250
    write_workbook(output,UNIQUE_HEADERS,unique)
    with MAP.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=MAP_HEADERS);w.writeheader();w.writerows(mapping)
    headers,readback=read_workbook(output)
    assert headers==UNIQUE_HEADERS and readback==[{k:str(v) for k,v in r.items()} for r in unique]
    assert all(r['relevance']=='' for r in readback)
    with ZipFile(output) as z:
        assert all(not any(token in z.read(n).decode('utf-8') for token in ['TF-IDF','BGE','Hybrid (current)','dense_bge','hybrid_rrf','Method A','Method B','Method C']) for n in z.namelist())
    assert all(sha(OUT/p)==h for p,h in protected.items())
    report=dict(status='WAITING FOR HUMAN ANNOTATION',original_annotation_rows=2250,unique_judgments_required=len(unique),duplicate_occurrences_removed=2250-len(unique),labels_generated_automatically=0,labels_filled=0,deduplication_key=['query_id','chunk_id'],row_number_convention='original_annotation_row is the original blinded workbook Excel row number: header=1, data=2..2251',original_workbook_sha256=sha(SOURCE),unique_workbook_sha256=sha(output),occurrence_map_sha256=sha(MAP),preserved_files_sha256=protected)
    (OUT/'audit/unique_judgments_preparation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    guide='''# Deduplicated human annotation helper

Use `retrieval_relevance_unique_human_judgments.xlsx` to judge each query/chunk pair once. Only `query_id + chunk_id` was used for deduplication. All duplicate occurrences were checked for identical dataset, prediction, query, document and chunk text. The same chunk under a different query remains a separate judgment. `occurrence_count` is the number of original rows restored from your judgment, not an indication of relevance.

Read ANNOTATION_GUIDE.md for the relevance definition. Enter human-provided 0 or 1 only in `relevance`; every other field must remain unchanged. The sheet name remains `Annotations`. All cells are initially blank. No retrieval method identity or rank is included in this judgment workbook. Expand rows or use the formula bar to read complete query/chunk text.

The custodian should retain duplicate_occurrence_map.csv. Its original_annotation_row uses Excel row numbers in the existing blinded workbook (header 1, data 2–2251). It contains anonymous method codes only, and is used solely to restore judgments to occurrences, preserving original rank and method order.

After a person completes every unique judgment, run from the project root:

```powershell
.venv/Scripts/python.exe -B supervisor_requested_experiments/table3_retrieval_quality/scripts/propagate_human_labels.py "PATH_TO_COMPLETED_UNIQUE_WORKBOOK.xlsx" --human-annotated --validate-only
```

After validation succeeds, omit `--validate-only` to create `annotation/retrieval_relevance_annotation_completed_from_unique.xlsx` and `.csv`. The helper stops on blanks, invalid labels, missing/duplicate/unknown pairs, altered source text/counts, or a changed occurrence map/original workbook. It copies only supplied human values; it never infers a missing label. Existing templates and completed outputs are not overwritten. No metrics are calculated. The expanded file is compatible with the existing Table-3 metrics script for a separately authorized later step.
'''
    guide+=f'\nOriginal rows: 2250. Unique judgments: {len(unique)}. Duplicate occurrences removed from manual workload: {2250-len(unique)}. Labels generated automatically: ZERO.\n\nWAITING FOR HUMAN ANNOTATION\n'
    (OUT/'annotation/UNIQUE_JUDGMENTS_GUIDE.md').write_text(guide,encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='preserved_files_sha256'},indent=2))
if __name__=='__main__':main()
