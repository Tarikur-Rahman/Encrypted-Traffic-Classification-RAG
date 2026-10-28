"""Seal the blank annotation package after successful before/after integrity audit."""
import hashlib,json
from pathlib import Path
from workbook_io import LABELS,read_workbook
OUT=Path(__file__).resolve().parents[1]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    report=(OUT/'audit/main_project_change_report.md').read_text(encoding='utf-8')
    assert 'working files modified/deleted: 0' in report
    assert 'Working files added outside explanation evaluation: 0' in report
    assert not list((OUT/'results').iterdir())
    for i in (1,2):
        _,rows=read_workbook(OUT/f'annotation/annotator_{i}.xlsx')
        assert len(rows)==150 and all(r[k]=='' for r in rows for k in LABELS)
    lines=['150 explanations prepared: YES',
        'Existing authoritative explanation rubric found: NO',
        'Rubric used: user-specified binary Evidence grounded / Relevant / Clear; Overall acceptable = A AND B AND C',
        'Rubric is an experiment-side operationalization, not a previously used human explanation rubric.',
        f'Annotator-1 workbook: {OUT / "annotation/annotator_1.xlsx"}',
        f'Annotator-2 workbook: {OUT / "annotation/annotator_2.xlsx"}',
        f'Annotation guide: {OUT / "annotation/ANNOTATION_GUIDE.md"}',
        f'Validation script: {OUT / "scripts/validate_explanation_annotations.py"}',
        'Main-project working files modified: ZERO',
        'Working files added outside this experiment: ZERO',
        'Completed Tables 1, 2, 3 and 6 and manuscript unchanged.',
        'Git checkpoint metadata differences, if any, reported separately in audit/main_project_change_report.md.',
        'Labels automatically generated: ZERO; explanations regenerated: ZERO; metrics calculated: NONE',
        'WAITING FOR TWO HUMAN ANNOTATORS']
    (OUT/'logs/preannotation_final_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    manifest_path=OUT/'run_manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['main_project_working_files_modified']=0
    manifest['artifact_sha256']={p.relative_to(OUT).as_posix():sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p!=manifest_path}
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('\n'.join(lines))
if __name__=='__main__':main()
