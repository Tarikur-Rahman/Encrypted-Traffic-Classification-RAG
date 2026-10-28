"""Strict validation only; never fill, copy, merge, adjudicate or score labels."""
import argparse, csv, hashlib, json
from pathlib import Path
from workbook_io import LABELS, read_workbook
OUT=Path(__file__).resolve().parents[1]

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def validate_one(path,role,expected):
    headers,rows=read_workbook(path)
    if headers!=['annotator_id',*expected[0],*LABELS]:raise ValueError(f'{role}: unexpected columns')
    if len(rows)!=150:raise ValueError(f'{role}: expected exactly 150 rows')
    byid={r['query_id']:r for r in rows}
    if len(byid)!=150 or set(byid)!={r['query_id'] for r in expected}:
        raise ValueError(f'{role}: duplicate/missing/foreign query IDs')
    ordered=[]
    for baseline in expected:
        r=byid[baseline['query_id']]
        if r['annotator_id']!=role:raise ValueError(f'{role}: wrong annotator role; copied/swapped file')
        for k,v in baseline.items():
            if r[k]!=v:raise ValueError(f"{role}/{r['query_id']}: modified source field {k}")
        for k in LABELS:
            if r[k] not in ('0','1'):raise ValueError(f"{role}/{r['query_id']}: {k} must be 0 or 1; missing labels STOP")
        if int(r['overall_acceptable'])!=int(all(r[k]=='1' for k in LABELS[:3])):
            raise ValueError(f"{role}/{r['query_id']}: overall must equal A AND B AND C; human correction required")
        ordered.append(r)
    return ordered

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotator-1',type=Path,default=OUT/'annotation/annotator_1.xlsx')
    p.add_argument('--annotator-2',type=Path,default=OUT/'annotation/annotator_2.xlsx')
    p.add_argument('--annotator-1-id',help='Distinct human name or stable pseudonym')
    p.add_argument('--annotator-2-id',help='Distinct human name or stable pseudonym')
    p.add_argument('--independent-human-annotations',action='store_true',help='Attest both people worked independently, without copied/merged or machine labels')
    a=p.parse_args()
    try:
        if a.annotator_1.resolve()==a.annotator_2.resolve() or sha(a.annotator_1)==sha(a.annotator_2):
            raise ValueError('Two separate role-specific human workbooks required; same file is not acceptable')
        manifest=json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
        baseline=OUT/'data/annotation_expected_rows.csv'
        if sha(baseline)!=manifest['expected_rows_sha256']:raise ValueError('Immutable source baseline hash mismatch')
        with baseline.open(encoding='utf-8-sig',newline='') as f:expected=list(csv.DictReader(f))
        if len(expected)!=150 or len({r['query_id'] for r in expected})!=150:raise ValueError('Invalid source baseline')
        validate_one(a.annotator_1,'annotator_1',expected)
        validate_one(a.annotator_2,'annotator_2',expected)
        identities=[(v or '').strip().casefold() for v in (a.annotator_1_id,a.annotator_2_id)]
        if not all(identities) or identities[0]==identities[1] or not a.independent_human_annotations:
            raise ValueError('Distinct human identifiers and --independent-human-annotations attestation required')
    except (ValueError,KeyError,OSError) as exc:
        p.exit(1,f'STOP: {exc}\nNo labels inferred, changed or propagated. No metrics calculated.\n')
    print('VALID: two complete independently attested human workbooks; source text intact. No metrics calculated.')
    print('Authorship cannot be established from label patterns alone; preserve original files and independent collection records.')

if __name__=='__main__':main()
