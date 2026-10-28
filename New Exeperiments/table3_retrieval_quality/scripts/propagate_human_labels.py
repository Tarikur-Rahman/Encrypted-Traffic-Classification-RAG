"""Copy complete human judgments to original blinded occurrences; never infer labels."""
import argparse,csv,hashlib,json,sys
from pathlib import Path
from collections import Counter
from workbook_io import read_workbook,write_workbook

OUT=Path(__file__).resolve().parents[1]
SOURCE=OUT/'annotation/retrieval_relevance_annotation_blinded.xlsx'
MAP=OUT/'annotation/duplicate_occurrence_map.csv'
UNIQUE_HEADERS=['query_id','dataset','predicted_class','structured_query','chunk_id','document_id','chunk_text','occurrence_count','relevance']
MAP_HEADERS=['query_id','chunk_id','original_annotation_row','blinded_method_code','rank']
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def derive_unique(original):
    unique={};mapping=[]
    for number,row in enumerate(original,2):
        key=(row['query_id'],row['chunk_id'])
        if row['relevance']!='':raise ValueError('Original template must remain unlabeled')
        value={h:row[h] for h in UNIQUE_HEADERS if h not in ('occurrence_count','relevance')}
        if key in unique:
            if any(unique[key][h]!=v for h,v in value.items()):raise ValueError('Conflicting source metadata/text for a query/chunk pair')
            unique[key]['occurrence_count']+=1
        else:unique[key]=dict(value,occurrence_count=1,relevance='')
        mapping.append(dict(query_id=key[0],chunk_id=key[1],original_annotation_row=str(number),blinded_method_code=row['retrieval_method'],rank=str(row['rank'])))
    return list(unique.values()),mapping
def propagate(original,headers,judgments,occurrence_map):
    expected,expected_map=derive_unique(original)
    if headers!=UNIQUE_HEADERS:raise ValueError('Unique-judgment columns must match the template exactly')
    if occurrence_map!=expected_map:raise ValueError('Occurrence map differs from original workbook row identities/order')
    lookup={(r['query_id'],r['chunk_id']):r for r in expected}
    if len(judgments)!=len(expected):raise ValueError('Missing or extra unique judgments')
    human={}
    for row in judgments:
        key=(row['query_id'],row['chunk_id'])
        if key not in lookup or key in human:raise ValueError('Unknown or duplicate query/chunk judgment')
        if row['relevance'] not in ('0','1'):raise ValueError('Missing or invalid HUMAN relevance for '+str(key)+'. STOP: every unique judgment must be exactly 0 or 1.')
        for h in UNIQUE_HEADERS[:-1]:
            if str(row[h])!=str(lookup[key][h]):raise ValueError('Altered source field '+h+' for '+str(key))
        human[key]=row['relevance']
    if set(human)!=set(lookup):raise ValueError('Missing human judgments')
    result=[dict(r,relevance=human[(r['query_id'],r['chunk_id'])]) for r in original]
    assert len(result)==len(original)
    assert all(r['relevance'] in ('0','1') for r in result)
    assert Counter((r['query_id'],r['chunk_id']) for r in result)==Counter({k:int(v['occurrence_count']) for k,v in lookup.items()})
    return result
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('completed_unique_workbook',type=Path)
    parser.add_argument('--human-annotated',action='store_true',help='Attest that a person supplied every relevance judgment')
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if not args.human_annotated:parser.error('--human-annotated is required; labels must be provided by a person')
    receipt=json.loads((OUT/'audit/unique_judgments_preparation.json').read_text(encoding='utf-8'))
    if sha(SOURCE)!=receipt['original_workbook_sha256'] or sha(MAP)!=receipt['occurrence_map_sha256']:raise ValueError('Original workbook or occurrence map changed')
    source_headers,original=read_workbook(SOURCE)
    if len(original)!=2250 or len({r['query_id'] for r in original})!=150:raise ValueError('Original 2250 rows / 150 queries required')
    if len({(r['query_id'],r['retrieval_method'],r['rank']) for r in original})!=2250:raise ValueError('Duplicate original occurrence')
    with MAP.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        if reader.fieldnames!=MAP_HEADERS:raise ValueError('Occurrence map headers changed')
        occurrence_map=list(reader)
    headers,judgments=read_workbook(args.completed_unique_workbook)
    result=propagate(original,headers,judgments,occurrence_map)
    if args.validate_only:print(f'PASS: {len(judgments)} complete human judgments map exactly to 2250 occurrences. No output written.');return
    xlsx=OUT/'annotation/retrieval_relevance_annotation_completed_from_unique.xlsx'
    csv_path=xlsx.with_suffix('.csv')
    if xlsx.exists() or csv_path.exists():raise ValueError('Completed output already exists; refusing to overwrite')
    # All labels and identity checks pass before any completed annotation is written.
    write_workbook(xlsx,source_headers,result)
    with csv_path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=source_headers);w.writeheader();w.writerows(result)
    h,roundtrip=read_workbook(xlsx)
    if h!=source_headers or roundtrip!=result:raise ValueError('Completed workbook failed serialization check')
    log=dict(status='HUMAN LABELS PROPAGATED; METRICS NOT CALCULATED',human_annotation_attested=True,input_sha256=sha(args.completed_unique_workbook),unique_human_judgments=len(judgments),output_occurrences=len(result),labels_inferred=0,output_sha256={xlsx.name:sha(xlsx),csv_path.name:sha(csv_path)})
    (OUT/'audit/human_label_propagation.json').write_text(json.dumps(log,indent=2),encoding='utf-8')
    print(json.dumps(log,indent=2))
if __name__=='__main__':
    try:main()
    except (ValueError,KeyError) as exc:print('STOP: '+str(exc),file=sys.stderr);sys.exit(2)
