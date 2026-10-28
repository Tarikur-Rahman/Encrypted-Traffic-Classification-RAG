"""Validate completed human CSV/XLSX annotations before producing any metric file."""
import argparse,csv,hashlib,json,math,sys
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from workbook_io import read_workbook

OUT=Path(__file__).resolve().parents[1]
METHODS=['TF-IDF','BGE','Hybrid (current)']
HEADERS=['query_id','dataset','predicted_class','structured_query','retrieval_method','rank','chunk_id','document_id','chunk_text','relevance']
TITLE='Table 3: Is retrieval finding relevant evidence?'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);return reader.fieldnames,list(reader)
def read_annotations(path):
    if path.suffix.lower()=='.xlsx':return read_workbook(path)
    if path.suffix.lower()=='.csv':return read_csv(path)
    raise ValueError('Input must be .csv or .xlsx')
def validate(headers,rows,expected,mapping):
    if headers!=HEADERS:raise ValueError('Annotation columns must match the supplied template exactly')
    if len(rows)!=2250:raise ValueError(f'Expected 2250 annotation rows; found {len(rows)}')
    method_values={r['retrieval_method'] for r in rows}
    blinded=method_values==set(mapping)
    if not blinded and method_values!=set(METHODS):raise ValueError('Use either all three blinded labels or all three original method names')
    if set(mapping.values())!=set(METHODS) or len(mapping)!=3:raise ValueError('Invalid method mapping')
    expected_map={(r['query_id'],r['retrieval_method'],str(r['rank'])):r for r in expected}
    if len(expected_map)!=2250:raise ValueError('Invalid canonical annotation identity file')
    clean=[];seen=set();groups=defaultdict(list);queries=set()
    for position,row in enumerate(rows,2):
        if None in row:raise ValueError(f'Extra CSV fields at row {position}')
        label=row['relevance']
        if label not in ('0','1'):raise ValueError(f'Human relevance label missing or invalid at row {position}; only 0 or 1 allowed. No metrics written.')
        r=dict(row)
        if blinded:r['retrieval_method']=mapping[r['retrieval_method']]
        key=(r['query_id'],r['retrieval_method'],str(r['rank']))
        if key in seen:raise ValueError(f'Duplicate query/method/rank at row {position}')
        seen.add(key)
        if key not in expected_map:raise ValueError(f'Unexpected query/method/rank at row {position}')
        for field in HEADERS[:-1]:
            if str(r[field])!=str(expected_map[key][field]):raise ValueError(f'Altered source field {field} at row {position}')
        r['relevance']=int(label);r['rank']=int(r['rank']);clean.append(r)
        groups[(r['query_id'],r['retrieval_method'])].append(r['rank']);queries.add(r['query_id'])
    if len(queries)!=150 or seen!=set(expected_map):raise ValueError('The exact 150 sampled query IDs are required')
    if len(groups)!=450 or any(sorted(ranks)!=[1,2,3,4,5] for ranks in groups.values()):raise ValueError('Every query/method must contain exactly ranks 1-5')
    if any({m for q,m in groups if q==i}!=set(METHODS) for i in queries):raise ValueError('Each query must have exactly three methods')
    return clean
def query_metrics(labels):
    if len(labels)!=5 or any(type(v) is not int or v not in (0,1) for v in labels):raise ValueError('Five binary labels required')
    dcg=sum(value/math.log2(rank+1) for rank,value in enumerate(labels,1))
    ideal=sum(value/math.log2(rank+1) for rank,value in enumerate(sorted(labels,reverse=True),1))
    return dict(relevant_count_top5=sum(labels),precision_at_5=sum(labels)/5,dcg_at_5=dcg,idcg_at_5=ideal,ndcg_at_5=dcg/ideal if ideal else 0.0)
def write(path,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def calculate(rows,query_ids):
    groups=defaultdict(list)
    for r in rows:groups[(r['query_id'],r['retrieval_method'])].append(r)
    per=[];summary=[];counts=[];cis=[]
    rng=np.random.default_rng(42);indices=rng.integers(0,150,size=(10000,150))
    for method in METHODS:
        values=[]
        for identity in query_ids:
            labels=[r['relevance'] for r in sorted(groups[(identity,method)],key=lambda r:r['rank'])]
            metrics=query_metrics(labels);values.append(metrics);per.append(dict(query_id=identity,retrieval_method=method,**metrics))
        mean={key:float(np.mean([v[key] for v in values])) for key in ['precision_at_5','ndcg_at_5']}
        summary.append(dict(retrieval_method=method,query_count=150,precision_at_5=mean['precision_at_5'],ndcg_at_5=mean['ndcg_at_5']))
        relevant=sum(v['relevant_count_top5'] for v in values)
        counts.append(dict(retrieval_method=method,annotation_rows=750,relevant=relevant,not_relevant=750-relevant,zero_relevant_queries=sum(v['relevant_count_top5']==0 for v in values)))
        for key in mean:
            samples=np.array([v[key] for v in values]);low,high=np.quantile(samples[indices].mean(axis=1),[.025,.975])
            cis.append(dict(retrieval_method=method,metric=key,estimate=mean[key],ci95_low=float(low),ci95_high=float(high),resamples=10000,seed=42))
    return per,summary,counts,cis
def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('annotation_file',type=Path)
    parser.add_argument('--human-annotated',action='store_true',help='Attest that final relevance labels were supplied by a person, not an LLM')
    parser.add_argument('--validate-only',action='store_true',help='Check labels and source identities without writing results')
    args=parser.parse_args(argv)
    if not args.human_annotated:parser.error('--human-annotated is required; final labels must come from a person')
    manifest=json.loads((OUT/'run_manifest.json').read_text(encoding='utf-8'))
    expected_path=OUT/'data/annotation_expected_rows.csv'
    if sha(expected_path)!=manifest['expected_rows_sha256']:raise ValueError('Canonical expected rows changed')
    sample_path=OUT/'data/query_sample_seed42.csv'
    if sha(sample_path)!=manifest['sample_file_sha256']:raise ValueError('Sample changed')
    _,expected=read_csv(expected_path)
    mapping=json.loads((OUT/'audit/blinded_method_mapping.json').read_text(encoding='utf-8'))
    headers,rows=read_annotations(args.annotation_file)
    clean=validate(headers,rows,expected,mapping)
    if args.validate_only:
        print('PASS: 2250 human-labeled rows, exact 150 queries, three methods, ranks 1-5, unchanged source fields. No metrics written.');return
    _,sample=read_csv(sample_path)
    per,summary,counts,cis=calculate(clean,[r['query_id'] for r in sample])
    table=[{'Retrieval method':r['retrieval_method'],'Precision@5':f"{r['precision_at_5']:.4f}",'nDCG@5':f"{r['ndcg_at_5']:.4f}"} for r in summary]
    # Nothing is published until the entire annotation file passes validation.
    results=OUT/'results'
    for name,values in [('table3_is_retrieval_finding_relevant_evidence.csv',table),('table3_metrics_full_precision.csv',summary),('per_query_retrieval_metrics.csv',per),('relevance_counts_by_method.csv',counts),('table3_bootstrap_ci.csv',cis)]:write(results/name,values)
    md=TITLE+'\n\n| Retrieval method | Precision@5 | nDCG@5 |\n|---|---|---|\n'+''.join(f"| {r['Retrieval method']} | {r['Precision@5']} | {r['nDCG@5']} |\n" for r in table)
    for ext in ['md','txt']:(results/f'table3_is_retrieval_finding_relevant_evidence.{ext}').write_text(md,encoding='utf-8')
    tex='\\begin{table}[t]\n\\centering\n\\caption{Is retrieval finding relevant evidence?}\n\\label{tab:retrieval-relevance}\n\\begin{tabular}{lcc}\n\\hline\nRetrieval method & Precision@5 & nDCG@5 \\\\\n\\hline\n'+''.join(r['Retrieval method']+' & '+r['Precision@5']+' & '+r['nDCG@5']+' \\\\\n' for r in table)+'\\hline\n\\end{tabular}\n\\end{table}\n'
    (results/'table3_is_retrieval_finding_relevant_evidence.tex').write_text(tex,encoding='utf-8')
    receipt=dict(status='COMPLETE',completed_at=datetime.now(timezone.utc).isoformat(),human_annotation_attested=True,input_file=str(args.annotation_file.resolve()),input_sha256=sha(args.annotation_file),rows=2250,queries=150,ndcg_convention='IDCG uses the same method/query five labels sorted descending; zero relevant implies nDCG=0',bootstrap='10000 paired query-level resamples; numpy default_rng(42); percentile 95% CI',output_sha256={p.name:sha(p) for p in results.iterdir() if p.is_file()})
    (OUT/'audit/human_annotation_metric_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    manifest.update(status='COMPLETE',labels_filled=2250,human_annotation_metric_receipt='audit/human_annotation_metric_receipt.json')
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(md)
if __name__=='__main__':
    try:main()
    except (ValueError,KeyError) as exc:print('VALIDATION FAILED: '+str(exc),file=sys.stderr);sys.exit(2)
