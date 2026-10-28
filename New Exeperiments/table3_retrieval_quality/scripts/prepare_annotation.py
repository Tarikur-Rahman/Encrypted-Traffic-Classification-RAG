"""Sample IDs independently, reuse authentic Stage-4 ranks, leave all labels blank."""
import csv,hashlib,json,random,secrets,sys
from collections import Counter,defaultdict
from pathlib import Path
from zipfile import ZipFile
from workbook_io import write_workbook,read_workbook
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1]
BASE='results/rag/test/final_4300_v5_contract_v2/'
QUERY=BASE+'stage01_structured_queries/test_query_records.csv'
COHORT=BASE+'stage01_structured_queries/cohort_manifest.csv'
CANDIDATES=BASE+'stage04_retrieval/retrieval_candidates.csv'
CHUNKS='data/processed/rag/v5f_validation/evidence_chunks.csv'
METHODS={'tfidf':('TF-IDF','tfidf'),'dense_bge':('BGE','bge'),'hybrid_rrf':('Hybrid (current)','hybrid')}
HEADERS=['query_id','dataset','predicted_class','structured_query','retrieval_method','rank','chunk_id','document_id','chunk_text','relevance']
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows,headers=None):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=headers or list(rows[0]));w.writeheader();w.writerows(rows)
def dump(p,value):p.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    target=OUT/'annotation/retrieval_relevance_annotation_blinded.xlsx'
    if target.exists():raise RuntimeError('Annotation package already exists; refusing to overwrite human work')
    source_paths=[QUERY,COHORT,CANDIDATES,CHUNKS,BASE+'phase2_test_stage_manifest.csv',BASE+'stage01_structured_queries/test_query_configuration.csv',BASE+'stage01_structured_queries/test_query_validation_checklist.csv',BASE+'stage04_retrieval/retrieval_configuration.csv',BASE+'stage04_retrieval/retrieval_validation_checklist.csv','src/rag/final_test/upstream.py','src/rag/run_rag_retrieval_v3.py','src/rag/run_rag_retrieval.py','metadata/rag/v5f_validation/stage03_index/evidence_index_configuration.csv']
    hashes={p:sha(ROOT/p) for p in source_paths}
    cohort=read(ROOT/COHORT);population=sorted(r['query_id'] for r in cohort)
    assert len(population)==len(set(population))==4300
    assert all(r['source_split']=='test' for r in cohort)
    # Only the sorted ID list is given to the random generator. No labels/results/scores.
    chosen=random.Random(42).sample(population,150)
    assert len(chosen)==len(set(chosen))==150
    queries={r['query_id']:r for r in read(ROOT/QUERY)};assert set(queries)==set(population)
    source_check=read(ROOT/(BASE+'stage04_retrieval/retrieval_validation_checklist.csv'))
    assert source_check[0]['status']=='PASS' and source_check[0]['processed_rows']=='4300'
    metadata=json.loads(source_check[0]['output_metadata_json'])
    identity=next(r for r in metadata if r['path']==CANDIDATES)
    assert (ROOT/CANDIDATES).stat().st_size==identity['size_bytes']
    sample=[dict(query_id=i,dataset=queries[i]['source_dataset'],predicted_class=queries[i]['predicted_class_label'],structured_query=queries[i]['retrieval_query_text'],source_query_artifact=QUERY,source_query_reference=QUERY+'#'+i) for i in chosen]
    assert all(r['structured_query'] for r in sample)
    write(OUT/'data/query_sample_seed42.csv',sample)
    # Project-safe query text is the actual registered retrieval_query_text field.
    # query_text contains confidence and is intentionally never exported.
    chunks={r['chunk_id']:r for r in read(ROOT/CHUNKS)}
    groups=defaultdict(list)
    with (ROOT/CANDIDATES).open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            if r['query_id'] in set(chosen):groups[(r['query_id'],r['retrieval_method'])].append(r)
    assert len(groups)==450
    annotation=[];all_raw=[]
    for method,(label,filename) in METHODS.items():
        raw=[]
        for query in sample:
            group=sorted(groups[(query['query_id'],method)],key=lambda r:int(r['rank']))
            assert [int(r['rank']) for r in group]==list(range(1,11)) and len({r['chunk_id'] for r in group})==10
            for r in group[:5]:
                c=chunks[r['chunk_id']]
                assert c['document_id']==r['document_id'] and c['source_type']==r['source_type']
                assert r['source_dataset']==query['dataset'] and r['classifier_prediction']==query['predicted_class']
                row={k:query[k] for k in ['query_id','dataset','predicted_class','structured_query']}
                row.update(retrieval_method=label,rank=int(r['rank']),chunk_id=r['chunk_id'],document_id=r['document_id'],chunk_text=c['chunk_text'],retrieval_score=r['retrieval_score'],source_type=c['source_type'],source_reference=c['canonical_url'] or c['doi'] or c['source_location'],source_location=c['source_location'],source_retrieval_artifact=CANDIDATES,source_retrieval_method=method)
                raw.append(row);annotation.append({**{h:row[h] for h in HEADERS if h!='relevance'},'relevance':''})
        assert len(raw)==750
        write(OUT/f'retrieval_outputs/{filename}_top5.csv',raw);all_raw.extend(raw)
    assert len(annotation)==2250 and len({(r['query_id'],r['retrieval_method'],r['rank']) for r in annotation})==2250
    write(OUT/'annotation/retrieval_relevance_annotation.csv',annotation,HEADERS)
    write(OUT/'data/annotation_expected_rows.csv',annotation,HEADERS)
    write_workbook(OUT/'annotation/retrieval_relevance_annotation.xlsx',HEADERS,annotation)
    actual=[value[0] for value in METHODS.values()];secrets.SystemRandom().shuffle(actual)
    mapping=dict(zip(['Method A','Method B','Method C'],actual))
    dump(OUT/'audit/blinded_method_mapping.json',mapping)
    inverse={real:blind for blind,real in mapping.items()}
    bygroup=defaultdict(list)
    for row in annotation:bygroup[(row['query_id'],row['retrieval_method'])].append(row)
    presentation=random.Random(42);blinded=[]
    for identity in chosen:
        order=['Method A','Method B','Method C'];presentation.shuffle(order)
        for name in order:
            blinded.extend(dict(r,retrieval_method=name) for r in bygroup[(identity,mapping[name])])
    write_workbook(target,HEADERS,blinded)
    # Verify serialized workbook cells, label blanks, full text, and no hidden metadata leak.
    for path,expected in [(target,blinded),(OUT/'annotation/retrieval_relevance_annotation.xlsx',annotation)]:
        headers,loaded=read_workbook(path)
        assert headers==HEADERS
        assert loaded==[{k:str(v) for k,v in r.items()} for r in expected]
        assert all(r['relevance']=='' for r in loaded)
    with ZipFile(target) as z:
        assert len(z.namelist())==6
        for name in z.namelist():
            text=z.read(name).decode('utf-8')
            # Real method identifiers must not be carried in annotations/metadata.
            assert not any(term in text for term in ['TF-IDF','Hybrid (current)','dense_bge','hybrid_rrf'])
            assert 'BGE' not in text
            assert '<f>' not in text and 'state="hidden"' not in text and 'state="veryHidden"' not in text
    id_hash=hashlib.sha256(json.dumps(chosen,separators=(',',':')).encode()).hexdigest()
    sample_hash=sha(OUT/'data/query_sample_seed42.csv')
    manifest=dict(status='WAITING FOR HUMAN ANNOTATION',random_seed=42,sampling_algorithm='Python random.Random(42).sample(sorted(authoritative_query_ids),150); without replacement; sample order retained',source_manifest=COHORT,source_manifest_sha256=hashes[COHORT],source_query_artifact=QUERY,source_artifact_sha256=hashes,sample_file_sha256=sample_hash,ordered_query_ids_sha256=id_hash,query_count=150,method_count=3,rows_per_method=750,annotation_rows=2250,labels_filled=0,retrieval_execution='Reused authentic frozen Stage-4 method-specific rank 1-5 artifacts; no downstream reranking/context packing or LLM',structured_query_field='retrieval_query_text',blinding='Random bijection stored only in audit/blinded_method_mapping.json; per-query method presentation randomized with independent seed42 RNG',expected_rows_sha256=sha(OUT/'data/annotation_expected_rows.csv'),python_version=sys.version,source_receipt_candidate_size_matches=True)
    dump(OUT/'run_manifest.json',manifest)
    (OUT/'audit/source_artifacts.md').write_text('# Source artifacts\n\nAuthoritative frozen final 4,300-query TEST comparison, not Table-1 balanced sample. Existing source and frozen retrieval artifacts read only. Stage-4 receipt PASS/4300 and registered candidate file size verified. Current source SHA-256 identities:\n\n'+''.join(f'- `{p}`: `{h}`\n' for p,h in hashes.items())+'\n`src/rag/final_test/upstream.py:stage4` calls `run_rag_retrieval_v3.balanced_batch`; `merge_intents` emits ten candidates per method with within-method hypothesis balancing and ID deduplication. These registered ranks are reused verbatim; retrieval_score is retained for traceability, not used to reorder them. Hybrid is the registered hybrid_rrf branch. No re-embedding, new retrieval, reranking, context packing, or LLM calls were needed.\n',encoding='utf-8')
    (OUT/'audit/sample_integrity.md').write_text('# Sample integrity\n\n150 unique IDs sampled without replacement from sorted 4,300 authoritative IDs using Python random.Random(42).sample. ID selection occurs before joining query metadata or reading retrieval results; no correctness, truth, status, confidence, or dataset outcomes used. No stratification. Same IDs for all methods.\n\nSample file SHA-256: `'+sample_hash+'`\n\nOrdered-ID SHA-256: `'+id_hash+'`\n\nSource manifest: `'+COHORT+'`; SHA-256 `'+hashes[COHORT]+'`. The 4,300-query source itself is a fixed historical comparison, not a naturally weighted traffic population.\n',encoding='utf-8')
    (OUT/'audit/retrieval_integrity.md').write_text('# Retrieval integrity\n\nPASS: 150 queries × 3 methods × ranks 1–5 = 2,250 rows. Exactly 750 per method, no duplicate query/method/rank keys. Ten authentic candidates per method/query verified before selecting saved ranks 1–5. Every chunk ID, parent document and source type matches the frozen corpus; chunk text copied verbatim. No manual selection, score sorting, reranking or context-token constraint applied.\n',encoding='utf-8')
    (OUT/'audit/annotation_integrity.md').write_text('# Annotation integrity\n\nBoth XLSX workbooks round-trip to exactly 2,250 expected records with complete, untruncated query/chunk text. All relevance cells BLANK. CSV labels also blank. Exact whitelisted columns exclude ground truth, correctness, confidence, Phase-II statuses, and retrieval scores. Actual evidence may discuss traffic classes/associations; that is corpus text, not query outcome metadata. Blinded workbook has only generic Method A/B/C labels, one visible Annotations sheet, no comments, hidden sheets, formulas, or real method identifiers in OOXML. Neutral formatting is identical for every method. Mapping exists only in audit/blinded_method_mapping.json. Each query retains ranks 1–5 within randomly ordered method groups. No human labels received and no machine-generated final labels.\n\nFinal metrics intentionally absent. WAITING FOR HUMAN ANNOTATION.\n',encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['status','sample_file_sha256','ordered_query_ids_sha256','query_count','rows_per_method','annotation_rows','labels_filled']},indent=2))

if __name__=='__main__':main()
