"""Independently verify exported scores against saved HTTP responses."""
import csv,json,base64,hashlib
from pathlib import Path
from sklearn.metrics import roc_auc_score
OUT=Path(__file__).resolve().parents[1]
def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))
def lines(path):return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
truth=rows(OUT.parent/'table1_error_detection/data/error_detection_balanced_set.csv')
ids=[r['query_id'] for r in truth]
metrics={r['variant']:r for r in rows(OUT/'results/table2_metrics_full_precision.csv')}
checks={}
for slug in ['no_reranking','tfidf_retrieval_only','no_provenance_check']:
    directory=OUT/'variant_outputs'/slug
    raw=lines(directory/'raw_responses.jsonl');accepted=lines(directory/'accepted_responses.jsonl')
    results=rows(directory/'query_results.csv');contexts=rows(directory/'prepared_contexts.csv')
    assert all([r['query_id'] for r in seq]==ids for seq in [raw,accepted,results,contexts])
    for raw_row,accept,result,context in zip(raw,accepted,results,contexts):
        content=base64.b64decode(raw_row['response_base64'])
        assert hashlib.sha256(content).hexdigest()==raw_row['response_sha256']
        response=json.loads(content);value=json.loads(response['choices'][0]['message']['content'])
        assert value==accept['raw_v2_model_response']
        assert value['evidence_status']==accept['validated_response']['evidence_status']==result['phase2_evidence_status']
        assert len(json.loads(context['selected_chunk_ids']))==5 and int(context['context_token_count'])<=2048
    auc=roc_auc_score([int(t['predicted_class']!=t['ground_truth']) for t in truth],[int(r['phase2_evidence_status']!='SUPPORTED') for r in results])
    assert abs(auc-float(metrics[slug]['auroc_full_precision']))<1e-15
    checks[slug]={'rows':188,'raw_hashes_status_and_exact_five_contexts':'PASS','independent_auroc':auc}
assert not (OUT/'variant_outputs/bge_retrieval_only/raw_responses.jsonl').exists()
(OUT/'audit/recovery_export_verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
