"""Extract frozen narratives and evidence; never generate judgments or explanations."""
import csv, hashlib, json
from pathlib import Path
from workbook_io import LABELS, read_workbook, write_workbook

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]
BASE=ROOT/'results/rag/test/final_4300_v5_contract_v2'
SAMPLE=ROOT/'supervisor_requested_experiments/table3_retrieval_quality/data/query_sample_seed42.csv'
SAMPLE_SHA='56bc840be3a68cf21577498325824ba94437c138b52333bfc177552c44c7f649'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def js(v):return json.dumps(v,ensure_ascii=False)

def main():
    if any((OUT/f'annotation/annotator_{i}.xlsx').exists() for i in (1,2)):
        raise RuntimeError('Refusing to overwrite human annotation workbooks')
    assert sha(SAMPLE)==SAMPLE_SHA
    sample=read(SAMPLE);ids=[r['query_id'] for r in sample]
    assert len(ids)==len(set(ids))==150
    rp=BASE/'stage08_llm/test_llm_responses.jsonl'
    cp=BASE/'stage07_trusted_contexts/trusted_context_records.csv'
    responses={};line_numbers={}
    with rp.open(encoding='utf-8') as f:
        for n,line in enumerate(f,1):
            r=json.loads(line);qid=r['query_id'];assert qid not in responses
            responses[qid]=r;line_numbers[qid]=n
    assert len(responses)==4300
    contexts={r['query_id']:r for r in read(cp)}
    assert len(contexts)==4300
    data=[];expected=[];no_citations=0
    for q in sample:
        qid=q['query_id'];r=responses[qid];raw=r['raw_v2_model_response'];ctx=contexts[qid]
        assert raw['query_id']==qid and r['validation_status']=='PASS'
        assert raw['explanation']==r['model_explanation'] and raw['explanation'].strip()
        assert ctx['source_dataset']==q['dataset'] and ctx['classifier_prediction']==q['predicted_class']
        evidence=json.loads(ctx['evidence_records_json']);chunks={e['chunk_id']:e for e in evidence}
        cited=raw['cited_chunk_ids'];claims=raw['claim_ids']
        assert len(cited)==len(set(cited)) and all(c in chunks for c in cited)
        assert all(e['chunk_text'] for e in evidence)
        if not cited:no_citations+=1
        data.append(dict(query_id=qid,dataset=q['dataset'],predicted_class=q['predicted_class'],
            evidence_status=raw['evidence_status'],explanation=raw['explanation'],cited_chunk_ids=js(cited),
            claim_ids=js(claims),source_phase_ii_artifact_reference=rp.relative_to(ROOT).as_posix()+f'#line={line_numbers[qid]}'))
        expected.append(dict(query_id=qid,dataset=q['dataset'],predicted_class=q['predicted_class'],
            structured_query=q['structured_query'],explanation=raw['explanation'],cited_chunk_ids=js(cited),
            claim_ids=js(claims),cited_chunk_texts='\n\n'.join(c+'\n'+chunks[c]['chunk_text'] for c in cited),
            retrieved_context_texts='\n\n'.join(e['chunk_id']+'\n'+e['chunk_text'] for e in evidence)))
    write(OUT/'data/explanation_sample_150.csv',data)
    write(OUT/'data/annotation_expected_rows.csv',expected)
    for i in (1,2):
        rows=[dict(annotator_id=f'annotator_{i}',**r,**{k:'' for k in LABELS}) for r in expected]
        path=OUT/f'annotation/annotator_{i}.xlsx';write_workbook(path,list(rows[0]),rows)
        headers,loaded=read_workbook(path);assert loaded==rows
        assert all(r[k]=='' for r in loaded for k in LABELS)
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in (SAMPLE,rp,cp)}
    manifest=dict(status='WAITING FOR TWO HUMAN ANNOTATORS',query_count=150,source_response_count=4300,
        source_sha256=sources,sample_sha256=SAMPLE_SHA,labels_generated_automatically=0,
        explanations_regenerated=0,queries_without_citations=no_citations,
        rubric='User-specified fallback: Evidence grounded, Relevant, Clear; Overall acceptable = conjunction',
        expected_rows_sha256=sha(OUT/'data/annotation_expected_rows.csv'),
        workbook_template_sha256={f'annotator_{i}':sha(OUT/f'annotation/annotator_{i}.xlsx') for i in (1,2)})
    (OUT/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (OUT/'audit/source_artifacts.md').write_text('# Source artifacts\n\n'+''.join(f'- `{p}` SHA-256 `{h}`\n' for p,h in sources.items())+'\nNarrative copied verbatim from raw_v2_model_response.explanation and checked against model_explanation. The derived validator explanation containing CLAIM_IDS is not the human narrative. Cited IDs and claim IDs are copied from the same frozen response. Full chunk texts come from that query\'s frozen Stage-7 evidence_records_json; every cited ID resolves there. Retrieval, LLM, and semantic grading were not rerun.\n',encoding='utf-8')
    (OUT/'audit/sample_integrity.md').write_text(f'# Sample integrity\n\nExactly 150 unique IDs in unchanged Table-3 sample order. Source sample SHA-256 `{SAMPLE_SHA}` verified. All 150 have nonempty frozen narratives, PASS schema/linkage status, matching dataset and predicted class, and resolvable cited IDs. PASS is a machine contract check, not a human quality label. Missing explanations: 0. Regenerated explanations: 0. Rows without citations: {no_citations}.\n',encoding='utf-8')
    (OUT/'audit/annotation_integrity.md').write_text('# Annotation integrity\n\nTwo distinct role-identified XLSX files, 150 rows each, round-trip checked against the immutable source baseline. All four judgment columns are blank in both files: 1,200 blank label cells. Labels generated automatically: ZERO. No ground truth, correctness, Table-1 labels, method identity, machine validation score, or other annotator scores are exported. Model evidence status is retained only in the custodian sample CSV, not the workbooks. Exact cited evidence and separately identified full retrieved context are shown. No formulas or external links.\n\nIndependent human completion is pending. Software cannot prove that humans did not copy judgments; distinct human identifiers and an explicit independence attestation are required by the validator. Equal judgments are not by themselves evidence of copying. Each annotator receives only their own workbook and the guide. No scores or agreement are computed during preparation.\n',encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
