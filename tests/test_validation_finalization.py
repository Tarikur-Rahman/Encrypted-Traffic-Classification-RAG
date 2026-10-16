import json
import pytest
from src.evaluation import finalize_validation_phase2 as target


def test_union_rejects_duplicates_missing_and_test(tmp_path, monkeypatch):
    monkeypatch.setattr(target, "SOURCES", ("results/rag/validation/input.jsonl",))
    path = tmp_path/target.SOURCES[0]
    path.parent.mkdir(parents=True)
    row = dict(query_id="USTC-VAL-000001", source_split="validation")
    path.write_text(json.dumps(row)+"\n")
    assert set(target.assemble(tmp_path, {row['query_id']})[0]) == {row['query_id']}
    with pytest.raises(ValueError, match="Missing"):
        target.assemble(tmp_path, {row['query_id'], 'USTC-VAL-000002'})
    path.write_text((json.dumps(row)+"\n")*2)
    with pytest.raises(ValueError, match="duplicate"):
        target.assemble(tmp_path, {row['query_id']})
    path.write_text(json.dumps(dict(row, source_split='test'))+'\n')
    with pytest.raises(ValueError, match="Non-validation"):
        target.assemble(tmp_path, {row['query_id']})


def test_reuses_fusion_and_retains_classifier_without_gt(phase2_case):
    q = dict(phase2_case.query, query_id='USTC-VAL-000001')
    context = dict(query_id=q['query_id'], source_dataset=q['source_dataset'], source_split='validation',
        rag_partition='development_validation', classifier_prediction=q['predicted_class_label'],
        classifier_confidence=q['confidence'], traffic_profile=q['retrieval_query_text'],
        selected_chunk_ids='[]', evidence_records_json='[]', context_text='')
    prompt = target.prompts.prompt_record(context, 'v4')
    response = dict(query_id=q['query_id'], source_dataset=q['source_dataset'], classifier_prediction=q['predicted_class_label'],
        evidence_status='INSUFFICIENT', reasoning_decision='ABSTAIN', supported_class=None, cited_chunk_ids=[], explanation='No evidence.')
    row = target.fuse_one(q, response, prompt)
    final = target.finalize(row)
    assert final['final_class'] == q['predicted_class_label']
    assert final['fusion_policy'] == 'retain_classifier' and final['source_split'] == 'validation'
    assert final['cited_chunk_ids'] == '[]' and final['explanation'] == 'No evidence.'
    assert not {'ground_truth', 'true_class_label', 'true_class_id'} & set(final)
    with pytest.raises(ValueError):
        target.finalize(dict(row, final_class='malware', classifier_prediction='benign'))
    with pytest.raises(ValueError):
        target.fuse_one(dict(q, source_split='test'), response, prompt)
