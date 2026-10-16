import pytest
from src.evaluation import finalize_disjoint_test_4319 as target


def inputs():
    query = dict(query_id="USTC-TEST-000001",source_dataset="USTC-TFC2016",source_split="test",
        rag_partition="final_test",predicted_class_label="benign",classifier_prediction="benign",
        confidence="0.9",classifier_confidence="0.9",class_probabilities_json='{"benign":0.9,"malware":0.1}',
        retrieval_query_text="Observed traffic.",evaluation_scope=target.c.SCOPE)
    context = dict(query_id=query["query_id"],source_dataset=query["source_dataset"],source_split="test",
        rag_partition="final_test",classifier_prediction="benign",classifier_confidence="0.9",
        traffic_profile=query["retrieval_query_text"],selected_chunk_ids="[]",evidence_records_json="[]",context_text="")
    response = dict(query_id=query["query_id"],source_dataset=query["source_dataset"],source_split="test",
        classifier_prediction="benign",reasoning_decision="ABSTAIN",evidence_status="INSUFFICIENT",
        supported_class=None,cited_chunk_ids=[],explanation="No qualifying evidence.")
    return query,response,context


def test_frozen_fusion_retention_and_final_projection():
    query,response,context = inputs()
    row = target.fuse_one(query,response,context)
    final = target.finalize(row)
    assert final["source_split"] == "test"
    assert final["final_class"] == final["classifier_prediction"] == "benign"
    assert final["fusion_policy"] == "retain_classifier"
    assert final["explanation"] == response["explanation"]
    assert final["cited_chunk_ids"] == "[]" and final["provenance"] == "{}"
    target.check_rows([final],{query["query_id"]})
    with pytest.raises(ValueError):
        target.check_rows([final,final],{query["query_id"]})
    with pytest.raises(ValueError):
        target.finalize(dict(row,final_class="malware"))


@pytest.mark.parametrize("change",["validation","ground_truth","citation"])
def test_unsafe_inputs_rejected(change):
    query,response,context = inputs()
    if change == "validation":
        response["source_split"] = "validation"
    elif change == "ground_truth":
        query["true_class_label"] = "benign"
    else:
        response["cited_chunk_ids"] = ["UNKNOWN"]
    with pytest.raises(ValueError):
        target.fuse_one(query,response,context)


def test_status_only_does_not_write(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(target,"build",lambda root:([],{"stage8_rows":4319},[]))
    monkeypatch.setattr(target.common,"write_bundle",lambda *a:pytest.fail("Status-only write"))
    target.main(["--project-root",str(tmp_path),"--status-only"])
    assert "READY_NO_WRITES" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


def test_existing_outputs_not_overwritten(tmp_path):
    path = tmp_path/target.STAGE9/"test_fusion_records.csv"
    path.parent.mkdir(parents=True)
    path.write_text("preserved",encoding="utf-8")
    with pytest.raises(ValueError,match="already exists"):
        target.common.write_bundle(tmp_path,[(target.STAGE9+path.name,("query_id",),[])])
    assert path.read_text() == "preserved"
