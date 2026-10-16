"""Offline tests of the separate accepted-only V5-v2 fusion adapter."""
import pytest

from src.evaluation import finalize_v5_contract_v2_stage9 as adapter


def inputs(status):
    query = "USTC-VAL-000001"
    positive = status in ("SUPPORTED", "CONTRADICTED")
    value = dict(query_id=query, reasoning_decision="SUPPORT_CLASS" if positive else "ABSTAIN",
                 evidence_status=status, supported_class="benign" if status == "SUPPORTED" else "malware" if positive else None,
                 cited_chunk_ids=["chunk"] if status != "INSUFFICIENT" else [], explanation="validated linkage")
    context = dict(query_id=query, source_dataset="USTC-TFC2016", source_split="validation",
                   rag_partition="development_validation", classifier_prediction="benign",
                   classifier_confidence="0.1", classifier_probabilities='{"benign":0.1,"malware":0.9}',
                   evidence_records_json=adapter.common.json_text([dict(chunk_id="chunk", document_id="doc",
                       source_type="curated", provenance_status="VERIFIED")]))
    record = dict(query_id=query, validation_status="PASS", response_contract_version="v2",
                  validated_response=value, derived_validator_payload=dict(value),
                  raw_v2_model_response=dict(claim_ids=["claim"] if status != "INSUFFICIENT" else []),
                  validated_claim_linkage=[], model_explanation="Original narrative.",
                  narrative_validation_status="NOT_SEMANTICALLY_VALIDATED")
    return context, record


@pytest.mark.parametrize("status", ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT", "CONFLICTING"])
def test_accepted_status_never_overrides_classifier(status):
    context, record = inputs(status)
    result = adapter.fuse_accepted(context, record)
    assert result["final_class"] == context["classifier_prediction"]
    assert result["fusion_policy"] == "retain_classifier"
    assert result["evidence_status"] == status
    assert result["model_explanation"] == "Original narrative."
    assert result["evidence_explanation"] == record["validated_response"]["explanation"]
    assert set(result) == set(adapter.HEADER)


def test_rejected_or_mismatched_record_is_not_coerced():
    context, record = inputs("SUPPORTED")
    record["validation_status"] = "REJECTED"
    with pytest.raises(ValueError):
        adapter.fuse_accepted(context, record)
    record["validation_status"] = "PASS"
    record["query_id"] = "another-id"
    with pytest.raises(ValueError):
        adapter.fuse_accepted(context, record)


def test_nonvalidation_and_unverified_citation_rejected():
    context, record = inputs("SUPPORTED")
    context["source_split"] = "train"
    with pytest.raises(ValueError):
        adapter.fuse_accepted(context, record)
    context["source_split"] = "validation"
    context["evidence_records_json"] = "[]"
    with pytest.raises(ValueError):
        adapter.fuse_accepted(context, record)


def test_status_only_never_publishes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(adapter, "build", lambda root: ([], {"status": "READY_FOR_STAGE9_PREFLIGHT"}, {}))
    monkeypatch.setattr(adapter, "snapshot", lambda root: {})
    def forbidden(*args, **kwargs):
        raise AssertionError("status-only must not publish")
    monkeypatch.setattr(adapter.common, "write_bundle", forbidden)
    assert adapter.main(["--project-root", str(tmp_path), "--status-only"]) == 0
    assert '"executed":false' in capsys.readouterr().out
    assert not list(tmp_path.iterdir())
