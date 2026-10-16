"""Offline compatibility checks; no inference or research artifacts."""
import pytest
from src.evaluation import finalize_v5_contract_v2_stage10 as stage10


@pytest.mark.parametrize("status", ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT", "CONFLICTING"])
def test_projection_retains_prediction_and_all_audit_fields(status):
    row = dict(query_id="USTC-VAL-1", source_split="validation", fusion_policy="retain_classifier",
               final_class="benign", classifier_prediction="benign", evidence_status=status,
               provenance_summary='{"chunk":"VERIFIED"}', evidence_explanation="CLAIM_IDS: []",
               model_explanation="Preserved narrative — UTF-8", claim_ids_json="[]")
    final = stage10.finalize(row)
    assert all(final[k] == v for k, v in row.items())
    assert final["provenance"] == row["provenance_summary"]
    assert final["explanation"] == row["evidence_explanation"]
    assert final["explanation_reference"] == stage10.SOURCE + "#USTC-VAL-1"
    row["final_class"] = "malware"
    with pytest.raises(ValueError):
        stage10.finalize(row)


def test_population_linkage_fail_closed(monkeypatch):
    ids = [f"USTC-VAL-{i}" for i in range(4205)]
    rows = [dict(query_id=q) for q in ids]
    contexts = {q: dict(query_id=q) for q in ids}
    accepted = dict(contexts)
    monkeypatch.setattr(stage10.upstream, "fuse_accepted", lambda context, record: dict(context))
    assert stage10.checked_rows(rows, ids, contexts, accepted) == rows
    rows[-1] = dict(rows[0])
    with pytest.raises(ValueError):
        stage10.checked_rows(rows, ids, contexts, accepted)
    rows[-1] = dict(query_id=ids[-1], model_explanation="changed")
    with pytest.raises(ValueError):
        stage10.checked_rows(rows, ids, contexts, accepted)


def test_status_only_does_not_write(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(stage10, "build", lambda root: ([], {}, {}, {}))
    monkeypatch.setattr(stage10, "snapshot", lambda root: {})
    monkeypatch.setattr(stage10.upstream, "snapshot", lambda root: {})
    def forbidden(*args, **kwargs):
        raise AssertionError("No publication allowed")
    monkeypatch.setattr(stage10.common, "write_bundle", forbidden)
    stage10.main(["--project-root", str(tmp_path), "--status-only"])
    assert '"executed":false' in capsys.readouterr().out
    assert not list(tmp_path.iterdir())
