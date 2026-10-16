from types import SimpleNamespace

import pytest

from src.rag import build_evidence_index_v2 as index
from src.rag import phase2_common as common


def test_only_keep_units_and_deterministic_chunks(phase2_index_project, phase2_policy):
    documents, units = index.eligible_units(phase2_index_project)
    assert len(documents) == len(units) == 12
    chunks = index.chunk_units(units, phase2_policy)
    assert chunks == index.chunk_units(units, phase2_policy)
    assert all(row["quality_status"] == "KEEP" and row["chunk_token_count"] <= 512 for row in chunks)
    assert all("Skip to Main" not in row["chunk_text"] and "Possibly bibliographic" not in row["chunk_text"] for row in chunks)
    assert all(set(row) == set(index.CHUNK_HEADER) for row in chunks)
    common.corpus.write_table(phase2_index_project/index.OUTPUTS[0], index.CHUNK_HEADER, chunks)
    assert index.validated_chunks(phase2_index_project, phase2_policy) == index.validated_chunks(phase2_index_project)


@pytest.mark.parametrize("field,value", [("quality_decision", "KEEP"), ("text_preview", "changed"), ("document_id", "UNKNOWN")])
def test_stale_or_altered_quality_blocked(phase2_index_project, field, value):
    quality = list(common.rows(phase2_index_project, index.INPUTS[2]))
    quality[1][field] = value
    common.corpus.write_table(phase2_index_project/index.INPUTS[2], index.corpus_v2.QUALITY_HEADER, quality)
    with pytest.raises(ValueError):
        index.eligible_units(phase2_index_project)


def test_unauthorized_document_blocked(phase2_index_project):
    docs = list(common.rows(phase2_index_project, index.INPUTS[0]))
    docs[0]["authorized_for_retrieval"] = "NO"
    common.corpus.write_table(phase2_index_project/index.INPUTS[0], index.corpus_v2.DOCUMENT_HEADER, docs)
    with pytest.raises(ValueError, match="Unauthorized"):
        index.eligible_units(phase2_index_project)


def test_dry_index_no_fitting_or_writes(phase2_index_project, phase2_policy, monkeypatch):
    before = {p: p.read_bytes() for p in phase2_index_project.rglob("*.csv")}
    monkeypatch.setattr(index.v1, "load_token_policy", lambda root: phase2_policy)
    monkeypatch.setattr(index.v1, "model_identity", lambda root: {})
    def forbidden(*args):
        pytest.fail("Dry mode fitted/encoded")
    monkeypatch.setattr(index.v1, "lexical_index", forbidden)
    monkeypatch.setattr(index.v1, "dense_index", forbidden)
    assert index.main(["--project-root", str(phase2_index_project)]) == 0
    assert before == {p: p.read_bytes() for p in phase2_index_project.rglob("*.csv")}
    assert not any((phase2_index_project/name).exists() for name in index.OUTPUTS)


def test_oversized_word_blocks_without_truncation(phase2_policy):
    policy = SimpleNamespace(limit=512, count=lambda text: 600)
    with pytest.raises(ValueError):
        index.v1.token_windows("unbreakable", policy)


def test_output_paths_protect_v1():
    assert not set(index.OUTPUTS) & set(index.v1.OUTPUTS)
    assert all("v2" in path for path in index.OUTPUTS)


def test_namespaced_index_reads_stage02_and_keeps_legacy_paths():
    assert index.input_paths() == index.INPUTS
    assert index.output_paths() == index.OUTPUTS
    inputs, outputs = index.input_paths("v5_validation"), index.output_paths("v5_validation")
    assert inputs[0] == "data/processed/rag/v5_validation/evidence_documents_v2.csv"
    assert outputs[0] == "data/processed/rag/v5_validation/evidence_chunks.csv"
    assert outputs[1].startswith("artifacts/rag/v5_validation/index/")
    assert all("/test/" not in path.lower() for path in (*inputs, *outputs))


def test_index_eligible_units_reads_resolved_stage02_inputs(phase2_index_project, monkeypatch):
    inputs = index.input_paths("v5_validation")
    legacy = index.INPUTS
    original = index.common.rows
    seen = []
    documents = list(original(phase2_index_project, legacy[0]))
    manifest = list(original(phase2_index_project, legacy[1]))
    old_quality = list(original(phase2_index_project, legacy[2]))
    quality = []
    for document in documents:
        prior_units = sorted((row for row in old_quality if row["document_id"] == document["document_id"]), key=lambda row: int(row["unit_order"]))
        source_text_units = index.corpus_v2.source_units(document["normalized_text"])
        assert len(prior_units) == len(source_text_units)
        located_units = [(prior["source_location"], source_text) for prior, (_, source_text) in zip(prior_units, source_text_units)]
        recomputed = index.corpus_v2.quality_rows(document["document_id"], located_units, profile="evidence_quality")
        assert [row["review_unit_id"] for row in recomputed] == [row["review_unit_id"] for row in prior_units]
        assert all(row["quality_profile"] == "evidence_quality" for row in recomputed)
        quality.extend(recomputed)
    for path, header, rows in zip(inputs, (index.corpus_v2.DOCUMENT_HEADER, index.corpus.MANIFEST_HEADER, index.corpus_v2.IMPROVED_QUALITY_HEADER), (documents, manifest, quality)):
        target = phase2_index_project/path
        target.parent.mkdir(parents=True, exist_ok=True)
        index.corpus.write_table(target, header, rows)
    def track_rows(root, path, required=()):
        seen.append(path)
        yield from original(root, path, required)
    monkeypatch.setattr(index.common, "rows", track_rows)
    index.eligible_units(phase2_index_project, inputs)
    assert set(inputs) <= set(seen)


def test_execute_writes_only_v2_with_synthetic_matrices(phase2_index_project, phase2_policy, monkeypatch):
    import numpy as np
    from scipy.sparse import csr_matrix
    root = phase2_index_project
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    monkeypatch.setattr(index.v1, "load_token_policy", lambda root: phase2_policy)
    monkeypatch.setattr(index.v1, "model_identity", lambda root: {})
    monkeypatch.setattr(index.v1, "lexical_index", lambda chunks: ({"synthetic": True}, csr_matrix(np.ones((len(chunks), 1)))))
    monkeypatch.setattr(index.v1, "dense_index", lambda root, chunks: np.full((len(chunks), 384), 1/np.sqrt(384), dtype=np.float32))
    assert index.main(["--project-root", str(root), "--execute"]) == 0
    after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert {p.as_posix() for p in set(after)-set(before)} == set(index.OUTPUTS)
    assert all(after[p] == data for p, data in before.items())
    assert index.main(["--project-root", str(root), "--execute"]) == 2
    assert after == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
