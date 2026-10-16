"""Synthetic provenance/streaming coverage tests; no research execution."""
import ast
import json
from pathlib import Path

import pytest

from src.rag import verify_retrieval_provenance as provenance
from src.rag import build_evidence_corpus as corpus


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def mutate(root, path, edit):
    header, rows = corpus.read_table(root, path)
    edit(rows)
    corpus.write_table(root/path, header, rows)


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "CHUNK_COUNT", 10)
    monkeypatch.setattr(provenance, "TOP_K", 2)
    monkeypatch.setattr(provenance, "QUERY_COUNTS", {"USTC-TFC2016": 1, "CIC-Darknet2020": 1})
    monkeypatch.setattr(provenance, "RETRIEVAL_ROWS", 12)
    manifest, documents, chunks = [], [], []
    for i in range(10):
        identity = f"RFC-{i:04d}"
        source = dict.fromkeys(corpus.MANIFEST_HEADER, "")
        source.update(document_id=identity, title="Synthetic source", source_type="standards_rfc", publisher="Synthetic organization",
                      local_path=f"data/rag/evidence_sources/trusted/{identity}.txt", corpus_partition="trusted", source_provenance_status="VERIFIED", authorized_for_retrieval="YES")
        path = tmp_path/source["local_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Synthetic evidence text.", encoding="utf-8")
        manifest.append(source)
        text = "Synthetic evidence text."
        documents.append(dict(source, extraction_status="PASS", extraction_review_status="MISSING", normalized_text=text, word_count=3, character_count=len(text), extraction_notes="Text-layer extraction"))
        chunk = {k: source[k] for k in provenance.CHUNK_METADATA}
        chunk.update(chunk_id=identity+"-CHUNK-0000", document_id=identity, chunk_order=0, chunk_word_count=3, chunk_character_count=len(text), chunk_token_count=5, chunk_text=text)
        chunks.append(chunk)
    corpus.write_table(tmp_path/corpus.MANIFEST, corpus.MANIFEST_HEADER, manifest)
    corpus.write_table(tmp_path/corpus.REVIEW, corpus.REVIEW_HEADER, [])
    corpus.write_table(tmp_path/corpus.OUTPUTS[0], corpus.DOCUMENT_HEADER, documents)
    corpus.write_table(tmp_path/corpus.OUTPUTS[1], corpus.SUMMARY_HEADER, corpus.summarize(documents))
    corpus.write_table(tmp_path/corpus.OUTPUTS[2], corpus.foundation.CHECKLIST_HEADER, [corpus.check("corpus", "nonempty_usable_documents", "PASS", "10")])
    corpus.write_table(tmp_path/provenance.CHUNKS, tuple(chunks[0]), chunks)
    config = dict(chunk_count="10", authorized_document_count="10", trusted_only="YES", adversarial_included="NO", corpus_document_ids_json=json.dumps([s["document_id"] for s in manifest]), maximum_chunk_words="320", chunk_token_limit="512")
    corpus.write_table(tmp_path/provenance.INDEX_CONFIG, provenance.CONFIG_HEADER, [dict(field=k, value=v) for k, v in config.items()])
    corpus.write_table(tmp_path/provenance.INDEX_CHECKS, corpus.foundation.CHECKLIST_HEADER, [corpus.check("index", "complete", "PASS", "Synthetic index")])
    retrieval_config = dict(experiment_id=provenance.EXPERIMENT, configuration_name=provenance.CONFIGURATION, stage="RAG_RETRIEVAL", rag_partition="development_validation", retrieval_top_k="2", chunk_count="10", authorized_document_count="10", test_access="NO", phase_i_models_loaded="NO", network_used="NO", llm_used="NO", provenance_verification_executed="NO", reranking_executed="NO")
    corpus.write_table(tmp_path/(provenance.BASE+"experiment_configuration.csv"), provenance.CONFIG_HEADER, [dict(field=k, value=v) for k, v in retrieval_config.items()])
    queries, retrieval = [], []
    for i, dataset in enumerate(provenance.QUERY_COUNTS):
        query = dict(experiment_id=provenance.EXPERIMENT, configuration_name=provenance.CONFIGURATION, query_id=f"QUERY-{i}", source_dataset=dataset, source_split="validation", rag_partition="development_validation", true_class_label="IGNORED_EVALUATION")
        queries.append(query)
        for method in provenance.METHODS:
            for rank in range(1, 3):
                chunk = chunks[rank-1]
                retrieval.append(dict(experiment_id=provenance.EXPERIMENT, configuration_name=provenance.CONFIGURATION, query_id=query["query_id"], source_dataset=dataset, retrieval_method=method, rank=rank, chunk_id=chunk["chunk_id"], document_id=chunk["document_id"]))
    corpus.write_table(tmp_path/(provenance.BASE+"query_records.csv"), tuple(queries[0]), queries)
    corpus.write_table(tmp_path/(provenance.BASE+"retrieval_results.csv"), tuple(retrieval[0]), retrieval)
    return tmp_path


def records(root):
    return provenance.chunk_records(root, provenance.validate_stage_inputs(root))


def test_one_sorted_record_per_chunk_and_pdf_pass_needs_no_review(project):
    rows = records(project)
    assert len(rows) == 10 and len({r["chunk_id"] for r in rows}) == 10
    assert [r["chunk_id"] for r in rows] == sorted(r["chunk_id"] for r in rows)
    assert all(r["provenance_verification_status"] == "VERIFIED" and r["authorized_for_downstream"] == "YES" for r in rows)
    assert all(r["extraction_review_status"] == "MISSING" for r in rows)
    assert all(set(r) == set(provenance.RECORD_HEADER) for r in rows)


def test_duplicate_chunk_rejected(project):
    mutate(project, provenance.CHUNKS, lambda rows: rows.append(dict(rows[0])))
    with pytest.raises(ValueError, match="duplicate"):
        records(project)


@pytest.mark.parametrize("path", [corpus.OUTPUTS[0], corpus.MANIFEST])
def test_parent_linkage_missing_blocks(project, path):
    mutate(project, path, lambda rows: rows.pop(0))
    row = records(project)[0]
    assert row["provenance_verification_status"] == "UNRESOLVED"
    assert row["authorized_for_downstream"] == "NO"


@pytest.mark.parametrize("field,value", [("corpus_partition", "adversarial"), ("source_provenance_status", "REQUIRES_REVIEW"), ("authorized_for_retrieval", "NO"), ("source_provenance_status", "REJECTED")])
def test_chunk_trust_authorization_enforced(project, field, value):
    mutate(project, provenance.CHUNKS, lambda rows: rows[0].update({field: value}))
    row = records(project)[0]
    assert row["provenance_verification_status"] == "REJECTED"
    assert row["authorized_for_downstream"] == "NO"


@pytest.mark.parametrize("review", ["MISSING", "REQUIRES_REVIEW", "REJECTED", "APPROVED"])
def test_required_extraction_review(project, review):
    mutate(project, corpus.OUTPUTS[0], lambda rows: rows[0].update(extraction_status="REQUIRES_REVIEW", extraction_review_status=review))
    if review != "MISSING":
        corpus.write_table(project/corpus.REVIEW, corpus.REVIEW_HEADER, [dict(document_id="RFC-0000", review_status=review, review_notes="Curated")])
    row = records(project)[0]
    assert (row["authorized_for_downstream"] == "YES") == (review == "APPROVED")


def test_tor_rejected_cannot_become_authorized(project):
    for name in (corpus.MANIFEST, corpus.OUTPUTS[0], provenance.CHUNKS):
        def edit(rows):
            rows[0]["document_id"] = "TOR-SPEC-001"
            if "chunk_id" in rows[0]:
                rows[0]["chunk_id"] = "TOR-SPEC-001-CHUNK-0000"
        mutate(project, name, edit)
    def replace_ids(rows):
        for row in rows:
            if row["field"] == "corpus_document_ids_json":
                ids = json.loads(row["value"])
                ids[0] = "TOR-SPEC-001"
                row["value"] = json.dumps(ids)
    mutate(project, provenance.INDEX_CONFIG, replace_ids)
    row = next(r for r in records(project) if r["document_id"] == "TOR-SPEC-001")
    assert row["authorized_for_downstream"] == "NO" and "REJECTED_SOURCE" in row["verification_reason"]


@pytest.mark.parametrize("method", provenance.METHODS)
def test_method_coverage_and_unknown_reference(project, method):
    verified = records(project)
    before = provenance.coverage(project, verified)
    matched = next(r for r in before if r["summary_level"] == "method" and r["retrieval_method"] == method)
    assert matched["retrieval_rows"] == 4 and matched["unknown_provenance_rows"] == 0
    def edit(rows):
        next(r for r in rows if r["retrieval_method"] == method)["chunk_id"] = "UNKNOWN-CHUNK"
    mutate(project, provenance.BASE+"retrieval_results.csv", edit)
    after = provenance.coverage(project, verified)
    matched = next(r for r in after if r["summary_level"] == "method" and r["retrieval_method"] == method)
    assert matched["unknown_provenance_rows"] == matched["unknown_retrieved_chunks"] == 1
    assert provenance.main(["--project-root", str(project), "--execute"]) == 2
    assert not any((project/name).exists() for name in provenance.OUTPUTS)


def test_unauthorized_and_adversarial_reference_counts(project):
    verified = records(project)
    verified[0].update(authorized_for_downstream="NO", corpus_partition="adversarial", provenance_verification_status="REJECTED")
    summary = provenance.coverage(project, verified)
    overall = next(r for r in summary if r["summary_level"] == "overall")
    assert overall["unauthorized_retrieval_rows"] == overall["adversarial_retrieval_rows"] == 6


@pytest.mark.parametrize("failure", ["missing_method", "duplicate_rank", "duplicate_chunk", "wrong_dataset", "wrong_document"])
def test_invalid_retrieval_coverage_blocks(project, failure):
    def edit(rows):
        if failure == "missing_method":
            rows[:] = [r for r in rows if r["retrieval_method"] != "dense_bge"]
        elif failure == "duplicate_rank":
            rows[1]["rank"] = rows[0]["rank"]
        elif failure == "duplicate_chunk":
            rows[1]["chunk_id"] = rows[0]["chunk_id"]
        elif failure == "wrong_dataset":
            rows[0]["source_dataset"] = "CIC-Darknet2020"
        else:
            rows[0]["document_id"] = "UNKNOWN"
    mutate(project, provenance.BASE+"retrieval_results.csv", edit)
    with pytest.raises(ValueError):
        provenance.validate(project)


def test_dry_write_free_and_execute_normalized_outputs(project, capsys):
    before = snapshot(project)
    assert provenance.main(["--project-root", str(project)]) == 0
    assert "READY_FOR_EXECUTION" in capsys.readouterr().out
    assert snapshot(project) == before
    assert provenance.main(["--project-root", str(project), "--execute"]) == 0
    after = snapshot(project)
    assert set(after)-set(before) == set(provenance.OUTPUTS)
    assert all(after[name] == data for name, data in before.items())
    assert len(corpus.read_table(project, provenance.OUTPUTS[0])[1]) == 10
    assert provenance.main(["--project-root", str(project), "--execute"]) == 0
    assert snapshot(project) == after


def test_ground_truth_ignored_and_datasets_separate(project):
    before = records(project)
    mutate(project, provenance.BASE+"query_records.csv", lambda rows: rows[0].update(true_class_label="another dataset label"))
    assert records(project) == before
    summary = provenance.coverage(project, before)
    assert {r["source_dataset"] for r in summary if r["summary_level"] == "dataset_method"} == set(provenance.QUERY_COUNTS)
    mutate(project, provenance.BASE+"query_records.csv", lambda rows: rows[0].update(source_split="TEST"))
    with pytest.raises(ValueError, match="TEST"):
        provenance.query_population(project)


def test_failure_never_overwrites_existing_stage_outputs(project):
    assert provenance.main(["--project-root", str(project), "--execute"]) == 0
    prior = {name: (project/name).read_bytes() for name in provenance.OUTPUTS}
    mutate(project, provenance.CHUNKS, lambda rows: rows[0].update(authorized_for_retrieval="NO"))
    assert provenance.main(["--project-root", str(project), "--execute"]) == 2
    assert all((project/name).read_bytes() == data for name, data in prior.items())


def test_static_boundaries_and_no_reasoning_statuses():
    source = Path(provenance.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports, calls = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    assert not imports & {"requests", "httpx", "urllib", "openai", "hashlib", "matplotlib", "sentence_transformers", "transformers", "torch", "numpy", "lightgbm"}
    assert not calls & {"search", "predict", "fit", "encode", "savefig", "sha256", "md5"}
    assert not any(state in source for state in ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT", "CONFLICTING"))
    assert "test.csv" not in source
