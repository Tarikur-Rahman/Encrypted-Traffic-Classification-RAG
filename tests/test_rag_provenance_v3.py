import pytest

from src.rag import verify_retrieval_provenance_v3 as provenance
from src.rag import phase2_common as common


def save_results(root, case, rows=None):
    values = case.results if rows is None else rows
    common.corpus.write_table(root/provenance.RESULTS, tuple(case.results[0]), values)


def test_full_coverage_is_deterministic(tmp_path, phase2_case):
    save_results(tmp_path, phase2_case)
    queries = {phase2_case.query["query_id"]: phase2_case.query}
    result = list(provenance.validated_retrieval(tmp_path, queries, phase2_case.verified))
    assert result == list(provenance.validated_retrieval(tmp_path, queries, phase2_case.verified))
    assert len(result) == 30
    assert all(set(r) == set(provenance.HEADER) for r in phase2_case.verified.values())


@pytest.mark.parametrize("mutation", ["unknown_query", "unknown_chunk", "wrong_document", "duplicate", "missing", "bad_rank"])
def test_retrieval_linkage_rejected(tmp_path, phase2_case, mutation):
    rows = [dict(r) for r in phase2_case.results]
    if mutation == "unknown_query": rows[0]["query_id"] = "UNKNOWN"
    if mutation == "unknown_chunk": rows[0]["chunk_id"] = "UNKNOWN"
    if mutation == "wrong_document": rows[0]["document_id"] = "UNKNOWN"
    if mutation == "duplicate": rows.append(rows[0])
    if mutation == "missing": rows.pop()
    if mutation == "bad_rank": rows[0]["rank"] = "11"
    save_results(tmp_path, phase2_case, rows)
    with pytest.raises(ValueError):
        list(provenance.validated_retrieval(tmp_path, {phase2_case.query["query_id"]: phase2_case.query}, phase2_case.verified))


@pytest.mark.parametrize("field,value", [("quality_status", "REQUIRES_REVIEW"), ("quality_status", "EXCLUDE_LOW_INFORMATION"), ("authorized_for_retrieval", "NO"), ("source_provenance_status", "REJECTED"), ("corpus_partition", "adversarial")])
def test_ineligible_evidence_rejected(phase2_case, field, value):
    chunk = dict(next(iter(phase2_case.chunks.values())), **{field: value})
    with pytest.raises(ValueError):
        provenance.verify_chunks([chunk])


def test_duplicate_chunk_rejected(phase2_case):
    chunk = next(iter(phase2_case.chunks.values()))
    with pytest.raises(ValueError, match="duplicate"):
        provenance.verify_chunks([chunk, chunk])
