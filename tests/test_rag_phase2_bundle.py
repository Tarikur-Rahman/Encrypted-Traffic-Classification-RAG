"""Synthetic integration and static boundaries; never uses canonical artifacts."""
import ast
from pathlib import Path

import pytest

from src.rag import phase2_common as common
from src.rag import build_evidence_index_v2 as index
from src.rag import run_rag_retrieval_v3 as retrieval
from src.rag import verify_retrieval_provenance_v3 as provenance
from src.rag import run_provenance_aware_reranking as reranking
from src.rag import build_trusted_contexts as contexts
from src.rag import materialize_constrained_llm_experiment as prompts
from src.rag import import_constrained_llm_responses as responses
from src.rag import run_evidence_supported_fusion as fusion

STAGES = (index, retrieval, provenance, reranking, contexts, prompts, responses, fusion)


def files(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_synthetic_downstream_pipeline_and_every_dry_mode(phase2_index_project, phase2_case, monkeypatch):
    root, case = phase2_index_project, phase2_case
    monkeypatch.setattr(common, "QUERY_COUNTS", {"USTC-TFC2016": 1})
    monkeypatch.setattr(index.v1, "load_token_policy", lambda root: case.policy)
    monkeypatch.setattr(index.v1, "model_identity", lambda root: {})
    _, units = index.eligible_units(root)
    chunks = index.chunk_units(units, case.policy)
    common.corpus.write_table(root/index.OUTPUTS[0], index.CHUNK_HEADER, chunks)
    common.corpus.write_table(root/common.QUERY_PATH, tuple(case.query), [case.query])
    result_rows = [dict(experiment_id="phase2_dev_retrieval_v3", query_id=case.query["query_id"], source_dataset=case.query["source_dataset"], retrieval_method=method,
                        rank=i, chunk_id=row["chunk_id"], document_id=row["document_id"], retrieval_score=1/i) for method in provenance.METHODS for i, row in enumerate(chunks[:10], 1)]
    common.corpus.write_table(root/provenance.RESULTS, tuple(result_rows[0]), result_rows)
    retrieval_inputs = [*index.INPUTS, index.OUTPUTS[0], common.QUERY_PATH, provenance.RESULTS]
    config = dict(experiment_id="phase2_dev_retrieval_v3", index_revision="V2", input_metadata_json=common.json_text(common.snapshot(root, retrieval_inputs)))
    common.corpus.write_table(root/provenance.RETRIEVAL_CONFIG, common.CONFIG_HEADER, common.config_rows(config))
    common.corpus.write_table(root/provenance.RETRIEVAL_CHECKS, common.corpus.foundation.CHECKLIST_HEADER, common.checks("synthetic_retrieval", {"coverage": 30}))
    protected = ("data/processed/rag/evidence_documents.csv", "artifacts/rag/index/sentinel.txt", "results/rag/phase2_dev_retrieval_v1/sentinel.txt", "results/rag/phase2_dev_retrieval_v2/sentinel.txt")
    for name in protected:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen", encoding="utf-8")
    original = files(root)
    argv = ["--project-root", str(root)]
    for stage in (provenance, reranking, contexts, prompts):
        before = files(root)
        assert stage.main(argv) == 0
        assert files(root) == before
        assert stage.main([*argv, "--execute"]) == 0
        assert set(files(root))-set(before) == set(stage.OUTPUTS)
    prompt_rows = list(common.rows(root, prompts.OUTPUTS[0]))
    external = "results/rag/llm/imports/external.jsonl"
    path = root/external
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(common.json_text(dict(query_id=row["query_id"], reasoning_decision="ABSTAIN", evidence_status="INSUFFICIENT", supported_class=None,
                                                   explanation="Supplied protocol descriptions do not establish this observation's class.", cited_chunk_ids=[])) for row in prompt_rows)+"\n", encoding="utf-8")
    for stage, extra in ((responses, ["--responses", external]), (fusion, [])):
        before = files(root)
        assert stage.main([*argv, *extra]) == 0
        assert files(root) == before
        assert stage.main([*argv, *extra, "--execute"]) == 0
        assert set(files(root))-set(before) == set(stage.OUTPUTS)
    decisions = list(common.rows(root, fusion.OUTPUTS[0], fusion.HEADER))
    assert len(decisions) == 1 and decisions[0]["final_class"] == case.query["predicted_class_label"]
    assert decisions[0]["evidence_status"] == "INSUFFICIENT"
    assert all(files(root)[name] == value for name, value in original.items())
    before = files(root)
    assert fusion.main([*argv, "--execute"]) == 2  # No silent overwrite.
    assert files(root) == before


def test_canonical_paths_disjoint_from_prior_versions():
    paths = [path for stage in STAGES for path in stage.OUTPUTS]
    assert len(paths) == len(set(paths))
    forbidden = ("results/final/", "results/rag/phase2_dev_retrieval_v1/", "results/rag/phase2_dev_retrieval_v2/", "artifacts/rag/index/")
    assert not any(path.startswith(forbidden) for path in paths)
    assert "data/processed/rag/evidence_chunks.csv" not in paths
    assert "data/processed/rag/evidence_documents.csv" not in paths


@pytest.mark.parametrize("stage", STAGES)
def test_static_no_network_hashes_figures_or_automatic_downstream(stage):
    source = Path(stage.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    attributes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import): imported.update(alias.name.split(".")[0] for alias in node.names)
        if isinstance(node, ast.ImportFrom): imported.add((node.module or "").split(".")[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute): attributes.add(node.func.attr)
    assert not imported & {"openai", "requests", "httpx", "urllib", "socket", "hashlib", "matplotlib", "seaborn"}
    assert not attributes & {"sha256", "md5", "savefig", "search", "download", "snapshot_download"}
    # Entrypoints call their own work or the shared CLI only, never another CLI.
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "main" for node in ast.walk(tree))


@pytest.mark.parametrize("path", ["data/processed/traffic/test.csv", "results/rag/imports/ustc_test_predictions.csv", "results/final/USTC-TFC2016/predictions.csv"])
def test_forbidden_inputs_block_before_file_access(tmp_path, path):
    with pytest.raises(ValueError):
        list(common.rows(tmp_path, path))
    with pytest.raises(ValueError):
        common.snapshot(tmp_path, [path])


def test_unknown_and_duplicate_queries_block(tmp_path, phase2_case, monkeypatch):
    monkeypatch.setattr(common, "QUERY_COUNTS", {"USTC-TFC2016": 1})
    row = phase2_case.query
    for rows in ([row, row], [dict(row, rag_partition="test")], [dict(row, source_dataset="UNKNOWN")]):
        common.corpus.write_table(tmp_path/common.QUERY_PATH, tuple(row), rows)
        with pytest.raises(ValueError):
            common.queries(tmp_path)
