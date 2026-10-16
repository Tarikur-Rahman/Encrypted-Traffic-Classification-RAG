"""Foundation tests use synthetic metadata only. Execute manually, never at import."""

import ast
import csv
from pathlib import Path

import pytest

from src.rag import phase2_foundation as foundation


def csv_file(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def project(tmp_path):
    for prefix, dataset in (("ustc", "USTC-TFC2016"), ("cic", "CIC-Darknet2020")):
        row = dict(dataset=dataset, stage_status="CLOSED_FROZEN", integrity_audit_status="PASS", finalization_status="PASS", frozen_for_retraining="YES", frozen_for_test_reuse="YES")
        if prefix == "cic":
            row.update(authoritative_run="20260920T145831Z", phase_i_status="COMPLETE")
        csv_file(tmp_path / "metadata" / "traffic" / f"{prefix}_experiment_closure.csv", [row])
        for split in ("train", "validation", "test") + (("per_class",) if prefix == "cic" else ()):
            csv_file(tmp_path / "results" / "final" / dataset / f"{prefix}_{split}_results.csv", [dict(model="synthetic", accuracy="0.5")])
    return tmp_path


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def checklist(result):
    return result.outputs[foundation.OUTPUT_NAMES[4]][1]


def validation_rows(project, prefix, **changes):
    row = dict(query_id=prefix + "-validation-0", source_dataset="USTC-TFC2016" if prefix == "ustc" else "CIC-Darknet2020", source_split="VALIDATION", source_row_index="0", classifier_model="synthetic", classifier_run_id="synthetic-ustc-run" if prefix == "ustc" else "20260920T145831Z", predicted_class_id="0", predicted_class_label="synthetic-label")
    row.update(changes)
    csv_file(project / "results" / "interface" / (prefix + "_validation_predictions.csv"), [row])


@pytest.mark.parametrize("prefix", ["ustc", "cic"])
def test_closed_frozen(project, prefix):
    assert foundation.closure_check(project, prefix)[0]


@pytest.mark.parametrize("prefix,field,value", [
    ("ustc", "stage_status", "OPEN"), ("cic", "stage_status", "OPEN"),
    ("ustc", "frozen_for_test_reuse", "NO"), ("cic", "frozen_for_test_reuse", "NO"),
    ("ustc", "frozen_for_retraining", "NO"), ("cic", "frozen_for_retraining", "NO"),
    ("ustc", "integrity_audit_status", "FAIL"), ("cic", "finalization_status", "FAIL"),
    ("cic", "phase_i_status", "INCOMPLETE"), ("cic", "authoritative_run", "other"),
    ("ustc", "dataset", "other"),
])
def test_protection_failures_block_all_writes(project, prefix, field, value):
    path = project / "metadata" / "traffic" / (prefix + "_experiment_closure.csv")
    _, rows = foundation.read_csv(path)
    rows[0][field] = value
    csv_file(path, rows)
    before = snapshot(project)
    assert foundation.main(["--project-root", str(project), "--execute"]) == 2
    assert snapshot(project) == before


def test_missing_and_ambiguous_closure_block(project):
    path = project / "metadata/traffic/cic_experiment_closure.csv"
    _, rows = foundation.read_csv(path)
    csv_file(path, rows + rows)
    assert not foundation.closure_check(project, "cic")[0]
    path.unlink()
    assert foundation.inspect_foundation(project).status == "BLOCKED"


def test_key_value_closure_and_duplicate_rejection(project):
    path = project / "metadata/traffic/ustc_experiment_closure.csv"
    _, rows = foundation.read_csv(path)
    pairs = [dict(field=k, value=v) for k, v in rows[0].items()]
    csv_file(path, pairs)
    assert foundation.closure_check(project, "ustc")[0]
    csv_file(path, pairs + [pairs[0]])
    assert not foundation.closure_check(project, "ustc")[0]


def test_phase_i_complete_and_missing_validation(project):
    result = foundation.inspect_foundation(project)
    assert result.status == "READY_FOR_INTERFACE_MATERIALIZATION"
    assert next(r for r in checklist(result) if r["check"] == "phase_i_complete")["status"] == "PASS"
    validation = [r for r in checklist(result) if r["check"].endswith("_validation_predictions")]
    assert len(validation) == 2
    assert all(r["status"] == "WARN" and "MATERIALIZATION" in r["details"] for r in validation)


def test_both_datasets_required_before_corpus_readiness(project):
    validation_rows(project, "ustc")
    assert foundation.inspect_foundation(project).status == "READY_FOR_INTERFACE_MATERIALIZATION"
    validation_rows(project, "cic")
    result = foundation.inspect_foundation(project)
    assert result.status == "READY_FOR_CORPUS_CONSTRUCTION"
    assert next(r for r in checklist(result) if r["check"] == "trusted_corpus_availability")["status"] == "WARN"


@pytest.mark.parametrize("changes", [dict(source_split="TEST"), dict(source_split="TRAIN"), dict(classifier_run_id="stale"), dict(query_id=""), dict(predicted_class_id="")])
def test_invalid_validation_is_not_assumed_available(project, changes):
    validation_rows(project, "ustc")
    validation_rows(project, "cic", **changes)
    assert foundation.inspect_foundation(project).status == "READY_FOR_INTERFACE_MATERIALIZATION"


def test_empty_validation_and_aggregate_metrics_are_not_predictions(project):
    csv_file(project / "results/interface/cic_validation_predictions.csv", [], list(foundation.INTERFACE_FIELDS))
    result = foundation.inspect_foundation(project)
    assert result.status == "READY_FOR_INTERFACE_MATERIALIZATION"


def test_duplicate_query_ids_do_not_establish_interface_availability(project):
    validation_rows(project, "ustc")
    validation_rows(project, "cic")
    path = project / "results/interface/cic_validation_predictions.csv"
    _, rows = foundation.read_csv(path)
    csv_file(path, rows + rows)
    assert foundation.inspect_foundation(project).status == "READY_FOR_INTERFACE_MATERIALIZATION"


@pytest.mark.parametrize("content", [
    "query_id,query_id\na,b\n",
    "query_id,source_split\na,VALIDATION,extra\n",
    "query_id,source_split\na\n",
])
def test_malformed_validation_metadata_remains_unverified(project, content):
    path = project / "results/interface/cic_validation_predictions.csv"
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    result = foundation.inspect_foundation(project)
    assert result.status == "READY_FOR_INTERFACE_MATERIALIZATION"
    row = next(r for r in checklist(result) if r["check"] == "results/interface/cic_validation_predictions.csv")
    assert row["status"] == "WARN"
    assert "invalid metadata" in row["details"]


def test_dry_mode_writes_nothing(project, capsys):
    before = snapshot(project)
    assert foundation.main(["--project-root", str(project)]) == 0
    assert snapshot(project) == before
    assert "PHASE2_FOUNDATION=READY_FOR_INTERFACE_MATERIALIZATION" in capsys.readouterr().out


def test_execute_exactly_six_overwrite_and_phase_i_untouched(project):
    before = snapshot(project)
    assert foundation.main(["--project-root", str(project), "--execute"]) == 0
    after = snapshot(project)
    expected = {"metadata/rag/" + name for name in foundation.OUTPUT_NAMES}
    assert set(after) - set(before) == expected
    assert all(after[name] == content for name, content in before.items())
    assert foundation.main(["--project-root", str(project), "--execute"]) == 0
    assert snapshot(project) == after
    target = project / "metadata/rag" / foundation.OUTPUT_NAMES[0]
    target.write_text("stale", encoding="utf-8")
    foundation.main(["--project-root", str(project), "--execute"])
    assert snapshot(project) == after
    assert not (project / "results/rag").exists()


def test_preflight_prevents_partial_writes_for_invalid_target(project):
    folder = project / "metadata/rag"
    folder.mkdir()
    (folder / foundation.OUTPUT_NAMES[-1]).mkdir()
    result = foundation.inspect_foundation(project)
    with pytest.raises(ValueError, match="Unsafe canonical"):
        foundation.write_foundation(project, result)
    assert not any(p.is_file() for p in folder.iterdir())


def test_failed_output_preflight_prints_exactly_one_status(project, capsys):
    (project / "metadata/rag" / foundation.OUTPUT_NAMES[-1]).mkdir(parents=True)
    assert foundation.main(["--project-root", str(project), "--execute"]) == 2
    statuses = [line for line in capsys.readouterr().out.splitlines() if line.startswith("PHASE2_FOUNDATION=")]
    assert statuses == ["PHASE2_FOUNDATION=BLOCKED"]


def test_output_hardlink_cannot_overwrite_phase_i(project):
    source = project / "metadata/traffic/cic_experiment_closure.csv"
    before = source.read_bytes()
    target = project / "metadata/rag" / foundation.OUTPUT_NAMES[0]
    target.parent.mkdir(parents=True)
    try:
        target.hardlink_to(source)
    except (OSError, NotImplementedError):
        pytest.skip("Filesystem does not support hardlinks")
    result = foundation.inspect_foundation(project)
    with pytest.raises(ValueError, match="Unsafe canonical"):
        foundation.write_foundation(project, result)
    assert source.read_bytes() == before
    assert set(p.name for p in target.parent.iterdir()) == {foundation.OUTPUT_NAMES[0]}


def test_symlink_candidate_is_not_followed(project, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside")
    document = outside / "source.txt"
    document.write_text("outside source", encoding="utf-8")
    link = project / "papers"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Filesystem does not permit symlinks")
    assert document not in list(foundation.project_files(project))
    assert not foundation.corpus_inventory(project, list(foundation.project_files(project)))


def test_interface_contract_privacy_and_no_observation_rows():
    header, rows = foundation.contracts()[foundation.OUTPUT_NAMES[0]]
    assert header == foundation.INTERFACE_HEADER
    required = "query_id source_dataset source_split source_row_index classifier_model classifier_run_id predicted_class_id predicted_class_label confidence class_probabilities_json traffic_representation_json traffic_provenance_json query_text rag_partition".split()
    by_name = {r["field_name"]: r for r in rows}
    assert set(required) == set(by_name)
    assert "TEST prohibited" in by_name["source_split"]["description"]
    assert "authorized" in by_name["traffic_representation_json"]["description"]
    assert "no endpoints" in by_name["traffic_provenance_json"]["description"]
    assert "non-identifying" in by_name["source_row_index"]["description"]
    assert not set(foundation.FORBIDDEN_IDENTIFIERS) & set(by_name)
    assert all(identifier in foundation.POLICIES["privacy"] for identifier in foundation.FORBIDDEN_IDENTIFIERS)


def test_query_components_and_evidence_vocabulary():
    _, rows = foundation.contracts()[foundation.OUTPUT_NAMES[1]]
    assert len({r["component"] for r in rows}) == 6
    assert foundation.EVIDENCE_STATUSES == ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT", "CONFLICTING")
    descriptions = " ".join(r["description"] for r in rows)
    assert all(state in descriptions for state in foundation.EVIDENCE_STATUSES)
    assert "supporting AND contradicting" in descriptions
    assert "foundation authorizes no features" in descriptions


def test_evidence_provenance_schema_and_source_categories():
    header, rows = foundation.contracts()[foundation.OUTPUT_NAMES[2]]
    assert header == foundation.EVIDENCE_HEADER
    expected = "document_id chunk_id title source_type publisher authors publication_date retrieval_date canonical_url doi local_path document_section chunk_text traffic_topics security_topics source_provenance_status content_integrity_status corpus_partition authorized_for_retrieval".split()
    assert {r["field_name"] for r in rows} == set(expected)
    assert set(foundation.SOURCE_CATEGORIES) == {"peer_reviewed_research", "standards_rfc", "government_security_guidance", "official_protocol_documentation", "recognized_security_knowledge_base", "dataset_documentation"}
    assert "Deterministic readable non-cryptographic" in foundation.POLICIES["identifiers"]
    assert "explicit and separate" in foundation.POLICIES["partition"]
    assert "never silently" in foundation.POLICIES["partition"]


def test_inventory_exclusions_and_candidates_never_trusted(project):
    paths = ["papers/research.pdf", "literature/bibliography.csv", "docs/project.md", "src/rag/example.py", "tests/test_rag_example.py", "models/papers/model.txt", ".git/docs/a.md", ".venv/docs/a.md", "venv/docs/a.md", "node_modules/docs/a.md", "papers/__pycache__/a.txt", "papers/.cache/a.txt", "evidence/adversarial/sample.txt", "metadata/rag/old.csv"]
    for name in paths:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic inventory placeholder", encoding="utf-8")
    rows = foundation.corpus_inventory(project, list(foundation.project_files(project)))
    names = {r["relative_path"] for r in rows}
    assert names == {"papers/research.pdf", "literature/bibliography.csv", "docs/project.md", "evidence/adversarial/sample.txt", "metadata/rag/old.csv"}
    assert all(r["status"] in {"REQUIRES_REVIEW", "PROJECT_CONTEXT_ONLY", "NOT_EVIDENCE"} for r in rows)
    assert next(r for r in rows if r["relative_path"] == "metadata/rag/old.csv")["status"] == "NOT_EVIDENCE"
    assert all(set(r) == set(foundation.INVENTORY_HEADER) for r in rows)


def test_no_document_parsing_models_or_raw_test_access(project, monkeypatch):
    paths = ["data/processed/ustc_test.csv", "results/predictions/cic_test_predictions.csv", "models/classifier.pkl", "papers/source.pdf"]
    for name in paths:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"must never be opened")
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.relative_to(project).as_posix() in paths:
            raise AssertionError("Forbidden content access: " + str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    result = foundation.inspect_foundation(project)
    assert result.status == "READY_FOR_INTERFACE_MATERIALIZATION"


@pytest.mark.parametrize("stage", foundation.STAGES)
def test_downstream_stages_not_started(project, stage):
    rows = checklist(foundation.inspect_foundation(project))
    assert next(r for r in rows if r["section"] == "downstream" and r["check"] == stage)["status"] == "NOT_STARTED"


@pytest.mark.parametrize("stage,expected", [
    ("corpus", "document_id chunk_id source_type corpus_partition publication_date content_integrity_status"),
    ("retrieval", "query_id retrieval_method retrieval_rank document_id chunk_id lexical_score vector_score hybrid_score retrieved relevant_if_known"),
    ("provenance", "source_identity_present allowed_source_category citation_complete parent_document_traceable authorized_for_retrieval provenance_check_status provenance_failure_reason source_type"),
    ("reranking", "initial_rank initial_retrieval_score provenance_status reranking_score final_rank retained_for_context reranking_components_json"),
    ("context", "candidate_evidence_count verified_evidence_count selected_evidence_count supporting_evidence_count contradicting_evidence_count context_token_count context_character_count evidence_source_count context_status"),
    ("reasoning", "classifier_predicted_class classifier_confidence reasoning_predicted_class reasoning_confidence evidence_status reasoning_abstained reasoning_output_valid citation_count reasoning_latency_ms"),
    ("fusion", "classifier_class classifier_confidence rag_class evidence_status fusion_class fusion_confidence classification_changed classifier_correct fusion_correct abstained decision_reason_code true_class"),
    ("robustness", "attack_type contamination_level adversarial_document_count adversarial_chunk_count defense_configuration retrieval_success final_prediction_correct evidence_status trusted_context_count attack_ground_truth"),
    ("ablation", "query_id metric_name metric_value numerator denominator"),
    ("latency", "query_construction_ms retrieval_ms provenance_verification_ms reranking_ms context_construction_ms llm_reasoning_ms fusion_ms total_ms"),
])
def test_logging_contract_fields(stage, expected):
    header, rows = foundation.contracts()[foundation.OUTPUT_NAMES[5]]
    assert header == foundation.LOG_HEADER
    selected = [r for r in rows if r["stage"] == stage]
    names = {r["field_name"] for r in selected}
    assert set(expected.split()) | {"experiment_id", "configuration_name"} <= names
    assert len(names) == len(selected)
    assert all(r["figure_use"] for r in selected)
    assert not set(foundation.FORBIDDEN_IDENTIFIERS) & names


def test_plot_reproducibility_denominators_truth_and_optional_confidence():
    _, rows = foundation.contracts()[foundation.OUTPUT_NAMES[5]]
    mapping = {(r["stage"], r["field_name"]): r for r in rows}
    for key in (("reasoning", "reasoning_confidence"), ("retrieval", "relevant_if_known"), ("robustness", "attack_ground_truth"), ("latency", "total_ms")):
        assert mapping[key]["required"] == "NO"
    assert ("query_records", "relevance_judgments_json") in mapping
    assert ("query_records", "total_relevant_count") in mapping
    assert ("query_records", "query_status") in mapping
    assert ("fusion", "true_class") in mapping
    assert "pandas, numpy, matplotlib" in foundation.POLICIES["figures"]
    assert "WITHOUT RERUNNING EXPENSIVE EXPERIMENTS" in foundation.POLICIES["figures"]
    assert "NOT frozen" in foundation.POLICIES["ablations"]
    assert len(foundation.FUTURE_RESULT_FAMILIES) == 13
    assert mapping[("corpus", "chunk_id")]["required"] == "CONDITIONAL"
    assert ("corpus", "record_level") in mapping
    assert ("experiment_configuration", "corpus_snapshot_csv") in mapping
    assert ("experiment_configuration", "query_snapshot_csv") in mapping
    assert "zero chunks" in foundation.POLICIES["corpus_snapshot"]


def test_implementation_has_no_pipeline_network_hash_or_plot_dependencies():
    tree = ast.parse(Path(foundation.__file__).read_text(encoding="utf-8"))
    imports = set()
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                calls.add(node.func.id)
    assert imports <= {"__future__", "argparse", "csv", "json", "os", "dataclasses", "pathlib", "typing"}
    assert not calls & {"fit", "train", "predict", "predict_proba", "load", "loads", "embed", "retrieve", "urlopen", "request", "savefig", "figure", "sha256", "md5", "hash", "system", "popen", "eval", "exec"}


def test_all_readiness_statuses_valid(project):
    rows = checklist(foundation.inspect_foundation(project))
    assert {r["status"] for r in rows} <= {"PASS", "WARN", "BLOCKED", "NOT_STARTED"}
    checks = {r["check"] for r in rows}
    assert {"no_model_loading", "no_inference", "no_training", "no_test_access", "test_frozen", "partition", "experiment_logging_contract", "index_boundary", "provenance_boundary"} <= checks


def test_authoritative_core_pipeline_has_no_standalone_screening(project):
    assert foundation.CORE_PIPELINE == (
        "Structured Security Query", "Cybersecurity Evidence Corpus",
        "Evidence Indexing", "RAG Retrieval", "Provenance Verification",
        "Provenance-Aware Evidence Reranking", "Trusted Context Construction",
        "Constrained LLM Reasoning", "Evidence-Supported Decision Fusion",
        "Final Classification",
    )
    result = foundation.inspect_foundation(project)
    downstream = {r["check"] for r in checklist(result) if r["section"] == "downstream"}
    assert downstream == {
        "indexing", "retrieval", "provenance_verification", "reranking",
        "context_construction", "llm_reasoning", "decision_fusion",
        "final_classification",
    }
    assert "screening" not in foundation.STAGES
    assert "screening" not in foundation.LOG_FIELDS
    assert "screening" not in foundation.FIGURE_USES
    assert len(result.outputs) == len(foundation.OUTPUT_NAMES) == 6
    assert set(result.outputs) == set(foundation.OUTPUT_NAMES)
    logs = result.outputs[foundation.OUTPUT_NAMES[5]][1]
    assert not any(r["stage"] == "screening" for r in logs)
    assert "screening_results.csv" not in foundation.FUTURE_RESULT_FAMILIES


def test_core_contracts_do_not_require_screening():
    contracts = foundation.contracts()
    forbidden = {"injection_screen_status", "poisoning_screen_status", "screening_status", "screening_ms"}
    for name in (foundation.OUTPUT_NAMES[1], foundation.OUTPUT_NAMES[2], foundation.OUTPUT_NAMES[5]):
        rows = contracts[name][1]
        assert not forbidden & {r["field_name"] for r in rows}
        assert all("screen" not in r["description"].lower() for r in rows)
    assert "screen" not in foundation.POLICIES["provenance_boundary"].lower()
    assert "no_injection_screen" not in foundation.POLICIES["ablations"]
    assert "no_poisoning_screen" not in foundation.POLICIES["ablations"]
    assert all("screen" not in use.lower() for use in foundation.FIGURE_USES.values())


def test_attack_conditions_remain_exclusively_robustness_metadata():
    rows = foundation.contracts()[foundation.OUTPUT_NAMES[5]][1]
    attack_rows = [r for r in rows if r["field_name"] == "attack_type"]
    assert len(attack_rows) == 1
    assert attack_rows[0]["stage"] == "robustness"
    for attack in ("prompt_injection", "knowledge_poisoning"):
        assert "attack_type=" + attack in attack_rows[0]["description"]
        assert attack not in foundation.STAGES
        assert attack not in foundation.LOG_FIELDS
        assert all(attack not in r["description"] for r in rows if r["stage"] != "robustness")
    assert "robustness_results.csv" in foundation.FUTURE_RESULT_FAMILIES
    assert "explicit and separate" in foundation.POLICIES["partition"]


# Interface materialization tests use synthetic models/data only.
from dataclasses import replace
import json
from src.rag import materialize_phase1_interface as interface


@pytest.fixture
def interface_project(project, monkeypatch):
    # Keep exact run identities and predictor counts; shrink only fixture rows.
    specs = tuple(replace(s, count=2) for s in interface.DATASETS)
    monkeypatch.setattr(interface, "DATASETS", specs)
    path = project / "metadata/traffic/ustc_experiment_closure.csv"
    _, rows = foundation.read_csv(path)
    rows[0]["authoritative_run"] = specs[0].run
    csv_file(path, rows)
    result = foundation.inspect_foundation(project)
    foundation.write_foundation(project, result)
    for spec in specs:
        features = [f"feature_{i}" for i in range(spec.predictors)]
        if spec.prefix == "ustc":
            parameters = [dict(feature_name=f, retained="True", final_feature_order=i + 1) for i, f in enumerate(features)]
            csv_file(project / "results/audits/USTC-TFC2016/integrity_audit/ustc_final_predictor_inventory.csv",
                     [dict(predictor=f, status="SAFE") for f in features])
        else:
            parameters = [dict(feature_name=f, retained_final="YES", final_predictor_order=i) for i, f in enumerate(features)]
            csv_file(project / "metadata/traffic/cic_label_mapping.csv",
                     [dict(class_id=i, canonical_label=label) for i, label in enumerate(spec.labels)])
        csv_file(project / spec.parameters, parameters)
        model = project / f"models/{spec.name}/{spec.run}/lightgbm.txt"
        model.parent.mkdir(parents=True)
        model.write_text("synthetic model placeholder; must not load", encoding="utf-8")
        records = []
        for i in range(2):
            row = dict.fromkeys(features, 0.25)
            row.update(dict.fromkeys(spec.metadata, "PRIVATE_METADATA"))
            row[spec.target] = i
            row["binary_label" if spec.prefix == "ustc" else "Label"] = spec.labels[i]
            records.append(row)
        csv_file(project / f"data/processed/{spec.name}/validation.csv", records)
    return project


def test_interface_frozen_configuration():
    ustc, cic = interface.DATASETS
    assert (ustc.run, ustc.count, ustc.predictors, ustc.labels) == ("20260919T183945Z", 97001, 50, ("benign", "malware"))
    assert (cic.run, cic.count, cic.predictors, cic.labels) == ("20260920T145831Z", 31723, 62, ("Non-Tor", "NonVPN", "Tor", "VPN"))
    for spec in interface.DATASETS:
        header = interface.interface_header(spec)
        assert len(header) == (19 if spec.prefix == "ustc" else 21)
        assert not set(foundation.FORBIDDEN_IDENTIFIERS) & set(header)
        assert {f"prob_class_{i}" for i in range(len(spec.labels))} <= set(header)


@pytest.mark.parametrize("prefix", ["ustc", "cic"])
@pytest.mark.parametrize("failure", ["missing", "run", "unfrozen"])
def test_interface_requires_both_exact_closures(interface_project, prefix, failure):
    path = interface_project / f"metadata/traffic/{prefix}_experiment_closure.csv"
    if failure == "missing":
        path.unlink()
    else:
        _, rows = foundation.read_csv(path)
        rows[0]["authoritative_run" if failure == "run" else "stage_status"] = "INVALID"
        csv_file(path, rows)
    with pytest.raises(ValueError):
        interface.preflight(interface_project)


@pytest.mark.parametrize("split", ["TEST", "test", "train", "Validation", "../test"])
def test_interface_rejects_every_nonvalidation_split(interface_project, split):
    for spec in interface.DATASETS:
        with pytest.raises(ValueError, match="source_split"):
            interface.validation_path(interface_project, spec, split)


def test_interface_dry_mode_never_loads_model_or_writes(interface_project, monkeypatch):
    before = snapshot(interface_project)
    def forbidden(*args, **kwargs):
        raise AssertionError("Inference in dry mode")
    monkeypatch.setattr(interface, "infer", forbidden)
    monkeypatch.setattr(interface, "load_validation", forbidden)
    assert interface.main(["--project-root", str(interface_project)]) == 0
    assert snapshot(interface_project) == before


@pytest.mark.parametrize("name", ["phase1_to_rag_interface_contract.csv", "structured_security_query_contract.csv", "phase2_readiness_checklist.csv", "phase2_experiment_logging_contract.csv"])
def test_interface_requires_foundation_artifacts(interface_project, name):
    (interface_project / "metadata/rag" / name).unlink()
    with pytest.raises(ValueError):
        interface.preflight(interface_project)


def test_interface_requires_materialization_readiness(interface_project):
    path = interface_project / "metadata/rag/phase2_readiness_checklist.csv"
    _, rows = foundation.read_csv(path)
    for row in rows:
        if row["check"] == "PHASE2_FOUNDATION":
            row["details"] = "READY_FOR_CORPUS_CONSTRUCTION"
    csv_file(path, rows)
    with pytest.raises(ValueError, match="not ready"):
        interface.preflight(interface_project)


def test_interface_rejects_validation_symlink_before_open(interface_project, tmp_path_factory):
    spec = interface.DATASETS[0]
    path = interface.validation_path(interface_project, spec)
    outside = tmp_path_factory.mktemp("unapproved_input") / "unapproved.csv"
    outside.write_text("must not open", encoding="utf-8")
    path.unlink()
    try:
        path.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Filesystem does not permit symlinks")
    with pytest.raises(ValueError):
        interface.validation_path(interface_project, spec)


@pytest.mark.parametrize("prefix", ["ustc", "cic"])
def test_interface_predictor_contract_order_required(interface_project, prefix):
    spec = next(s for s in interface.DATASETS if s.prefix == prefix)
    path = interface_project / spec.parameters
    _, rows = foundation.read_csv(path)
    key = "final_feature_order" if prefix == "ustc" else "final_predictor_order"
    rows[1][key] = rows[0][key]
    csv_file(path, rows)
    with pytest.raises(ValueError):
        interface.preflight(interface_project)


def test_interface_probability_conversion_and_validation():
    np = pytest.importorskip("numpy")
    p = interface.probabilities([0.2, 0.8], 2, 2)
    np.testing.assert_allclose(p, [[0.8, 0.2], [0.2, 0.8]])
    multiclass = [[0.1, 0.2, 0.3, 0.4]]
    np.testing.assert_array_equal(interface.probabilities(multiclass, 1, 4), multiclass)
    for raw, count, classes in [([float("nan")], 1, 2), ([1.1], 1, 2), ([[0.2]*4], 1, 4), ([[0.5, 0.5]], 1, 4), ([[0, 0, 0, float("inf")]], 1, 4)]:
        with pytest.raises(ValueError):
            interface.probabilities(raw, count, classes)


def test_interface_safe_observations_and_evaluation_separation():
    ids = set()
    for spec in interface.DATASETS:
        features = [f"feature_{i}" for i in range(spec.predictors)]
        probability = [0.25, 0.75] if spec.prefix == "ustc" else [0.1, 0.2, 0.3, 0.4]
        row = interface.observation(spec, features, 0, [0.5]*spec.predictors, probability)
        assert row["query_id"] == spec.prefix.upper() + "-VAL-000000"
        ids.add(row["query_id"])
        assert row["confidence"] == max(probability)
        assert row["predicted_class_id"] == len(spec.labels) - 1
        assert row["rag_partition"] == "development_validation"
        assert list(json.loads(row["class_probabilities_json"])) == list(spec.labels)
        representation = json.loads(row["traffic_representation_json"])
        assert representation["predictor_count"] == spec.predictors
        assert list(representation["features"]) == features
        assert set(json.loads(row["traffic_provenance_json"])) == set(interface.SAFE_PROVENANCE)
        assert set(json.loads(row["query_text"])["evidence_states"]) == set(foundation.EVIDENCE_STATUSES)
        assert "SUPPORTING, CONTRADICTING" in row["query_text"]
        assert not set(interface.EVALUATION_ONLY) & set(row)
        for field in ("query_text", "class_probabilities_json", "traffic_representation_json", "traffic_provenance_json"):
            assert not any(term in row[field] for term in (*foundation.FORBIDDEN_IDENTIFIERS, *interface.EVALUATION_ONLY))
        changed_truth = [list(interface.records(spec, features, [[0.5]*spec.predictors], [truth], [probability]))[0] for truth in (0, 1)]
        assert changed_truth[0]["query_text"] == changed_truth[1]["query_text"]
        assert changed_truth[0]["true_class_id"] != changed_truth[1]["true_class_id"]
    assert len(ids) == 2


def test_interface_binary_tie_preserves_frozen_threshold():
    spec = interface.DATASETS[0]
    row = interface.observation(spec, [f"f{i}" for i in range(50)], 0, [0]*50, [0.5, 0.5])
    assert row["predicted_class_id"] == 1


@pytest.mark.parametrize("identifier", [*foundation.FORBIDDEN_IDENTIFIERS, *interface.EVALUATION_ONLY])
def test_interface_rejects_unsafe_features(identifier):
    assert not interface.safe_feature(identifier)


def test_interface_summary_statistics():
    spec = replace(interface.DATASETS[0], count=3)
    rows = [dict(query_id=str(i), true_class_id=0, predicted_class_id=p, classifier_correct=p == 0, confidence=c)
            for i, (p, c) in enumerate([(0, 0.6), (0, 0.8), (1, 0.9)])]
    summary = interface.summarize(spec, rows)
    overall = next(r for r in summary if r["summary_level"] == "dataset")
    assert overall["row_count"] == 3 and overall["correct_count"] == 2
    assert overall["median_confidence"] == 0.8
    assert overall["mean_confidence"] == pytest.approx(2.3/3)
    assert overall["true_class_id"] == overall["predicted_class_id"] == ""
    with pytest.raises(ValueError, match="Duplicate"):
        interface.summarize(spec, rows + [rows[0]])


def test_interface_execute_validation_only_overwrite_and_phase_i_untouched(interface_project, monkeypatch):
    np = pytest.importorskip("numpy")
    before = snapshot(interface_project)
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name.lower() != "test.csv"
        assert path.suffix != ".txt", "Real model must not be opened in tests"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    def fake_infer(root, spec, features, x):
        assert len(x) == 2 and x.shape[1] == spec.predictors
        raw = [0.2, 0.8] if spec.prefix == "ustc" else [[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]]
        return interface.probabilities(raw, 2, len(spec.labels))
    monkeypatch.setattr(interface, "infer", fake_infer)
    assert interface.main(["--project-root", str(interface_project), "--execute"]) == 0
    outputs = interface_project / "results/rag/interface"
    assert {p.name for p in outputs.iterdir()} == set(interface.OUTPUTS)
    saved = {p.name: p.read_bytes() for p in outputs.iterdir()}
    assert interface.main(["--project-root", str(interface_project), "--execute"]) == 0
    assert {p.name: p.read_bytes() for p in outputs.iterdir()} == saved
    # Restore read guard for byte comparison of fixture model placeholders.
    monkeypatch.setattr(Path, "open", original_open)
    after = snapshot(interface_project)
    assert all(after[name] == value for name, value in before.items())
    path = outputs / interface.OUTPUTS[0]
    _, rows = foundation.read_csv(path)
    rows[0]["classifier_run_id"] = "different"
    csv_file(path, rows)
    with pytest.raises(ValueError, match="different identity"):
        interface.preflight(interface_project)


def test_interface_row_count_and_nonfinite_fail_before_output(interface_project):
    pytest.importorskip("numpy")
    spec = interface.DATASETS[0]
    features = interface.predictor_order(interface_project, spec)
    path = interface.validation_path(interface_project, spec)
    _, rows = foundation.read_csv(path)
    csv_file(path, rows[:1])
    with pytest.raises(ValueError, match="row count"):
        interface.load_validation(interface_project, spec, features)
    rows[0][features[0]] = "nan"
    csv_file(path, rows)
    with pytest.raises(ValueError, match="Nonfinite"):
        interface.load_validation(interface_project, spec, features)
    assert not (interface_project / "results/rag/interface").exists()


def test_interface_no_training_selection_network_plot_or_test_path():
    source = Path(interface.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not calls & {"fit", "fit_transform", "train", "save_model", "savefig", "urlopen", "sha256", "md5"}
    strings = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]
    assert not any("test.csv" in value.lower() for value in strings)
    assert "--classifier" not in source and "--model" not in source
    assert "import lightgbm" not in source.split("def infer(")[0]
