"""Phase-II contracts and read-only inventory; never runs a research pipeline.

Run manually with ``python -m src.rag.phase2_foundation --project-root .``.
Only --execute writes the six canonical metadata CSVs. No third-party packages.
Unknown metadata stays unknown. Inventory candidates are never trusted evidence.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EVIDENCE_STATUSES = ("SUPPORTED", "CONTRADICTED", "INSUFFICIENT", "CONFLICTING")
SOURCE_CATEGORIES = (
    "peer_reviewed_research", "standards_rfc", "government_security_guidance",
    "official_protocol_documentation", "recognized_security_knowledge_base",
    "dataset_documentation",
)
FORBIDDEN_IDENTIFIERS = ("Flow ID", "Src IP", "Dst IP", "Timestamp", "Src Port", "Dst Port")
EXCLUDED = {".git", ".venv", "venv", "__pycache__", "models", "node_modules", ".pytest_cache", ".mypy_cache"}
CONCEPTS = ("rag", "evidence", "corpus", "literature", "paper", "reference", "knowledge", "docs", "document", "provenance")
EXTENSIONS = {".pdf", ".txt", ".md", ".csv", ".json", ".jsonl", ".html", ".htm", ".docx"}
CORE_PIPELINE = (
    "Structured Security Query", "Cybersecurity Evidence Corpus",
    "Evidence Indexing", "RAG Retrieval", "Provenance Verification",
    "Provenance-Aware Evidence Reranking", "Trusted Context Construction",
    "Constrained LLM Reasoning", "Evidence-Supported Decision Fusion",
    "Final Classification",
)
STAGES = ("indexing", "retrieval", "provenance_verification", "reranking", "context_construction", "llm_reasoning", "decision_fusion", "final_classification")
OUTPUT_NAMES = (
    "phase1_to_rag_interface_contract.csv", "structured_security_query_contract.csv",
    "evidence_corpus_contract.csv", "evidence_corpus_inventory.csv",
    "phase2_readiness_checklist.csv", "phase2_experiment_logging_contract.csv",
)
FUTURE_RESULT_FAMILIES = (
    "query_records.csv", "retrieval_results.csv", "provenance_results.csv",
    "reranking_results.csv", "context_results.csv",
    "reasoning_results.csv", "fusion_results.csv", "experiment_metrics.csv",
    "per_class_metrics.csv", "robustness_results.csv", "ablation_results.csv",
    "latency_results.csv", "experiment_configuration.csv",
)
POLICIES = {
    "test_isolation": "PHASE-I TEST SETS ARE CLOSED. No fresh TEST inference or TEST use for query development, retrieval, reranking, prompts, evidence selection, thresholds, LLM configuration, fusion, or ablation design. Develop only from VALIDATION or independent external/query benchmarks.",
    "privacy": "Never include Flow ID, Src IP, Dst IP, Timestamp, Src Port or Dst Port in query text, traffic representation or provenance. Use an explicitly authorized non-identifying feature allowlist; no feature is authorized by this foundation.",
    "partition": "Trusted and adversarial corpus partitions must be explicit and separate. Adversarial evidence must never silently enter trusted retrieval. Filename, directory and publisher alone confer no trust.",
    "core_pipeline": " -> ".join(CORE_PIPELINE),
    "robustness_conditions": "Possible robustness/adversarial experiment conditions include attack_type=prompt_injection and attack_type=knowledge_poisoning. These are experimental conditions only, not ordinary pipeline stages or prerequisites for provenance verification, reranking or retrieval authorization.",
    "provenance_boundary": "Future checks: source identity present; source category allowed; citation complete; chunk traceable to parent; retrieval authorization; explicit trusted/adversarial partition. No scores or formulas defined.",
    "index_boundary": "Future only: ingestion, normalization, chunking, embedding, vector index, lexical index, hybrid retrieval. No embedding model, database or LLM selected.",
    "identifiers": "Deterministic readable non-cryptographic IDs: curated document namespace plus stable local document key; chunk ID adds section key and ordinal. Persist the mapping; detect collisions; never derive authority from IDs. No cryptographic hashes.",
    "figures": "ALL FINAL PAPER FIGURES MUST BE REGENERATABLE FROM SAVED CANONICAL CSV ARTIFACTS WITHOUT RERUNNING EXPENSIVE EXPERIMENTS. Later plotting requires only pandas, numpy, matplotlib and saved CSVs, without inference, retrieval, LLM/API, network or hidden runtime objects.",
    "missing_values": "Blank means unavailable, never zero or false. Save reasons for missing values. Never fabricate relevance, attack truth, timings or confidence; reasoning confidence requires a documented valid measure.",
    "logging": "Save immutable experiment/configuration identities, corpus/query snapshot identities, seeds, method versions, parameters, denominators, ground truth where authorized, and per-query outputs. All joins include experiment_id and configuration_name; candidate-stage joins also include retrieval_method, query_id, document_id, chunk_id. Preserve failed/empty queries in query_records for denominators. Record only performed outputs under results/rag/<EXPERIMENT_ID>/.",
    "logging_grain": "retrieval_results: one row per query x retrieved chunk x retrieval method. provenance/reranking: one row per candidate and method. context/reasoning/fusion/latency: one row per query per experiment/configuration; fusion must retain abstentions and failures. query_records preserves every eligible query, including zero-result queries. Robustness and ablation conditions require distinct configuration identities. Preserve selected citation/chunk IDs for joins; do not store raw endpoint identifiers.",
    "corpus_snapshot": "Future corpus construction must save a canonical CSV corpus ledger and reference its path and readable version identity in experiment_configuration. Preserve document records even when they have zero chunks, and separate chunk records using record_level=document or chunk. Save the ledger with the experiment's reproducibility inputs; do not depend on a mutable current corpus. Foundation creates no ledger or experiment outputs.",
    "relevance": "Save independent graded relevance judgments and the complete judged candidate universe in query_records as JSON, including relevant document/chunk IDs, judgment coverage and total relevant count. Unknown judgments remain blank. Recall@K and nDCG require this denominator/universe, not only retrieved rows.",
    "ablations": "Possible configurations: full_pipeline, no_provenance_verification, no_provenance_reranking, retrieval_only. Final list is NOT frozen; design using validation/external material only.",
}

INTERFACE_FIELDS = {
    "query_id": "Stable readable query identity.",
    "source_dataset": "USTC-TFC2016, CIC-Darknet2020, or named independent benchmark.",
    "source_split": "VALIDATION or EXTERNAL only during development; TEST prohibited.",
    "source_row_index": "Internal non-identifying processed row reference; never a raw flow identifier.",
    "classifier_model": "Classifier identity from frozen metadata; no model loading.",
    "classifier_run_id": "Frozen authoritative classifier run identity.",
    "predicted_class_id": "Class ID using the saved label mapping.",
    "predicted_class_label": "Class label using the saved label mapping.",
    "confidence": "Optional valid classifier confidence in [0,1]; blank when unavailable.",
    "class_probabilities_json": "Optional class-to-probability mapping, aligned to saved class mapping; blank if unavailable.",
    "traffic_representation_json": "Only explicitly authorized non-identifying behavior features or derived summaries; no endpoint identifiers.",
    "traffic_provenance_json": "Safe dataset, split, processed row index, predictor/preprocessing identities, classifier model and run only; no endpoints.",
    "query_text": "Sanitized behavior and evidence intent only; forbidden identifiers never included. No prompt is defined here.",
    "rag_partition": "Explicit validation-development or independent-benchmark partition; never TEST.",
}
EVIDENCE_FIELDS = (
    "document_id chunk_id title source_type publisher authors publication_date retrieval_date canonical_url doi local_path document_section chunk_text traffic_topics security_topics source_provenance_status content_integrity_status corpus_partition authorized_for_retrieval"
).split()
INTERFACE_HEADER = ("field_order", "field_name", "data_type", "required", "sensitivity", "source", "description")
EVIDENCE_HEADER = ("field_order", "field_name", "data_type", "required", "description")
QUERY_HEADER = ("component",) + EVIDENCE_HEADER
INVENTORY_HEADER = ("relative_path", "extension", "size_bytes", "candidate_role", "status", "reason")
CHECKLIST_HEADER = ("section", "check", "status", "details")
LOG_HEADER = ("stage", "field_order", "field_name", "data_type", "required", "description", "figure_use")

# Each stage specifies saved data, not an implementation or a choice of method.
LOG_FIELDS = {
    "corpus": "record_level document_id chunk_id source_type publisher publication_date traffic_topics security_topics corpus_partition source_provenance_status content_integrity_status authorized_for_retrieval",
    "query_records": "query_id source_dataset source_split classifier_model classifier_run_id query_class true_class class_mapping_json classifier_probabilities_json query_status relevance_judgments_json relevance_coverage total_relevant_count query_snapshot_id",
    "retrieval": "query_id source_dataset source_split classifier_model query_class retrieval_method retrieval_rank document_id chunk_id lexical_score vector_score hybrid_score retrieved relevant_if_known",
    "provenance": "query_id retrieval_method document_id chunk_id source_type retrieval_rank source_identity_present allowed_source_category citation_complete parent_document_traceable authorized_for_retrieval provenance_check_status provenance_failure_reason retained_after_provenance",
    "reranking": "query_id retrieval_method document_id chunk_id source_type corpus_partition initial_rank initial_retrieval_score provenance_status reranking_score reranking_components_json final_rank retained_for_context",
    "context": "query_id candidate_evidence_count verified_evidence_count selected_evidence_count supporting_evidence_count contradicting_evidence_count context_token_count context_character_count evidence_source_count context_status selected_chunk_ids_json",
    "reasoning": "query_id classifier_predicted_class classifier_confidence reasoning_predicted_class reasoning_confidence evidence_status supporting_evidence_count contradicting_evidence_count reasoning_abstained reasoning_output_valid citation_count reasoning_latency_ms confidence_method citation_chunk_ids_json",
    "fusion": "query_id source_dataset source_split classifier_model classifier_class classifier_confidence rag_class evidence_status fusion_class fusion_confidence classification_changed classifier_correct fusion_correct abstained decision_reason_code true_class fusion_probabilities_json confidence_method",
    "robustness": "query_id attack_type contamination_level adversarial_document_count adversarial_chunk_count defense_configuration retrieval_success final_prediction_correct evidence_status selected_evidence_count trusted_context_count contamination_unit attack_ground_truth",
    "ablation": "query_id true_class classifier_class fusion_class evidence_status abstained classifier_correct fusion_correct metric_name metric_value numerator denominator",
    "latency": "query_id query_construction_ms retrieval_ms provenance_verification_ms reranking_ms context_construction_ms llm_reasoning_ms fusion_ms total_ms timing_method",
    "experiment_metrics": "source_dataset source_split classifier_model metric_name metric_value k numerator denominator averaging_method evaluation_population uncertainty_method uncertainty_lower uncertainty_upper sample_count",
    "per_class_metrics": "source_dataset source_split classifier_model class_id class_label precision recall f1 support true_positive false_positive false_negative",
    "experiment_configuration": "corpus_snapshot_id corpus_snapshot_csv query_snapshot_id query_snapshot_csv seed code_version method_versions_json parameters_json model_identity prompt_version feature_allowlist_json confidence_method relevance_protocol attack_protocol partition_policy missing_value_policy stage_execution_status_json",
}
FIGURE_USES = {
    "corpus": "Document/chunk counts; source, topic, date, partition, provenance and content-integrity distributions. Deduplicate document_id for document counts.",
    "query_records": "Denominators, empty retrievals, independent relevance ground truth and valid calibration.",
    "retrieval": "Precision@K, Recall@K, MRR, nDCG; score/rank distributions; method and source comparison. Join complete judgments from query_records.",
    "provenance": "PASS/FAIL rate, failure reasons by source, removed evidence.",
    "reranking": "Pre/post rank, component scores, trusted promotion, adversarial demotion and top-K composition.",
    "context": "Evidence-retention funnel, context size, diversity, support versus contradiction.",
    "reasoning": "Evidence states, abstention, valid outputs and latency; valid confidence only.",
    "fusion": "Classifier/fusion performance; corrected/introduced errors; intervention, abstention, class distributions, confusion matrices and valid calibration.",
    "robustness": "Performance, retrieval success and trusted-context retention versus documented attack conditions and contamination.",
    "ablation": "Named configuration comparisons with common populations and denominators.",
    "latency": "Stage latency distributions; missing timing is blank, not zero.",
    "experiment_metrics": "Aggregate comparisons across datasets only when populations, metrics and protocols are comparable.",
    "per_class_metrics": "Per-class performance and support.",
    "experiment_configuration": "Reproduce figure grouping, configuration selection, population and methodological validity from CSV only.",
}


def data_type(field: str) -> str:
    if field.endswith("_json"):
        return "json"
    if field.endswith(("_count", "_rank")) or field in {"field_order", "source_row_index", "k", "seed", "support", "true_positive", "false_positive", "false_negative"}:
        return "integer"
    if field.endswith(("_score", "_confidence", "_ms")) or field in {"confidence", "metric_value", "precision", "recall", "f1", "contamination_level", "numerator", "denominator", "uncertainty_lower", "uncertainty_upper"}:
        return "number"
    if field.startswith("retained_") or field.endswith(("_correct", "_abstained", "_success")) or field in {"authorized_for_retrieval", "retrieved", "classification_changed", "abstained", "reasoning_output_valid", "source_identity_present", "allowed_source_category", "citation_complete", "parent_document_traceable"}:
        return "boolean"
    return "string"


def contracts() -> dict[str, tuple[tuple[str, ...], list[dict]]]:
    interface = [dict(zip(INTERFACE_HEADER, (i, name, data_type(name), "NO" if name in {"confidence", "class_probabilities_json"} else "YES", "INTERNAL_NON_IDENTIFYING", "frozen metadata / future authorized validation materialization", description))) for i, (name, description) in enumerate(INTERFACE_FIELDS.items(), 1)]
    components = {
        "A. QUERY IDENTITY": {"query_id": "Stable non-identifying identity.", "rag_partition": "Validation-development or independent benchmark only."},
        "B. CLASSIFIER OBSERVATION": {"predicted_class_label": "Observation, not an instruction to agree.", "confidence": "Optional valid classifier confidence.", "class_probabilities_json": "Optional competing-class probabilities using canonical mapping."},
        "C. TRAFFIC REPRESENTATION": {"traffic_representation_json": POLICIES["privacy"], "authorized_feature_names_json": "Explicit reviewed non-identifying protocol/statistical behavior allowlist; foundation authorizes no features."},
        "D. RETRIEVAL INTENT": {"evidence_intent": "Seek supporting AND contradicting evidence; permit failure to establish and conflicting evidence."},
        "E. PROVENANCE": {"source_dataset": "Named dataset or independent benchmark.", "source_split": "VALIDATION or EXTERNAL; never TEST.", "traffic_provenance_json": INTERFACE_FIELDS["traffic_provenance_json"]},
        "F. EVIDENCE CONSTRAINTS": {"allowed_source_categories_json": ", ".join(SOURCE_CATEGORIES), "provenance_requirements": POLICIES["provenance_boundary"], "evidence_status": "Exact evidence states (not traffic classes): " + ", ".join(EVIDENCE_STATUSES), "insufficient_evidence_policy": "Record INSUFFICIENT; permit abstention; never manufacture support."},
    }
    query = []
    for component, fields in components.items():
        for name, description in fields.items():
            query.append(dict(zip(QUERY_HEADER, (component, len(query) + 1, name, data_type(name), "NO" if name in {"confidence", "class_probabilities_json"} else "YES", description))))
    evidence_descriptions = {
        "document_id": POLICIES["identifiers"], "chunk_id": "Stable readable parent-document ID plus section and chunk ordinal; collision checked.",
        "source_type": "Allowed categories: " + ", ".join(SOURCE_CATEGORIES),
        "corpus_partition": POLICIES["partition"],
        "authorized_for_retrieval": "Explicit reviewed boolean; absent authorization means not authorized.",
        "source_provenance_status": "Recorded verification state, unknown until verified; no numeric trust score.",
        "content_integrity_status": "Recorded review outcome, unknown until checked; do not calculate hashes.",
        "publication_date": "Known source publication date with known precision; blank if unknown.",
        "retrieval_date": "Actual acquisition date, never inferred from filesystem timestamps.",
    }
    evidence = [dict(zip(EVIDENCE_HEADER, (i, name, data_type(name), "NO" if name in {"doi", "authors", "publication_date", "canonical_url"} else "YES", evidence_descriptions.get(name, "Source-recorded " + name.replace("_", " ") + "; do not invent missing metadata.")))) for i, name in enumerate(EVIDENCE_FIELDS, 1)]
    logs = []
    optional = {"relevant_if_known", "true_class", "classifier_correct", "fusion_correct", "final_prediction_correct", "attack_ground_truth", "relevance_judgments_json", "total_relevant_count", "reasoning_confidence", "fusion_confidence", "classifier_confidence", "classifier_probabilities_json", "fusion_probabilities_json", "confidence_method", "publication_date", "uncertainty_lower", "uncertainty_upper"}
    for stage, fields in LOG_FIELDS.items():
        names = list(dict.fromkeys(["experiment_id", "configuration_name"] + fields.split()))
        for i, name in enumerate(names, 1):
            required = "NO" if name in optional or name.endswith(("_ms", "_score")) else "YES"
            description = name.replace("_", " ") + ". "
            if required == "NO":
                description += POLICIES["missing_values"]
            if name == "reranking_components_json":
                description += "Save every component name and numeric value separately within the JSON mapping; no formula selected."
            if name == "relevance_judgments_json":
                description += POLICIES["relevance"]
            if name == "evidence_status":
                description += " | ".join(EVIDENCE_STATUSES)
            if stage == "robustness" and name == "attack_type":
                description += POLICIES["robustness_conditions"]
            if stage == "corpus" and name == "chunk_id":
                required = "CONDITIONAL"
                description += "Required for chunk records; blank on document records, including documents with zero chunks."
            if stage == "corpus" and name == "record_level":
                description += "document or chunk; count each population separately and deduplicate document IDs."
            if name in {"corpus_snapshot_csv", "query_snapshot_csv"}:
                description += "Portable path to a saved canonical CSV snapshot available to offline plotting."
            logs.append(dict(zip(LOG_HEADER, (stage, i, name, data_type(name), required, description, FIGURE_USES[stage]))))
    return {
        OUTPUT_NAMES[0]: (INTERFACE_HEADER, interface),
        OUTPUT_NAMES[1]: (QUERY_HEADER, query),
        OUTPUT_NAMES[2]: (EVIDENCE_HEADER, evidence),
        OUTPUT_NAMES[5]: (LOG_HEADER, logs),
    }


def linked_path(path: Path) -> bool:
    """Junctions are links too on Windows (is_junction is Python 3.12+)."""
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def project_files(root: Path) -> Iterable[Path]:
    """Never follow links or traverse model/cache directories."""
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d.lower() not in EXCLUDED and "cache" not in d.lower() and not linked_path(Path(directory) / d) and (Path(directory) / d).resolve().is_relative_to(root))
        for name in sorted(files):
            path = Path(directory) / name
            if not linked_path(path) and path.resolve().is_relative_to(root):
                yield path


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Missing or duplicate CSV headers")
        rows = list(reader)
        if any(None in row or any(value is None for value in row.values()) for row in rows):
            raise ValueError("Malformed CSV row")
        return list(reader.fieldnames), rows


def closure_check(root: Path, prefix: str) -> tuple[bool, str]:
    path = root / "metadata" / "traffic" / (prefix + "_experiment_closure.csv")
    expected = {"dataset": "USTC-TFC2016" if prefix == "ustc" else "CIC-Darknet2020", "stage_status": "CLOSED_FROZEN", "integrity_audit_status": "PASS", "finalization_status": "PASS", "frozen_for_retraining": "YES", "frozen_for_test_reuse": "YES"}
    if prefix == "cic":
        expected.update(authoritative_run="20260920T145831Z", phase_i_status="COMPLETE")
    try:
        if any(linked_path(p) for p in [path, *path.parents]) or not path.resolve().is_relative_to(root):
            raise ValueError("Linked closure is not accepted")
        headers, rows = read_csv(path)
        # Accept unambiguous wide records or conventional field/value contracts.
        key = next((k for k in ("field", "field_name", "key", "check", "property") if k in headers and "value" in headers), None)
        if key:
            values = {}
            for row in rows:
                name = row[key].strip()
                if name in values:
                    raise ValueError("Duplicate closure field: " + name)
                values[name] = row["value"].strip()
            records = [values]
        else:
            records = rows
        if len(records) != 1:
            raise ValueError("Expected exactly one canonical closure record")
        failures = [f"{k}: expected {v}, found {records[0].get(k, '<missing>')}" for k, v in expected.items() if records[0].get(k, "").strip() != v]
        return not failures, "; ".join(failures) if failures else str(path.relative_to(root)) + ": protected"
    except (OSError, ValueError, csv.Error, UnicodeError) as exc:
        return False, str(exc)


def corpus_inventory(root: Path, files: list[Path]) -> list[dict]:
    rows = []
    for path in files:
        relative = path.relative_to(root)
        name = relative.as_posix().lower()
        if path.suffix.lower() not in EXTENSIONS or not any(word in name for word in CONCEPTS):
            continue
        # Own derived outputs must not change the next inventory/readiness result.
        if relative.parent == Path("metadata/rag") and path.name in OUTPUT_NAMES:
            continue
        if relative.parts[0].lower() in {"src", "tests", "metadata", "results"}:
            role, status, reason = "existing_rag_artifact", "NOT_EVIDENCE", "Code-adjacent metadata/results; inventory context only, not source evidence."
        elif path.suffix.lower() == ".md" or relative.parts[0].lower() in {"docs", "doc"}:
            role, status, reason = "project_documentation", "PROJECT_CONTEXT_ONLY", "Documentation needs manual review; no source authority inferred."
        elif "literature" in name and path.suffix.lower() in {".csv", ".json", ".jsonl"}:
            role, status, reason = "literature_metadata", "REQUIRES_REVIEW", "Potential bibliography; metadata and authorization unverified."
        else:
            role, status, reason = "evidence_candidate", "REQUIRES_REVIEW", "Potential source only; contents, provenance and authorization unverified."
        rows.append(dict(zip(INVENTORY_HEADER, (relative.as_posix(), path.suffix.lower(), path.stat().st_size, role, status, reason))))
    return rows


def phase1_inventory(root: Path, files: list[Path]) -> tuple[list[dict], set[str]]:
    """Inspect metadata and CSV headers, never raw traffic or row-level TEST.

    Validation availability requires nonempty CSV rows with explicit validation
    split and dataset/model/prediction identity, not a suggestive filename.
    """
    findings = []
    available = set()
    for path in files:
        relative = path.relative_to(root)
        if relative.parts[0].lower() not in {"metadata", "results"}:
            continue
        if relative.parts[:2] == ("metadata", "rag") and path.name in OUTPUT_NAMES:
            continue
        lower = relative.as_posix().lower()
        if path.suffix.lower() not in {".csv", ".json", ".jsonl", ".txt"}:
            continue
        role = []
        for term, label in (("closure", "closure"), ("label", "class_mapping"), ("mapping", "class_mapping"), ("predictor", "predictor_contract"), ("preprocess", "preprocessing_contract"), ("split", "split_identity"), ("inventory", "artifact_inventory"), ("manifest", "artifact_inventory"), ("model", "model_identity")):
            if term in lower:
                role.append(label)
        aggregate = relative.as_posix() in {
            "results/final/USTC-TFC2016/ustc_train_results.csv", "results/final/USTC-TFC2016/ustc_validation_results.csv", "results/final/USTC-TFC2016/ustc_test_results.csv",
            "results/final/CIC-Darknet2020/cic_train_results.csv", "results/final/CIC-Darknet2020/cic_validation_results.csv", "results/final/CIC-Darknet2020/cic_test_results.csv", "results/final/CIC-Darknet2020/cic_per_class_results.csv",
        }
        if aggregate:
            role.append("aggregate_model_metrics")
        details = "Path-level discovery only; contents not verified."
        status = "WARN"
        # Never open TEST predictions, raw datasets, or unknown result payloads.
        validation_candidate = "validation" in lower and path.suffix.lower() == ".csv" and not aggregate and "test" not in lower
        if validation_candidate or aggregate or (relative.parts[0] == "metadata" and "test" not in lower and path.suffix.lower() == ".csv"):
            try:
                if validation_candidate:
                    headers, rows = read_csv(path)
                else:
                    with path.open("r", encoding="utf-8-sig", newline="") as stream:
                        headers = next(csv.reader(stream), [])
                    rows = []
                details = "CSV fields: " + ", ".join(headers)
                if headers:
                    status = "PASS"
                for field, label in (("classifier_model", "model_identity"), ("classifier_run_id", "model_identity"), ("source_split", "split_identity"), ("predicted_class_id", "row_predictions"), ("class_probabilities_json", "row_probabilities")):
                    if field in headers:
                        role.append(label)
                required = {"query_id", "source_dataset", "source_split", "source_row_index", "classifier_model", "classifier_run_id", "predicted_class_id", "predicted_class_label"}
                if validation_candidate and rows and required.issubset(headers):
                    valid = all(all(row[k].strip() for k in required) and row["source_split"].strip().upper() == "VALIDATION" for row in rows)
                    valid = valid and len({row["query_id"] for row in rows}) == len(rows)
                    if valid:
                        available.update(row["source_dataset"].strip() for row in rows if row["source_dataset"].strip() in {"USTC-TFC2016", "CIC-Darknet2020"} and (row["source_dataset"].strip() != "CIC-Darknet2020" or row["classifier_run_id"].strip() == "20260920T145831Z"))
                        details += "; explicit row-level VALIDATION prediction identity verified; feature/query materialization still requires review."
                    else:
                        details += "; validation identity invalid or ambiguous."
                elif validation_candidate:
                    details += "; no compatible nonempty row-level validation prediction schema verified."
            except (OSError, ValueError, UnicodeError, csv.Error) as exc:
                status, details = "WARN", "Unreadable or invalid metadata: " + str(exc)
        if role or validation_candidate:
            findings.append(dict(section="phase1_artifact", check=relative.as_posix(), status=status, details="Roles: " + ", ".join(sorted(set(role))) + ". " + details))
    return findings, available


@dataclass
class Foundation:
    status: str
    outputs: dict[str, tuple[tuple[str, ...], list[dict]]]


def inspect_foundation(project_root: Path) -> Foundation:
    root = project_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("Project root must be a directory")
    checks = []

    def check(section: str, name: str, status: str, details: str) -> None:
        checks.append(dict(section=section, check=name, status=status, details=details))

    protected = True
    for prefix in ("ustc", "cic"):
        passed, details = closure_check(root, prefix)
        protected &= passed
        check("protection", prefix + "_closure", "PASS" if passed else "BLOCKED", details)
    check("protection", "phase_i_complete", "PASS" if protected else "BLOCKED", "Both canonical closures must pass, including CIC phase_i_status=COMPLETE.")
    files = list(project_files(root))
    inventory = corpus_inventory(root, files)
    artifacts, available = phase1_inventory(root, files)
    checks.extend(artifacts)
    for dataset, prefix in (("USTC-TFC2016", "ustc"), ("CIC-Darknet2020", "cic")):
        for split in ("train", "validation", "test") + (("per_class",) if prefix == "cic" else ()):
            name = f"results/final/{dataset}/{prefix}_{split}_results.csv"
            found = next((a for a in artifacts if a["check"] == name), None)
            check("phase1_results", name, "PASS" if found and found["status"] == "PASS" else "WARN", "Existing aggregate CSV headers only; no evaluation." if found else "Expected final aggregate result missing.")
        check("interface", dataset + "_validation_predictions", "PASS" if dataset in available else "WARN", "Compatible row-level VALIDATION predictions found; full interface review remains required." if dataset in available else "No compatible row-level VALIDATION predictions verified. Required future PHASE-I -> PHASE-II INTERFACE MATERIALIZATION; do not generate in foundation. Other schemas remain unverified, not assumed absent.")
    for role in ("class_mapping", "predictor_contract", "preprocessing_contract", "split_identity", "model_identity", "artifact_inventory"):
        for dataset, prefix in (("USTC-TFC2016", "ustc"), ("CIC-Darknet2020", "cic")):
            found = [a["check"] for a in artifacts if role in a["details"] and prefix in a["check"].lower()]
            check("phase1_metadata", dataset + "_" + role, "WARN", "Discovered candidates require semantic review: " + ", ".join(found) if found else "No candidate metadata discovered.")
    for name in ("no_model_loading", "no_inference", "no_training", "no_test_access", "test_frozen"):
        check("safety", name, "PASS", "Foundation has no execution path for models, training, inference or row-level TEST. Existing final aggregate TEST metrics headers only are permitted.")
    for name, details in POLICIES.items():
        check("policy", name, "PASS", details)
    check("policy", "future_result_families", "PASS", "results/rag/<EXPERIMENT_ID>/: " + ", ".join(FUTURE_RESULT_FAMILIES) + "; create only performed-experiment outputs in future, none now.")
    for name in ("interface_contract", "structured_query_contract", "evidence_corpus_contract", "experiment_logging_contract"):
        check("contracts", name, "PASS", "Schema defined; no observations, prompts or research outputs generated.")
    check("corpus", "candidate_inventory", "PASS", f"{len(inventory)} candidate files inventoried without parsing documents.")
    check("corpus", "trusted_corpus_availability", "WARN", "No trusted corpus certified by foundation. Candidates are not authorization; construction or existing-corpus provenance/partition review is required.")
    for stage in STAGES:
        check("downstream", stage, "NOT_STARTED", "Boundary only; not implemented or executed by foundation. Existing artifacts do not establish an executed stage.")
    status = "BLOCKED" if not protected else ("READY_FOR_CORPUS_CONSTRUCTION" if available >= {"USTC-TFC2016", "CIC-Darknet2020"} else "READY_FOR_INTERFACE_MATERIALIZATION")
    check("foundation", "PHASE2_FOUNDATION", "BLOCKED" if not protected else "PASS", status)
    outputs = contracts()
    outputs[OUTPUT_NAMES[3]] = (INVENTORY_HEADER, inventory)
    outputs[OUTPUT_NAMES[4]] = (CHECKLIST_HEADER, checks)
    return Foundation(status, outputs)


def write_foundation(root: Path, result: Foundation) -> None:
    """Overwrite only canonical derived files after preflighting every target.

    A failed protection gate never writes. No backups or timestamp copies.
    Reject links/hardlinks so output aliases cannot modify Phase-I artifacts.
    """
    if result.status == "BLOCKED":
        raise ValueError("Phase-I protection failed; no outputs written")
    root = root.resolve(strict=True)
    folder = root / "metadata" / "rag"
    for parent in (root / "metadata", folder):
        if linked_path(parent) or (parent.exists() and not parent.is_dir()) or not parent.resolve().is_relative_to(root):
            raise ValueError("Unsafe output directory")
    if set(result.outputs) != set(OUTPUT_NAMES):
        raise ValueError("Exactly six canonical outputs required")
    for name in OUTPUT_NAMES:
        path = folder / name
        if linked_path(path) or (path.exists() and (not path.is_file() or path.stat().st_nlink != 1)):
            raise ValueError("Unsafe canonical output: " + name)
    folder.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_NAMES:
        header, rows = result.outputs[name]
        with (folder / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = inspect_foundation(args.project_root)
        for row in result.outputs[OUTPUT_NAMES[4]][1]:
            print(f"{row['status']} {row['section']}/{row['check']}: {row['details']}")
        if result.status == "BLOCKED":
            print("PHASE2_FOUNDATION=BLOCKED")
            return 2
        if args.execute:
            write_foundation(args.project_root, result)
        print("PHASE2_FOUNDATION=" + result.status)
        return 0
    except (OSError, ValueError, csv.Error) as exc:
        print("PHASE2_FOUNDATION=BLOCKED")
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
