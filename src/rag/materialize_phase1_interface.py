"""Validation interface CLI; scope-explicit observation helpers also serve TEST.

No training, preprocessing fitting, classifier selection, TEST input, network,
retrieval or plotting in this CLI. The final TEST orchestrator separately opts
into TEST observation construction. Evaluation labels stay outside payloads.
Probability policy: no normalization; row sums must agree with one within 1e-8.
USTC uses its frozen >=0.5 binary threshold (including ties); CIC uses argmax.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from src.rag import phase2_foundation as foundation


@dataclass(frozen=True)
class Dataset:
    name: str
    prefix: str
    run: str
    count: int
    predictors: int
    labels: tuple[str, ...]
    target: str
    metadata: tuple[str, ...]
    parameters: str


DATASETS = (
    Dataset("USTC-TFC2016", "ustc", "20260919T183945Z", 97001, 50,
            ("benign", "malware"), "target_binary",
            ("target_binary", "binary_label", "semantic_family", "capture_group", "source_pcap"),
            "data/processed/USTC-TFC2016/preprocessing_parameters.csv"),
    Dataset("CIC-Darknet2020", "cic", "20260920T145831Z", 31723, 62,
            ("Non-Tor", "NonVPN", "Tor", "VPN"), "target_id",
            ("target_id", "Label", "Flow ID", "Src IP", "Dst IP", "Timestamp", "Src Port", "Dst Port", "Label.1"),
            "metadata/traffic/cic_preprocessing_parameters.csv"),
)
OUTPUTS = ("ustc_validation_interface.csv", "cic_validation_interface.csv",
           "interface_summary.csv", "interface_validation_checklist.csv")
EVALUATION_ONLY = ("true_class_id", "true_class_label", "classifier_correct")
SUMMARY_HEADER = ("summary_level", "source_dataset", "classifier_model", "classifier_run_id",
                  "true_class_id", "true_class_label", "predicted_class_id", "predicted_class_label",
                  "row_count", "correct_count", "mean_confidence", "median_confidence",
                  "min_confidence", "max_confidence")
SAFE_PROVENANCE = ("source_dataset", "source_split", "source_row_index", "predictor_count",
                   "classifier_model", "classifier_run_id", "preprocessing_contract",
                   "predictor_contract", "phase_i_stage")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_path(root: Path, relative: str, *, existing=True) -> Path:
    """Reject aliases, links and escaping paths before any content access."""
    path = root / relative
    require(path.is_relative_to(root) and path.resolve().is_relative_to(root), "Path escapes project")
    for part in (path, *path.parents):
        if part == root:
            break
        require(not foundation.linked_path(part), "Linked paths prohibited")
    if existing:
        require(path.is_file() and path.stat().st_nlink == 1, "Missing or aliased input: " + relative)
    return path


def validation_path(root: Path, spec: Dataset, source_split="validation") -> Path:
    require(source_split == "validation", "Only source_split=validation is authorized")
    require(spec in DATASETS, "Unknown frozen dataset identity")
    # No caller-supplied split or path is interpolated into a data-file name.
    return checked_path(root, f"data/processed/{spec.name}/validation.csv")


def model_path(root, spec):
    return checked_path(root, f"models/{spec.name}/{spec.run}/lightgbm.txt")


def table(root, relative):
    return foundation.read_csv(checked_path(root, relative))


def interface_header(spec):
    return ("query_id", "source_dataset", "source_split", "source_row_index", "classifier_model",
            "classifier_run_id", "true_class_id", "true_class_label", "predicted_class_id",
            "predicted_class_label", "classifier_correct", "confidence",
            *(f"prob_class_{i}" for i in range(len(spec.labels))), "class_probabilities_json",
            "traffic_representation_json", "traffic_provenance_json", "query_text", "rag_partition")


def safe_feature(name):
    normalized = "".join(c for c in name.lower() if c.isalnum())
    forbidden = {"flowid", "srcip", "dstip", "timestamp", "srcport", "dstport", "sourceip", "destinationip", "sourceport", "destinationport"}
    evaluation = {"".join(c for c in n.lower() if c.isalnum()) for n in EVALUATION_ONLY}
    return bool(name) and normalized not in forbidden | evaluation and name not in {n for s in DATASETS for n in s.metadata}


def predictor_order(root, spec):
    _, rows = table(root, spec.parameters)
    flag, order, yes, allowed, start = ("retained", "final_feature_order", "true", {"true", "false"}, 1) if spec.prefix == "ustc" else ("retained_final", "final_predictor_order", "yes", {"yes", "no"}, 0)
    require(rows and all(r.get(flag, "").lower() in allowed for r in rows), "Invalid retention flags")
    names = [r.get("feature_name", "") for r in rows]
    require(all(names) and len(names) == len(set(names)), "Duplicate or missing predictor names")
    kept = sorted((r for r in rows if r[flag].lower() == yes), key=lambda r: int(r[order]))
    require([int(r[order]) for r in kept] == list(range(start, start + spec.predictors)), "Frozen predictor count/order differs")
    features = [r["feature_name"] for r in kept]
    require(all(safe_feature(n) for n in features), "Unsafe predictor in authoritative contract")
    if spec.prefix == "ustc":
        _, audit = table(root, "results/audits/USTC-TFC2016/integrity_audit/ustc_final_predictor_inventory.csv")
        require([r["predictor"] for r in audit] == features and all(r["status"] == "SAFE" for r in audit), "USTC predictor audit differs")
    else:
        _, mapping = table(root, "metadata/traffic/cic_label_mapping.csv")
        require([(r["class_id"], r["canonical_label"]) for r in mapping] == [(str(i), label) for i, label in enumerate(spec.labels)], "CIC class order differs")
    return features


def preflight(root: Path):
    root = root.resolve(strict=True)
    expected_contracts = foundation.contracts()
    for name in (foundation.OUTPUT_NAMES[0], foundation.OUTPUT_NAMES[1], foundation.OUTPUT_NAMES[5]):
        header, rows = table(root, "metadata/rag/" + name)
        expected_header, expected_rows = expected_contracts[name]
        require(header == list(expected_header) and rows, "Missing foundation contract: " + name)
        key = (lambda r: (r["stage"], r["field_name"])) if name == foundation.OUTPUT_NAMES[5] else (lambda r: r["field_name"])
        require(len({key(r) for r in rows}) == len(rows) and {key(r) for r in rows} == {key(r) for r in expected_rows}, "Foundation schema differs: " + name)
    _, readiness = table(root, "metadata/rag/phase2_readiness_checklist.csv")
    status = [r for r in readiness if r["section"] == "foundation" and r["check"] == "PHASE2_FOUNDATION"]
    require(len(status) == 1 and status[0]["status"] == "PASS" and status[0]["details"] == "READY_FOR_INTERFACE_MATERIALIZATION", "Foundation is not ready for interface materialization")
    require(not any(r["status"] == "BLOCKED" for r in readiness), "Foundation contains blocked checks")
    plans = []
    for spec in DATASETS:
        closure = f"metadata/traffic/{spec.prefix}_experiment_closure.csv"
        _, rows = table(root, closure)
        require(len(rows) == 1, "Ambiguous closure")
        expected = dict(dataset=spec.name, authoritative_run=spec.run, stage_status="CLOSED_FROZEN", integrity_audit_status="PASS", finalization_status="PASS", frozen_for_retraining="YES", frozen_for_test_reuse="YES")
        if spec.prefix == "cic":
            expected["phase_i_status"] = "COMPLETE"
        require(all(rows[0].get(k) == v for k, v in expected.items()), "Frozen closure identity/status differs: " + spec.name)
        features = predictor_order(root, spec)
        with validation_path(root, spec).open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        require(header == features + list(spec.metadata), "Processed validation schema/order differs: " + spec.name)
        model_path(root, spec)  # Existence only; no deserialization in preflight.
        plans.append((spec, features))
    check_outputs(root)
    return plans


def check_outputs(root):
    folder = checked_path(root, "results/rag/interface", existing=False)
    require(not folder.exists() or folder.is_dir(), "Invalid output directory")
    for name in OUTPUTS:
        path = checked_path(root, "results/rag/interface/" + name, existing=False)
        if not path.exists():
            continue
        require(path.is_file() and path.stat().st_nlink == 1, "Unsafe output target")
        # Every existing identity-bearing row must match this exact operation.
        if name.endswith("_interface.csv") or name == "interface_summary.csv":
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                count = 0
                for row in reader:
                    spec = next((s for s in DATASETS if row.get("source_dataset") == s.name), None)
                    require(spec is not None and row.get("classifier_run_id") == spec.run and row.get("classifier_model") == "lightgbm", "Existing output has different identity")
                    if name.endswith("_interface.csv"):
                        require(name == spec.prefix + "_validation_interface.csv" and row.get("source_split") == "validation", "Existing output has different split/dataset")
                    count += 1
                require(count > 0, "Existing output identity cannot be established")
        else:
            _, rows = foundation.read_csv(path)
            require(any(r.get("check") == "operation_identity" and r.get("details") == operation_identity() for r in rows), "Existing checklist identity differs")


def operation_identity():
    return ";".join(f"{s.name}/validation/lightgbm/{s.run}" for s in DATASETS)


def probabilities(raw, count, classes):
    import numpy as np
    p = np.asarray(raw, dtype=np.float64)
    if classes == 2 and p.shape == (count,):
        p = np.column_stack((1.0 - p, p))
    require(p.shape == (count, classes), "Probability matrix shape differs")
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid probability values")
    require(np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-8), "Probability row sums differ; no normalization performed")
    return p


def load_validation(root, spec, features):
    import numpy as np
    x = np.empty((spec.count, len(features)), dtype=np.float64)
    y = np.empty(spec.count, dtype=np.int64)
    with validation_path(root, spec).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == features + list(spec.metadata), "Validation schema changed")
        count = 0
        for i, row in enumerate(reader):
            require(i < spec.count and None not in row and all(v is not None for v in row.values()), "Malformed or excess validation rows")
            x[i] = [float(row[f]) for f in features]
            target = int(row[spec.target])
            require(0 <= target < len(spec.labels), "Invalid evaluation class ID")
            label_column = "binary_label" if spec.prefix == "ustc" else "Label"
            require(row[label_column] == spec.labels[target], "Evaluation class mapping differs")
            y[i] = target
            count += 1
    require(count == spec.count, "Validation row count differs: " + spec.name)
    require(np.isfinite(x).all(), "Nonfinite processed predictor values")
    return x, y


def infer(root, spec, features, x):
    # Lazy import and load: called only from execute, never from preflight.
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(model_path(root, spec)))
    require(model.num_feature() == spec.predictors, "Frozen model predictor count differs")
    # LightGBM replaces spaces in pandas feature names with underscores.
    require(model.feature_name() == [name.replace(" ", "_") for name in features], "Frozen model predictor order differs")
    objective = str(model.params.get("objective", ""))
    require(objective.startswith("binary" if len(spec.labels) == 2 else "multiclass"), "Frozen model objective differs")
    return probabilities(model.predict(x), spec.count, len(spec.labels))


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def observation(spec, features, index, values, probability, *, source_split="validation"):
    """This function has no ground-truth argument: query construction is blind."""
    require(len(features) == spec.predictors and len(values) == spec.predictors and all(safe_feature(f) for f in features), "Unsafe representation schema")
    require(all(math.isfinite(float(v)) for v in values), "Nonfinite representation")
    p = [float(v) for v in probability]
    require(len(p) == len(spec.labels) and all(math.isfinite(v) and 0 <= v <= 1 for v in p) and abs(sum(p) - 1) <= 1e-8, "Invalid row probabilities")
    predicted = int(p[1] >= 0.5) if len(p) == 2 else max(range(len(p)), key=p.__getitem__)
    require(source_split in {"validation", "test"}, "Unknown explicit observation scope")
    query_id = f"{spec.prefix.upper()}-{'VAL' if source_split == 'validation' else 'TEST'}-{index:06d}"
    distribution = dict(zip(spec.labels, p))
    representation = dict(predictor_count=len(features), features=dict(zip(features, map(float, values))))
    provenance = dict(source_dataset=spec.name, source_split=source_split, source_row_index=index,
                      predictor_count=len(features), classifier_model="lightgbm", classifier_run_id=spec.run,
                      preprocessing_contract=spec.parameters, predictor_contract=spec.parameters,
                      phase_i_stage="CLOSED_FROZEN")
    query = dict(source_dataset=spec.name, classifier="LightGBM", predicted_traffic_class=spec.labels[predicted],
                 classifier_confidence=max(p), class_probability_distribution=distribution,
                 traffic_representation_reference=query_id + ":traffic_representation_json",
                 retrieval_intent="Retrieve authorized cybersecurity evidence capable of SUPPORTING, CONTRADICTING, or establishing INSUFFICIENT/CONFLICTING evidence for the classifier observation.",
                 evidence_states=list(foundation.EVIDENCE_STATUSES), provenance_requirement=foundation.POLICIES["provenance_boundary"])
    result = dict(query_id=query_id, source_dataset=spec.name, source_split=source_split, source_row_index=index,
                  classifier_model="lightgbm", classifier_run_id=spec.run, predicted_class_id=predicted,
                  predicted_class_label=spec.labels[predicted], confidence=max(p),
                  class_probabilities_json=encode(distribution), traffic_representation_json=encode(representation),
                  traffic_provenance_json=encode(provenance), query_text=encode(query),
                  rag_partition="development_validation" if source_split == "validation" else "final_test")
    result.update({f"prob_class_{i}": v for i, v in enumerate(p)})
    for field in ("query_text", "class_probabilities_json", "traffic_representation_json", "traffic_provenance_json"):
        require(not any(term.lower() in result[field].lower() for term in (*foundation.FORBIDDEN_IDENTIFIERS, *EVALUATION_ONLY)), "Unsafe payload")
    require(tuple(provenance) == SAFE_PROVENANCE, "Unsafe provenance keys")
    return result


def records(spec, features, x, y, p):
    for i in range(len(y)):
        row = observation(spec, features, i, x[i], p[i])
        # Evaluation annotations are added only after all RAG payloads are built.
        row.update(true_class_id=int(y[i]), true_class_label=spec.labels[int(y[i])],
                   classifier_correct=int(y[i]) == row["predicted_class_id"])
        yield row


def summarize(spec, rows):
    groups = {}
    all_confidence = []
    correct = 0
    ids = set()
    for row in rows:
        require(row["query_id"] not in ids, "Duplicate query ID")
        ids.add(row["query_id"])
        groups.setdefault((row["true_class_id"], row["predicted_class_id"]), []).append(row["confidence"])
        all_confidence.append(row["confidence"])
        correct += int(row["classifier_correct"])
    require(len(ids) == spec.count, "Materialized row count differs")
    output = []
    for key, values in [*sorted(groups.items()), (None, all_confidence)]:
        truth, predicted = key if key is not None else (None, None)
        output.append(dict(summary_level="dataset" if key is None else "class_pair", source_dataset=spec.name,
                           classifier_model="lightgbm", classifier_run_id=spec.run,
                           true_class_id="" if key is None else truth, true_class_label="" if key is None else spec.labels[truth],
                           predicted_class_id="" if key is None else predicted, predicted_class_label="" if key is None else spec.labels[predicted],
                           row_count=len(values), correct_count=correct if key is None else (len(values) if truth == predicted else 0),
                           mean_confidence=statistics.mean(values), median_confidence=statistics.median(values),
                           min_confidence=min(values), max_confidence=max(values)))
    return output


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)


def execute(root, plans):
    prepared, summaries, checks = [], [], []
    def passed(section, check, details):
        checks.append(dict(section=section, check=check, status="PASS", details=details))
    # Validate both datasets and every payload before opening any output file.
    for spec, features in plans:
        x, y = load_validation(root, spec, features)
        p = infer(root, spec, features, x)
        summaries.extend(summarize(spec, records(spec, features, x, y, p)))
        prepared.append((spec, features, x, y, p))
        for name, details in {"closure_valid": spec.run, "model_artifact_exists": "Frozen LightGBM only",
                              "validation_rows": str(len(y)), "predictor_count": str(len(features)),
                              "predictors_finite": "All processed values finite", "probability_matrix_valid": "Finite [0,1]; sums within 1e-8; no normalization",
                              "class_order": encode(spec.labels), "confidence": "Maximum class probability; [0,1]"}.items():
            passed(spec.name, name, details)
    for name, details in {
        "operation_identity": operation_identity(), "only_validation_source": "source_split=validation; fixed input paths",
        "test_files_never_opened": "No TEST data path constructed or opened by this module",
        "query_ids_unique": "Distinct USTC/CIC prefixes and unique zero-based row indices",
        "no_forbidden_output_identifiers": "Explicit output column allowlist",
        "no_forbidden_payload_identifiers": "Every JSON/text payload checked",
        "evaluation_only": "true_class_id, true_class_label, classifier_correct: EVALUATION_ONLY; never RAG input",
        "ground_truth_absent_from_query": "Observation constructor receives no ground truth",
        "classifier_correct_absent_from_query": "Evaluation annotations added after query construction",
        "representation_predictors_only": "Full frozen ordered predictor vector; no inverse transform",
        "provenance_safe_only": encode(SAFE_PROVENANCE), "rag_partition": "development_validation",
        "no_training": "Prediction-only Booster", "no_preprocessing_fitting": "Existing processed values only",
        "no_test_inference": "Only fixed validation inputs", "no_hashes": "Readable deterministic IDs",
        "figure_reproducibility": "Saved numeric probabilities, confidence, evaluation annotations and summaries; no model reload required",
    }.items():
        passed("policy", name, details)
    check_outputs(root)
    folder = root / "results/rag/interface"
    folder.mkdir(parents=True, exist_ok=True)
    for spec, features, x, y, p in prepared:
        write_csv(folder / (spec.prefix + "_validation_interface.csv"), interface_header(spec), records(spec, features, x, y, p))
    write_csv(folder / OUTPUTS[2], SUMMARY_HEADER, summaries)
    write_csv(folder / OUTPUTS[3], foundation.CHECKLIST_HEADER, checks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = args.project_root.resolve(strict=True)
        plans = preflight(root)
        if not args.execute:
            print("PHASE1_RAG_INTERFACE=READY_FOR_MATERIALIZATION")
            print("ROW_COUNTS_AND_VALUES=CHECKED_DURING_EXECUTE")
            return 0
        execute(root, plans)
        print("PHASE1_RAG_INTERFACE=PASS\nUSTC_ROWS=97001\nCIC_ROWS=31723\nTEST_ACCESS=NO\nNEXT_STAGE=EVIDENCE_CORPUS_CONSTRUCTION")
        return 0
    except (OSError, ValueError, KeyError, csv.Error, ImportError) as exc:
        print("PHASE1_RAG_INTERFACE=BLOCKED\n" + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
