"""Finalize the exact frozen validation union; no inference or evaluation labels."""
import argparse
from collections import Counter
from pathlib import Path

from src.rag import run_evidence_supported_fusion as fusion

common, prompts, importer = fusion.common, fusion.prompts, fusion.responses
FREEZE = "results/rag/llm/v4/validation_freeze/validation_freeze.json"
SOURCES = (
    "results/rag/llm/v4/execution/stratified_1200_semantic_fail_closed_5120_1024/llm_responses.jsonl",
    "results/rag/llm/v4/diagnostics/fail_closed_abstention_5120_1024/six_class_grounding_diagnostic.jsonl",
    "results/rag/llm/v4/validation_continuation/validated_responses.jsonl",
)
STAGE9 = "results/rag/validation/stage09_fusion/"
STAGE10 = "results/rag/validation/stage10_final/"


def assemble(root, expected):
    records, origins = {}, {}
    for source in SOURCES:
        for row in importer.external_rows(root, source):
            identity = row["query_id"]
            common.require(identity in expected and identity not in records, "Unknown/duplicate Stage-8 ID")
            common.require("-VAL-" in identity and row.get("source_split", "validation") == "validation", "Non-validation Stage-8 row")
            if source != SOURCES[-1]:
                common.require(row.get("response_valid") is True, "Unvalidated prior response")
            records[identity], origins[identity] = row, source
    common.require(set(records) == expected, "Missing frozen Stage-8 IDs")
    return records, origins


def fuse_one(query, response, prompt):
    common.validate_query(query)
    common.require("-VAL-" in query["query_id"], "Non-validation query ID")
    common.require(response["query_id"] == prompt["query_id"] == query["query_id"], "Query identity differs")
    common.require(response["source_dataset"] == prompt["source_dataset"] == query["source_dataset"], "Dataset differs")
    common.require(response["classifier_prediction"] == prompt["classifier_prediction"] == query["predicted_class_label"], "Classifier prediction differs")
    common.require(float(prompt["classifier_confidence"]) == float(query["confidence"]), "Classifier confidence differs")
    payload = common.parse_json(prompt["prompt_text"].partition("INPUT_JSON:\n")[2])
    context = dict(query_id=query["query_id"], source_dataset=query["source_dataset"],
        source_split="validation", rag_partition="development_validation",
        classifier_prediction=query["predicted_class_label"], classifier_confidence=query["confidence"],
        traffic_profile=query["retrieval_query_text"], selected_chunk_ids=prompt["allowed_chunk_ids"],
        evidence_records_json=common.json_text(payload["trusted_evidence"]), context_text=prompt["trusted_context"])
    common.require(prompts.prompt_record(context, "v4") == prompt, "Frozen prompt/query/context differs")
    value = {k: response[k] for k in prompts.REQUIRED_RESPONSE_FIELDS}
    validated = importer.validate_response(value, prompt)
    row = fusion.fuse(query, validated, context, policy="retain_classifier", prompt_version="v4")
    common.require(row["final_class"] == row["classifier_prediction"], "Classifier override prohibited")
    row.update(source_split="validation", reasoning_decision=value["reasoning_decision"],
        supported_class=value["supported_class"] or "", fusion_policy="retain_classifier",
        fusion_action=row["fusion_reason"], decision_source="retain_classifier")
    return row


def finalize(row, *, source_split="validation", response_reference=None):
    common.require(source_split in {"validation", "test"}, "Invalid explicit finalization split")
    common.require(source_split == "validation" or response_reference is not None, "TEST fusion reference required")
    common.require(row["source_split"] == source_split and row["fusion_policy"] == "retain_classifier"
        and row["final_class"] == row["classifier_prediction"], "Final classification policy/split differs")
    # Existing Stage-10 projection, retaining citations and classifier prediction.
    return dict(row, provenance=row["provenance_summary"], explanation=row["evidence_explanation"],
        explanation_reference=(response_reference or STAGE9+"validation_fusion_records.csv")+"#"+row["query_id"])


def build(root):
    receipt = common.parse_json((root/FREEZE).read_text(encoding="utf-8"))
    expected = set(receipt["completed_validation_ids"])
    common.require(len(expected) == len(receipt["completed_validation_ids"]) == 4205, "Frozen population differs")
    common.require(receipt["configuration"]["fusion_policy"] == "retain_classifier", "Frozen policy differs")
    paths = [common.QUERY_PATH, prompts.V4_OUTPUTS[0], *SOURCES]
    frozen = {r["path"]: r for r in receipt["input_metadata"]}
    common.require(all(r == frozen[r["path"]] for r in common.snapshot(root, paths)), "Frozen input identity changed")
    before = common.snapshot(root, [FREEZE, *paths])
    records, origins = assemble(root, expected)
    queries = {}
    for row in common.rows(root, common.QUERY_PATH, prompts.QUERY_FIELDS):
        if row["query_id"] in expected:
            common.require(row["query_id"] not in queries, "Duplicate query")
            query = {k: row[k] for k in prompts.QUERY_FIELDS}  # GT never enters fusion.
            common.validate_query(query)
            queries[row["query_id"]] = query
    common.require(set(queries) == expected, "Missing canonical validation query")
    result = {}
    for prompt in common.rows(root, prompts.V4_OUTPUTS[0], prompts.HEADER):
        identity = prompt["query_id"]
        if identity in expected:
            common.require(identity not in result, "Duplicate canonical prompt")
            result[identity] = fuse_one(queries[identity], records[identity], prompt)
    common.require(set(result) == expected, "Missing Stage-9 IDs")
    fused = [result[k] for k in sorted(expected)]
    final = [finalize(row) for row in fused]
    common.require(before == common.snapshot(root, [FREEZE, *paths]), "Inputs changed during finalization")
    checks = dict(rows=4205, unique_ids=4205, validation_rows=4205, test_rows=0,
        missing_stage8_ids=0, duplicate_ids=0, classifier_overrides=0, policy="retain_classifier",
        ground_truth_used_in_fusion=0, status="PASS")
    config = dict(policy="retain_classifier", source_split="validation", input_metadata_json=common.json_text(before),
        ground_truth_used="NO", llm_calls=0, frozen_query_count=4205,
        evidence_status_counts=common.json_text(Counter(r["evidence_status"] for r in fused)),
        fusion_action_counts=common.json_text(Counter(r["fusion_action"] for r in fused)))
    tables = [(STAGE9+"validation_fusion_records.csv", tuple(fused[0]), fused),
        (STAGE10+"validation_final_classification_records.csv", tuple(final[0]), final),
        (STAGE9+"stage8_input_manifest.csv", ("query_id", "source_split", "response_source"),
            [dict(query_id=k, source_split="validation", response_source=origins[k]) for k in sorted(expected)]),
        (STAGE9+"fusion_configuration.csv", common.CONFIG_HEADER, common.config_rows(config))]
    tables.extend((base+"validation_checklist.csv", tuple(checks), [checks]) for base in (STAGE9, STAGE10))
    return tables, expected


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    tables, expected = build(root)
    if args.execute:
        common.write_bundle(root, tables)  # Exclusive creation; no historical overwrite.
        for name, _, _ in tables[:2]:
            saved = list(common.rows(root, name))
            common.require(len(saved) == 4205 and {r["query_id"] for r in saved} == expected, "Published ID set differs")
            common.require(all(r["source_split"] == "validation" and r["final_class"] == r["classifier_prediction"]
                and r["fusion_policy"] == "retain_classifier" for r in saved), "Published policy differs")
    print(common.json_text(dict(status="PASS", executed=args.execute, rows=len(expected),
        evidence_status_counts=dict(Counter(r["evidence_status"] for r in tables[0][2])),
        fusion_action_counts=dict(Counter(r["fusion_action"] for r in tables[0][2])))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
