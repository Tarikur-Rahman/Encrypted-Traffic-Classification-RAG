"""Offline retain-classifier finalization of the completed disjoint TEST run."""
import argparse
from collections import Counter
from pathlib import Path

from src.evaluation import finalize_validation_phase2 as validation
from src.evaluation import prepare_disjoint_test_subset as prepared

common, prompts, importer, fusion = validation.common, validation.prompts, validation.importer, validation.fusion
c, freeze = prepared.c, prepared.freeze
INPUT = prepared.OUTPUT
STAGE9 = "results/rag/test/stage09_fusion/final_disjoint_4319_v4/"
STAGE10 = "results/rag/test/stage10_final/final_disjoint_4319_v4/"
EXPECTED = {"INSUFFICIENT":4053, "CONFLICTING":266}


def fuse_one(query, response, context):
    c.query_check(query)
    c.assert_safe(context)
    common.require(response["source_split"] == context["source_split"] == "test", "Non-TEST response/context")
    common.require(query["query_id"] == response["query_id"] == context["query_id"], "Query identity differs")
    common.require(query["source_dataset"] == response["source_dataset"] == context["source_dataset"], "Dataset differs")
    common.require(query["predicted_class_label"] == response["classifier_prediction"] == context["classifier_prediction"], "Classifier prediction differs")
    common.require(float(query["confidence"]) == float(context["classifier_confidence"]), "Classifier confidence differs")
    # Explicit projections keep all execution metadata and evaluation labels out
    # of the existing scientific fusion function.
    safe_query = {k:query[k] for k in prompts.QUERY_FIELDS if k in query}
    value = {k:response[k] for k in prompts.REQUIRED_RESPONSE_FIELDS}
    c.assert_safe(value)
    prompt = prompts.prompt_record(context,"v4",source_split="test")
    validated = importer.validate_response(value,prompt)
    row = fusion.fuse(safe_query,validated,context,policy="retain_classifier",source_split="test",prompt_version="v4")
    common.require(row["final_class"] == row["classifier_prediction"], "Classifier override prohibited")
    row.update(source_split="test",reasoning_decision=value["reasoning_decision"],
        supported_class=value["supported_class"] or "",fusion_policy="retain_classifier",
        fusion_action=row["fusion_reason"],decision_source="retain_classifier")
    return row


def finalize(row):
    common.require(row["source_split"] == "test" and row["fusion_policy"] == "retain_classifier"
        and row["final_class"] == row["classifier_prediction"], "Final classification policy/split differs")
    c.assert_safe(row)
    # Same final-record projection as validation, with the TEST-only reference.
    return dict(row,provenance=row["provenance_summary"],explanation=row["evidence_explanation"],
        explanation_reference=STAGE9+"test_fusion_records.csv#"+row["query_id"])


def check_rows(rows, ids):
    common.require(len(rows) == len({r["query_id"] for r in rows}) == len(ids)
        and {r["query_id"] for r in rows} == ids, "Missing/duplicate finalization IDs")
    common.require(all(r["source_split"] == "test" and r["fusion_policy"] == "retain_classifier"
        and r["final_class"] == r["classifier_prediction"] for r in rows), "Finalization policy/split differs")


def build(root):
    config = common.parse_json(c.safe_path(root,INPUT+"configuration.json").read_text(encoding="utf-8"))
    common.require(config["freeze_path"] == freeze.RECORD and config["freeze_sha256"] == freeze.digest(root/freeze.RECORD), "Freeze identity differs")
    receipt = common.parse_json((root/freeze.RECORD).read_text(encoding="utf-8"))
    common.require(receipt["source_sha256"] == freeze.source_hashes(root), "Frozen scientific source differs")
    common.require(config["configuration"] == receipt["configuration"]
        and receipt["configuration"]["fusion_policy"] == "retain_classifier", "Frozen configuration differs")
    common.require(config["input_metadata"] == c.snapshot(root,[r["path"] for r in config["input_metadata"]]), "Prepared inputs changed")
    paths = [freeze.RECORD,INPUT+"configuration.json",INPUT+"subset_manifest.csv",
        INPUT+"validated_responses.jsonl",c.rel(1),c.rel(7),*receipt["source_sha256"]]
    before = c.snapshot(root,paths)
    manifest = common.unique(c.rows(root,INPUT+"subset_manifest.csv"),"query_id")
    ids = set(manifest)
    common.require(len(ids) == 4319 and all(r["source_split"] == "test" and "-TEST-" in r["query_id"]
        for r in manifest.values()), "Invalid TEST manifest")
    common.require(Counter((r["source_dataset"],r["classifier_prediction"]) for r in manifest.values()) == prepared.TARGETS,
        "Manifest strata differ")
    common.require(not ids & (set(receipt["completed_validation_ids"]) | set(config["excluded_previous_test_ids"])), "Excluded query in TEST set")
    responses = common.unique(c.rows(root,INPUT+"validated_responses.jsonl"),"query_id")
    common.require(set(responses) == ids and all(r["source_split"] == "test" for r in responses.values()), "Incomplete/non-TEST Stage 8")
    common.require(Counter(r["evidence_status"] for r in responses.values()) == EXPECTED, "Completed Stage-8 status counts differ")
    queries = {q["query_id"]:q for q in c.queries(root) if q["query_id"] in ids}
    common.require(set(queries) == ids, "Missing frozen classifier query")
    fused = {}
    for context in c.rows(root,c.rel(7)):
        identity = context["query_id"]
        if identity not in ids:
            continue
        common.require(identity not in fused, "Duplicate context")
        common.require(all(context[k] == manifest[identity][k] for k in ("source_dataset","classifier_prediction")), "Manifest/context differs")
        fused[identity] = fuse_one(queries[identity],responses[identity],context)
    stage9 = [fused[k] for k in sorted(fused)]
    check_rows(stage9,ids)
    stage10 = [finalize(row) for row in stage9]
    check_rows(stage10,ids)
    common.require(before == c.snapshot(root,paths), "Inputs changed during finalization")
    checks = dict(stage8_rows=4319,stage8_unique_ids=4319,stage9_rows=4319,stage9_unique_ids=4319,
        stage10_rows=4319,stage10_unique_ids=4319,test_rows=4319,validation_rows=0,
        missing_stage8_ids=0,duplicate_ids=0,classifier_overrides=0,policy="retain_classifier",ground_truth_used=0)
    metadata = dict(policy="retain_classifier",source_split="test",ground_truth_used="NO",llm_calls=0,
        input_metadata_json=common.json_text(before),evidence_status_counts=common.json_text(EXPECTED),
        fusion_action_counts=common.json_text(Counter(r["fusion_action"] for r in stage9)))
    tables = [(STAGE9+"test_fusion_records.csv",tuple(stage9[0]),stage9),
        (STAGE10+"test_final_classification_records.csv",tuple(stage10[0]),stage10),
        (STAGE9+"fusion_configuration.csv",common.CONFIG_HEADER,common.config_rows(metadata))]
    tables.extend((base+"validation_checklist.csv",tuple(checks),[checks]) for base in (STAGE9,STAGE10))
    return tables,checks,before


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--project-root",type=Path,default=Path("."))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status-only",action="store_true")
    mode.add_argument("--execute",action="store_true")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    tables,checks,before = build(root)
    if args.execute:
        # Reuse exclusive publication: refuse all existing outputs before writing.
        common.write_bundle(root,tables)
        ids = {r["query_id"] for r in tables[0][2]}
        for name,_,_ in tables[:2]:
            check_rows(list(c.rows(root,name)),ids)
        common.require(before == c.snapshot(root,[r["path"] for r in before]), "Inputs changed during publication")
    print("FINALIZATION="+("COMPLETE" if args.execute else "READY_NO_WRITES"))
    for key,value in checks.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
