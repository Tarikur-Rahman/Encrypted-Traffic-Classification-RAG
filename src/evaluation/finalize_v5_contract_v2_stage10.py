"""Create-only final classification of the accepted V5-v2 development cohort."""
import argparse
import ast
from collections import Counter
from pathlib import Path

from src.evaluation import finalize_v5_contract_v2_stage9 as upstream
from src.evaluation import finalize_validation_phase2 as historical

common = upstream.common
require = common.require
BASE = "results/rag/v5f_validation/stage10_final/" + upstream.stage8.COMPARISON + "/"
SOURCE = upstream.BASE + "final_decisions.csv"


def snapshot(root):
    return {name: upstream.runtime.method.previous.sha(upstream.runtime.method.path(root, name))
            for name in (SOURCE, upstream.BASE + "fusion_configuration.csv")}


def checked_rows(rows, ids, contexts, accepted):
    require(len(rows) == len(ids) == len(set(ids)) == 4205, "Final population differs")
    require([row["query_id"] for row in rows] == ids, "Missing/duplicate/reordered Stage9 IDs")
    for row in rows:
        q = row["query_id"]
        # Pure comparison, never rerun or publish Stage9. This also checks narrative
        # separately from the frozen validator's serialized claim linkage.
        require(row == upstream.fuse_accepted(contexts[q], accepted[q]), "Stage8/Stage9 linkage differs: " + q)
    return rows


def finalize(row):
    result = historical.finalize(row)
    result["explanation_reference"] = SOURCE + "#" + row["query_id"]
    return result


def build(root):
    before = snapshot(root)
    config = common.unique(common.rows(root, upstream.BASE + "fusion_configuration.csv", common.CONFIG_HEADER), "field")
    config = {k: v["value"] for k, v in config.items()}
    require(config["policy"] == "retain_classifier" and config["ground_truth_used"] == "False"
            and config["response_contract_version"] == "v2"
            and config["rejection_mapping"] == "NONE_ACCEPTED_ONLY", "Stage9 configuration differs")
    require(config["adapter_sha256"] == upstream.runtime.method.previous.sha(
        upstream.runtime.method.path(root, "src/evaluation/finalize_v5_contract_v2_stage9.py")), "Stage9 source changed")
    ids, contexts, accepted, stage8_hashes = upstream.accepted_inputs(root)
    require(ast.literal_eval(config["stage8_sha256"]) == stage8_hashes, "Stage9 input hashes differ")
    rows = checked_rows(list(common.rows(root, SOURCE, upstream.HEADER)), ids, contexts, accepted)
    final = [finalize(row) for row in rows]
    require(before == snapshot(root) and stage8_hashes == upstream.snapshot(root), "Inputs changed")
    checks = dict(stage9_rows=len(rows), stage10_rows=len(final), unique_ids=len(ids),
                  validation_rows=len(final), test_rows=0, missing_ids=0, duplicate_ids=0,
                  classifier_overrides=0, policy="retain_classifier", ground_truth_used=0,
                  evidence_status_counts=dict(Counter(row["evidence_status"] for row in final)))
    configuration = dict(policy="retain_classifier", source_split="validation", ground_truth_used=False,
                         response_contract_version="v2", input_sha256=common.json_text(before),
                         stage8_sha256=common.json_text(stage8_hashes),
                         development_disclosure=upstream.stage8.contract.DISCLOSURE)
    tables = [(BASE + "final_classification_records.csv", tuple(final[0]), final),
              (BASE + "final_classification_configuration.csv", common.CONFIG_HEADER, common.config_rows(configuration)),
              (BASE + "validation_checklist.csv", common.corpus.foundation.CHECKLIST_HEADER,
               common.checks("FINAL_CLASSIFICATION", checks))]
    return tables, checks, before, stage8_hashes


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--status-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    tables, checks, before, stage8_hashes = build(root)
    require(before == snapshot(root) and stage8_hashes == upstream.snapshot(root), "Inputs changed before publication")
    if args.execute:
        target = upstream.runtime.method.path(root, BASE + "final_classification_records.csv", False)
        require(not target.parent.exists(), "Stage10 namespace exists; overwrite prohibited")
        common.write_bundle(root, tables)
    print(common.json_text(dict(checks, executed=args.execute,
                               status="STAGE10_COMPLETE" if args.execute else "READY_FOR_STAGE10_EXECUTION")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
