"""Offline-only, immutable disjoint TEST subset; frozen prompts remain unchanged."""
import argparse
import random
from collections import Counter
from pathlib import Path

from src.rag.final_test import freeze

c, common, runtime, prompts = freeze.c, freeze.common, freeze.runtime, freeze.prompts
OUTPUT = "results/rag/test/stage08_llm/final_disjoint_4319_v4/"
TARGETS = {("CIC-Darknet2020", "Non-Tor"):850, ("CIC-Darknet2020", "NonVPN"):850,
    ("CIC-Darknet2020", "Tor"):69, ("CIC-Darknet2020", "VPN"):850,
    ("USTC-TFC2016", "benign"):850, ("USTC-TFC2016", "malware"):850}
PREVIOUS = c.rel(8)


def select(rows, excluded):
    population = {}
    for row in rows:
        identity = row["query_id"]
        common.require(row["source_split"] == "test" and "-TEST-" in identity, "Non-TEST selection row")
        common.require(identity not in population, "Duplicate population ID")
        pair = row["source_dataset"], row["classifier_prediction"]
        common.require(pair in TARGETS, "Unknown prediction stratum")
        population[identity] = pair  # No confidence, outcome, or ground truth enters sampling.
    rng, counts = random.Random(42), Counter()
    buckets = {pair:[] for pair in TARGETS}
    for identity, pair in sorted(population.items()):
        if identity in excluded:
            continue
        counts[pair] += 1
        if len(buckets[pair]) < TARGETS[pair]:
            buckets[pair].append(identity)
        else:
            position = rng.randrange(counts[pair])
            if position < TARGETS[pair]:
                buckets[pair][position] = identity
    common.require(all(len(buckets[pair]) == n for pair,n in TARGETS.items()), "Insufficient eligible predictions: "+str(counts))
    chosen = sorted((identity,*pair) for pair, ids in buckets.items() for identity in ids)
    return [dict(query_id=i,source_dataset=d,classifier_prediction=p,source_split="test",
        stratum=d+"/"+p,selection_ordinal=j,sampling_seed=42) for j,(i,d,p) in enumerate(chosen,1)]


def prepare(root):
    common.require(not (root/OUTPUT).exists(), "Immutable subset namespace already exists")
    record = common.parse_json((root/freeze.RECORD).read_text(encoding="utf-8"))
    common.require(record["source_sha256"] == freeze.source_hashes(root), "Frozen source hash mismatch")
    before = record["input_metadata"]
    common.require(before == c.snapshot(root,[r["path"] for r in before]), "Frozen artifact changed")
    configuration = record["configuration"]
    common.require(all(configuration[k] == v for k,v in dict(version="v4",prompt_schema="constrained_evidence_v4",
        runtime_configuration=runtime.SEMANTIC_STRATIFIED_1200,model=runtime.MODEL,max_model_len=5120,
        max_completion_tokens=1024,temperature=0,concurrency=1,sampling_seed=42,fusion_policy="retain_classifier").items()), "Frozen configuration mismatch")
    validation = set(record["completed_validation_ids"])
    previous_rows = list(c.rows(root,PREVIOUS))
    previous = {r["query_id"] for r in previous_rows}
    common.require(len(validation) == 4205 and len(previous) == len(previous_rows) == 1200, "Exclusion populations differ")
    common.require(previous == {r["query_id"] for r in record["test_subset"]}, "Historical TEST identity differs")
    execution_paths = [PREVIOUS]
    additional = freeze.OUTPUT+"raw_responses.jsonl"
    if (root/additional).exists():
        for row in c.rows(root,additional):
            common.require(row["source_split"] == "test" and row["query_id"] in previous,
                "Additional execution contains IDs outside the previous 1200")
        execution_paths.append(additional)
    execution_identity = c.snapshot(root,execution_paths)
    selected = select(c.queries(root),validation|previous)
    ids = {r["query_id"] for r in selected}
    common.require(len(selected) == len(ids) == 4319 and not ids & (validation|previous), "Invalid disjoint subset")
    by_id = {r["query_id"]:r for r in selected}
    tokenizer = runtime.audit.load_tokenizer(root)
    audit, found = [], set()
    for context in c.rows(root,c.rel(7)):
        identity = context["query_id"]
        if identity not in ids:
            continue
        common.require(identity not in found and context["source_split"] == "test", "Duplicate/non-TEST context")
        c.assert_safe(context)
        common.require(all(context[k] == by_id[identity][k] for k in ("source_dataset","classifier_prediction")), "Context identity differs")
        prompt = prompts.prompt_record(context,"v4",source_split="test")
        tokens = runtime.audit.prompt_tokens(tokenizer,prompt["prompt_text"])
        common.require(0 < tokens <= 4096, "Frozen token envelope exceeded: "+identity)
        audit.append(dict(query_id=identity,prompt_tokens=tokens,max_completion_tokens=1024,max_model_len=5120,gate_status="PASS"))
        found.add(identity)
    common.require(found == ids, "Missing frozen context")
    common.require(before == c.snapshot(root,[r["path"] for r in before]) and record["source_sha256"] == freeze.source_hashes(root), "Frozen inputs changed")
    common.require(execution_identity == c.snapshot(root,execution_paths), "Historical execution changed during selection")
    report = dict(test_subset_rows=4319,test_subset_unique_ids=4319,validation_rows_in_test_queue=0,
        overlap_with_validation_ids=0,overlap_with_previous_test_1200=0,duplicate_test_ids=0,
        ground_truth_in_selection=0,prompt_token_audit="PASS",frozen_validator_hash="MATCH",
        frozen_runtime_configuration="MATCH",completed_current_test=0,remaining_test=4319)
    # All checks complete before exclusive publication. No prompt copies or requests.
    c.table(root,OUTPUT+"subset_manifest.csv",selected)
    c.table(root,OUTPUT+"prompt_token_audit.csv",sorted(audit,key=lambda r:r["query_id"]))
    c.table(root,OUTPUT+"preflight.csv",[report])
    freeze.exclusive_json(root,OUTPUT+"configuration.json",dict(configuration=configuration,
        freeze_path=freeze.RECORD,freeze_sha256=freeze.digest(root/freeze.RECORD),
        selection="Algorithm R; sorted query_id; Random(42); dataset/prediction only",
        excluded_validation_ids=sorted(validation),excluded_previous_test_ids=sorted(previous),
        selected_counts={d+"/"+p:n for (d,p),n in TARGETS.items()},
        input_metadata=c.snapshot(root,[freeze.RECORD,*execution_paths,c.rel(1),c.rel(7)]),
        launch_ready=False,launch_limitation="Existing frozen executor is fixed to the previous 1200-query namespace"))
    for key,value in report.items():
        print(f"{key}={value}")


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--project-root",type=Path,default=Path("."))
    args = parser.parse_args(argv)
    prepare(args.project_root.resolve())


if __name__ == "__main__":
    main()
