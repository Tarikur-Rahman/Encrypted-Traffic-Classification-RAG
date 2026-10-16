"""Exclusive freeze receipt and read-only adapter to existing TEST inputs.

No scientific transformation, inference, or rewriting of historical receipts.
Execution reuses run_constrained_llm_v3's validation-frozen durable loop.
"""
import csv
import hashlib
import json
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

from . import contracts as c, llm
from src.rag import run_constrained_llm_v3 as execution

runtime, common, prompts = execution.runtime, execution.common, execution.prompts
RECORD = "results/rag/llm/v4/validation_freeze/validation_freeze.json"
OUTPUT = "results/rag/test/stage08_llm/final_frozen_v4/"
TEST_PROMPTS = llm.PROMPTS
ADAPTER_SOURCES = {"src/rag/llm_v3_runtime.py", "src/rag/run_constrained_llm_v3.py"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes(root):
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted((root/"src/rag").rglob("*.py"))}


def upstream_audit(root):
    """Check all old outputs/inputs; report the two known downstream source changes."""
    report = []
    for stage in range(1, 8):
        receipt = list(c.rows(root, c.rel(stage, c.CHECKS[stage-1])))
        common.require(len(receipt) == 1 and receipt[0]["status"] == "PASS", "Upstream TEST receipt not PASS")
        old = json.loads(receipt[0]["output_metadata_json"])
        common.require(c.snapshot(root, [r["path"] for r in old]) == old, "Historical TEST outputs changed")
        cfg = c.config(root, c.rel(stage, c.CONFIG[stage-1]))
        common.require(cfg["frozen_configuration_json"] == c.json_text(c.FROZEN), "Upstream scientific settings changed")
        changed = []
        for item in json.loads(cfg["input_metadata_json"]):
            now = c.snapshot(root, [item["path"]])[0]
            if now != item:
                common.require(item["path"] in ADAPTER_SOURCES, "Unexplained upstream input change: " + item["path"])
                changed.append(item["path"])
        report.append(dict(stage=stage, outputs="UNCHANGED", scientific_inputs="UNCHANGED", historical_downstream_source_mismatches=changed))
    return report


def selected_queries(root, validation_ids):
    # Existing sampler: Algorithm R over sorted IDs, only dataset/prediction.
    with sqlite3.connect(":memory:") as connection:
        chosen, _ = llm.select(c.queries(root), connection)
    old = list(c.rows(root, llm.SUBSET))
    common.require(old == [{k:str(v) for k,v in r.items()} for r in chosen], "Historical TEST subset differs from seed-42 sampling")
    common.require(not ({r["query_id"] for r in chosen} & set(validation_ids)), "TEST/validation identity overlap")
    return chosen


def validation_state(root):
    continuation = execution.CONTINUATION
    records = list(execution.continuation_jsonl(root, continuation+"validated_responses.jsonl"))
    common.require(len(records) == len({r["query_id"] for r in records}) == 2999, "Validation continuation count changed")
    common.require(all(r["source_split"] == "validation" for r in records), "Non-validation continuation record")
    counts = Counter(r["evidence_status"] for r in records)
    common.require(counts == {"INSUFFICIENT":2823, "CONFLICTING":176}, "Validation outcomes changed")
    common.require(all(r["reasoning_decision"] == "ABSTAIN" for r in records), "Validation decision changed")
    ids = {r["query_id"] for r in records}
    prior = set()
    for relative in (c.DEV+"llm_responses.jsonl",
                     "results/rag/llm/v4/diagnostics/fail_closed_abstention/six_class_grounding_diagnostic.jsonl",
                     "results/rag/llm/v4/diagnostics/fail_closed_abstention_5120_1024/six_class_grounding_diagnostic.jsonl"):
        prior.update(r["query_id"] for r in execution.continuation_jsonl(root, relative))
    common.require(len(prior) == 1206 and not (prior & ids), "Prior validation union changed")
    all_ids = prior | ids
    found = set()
    for row in common.rows(root, common.QUERY_PATH, ("query_id", "source_split")):
        if row["query_id"] in all_ids:
            common.require(row["source_split"] == "validation" and row["query_id"] not in found, "Validation identity/split differs")
            found.add(row["query_id"])
    common.require(found == all_ids and len(all_ids) == 4205, "Completed validation IDs missing")
    retries = list(execution.continuation_jsonl(root, continuation+"positive_retry_audit.jsonl"))
    common.require(len(retries) == 1574 and sum(r["retry_attempted"] == "YES" for r in retries) == 1153, "Retry audit changed")
    return sorted(all_ids), dict(prior_completed_validation=1206, new_validation_completed=2999,
        cumulative_validation_completed=4205, remaining_validation=124519, completed_test_by_continuation=0,
        evidence_status_counts=dict(counts, SUPPORTED=0, CONTRADICTED=0), ABSTAIN=2999, SUPPORT_CLASS=0,
        positive_retry_candidates=1574, correction_attempts=1153, skipped_corrections=421, validated_positive_outcomes=0)


def exclusive_json(root, relative, value):
    path = common.corpus.local_path(root, relative, must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        runtime.flush_record(stream, value)


def create(root):
    common.require(not (root/RECORD).exists() and not (root/OUTPUT).exists(), "Freeze/namespace already exists; never overwrite")
    upstream = upstream_audit(root)
    ids, state = validation_state(root)
    chosen = selected_queries(root, ids)
    # No model weights or evaluation-only labels are read for this receipt.
    sources = source_hashes(root)
    paths = [p.relative_to(root).as_posix() for directory in ("results/rag/test", execution.CONTINUATION, c.DEV,
             "results/rag/llm/v4/diagnostics") for p in sorted((root/directory).rglob("*")) if p.is_file()]
    paths += [common.QUERY_PATH, *prompts.V4_OUTPUTS]
    for stage in range(1, 8):
        cfg = c.config(root, c.rel(stage, c.CONFIG[stage-1]))
        paths.extend(row["path"] for row in json.loads(cfg["input_metadata_json"]) if not row["path"].startswith("src/"))
    config = dict(version="v4", prompt_schema="constrained_evidence_v4", runtime_configuration=runtime.SEMANTIC_STRATIFIED_1200,
        model=runtime.MODEL, model_directory=runtime.audit.MODEL, max_model_len=5120, max_completion_tokens=1024,
        temperature=0, concurrency=1, sampling_seed=42, fusion_policy="retain_classifier", positive_correction="exactly_one_if_budget_allows")
    record = dict(freeze_timestamp=datetime.now(timezone(timedelta(hours=6))).isoformat(), timezone="Asia/Dhaka",
        configuration=config, validation_state=state, completed_validation_ids=ids, test_subset=chosen,
        source_sha256=sources, input_metadata=c.snapshot(root, paths), upstream_audit=upstream,
        forensic_conclusion="Zero of 1574 validation retry-candidate contexts contained a qualifying positive-evidence line under the frozen V4 grammar; no grammar/prompt/corpus change authorized.",
        test_tuning_prohibited=True, historical_test=dict(unique_queries=1200, source_split="test", prefreeze=True,
            completeness="1200 historical responses validate under the old generate path",
            method_difference="Old run lacked the current single positive correction and raw-before-validation protocol",
            server_max_num_seqs="Not attested by saved response/configuration artifacts; client loop was sequential"),
        final_test_namespace=OUTPUT, scientific_limitation="Same deterministic TEST subset was evaluated before this freeze; a new run cannot restore never-seen holdout status.")
    exclusive_json(root, RECORD, record)
    exclusive_json(root, OUTPUT+"freeze_reference.json", dict(freeze_path=RECORD, sha256=digest(root/RECORD)))
    print("VALIDATION_FREEZE=CREATED\nFINAL_TEST_EXECUTED=NO")


def verify(root, args):
    runtime.validate_options(args)
    common.require(args.retry_limit == 3 and float(args.timeout) == 120, "Frozen transport configuration differs")
    record = json.loads(common.corpus.local_path(root, RECORD).read_text(encoding="utf-8"))
    reference = json.loads(common.corpus.local_path(root, OUTPUT+"freeze_reference.json").read_text(encoding="utf-8"))
    common.require(reference == dict(freeze_path=RECORD, sha256=digest(root/RECORD)), "Freeze receipt hash differs")
    common.require(record["source_sha256"] == source_hashes(root), "Frozen validator/runtime/source hash differs")
    old = record["input_metadata"]
    common.require(old == c.snapshot(root, [r["path"] for r in old]), "Frozen input/output identity differs")
    common.require(record["test_tuning_prohibited"] is True and record["final_test_namespace"] == OUTPUT, "Frozen policy differs")
    return record


def prepare(root, connection, args):
    record = verify(root, args)
    chosen = selected_queries(root, record["completed_validation_ids"])
    common.require(chosen == record["test_subset"], "Frozen subset differs")
    selected = {r["query_id"]:r for r in chosen}
    queries = {r["query_id"]:r for r in c.queries(root) if r["query_id"] in selected}
    common.require(len(queries) == 1200, "Missing TEST queries")
    connection.execute("CREATE TABLE population (id TEXT PRIMARY KEY, dataset TEXT, prediction TEXT, query TEXT, position TEXT, tokens INTEGER)")
    tokenizer = runtime.audit.load_tokenizer(root)  # Local tokenizer only; no weights/network.
    saved = {}
    with common.corpus.local_path(root, TEST_PROMPTS).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(iter(stream.readline, ""))
        common.require(tuple(reader.fieldnames) == (*prompts.HEADER, "qwen_prompt_tokens"), "TEST prompt header differs")
        while True:
            offset = stream.tell()
            row = next(reader, None)
            if row is None:
                break
            identity = row["query_id"]
            common.require(identity in selected and identity not in saved, "Unknown/duplicate TEST prompt")
            query = queries[identity]
            c.query_check(query)
            payload = common.parse_json(row["prompt_text"].partition("INPUT_JSON:\n")[2])
            c.assert_safe(payload)
            runtime.audit.validate_prompt(row, query, "v4")
            tokens = runtime.audit.prompt_tokens(tokenizer, row["prompt_text"])
            common.require(tokens == int(row["qwen_prompt_tokens"]) and 0 < tokens <= 4096, "Frozen TEST token audit failed")
            connection.execute("INSERT INTO population VALUES (?, ?, ?, ?, ?, ?)",
                (identity, row["source_dataset"], row["classifier_prediction"], common.json_text(query), str(offset), tokens))
            saved[identity] = {k:row[k] for k in prompts.HEADER}
    common.require(set(saved) == set(selected), "Incomplete TEST prompts")
    # Prove exact frozen prompt construction from existing TEST contexts, without rewriting them.
    found = set()
    for context in c.rows(root, c.rel(7)):
        identity = context["query_id"]
        if identity not in selected:
            continue
        c.assert_safe(context)
        common.require(context["source_split"] == "test" and identity not in found, "Non-TEST/duplicate trusted context")
        common.require(prompts.prompt_record(context, "v4", source_split="test") == saved[identity], "Frozen TEST prompt/context mismatch")
        found.add(identity)
    common.require(found == set(selected), "Missing selected TEST contexts")
    connection.commit()
    verify(root, args)
    # Historical TEST IDs are deliberately not progress in this distinct protocol.
    # Validation IDs are checked for overlap, never mixed into TEST denominators.
    return set(), set(), [RECORD, OUTPUT+"freeze_reference.json", TEST_PROMPTS, llm.SUBSET, c.rel(1), c.rel(7)], [r["path"] for r in record["input_metadata"]]
