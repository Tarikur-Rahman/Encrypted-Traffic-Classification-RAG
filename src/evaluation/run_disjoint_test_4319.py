"""Sequential adapter for the immutable disjoint TEST manifest; no scientific changes."""
import argparse
import csv
import math
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from src.evaluation import prepare_disjoint_test_subset as prepared

freeze = prepared.freeze
c, common, runtime, prompts = prepared.c, prepared.common, prepared.runtime, prepared.prompts
execution = freeze.execution
OUTPUT = prepared.OUTPUT
ARTIFACTS = ("raw_responses.jsonl", "validated_responses.jsonl", "positive_retry_audit.jsonl",
             "execution_errors.csv", "execution_summary.csv", "validation_checklist.csv")


def runtime_args():
    return SimpleNamespace(version="v4", runtime_configuration=runtime.SEMANTIC_STRATIFIED_1200,
        final_frozen_test=True, validation_continuation=False, model=runtime.MODEL,
        endpoint=runtime.ENDPOINT, max_model_len=5120, max_completion_tokens=1024,
        retry_limit=3, timeout=120)


def prepare(root, args):
    """Verify preparation and regenerate identical prompts in memory, never on disk."""
    runtime.validate_options(args)
    preparation_paths = [OUTPUT+n for n in
        ("configuration.json","subset_manifest.csv","prompt_token_audit.csv","preflight.csv")]
    preparation_identity = c.snapshot(root,preparation_paths)
    config = common.parse_json(c.safe_path(root, OUTPUT+"configuration.json").read_text(encoding="utf-8"))
    common.require(config["freeze_path"] == freeze.RECORD and
        config["freeze_sha256"] == freeze.digest(root/freeze.RECORD), "Freeze identity differs")
    record = common.parse_json((root/freeze.RECORD).read_text(encoding="utf-8"))
    common.require(record["source_sha256"] == freeze.source_hashes(root), "Frozen source hash differs")
    common.require(config["configuration"] == record["configuration"], "Prepared/frozen configuration differs")
    required = dict(version="v4", prompt_schema="constrained_evidence_v4",
        runtime_configuration=runtime.SEMANTIC_STRATIFIED_1200, model=runtime.MODEL,
        max_model_len=5120, max_completion_tokens=1024, temperature=0,
        concurrency=1, sampling_seed=42, fusion_policy="retain_classifier")
    common.require(all(record["configuration"][k] == v for k,v in required.items()), "Frozen runtime configuration differs")
    for old in (record["input_metadata"], config["input_metadata"]):
        common.require(old == c.snapshot(root,[r["path"] for r in old]), "Frozen inputs changed")
    validation, previous = set(config["excluded_validation_ids"]), set(config["excluded_previous_test_ids"])
    common.require(len(validation) == 4205 and validation == set(record["completed_validation_ids"]), "Validation exclusions differ")
    common.require(len(previous) == 1200 and previous == {r["query_id"] for r in record["test_subset"]}, "TEST exclusions differ")
    manifest = list(c.rows(root,OUTPUT+"subset_manifest.csv"))
    expected = prepared.select(c.queries(root),validation|previous)
    common.require(manifest == [{k:str(v) for k,v in r.items()} for r in expected], "Manifest differs from deterministic selection")
    ids = {r["query_id"] for r in manifest}
    common.require(len(manifest) == len(ids) == 4319 and not ids & (validation|previous), "Invalid disjoint population")
    audits = common.unique(c.rows(root,OUTPUT+"prompt_token_audit.csv"),"query_id")
    common.require(set(audits) == ids, "Token audit coverage differs")
    tokenizer = runtime.audit.load_tokenizer(root)  # Existing local-only tokenizer loader.
    selected, by_id = {}, {r["query_id"]:r for r in manifest}
    for context in c.rows(root,c.rel(7)):
        identity = context["query_id"]
        if identity not in ids:
            continue
        c.assert_safe(context)
        common.require(context["source_split"] == "test" and identity not in selected, "Duplicate/non-TEST context")
        common.require(all(context[k] == by_id[identity][k] for k in ("source_dataset","classifier_prediction")), "Context identity differs")
        prompt = prompts.prompt_record(context,"v4",source_split="test")
        tokens = runtime.audit.prompt_tokens(tokenizer,prompt["prompt_text"])
        common.require(audits[identity] == dict(query_id=identity,prompt_tokens=str(tokens),
            max_completion_tokens="1024",max_model_len="5120",gate_status="PASS") and 0 < tokens <= 4096,
            "Frozen token audit differs")
        selected[identity] = prompt,tokens
    common.require(set(selected) == ids, "Missing selected context")
    paths = [freeze.RECORD,*preparation_paths,*record["source_sha256"],
        *[r["path"] for r in config["input_metadata"]]]
    for old in (record["input_metadata"], config["input_metadata"]):
        common.require(old == c.snapshot(root,[r["path"] for r in old]), "Inputs changed during preflight")
    common.require(preparation_identity == c.snapshot(root,preparation_paths)
        and record["source_sha256"] == freeze.source_hashes(root), "Preparation/source identity changed")
    return selected,tokenizer,c.snapshot(root,paths)


def ledger(root, name, ids, raw=False, stream=None):
    path = OUTPUT+name
    records = {}
    if not (root/path).exists():
        return records
    if stream is not None:
        # Windows denies a second open while response_file holds its lock.
        # Read through the owned handle; exhaustion leaves it ready to append.
        stream.seek(0)
        rows = (common.parse_json(line) for line in stream)
    else:
        rows = c.rows(root,path)
    for row in rows:
        identity = row["query_id"]
        common.require(identity in ids, "Unknown/nonselected execution ID")
        key = (identity,row["phase"]) if raw else identity
        common.require(key not in records, "Duplicate durable record")
        if raw:
            common.require(row["phase"] in {"initial","correction"} and row["source_split"] == "test", "Invalid raw phase/split")
            common.require(row["phase"] == "initial" or (identity,"initial") in records, "Orphan correction")
        records[key] = row
    return records


def resolve(prompt, tokens, args, tokenizer, events, persist, on_error, allow_request):
    """Use the frozen request, validator and single-correction functions verbatim."""
    identity = prompt["query_id"]
    initial = events.get((identity,"initial"))
    recovered = initial is not None
    if initial is None:
        common.require(allow_request, "Status-only cannot request a missing response")
        initial = runtime.continuation_request(prompt,tokens,args,math.inf,"initial",persist,on_error)
    common.require(initial["prompt_tokens"] == tokens, "Durable initial token count differs")
    first = runtime.continuation_result(prompt,initial,args)
    correction = events.get((identity,"correction"))
    if first.get("validator_fallback_reason") != "POSITIVE_EVIDENCE":
        common.require(correction is None, "Unjustified correction")
        return first,None
    corrected = runtime.correction_prompt(prompt,first["grounding_gate_error_message"])
    correction_tokens = runtime.audit.prompt_tokens(tokenizer,corrected["prompt_text"])
    if correction is not None:
        common.require(correction["prompt_tokens"] == correction_tokens and correction_tokens+1024 <= 5120,
            "Durable correction token audit differs")
    elif not recovered and allow_request and correction_tokens+1024 <= 5120:
        correction = runtime.continuation_request(corrected,correction_tokens,args,math.inf,"correction",persist,on_error)
    # Match frozen recovery: never reissue a durable initial response or invent
    # a correction after recovering an interrupted query.
    result = runtime.continuation_result(prompt,correction,args) if correction else first
    skip = "" if correction else "RECOVERED_RAW_NO_REISSUE" if recovered else "TOKEN_BUDGET_OR_DEADLINE"
    audit = execution.continuation_retry_record(first,correction,result,correction_tokens,skip)
    return result,audit


def run(root, selected, tokenizer, identity, args, execute=False, resume=False):
    ids = set(selected)
    raw_path = c.safe_path(root,OUTPUT+ARTIFACTS[0],write=True)
    lock = execution.response_file(raw_path,True) if execute else nullcontext(None)
    with lock as raw_stream:
        common.require(identity == c.snapshot(root,[r["path"] for r in identity]), "Prepared inputs changed before execution")
        events = ledger(root,ARTIFACTS[0],ids,raw=True,stream=raw_stream)
        validated = ledger(root,ARTIFACTS[1],ids)
        retries = ledger(root,ARTIFACTS[2],ids)
        initial_ids = {i for i,phase in events if phase == "initial"}
        common.require(set(validated) <= initial_ids and set(retries) <= initial_ids, "Orphan validated/retry record")
        common.require(not execute or not (events or validated or retries) or resume, "Existing execution requires --resume")
        def persist(event):
            key = event["query_id"],event["phase"]
            common.require(key not in events, "Duplicate request response")
            execution.append_response(raw_stream,event)  # write, flush, fsync BEFORE validation.
            events[key] = event
        def append(name, row):
            with c.safe_path(root,OUTPUT+name,write=True).open("a",encoding="utf-8",newline="\n") as stream:
                runtime.flush_record(stream,row)
        def on_error(row):
            import os
            path = c.safe_path(root,OUTPUT+ARTIFACTS[3],write=True)
            with path.open("a",encoding="utf-8",newline="") as stream:
                csv.DictWriter(stream,fieldnames=execution.ERROR_HEADER).writerow(row)
                stream.flush()
                os.fsync(stream.fileno())
        recovered = {}
        # Validate every durable record before sending any further request.
        for query_id in sorted(initial_ids):
            prompt,tokens = selected[query_id]
            result,audit = resolve(prompt,tokens,args,tokenizer,events,persist,on_error,False)
            if query_id in validated:
                common.require(validated[query_id] == result, "Saved validation differs from durable response")
            if query_id in retries:
                common.require(audit is not None and all(retries[query_id].get(k) == v for k,v in audit.items()
                    if k not in {"skip_reason","correction_prompt_tokens"}), "Saved retry audit differs")
            recovered[query_id] = result,audit
        stats = dict(test_subset_rows=len(ids),test_subset_unique_ids=len(ids),validation_rows_in_test_queue=0,
            overlap_with_validation_ids=0,overlap_with_previous_test_1200=0,duplicate_test_ids=0,
            ground_truth_in_selection=0,prompt_token_audit="PASS",frozen_validator_hash="MATCH",
            frozen_runtime_configuration="MATCH",completed_current_test=len(validated),
            remaining_test=len(ids)-len(validated),raw_pending_validation=len(initial_ids-set(validated)))
        if not execute:
            common.require(identity == c.snapshot(root,[r["path"] for r in identity]), "Prepared inputs changed")
            for k,v in stats.items():
                print(f"{k}={v}")
            return
        error_path = c.safe_path(root,OUTPUT+ARTIFACTS[3],write=True)
        if not error_path.exists():
            common.corpus.write_table(error_path,execution.ERROR_HEADER,[],mode="x")
        def finish(query_id,result,audit):
            if audit is not None and query_id not in retries:
                append(ARTIFACTS[2],audit)
                retries[query_id] = audit
            if query_id not in validated:
                append(ARTIFACTS[1],result)
                validated[query_id] = result
        state = "INCOMPLETE"
        try:
            for query_id,(result,audit) in recovered.items():
                finish(query_id,result,audit)
            streak = 0
            for query_id in reversed([i for i,p in events if p == "initial"]):
                if validated[query_id].get("runtime_fallback_reason") != "TRANSPORT_RETRIES_EXHAUSTED":
                    break
                streak += 1
            common.require(streak < 5, "Frozen transport circuit open")
            for query_id in sorted(ids-set(validated)):
                prompt,tokens = selected[query_id]
                result,audit = resolve(prompt,tokens,args,tokenizer,events,persist,on_error,True)
                finish(query_id,result,audit)
                streak = streak+1 if result.get("runtime_fallback_reason") == "TRANSPORT_RETRIES_EXHAUSTED" else 0
                common.require(streak < 5, "Frozen transport circuit open after five exhausted queries")
            state = "COMPLETE"
        finally:
            common.require(identity == c.snapshot(root,[r["path"] for r in identity]), "Prepared inputs changed")
            summary = dict(execution_status=state,source_split="test",selected_test_queries=len(ids),
                completed_current_test=len(validated),remaining_test=len(ids)-len(validated),
                evidence_status_counts=common.json_text(Counter(r["evidence_status"] for r in validated.values())),
                evaluation_scope="disjoint_prediction_stratified_test_subset",temperature=0,concurrency=1,
                sampling_seed=42,max_model_len=5120,max_completion_tokens=1024)
            execution.replace_table(root,OUTPUT+ARTIFACTS[4],tuple(summary),[summary])
            check = dict(status=state,unique_validated_ids=len(validated),duplicate_ids=0,test_rows=len(validated),
                validation_rows=0,missing_ids=len(ids)-len(validated),ground_truth_used=0,
                frozen_configuration="MATCH",preparation_files="UNCHANGED")
            execution.replace_table(root,OUTPUT+ARTIFACTS[5],tuple(check),[check])
        print(common.json_text(summary))


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--project-root",type=Path,default=Path("."))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--status-only",action="store_true")
    mode.add_argument("--execute",action="store_true")
    parser.add_argument("--resume",action="store_true")
    cli = parser.parse_args(argv)
    root,args = cli.project_root.resolve(),runtime_args()
    selected,tokenizer,identity = prepare(root,args)
    run(root,selected,tokenizer,identity,args,cli.execute,cli.resume)


if __name__ == "__main__":
    main()
