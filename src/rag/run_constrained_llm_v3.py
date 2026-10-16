"""Opt-in, resumable V3/V4 loopback Qwen execution. Dry mode never sends requests.

Completed records form a strict prefix of canonical prompt order. A malformed
unterminated last line blocks by default; --repair-trailing-partial explicitly
discards only that fragment and records its byte count in the error ledger.
"""
import csv
import math
import os
import sqlite3
import tempfile
import sys
import random
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from src.rag import llm_v3_runtime as runtime
from src.rag import materialize_llm_evaluation_subset as subset
from src.rag import llm_subset_preflight

common, audit, prompts = runtime.common, runtime.audit, runtime.prompts
require = common.require
BASE = "results/rag/llm/v3/execution/"
OUTPUTS = tuple(BASE + name for name in ("llm_responses.jsonl", "llm_execution_summary.csv", "llm_execution_configuration.csv", "llm_execution_errors.csv", "llm_execution_validation_checklist.csv"))
EXPORT = BASE + "llm_responses_for_import.jsonl"
ERROR_HEADER = ("query_id", "attempt", "retry_count", "error_type", "error_message", "will_retry")
SUMMARY_HEADER = ("metric", "source_dataset", "classifier_prediction", "evidence_status", "value")
CONTINUATION = "results/rag/llm/v4/validation_continuation/"
CONTINUATION_STRATA = (("USTC-TFC2016", "benign"), ("USTC-TFC2016", "malware"),
                       ("CIC-Darknet2020", "Non-Tor"), ("CIC-Darknet2020", "NonVPN"),
                       ("CIC-Darknet2020", "Tor"), ("CIC-Darknet2020", "VPN"))


def output_paths(version="v3", configuration="legacy"):
    require(version in {"v3", "v4"}, "Unknown execution version")
    suffix = runtime.configuration_directory(version, configuration)
    return tuple(path.replace("/v3/", "/" + version + "/").replace("/execution/", "/execution/" + suffix) for path in OUTPUTS)


def selected_version(args):
    return getattr(args, "version", "v3")


def runtime_temp_root(root):
    base = Path(os.environ.get("RAG_RUNTIME_TMPDIR") or ("/tmp/project2_rag" if sys.platform.startswith("linux") else str(Path(tempfile.gettempdir())/"project2_rag"))).resolve()
    require(not base.is_relative_to(root.resolve()), "Runtime temp directory must be outside project_root")
    if sys.platform.startswith("linux"):
        require(not base.is_relative_to(Path("/mnt").resolve()), "Runtime temp directory must use native Linux storage, not /mnt")
    base.mkdir(parents=True, exist_ok=True)
    return base


def audit_gate(root, connection, args):
    runtime.validate_configuration_envelope(args)
    version = selected_version(args)
    paths = audit.output_paths(args.max_model_len, args.max_completion_tokens, version)
    summaries = list(common.rows(root, paths[1], audit.SUMMARY_HEADER))
    require(len(summaries) == 1, "Exactly one audit summary required")
    row = summaries[0]
    require(row["gate_status"] == "PASS" and int(row["over_limit_with_reservation"]) == 0, "Token audit PASS prerequisite failed")
    require(row["configuration_key"] == audit.configuration_key(args.max_model_len, args.max_completion_tokens) and int(row["max_model_len"]) == args.max_model_len and int(row["reserved_completion_tokens"]) == args.max_completion_tokens and int(row["safe_prompt_limit"]) == args.max_model_len-args.max_completion_tokens, "Token-audit configuration mismatch")
    require(int(row["rows"]) == sum(common.QUERY_COUNTS.values()) and common.parse_json(row["dataset_counts_json"]) == common.QUERY_COUNTS, "Audit population differs")
    identity = common.parse_json(row["input_metadata_json"])
    expected_paths = [common.QUERY_PATH, *prompts.version_settings(version)[0]]
    require(identity == common.snapshot(root, expected_paths), "Audit prompt identity changed")
    require(row["tokenizer_metadata_json"] == common.json_text(audit.tokenizer_files(root)), "Audit tokenizer identity changed")
    counts = Counter()
    for item in common.rows(root, paths[0], audit.HEADER):
        found = connection.execute("SELECT payload, tokens FROM prompts WHERE id=?", (item["query_id"],)).fetchone()
        require(found is not None and found[1] is None, "Unknown/duplicate audit query")
        prompt = common.parse_json(found[0])
        count = int(item["qwen_prompt_tokens"])
        require(count > 0 and count+args.max_completion_tokens <= args.max_model_len and item["within_limit"] == "YES", "Unsafe audit row")
        require(int(item["max_model_len"]) == args.max_model_len and int(item["reserved_completion_tokens"]) == args.max_completion_tokens and int(item["safe_prompt_limit"]) == args.max_model_len-args.max_completion_tokens and int(item["reserved_total_tokens"]) == count+args.max_completion_tokens, "Audit row limits differ")
        require(item["source_dataset"] == prompt["source_dataset"] and item["classifier_prediction"] == prompt["classifier_prediction"], "Audit/query mismatch")
        connection.execute("UPDATE prompts SET tokens=? WHERE id=?", (count, item["query_id"]))
        counts[item["source_dataset"]] += 1
    require(dict(counts) == common.QUERY_COUNTS and connection.execute("SELECT id FROM prompts WHERE tokens IS NULL LIMIT 1").fetchone() is None, "Incomplete audit rows")
    connection.commit()
    return paths


def configuration(args, before, audit_paths, total):
    result = dict(model_path=audit.MODEL, model_name=args.model, served_model_name=args.model, endpoint=args.endpoint,
                prompt_schema=prompts.version_settings(selected_version(args))[1], query_count=str(total), max_model_len=str(args.max_model_len), max_completion_tokens=str(args.max_completion_tokens),
                temperature="0", seed_expectation="42", retry_limit=str(args.retry_limit), timeout=str(args.timeout),
                retry_policy="one initial request plus at most retry_limit additional attempts; permanent HTTP failures stop immediately",
                latency_policy="perf_counter seconds across all attempts for each query", percentile_policy="nearest rank; median averages middle pair",
                resume_policy="strict canonical prefix; validate every saved line; flush/fsync before next query",
                structured_output_policy="six exact fields; json_schema; citation uniqueness checked after generation; no content rewriting",
                token_audit_configuration=audit.configuration_key(args.max_model_len, args.max_completion_tokens), token_audit_summary=audit_paths[1],
                source_prompt_path=prompts.version_settings(selected_version(args))[0][0], input_metadata_json=common.json_text(before))
    if selected_version(args) == "v4":
        result.update(positive_evidence_gate=prompts.positive_evidence_v4.POLICY,
                      diagnostic_gate="six current validated responses; human acceptance is not inferred",
                      human_grounding_acceptance="NOT_INFERRED")
    if runtime.configuration_name(args) in runtime.FAIL_CLOSED_CONFIGURATIONS:
        result.update(runtime_configuration=runtime.configuration_name(args),
                      structured_output_policy="six exact fields; raw visible JSON retained; explicit validator-enforced abstention on V4 positive-grounding rejection only",
                      retry_policy="retry only retryable RequestFailure transport/runtime failures; no retry for deterministic gate rejection or invalid schema/tokens",
                      validator_fallback_explanation=runtime.FALLBACK_EXPLANATION)
    if runtime.configuration_name(args) in runtime.SUBSET_CONFIGURATIONS:
        result.update(evaluation_scope=subset.SCOPE, population_queries=str(sum(common.QUERY_COUNTS.values())),
                      evaluation_target_queries=str(subset.TOTAL), sampling_seed=str(subset.SEED), sampling_strategy=subset.STRATEGY,
                      subset_manifest=subset.MANIFEST, sampling_algorithm="Algorithm R; Random(42); canonical query_id order; manifest ordinals one-based",
                      diagnostic_gate="reuse current 5120/1024 fail-closed six-class diagnostic; identical grounding and generation envelope",
                      reporting_policy="per-stratum sample counts; balanced subset is not the natural population distribution")
    if runtime.configuration_name(args) == runtime.SEMANTIC_STRATIFIED_1200:
        result.update(validator_fallback_explanation=runtime.SEMANTIC_FALLBACK_EXPLANATION,
                      structured_output_policy="six exact typed fields and expected query required; explicit semantic or positive-evidence abstention; visible raw JSON and exact errors retained",
                      diagnostic_gate="reuse 5120/1024 positive-gate diagnostic for unchanged prompts/envelope; semantic runtime acceptance is not inferred")
    return result


def diagnostic_gate(root, connection, args):
    """Require the six saved, current V4 diagnostics; never infer human acceptance."""
    from src.rag import run_v3_grounding_diagnostic as diagnostic
    if runtime.configuration_name(args) in runtime.SUBSET_CONFIGURATIONS:
        prior = SimpleNamespace(**vars(args))
        prior.runtime_configuration = runtime.FAIL_CLOSED_5120_1024
        return diagnostic_gate(root, connection, prior)
    configuration = runtime.configuration_name(args)
    paths = diagnostic.output_paths("v4", configuration)
    selected = diagnostic.select_six(common.parse_json(row[0]) for row in connection.execute("SELECT payload FROM prompts ORDER BY ordinal"))
    identity = common.json_text(common.snapshot(root, [common.QUERY_PATH, *prompts.V4_OUTPUTS]))
    tokenizer_identity = common.json_text(audit.tokenizer_files(root))
    header = diagnostic.summary_header(configuration)
    summaries = iter(common.rows(root, paths[1], header))
    common.safe_input(paths[0])
    with common.corpus.local_path(root, paths[0]).open("r", encoding="utf-8") as stream:
        for prompt in selected:
            line = stream.readline()
            require(line.strip(), "Six-class diagnostic incomplete")
            row = common.parse_json(line)
            require(isinstance(row, dict) and row.get("query_id") == prompt["query_id"], "Diagnostic order/query differs")
            require(row.get("input_metadata_json") == identity and row.get("tokenizer_metadata_json") == tokenizer_identity, "Diagnostic inputs changed")
            require(row.get("human_grounding_status") == "REVIEW_REQUIRED", "Diagnostic cannot imply human acceptance")
            tokens = connection.execute("SELECT tokens FROM prompts WHERE id=?", (prompt["query_id"],)).fetchone()[0]
            saved = {key: value for key, value in row.items() if key not in {"input_metadata_json", "tokenizer_metadata_json", "api_prompt_tokens", "retry_errors", "human_grounding_status"}}
            validate_saved(saved, prompt, tokens, args)
            require(row.get("api_prompt_tokens") == tokens, "Diagnostic token accounting differs")
            summary = next(summaries, None)
            require(summary is not None, "Missing diagnostic summary")
            expected = dict({key: row[key] for key in header if key in row}, citation_count=len(row["cited_chunk_ids"]), structural_status="PASS", human_grounding_status="REVIEW_REQUIRED")
            require(summary == {key: "" if value is None else str(value) for key, value in expected.items()}, "Diagnostic summary differs from validated response")
        require(not stream.read(1) and next(summaries, None) is None, "Diagnostic must contain exactly six responses")
    return paths


def check_configuration(root, expected):
    outputs = output_paths(expected["prompt_schema"].rsplit("_", 1)[1], expected.get("runtime_configuration", "legacy"))
    path = common.corpus.local_path(root, outputs[2], must_exist=False)
    if path.exists():
        old = {k: r["value"] for k, r in common.unique(common.rows(root, outputs[2], common.CONFIG_HEADER), "field").items()}
        old.pop("execution_status", None)
        require(old == expected, "Resume configuration/input identity differs")
    else:
        raw = common.corpus.local_path(root, outputs[0], must_exist=False)
        require(not raw.exists() or raw.stat().st_size == 0, "Existing responses lack execution configuration")


def validate_saved(row, prompt, tokens, args):
    require(isinstance(row, dict), "Invalid saved response object")
    value = {key: row[key] for key in runtime.FIELDS}
    runtime.validate_logical(value, prompt)
    expected = dict(source_dataset=prompt["source_dataset"], classifier_prediction=prompt["classifier_prediction"], prompt_schema_version=prompts.version_settings(selected_version(args))[1],
                    model_name=args.model, temperature=0, max_completion_tokens=args.max_completion_tokens, max_model_len=args.max_model_len, response_valid=True, finish_reason="stop", error_type="", error_message="")
    if runtime.configuration_name(args) in runtime.FAIL_CLOSED_CONFIGURATIONS:
        runtime.validate_configuration_envelope(args)
        if runtime.configuration_name(args) == runtime.SEMANTIC_STRATIFIED_1200 and row.get("runtime_fallback_applied") == "YES":
            require("raw_model_response" in row and "original_finish_reason" in row, "Missing runtime fallback provenance")
            reason = row.get("runtime_fallback_reason")
            if reason == "TRUNCATED_COMPLETION":
                constrained, metadata = runtime.truncated_response(row["raw_model_response"], row["original_finish_reason"], prompt)
            elif reason == "UNPARSEABLE_MODEL_OUTPUT":
                constrained, metadata = runtime.completed_model_response(row["raw_model_response"], prompt)
            elif reason == "TRANSPORT_RETRIES_EXHAUSTED":
                constrained, metadata = runtime.transport_abstention(prompt, row.get("transport_attempt_errors", []))
                require(row["retry_count"] == len(row["transport_attempt_errors"])-1 and row["completion_tokens"] == 0 and row["total_tokens"] == tokens,
                        "Invalid unavailable transport usage")
            else:
                raise ValueError("Unknown runtime fallback reason")
            expected["finish_reason"] = row["original_finish_reason"]
        else:
            constrained, metadata = runtime.constrain_response(row.get("raw_model_response"), prompt, runtime.configuration_name(args))
        require(value == constrained, "Saved constrained output differs from raw response and V4 gate")
        expected.update(metadata)
    require(all(row.get(key) == value for key, value in expected.items()), "Saved response runtime metadata differs")
    require(type(row.get("response_valid")) is bool, "Invalid response_valid")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "retry_count"):
        require(type(row.get(key)) is int and row[key] >= 0, "Invalid saved count")
    require(row["prompt_tokens"] == tokens and row["completion_tokens"] <= args.max_completion_tokens and row["total_tokens"] == tokens+row["completion_tokens"] <= args.max_model_len and row["retry_count"] <= args.retry_limit, "Saved token/retry counts differ")
    require(common.finite(row["latency_seconds"]) >= 0, "Invalid latency")
    # No server hidden reasoning or raw transport payload is persisted.
    require(set(row) == set(runtime.FIELDS) | set(expected) | {"prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds", "retry_count"}, "Unknown saved response fields")


def remember(connection, row):
    connection.execute("INSERT INTO done VALUES (?, ?, ?, ?, ?, ?)", (row["query_id"], common.json_text(row), row["prompt_tokens"], row["completion_tokens"], row["latency_seconds"], row["retry_count"]))


def transport_streak(connection):
    count = 0
    for (payload,) in connection.execute("SELECT done.payload FROM done JOIN prompts USING(id) ORDER BY prompts.ordinal DESC LIMIT 5"):
        if common.parse_json(payload).get("runtime_fallback_reason") != "TRANSPORT_RETRIES_EXHAUSTED":
            break
        count += 1
    return count


def scan_saved(stream, connection, args, on_error=lambda row: None):
    connection.execute("CREATE TABLE done (id TEXT PRIMARY KEY, payload TEXT, prompt_tokens INTEGER, completion_tokens INTEGER, latency REAL, retries INTEGER)")
    count = 0
    if stream is None:
        return count
    stream.seek(0)
    while True:
        offset = stream.tell()
        line = stream.readline()
        if not line:
            break
        try:
            row = common.parse_json(line.decode("utf-8"))
        except (ValueError, UnicodeError) as exc:
            trailing = not line.endswith(b"\n") and stream.read(1) == b""
            require(trailing and args.execute and args.repair_trailing_partial and not args.export, "Malformed JSONL; unterminated last fragment requires explicit --repair-trailing-partial")
            on_error(dict(query_id="", attempt=0, retry_count=0, error_type="TRAILING_PARTIAL_DISCARDED", error_message=f"Explicit repair discarded {len(line)} bytes at offset {offset}", will_retry="NO"))
            stream.seek(offset)
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
            break
        require(isinstance(row, dict) and isinstance(row.get("query_id"), str), "Malformed saved response")
        found = connection.execute("SELECT ordinal, payload, tokens FROM prompts WHERE id=?", (row["query_id"],)).fetchone()
        require(found is not None, "Unknown saved query_id")
        require(found[0] == count, "Duplicate/out-of-order/gapped saved query_id")
        validate_saved(row, common.parse_json(found[1]), found[2], args)
        remember(connection, row)
        count += 1
    connection.commit()
    return count


@contextmanager
def response_file(path, writing):
    if not writing and not path.exists():
        yield None
        return
    if writing:
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b" if writing else "rb") as stream:
        if os.name == "nt":
            import msvcrt
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield stream
        finally:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def append_response(stream, row):
    stream.seek(0, os.SEEK_END)
    if stream.tell():
        stream.seek(-1, os.SEEK_END)
        if stream.read(1) != b"\n":
            stream.write(b"\n")
    stream.write((common.json_text(row)+"\n").encode("utf-8"))
    stream.flush()
    os.fsync(stream.fileno())


def error_writer(root, version="v3", configuration="legacy"):
    outputs = output_paths(version, configuration)
    path = common.corpus.local_path(root, outputs[3], must_exist=False)
    if path.exists():
        # Stream validation; do not load a potentially large retry ledger.
        for _ in common.rows(root, outputs[3], ERROR_HEADER):
            pass
    else:
        common.corpus.write_table(path, ERROR_HEADER, [], mode="x")
    def write(row):
        with path.open("a", encoding="utf-8", newline="") as stream:
            csv.DictWriter(stream, fieldnames=ERROR_HEADER).writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
    return write


def replace_table(root, relative, header, records):
    target = common.corpus.local_path(root, relative, must_exist=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rag_execution_metadata_", dir=root) as directory:
        staged = Path(directory)/"table.csv"
        common.corpus.write_table(staged, header, records, mode="x")
        os.replace(staged, target)


def summary_rows(root, connection, total, status, version="v3", configuration="legacy"):
    outputs = output_paths(version, configuration)
    count = connection.execute("SELECT COUNT(*) FROM done").fetchone()[0]
    counts, breakdown, fallbacks = Counter(), Counter(), Counter()
    semantic_fallbacks, positive_fallbacks, truncated_fallbacks, native = Counter(), Counter(), Counter(), Counter()
    format_fallbacks, transport_fallbacks = Counter(), Counter()
    observed_completions = []
    for (payload,) in connection.execute("SELECT payload FROM done ORDER BY id"):
        row = common.parse_json(payload)
        counts[row["evidence_status"]] += 1
        breakdown[(row["source_dataset"], row["classifier_prediction"], row["evidence_status"])] += 1
        fallbacks[(row["source_dataset"], row["classifier_prediction"])] += row.get("validator_fallback_applied") == "YES"
        semantic_fallbacks[(row["source_dataset"], row["classifier_prediction"])] += row.get("validator_fallback_reason") == "SEMANTIC_VALIDATION"
        positive_fallbacks[(row["source_dataset"], row["classifier_prediction"])] += row.get("validator_fallback_reason") == "POSITIVE_EVIDENCE"
        truncated_fallbacks[(row["source_dataset"], row["classifier_prediction"])] += row.get("runtime_fallback_reason") == "TRUNCATED_COMPLETION"
        native[(row["source_dataset"], row["classifier_prediction"])] += row.get("validator_fallback_applied") != "YES" and row.get("runtime_fallback_applied") != "YES"
        format_fallbacks[(row["source_dataset"], row["classifier_prediction"])] += row.get("runtime_fallback_reason") == "UNPARSEABLE_MODEL_OUTPUT"
        transport_fallbacks[(row["source_dataset"], row["classifier_prediction"])] += row.get("runtime_fallback_reason") == "TRANSPORT_RETRIES_EXHAUSTED"
        if configuration == runtime.SEMANTIC_STRATIFIED_1200 and row.get("token_usage_status") != "UNAVAILABLE_TRANSPORT":
            observed_completions.append(row["completion_tokens"])
    retries = 0
    invalid = set()
    if common.corpus.local_path(root, outputs[3], must_exist=False).exists():
        for row in common.rows(root, outputs[3], ERROR_HEADER):
            retries += row["will_retry"] == "YES"
            if row["query_id"] and connection.execute("SELECT 1 FROM done WHERE id=?", (row["query_id"],)).fetchone() is None:
                invalid.add(row["query_id"])
    def record(metric, value, dataset="", prediction="", evidence=""):
        return dict(metric=metric, source_dataset=dataset, classifier_prediction=prediction, evidence_status=evidence, value=value)
    values = dict(expected_queries=total, completed_queries=count, remaining_queries=total-count, valid_queries=count, invalid_queries=len(invalid), retry_count_total=retries, execution_status=status)
    output = [record(k, v) for k, v in values.items()]
    output.extend(record("status_count", counts[state], evidence=state) for state in common.STATUSES)
    output.extend(record("dataset_prediction_status_count", n, *key) for key, n in sorted(breakdown.items()))
    if configuration in runtime.SUBSET_CONFIGURATIONS:
        output.extend(record(key, value) for key, value in dict(
            evaluation_scope=subset.SCOPE, population_queries=sum(common.QUERY_COUNTS.values()),
            evaluated_queries=count, evaluation_target_queries=subset.TOTAL,
            sampling_seed=subset.SEED, sampling_strategy=subset.STRATEGY,
            reporting_policy="Balanced development subset; counts describe this sample, not natural population rates").items())
        # Include zero cells as well as observed outcomes for every stratum.
        output = [row for row in output if row["metric"] != "dataset_prediction_status_count"]
        output.extend(record("dataset_prediction_status_count", breakdown[(dataset, prediction, state)], dataset, prediction, state)
                      for dataset, prediction in subset.STRATA for state in common.STATUSES)
    if configuration in runtime.FAIL_CLOSED_CONFIGURATIONS:
        output.append(record("validator_fallback_count", sum(fallbacks.values())))
        output.extend(record("dataset_prediction_validator_fallback_count", fallbacks[(dataset, prediction)], dataset, prediction)
                      for dataset, classes in common.CLASSES.items() for prediction in classes)
    if configuration == runtime.SEMANTIC_STRATIFIED_1200:
        for metric, counter in (("model_native_count", native), ("semantic_validator_fallback_count", semantic_fallbacks), ("positive_evidence_fallback_count", positive_fallbacks), ("total_validator_fallback_count", fallbacks), ("truncated_completion_fallback_count", truncated_fallbacks), ("unparseable_model_output_fallback_count", format_fallbacks), ("transport_retries_exhausted_count", transport_fallbacks), ("total_validator_or_runtime_fallback_count", fallbacks + truncated_fallbacks + format_fallbacks + transport_fallbacks)):
            output.append(record(metric, sum(counter.values())))
            output.extend(record("dataset_prediction_" + metric, counter[pair], *pair) for pair in subset.STRATA)
    for column, label in (("prompt_tokens", "prompt_tokens"), ("completion_tokens", "completion_tokens"), ("latency", "latency_seconds")):
        if column == "completion_tokens" and configuration == runtime.SEMANTIC_STRATIFIED_1200:
            # Transport exhaustion has no measured API usage. Its stored zero
            # sentinel must not bias completion-token statistics.
            measured = sorted(observed_completions)
            n = len(measured)
            stats = dict(min=measured[0], mean=sum(measured)/n, median=(measured[(n-1)//2]+measured[n//2])/2,
                         p95=measured[math.ceil(.95*n)-1], p99=measured[math.ceil(.99*n)-1], max=measured[-1]) if n else dict.fromkeys(("min", "mean", "median", "p95", "p99", "max"), "")
            output.append(record("completion_token_usage_unavailable_count", count-n))
        elif count:
            low, mean, high = connection.execute(f"SELECT MIN({column}), AVG({column}), MAX({column}) FROM done").fetchone()
            def at(position):
                return connection.execute(f"SELECT {column} FROM done ORDER BY {column}, id LIMIT 1 OFFSET ?", (position,)).fetchone()[0]
            stats = dict(min=low, mean=mean, median=(at((count-1)//2)+at(count//2))/2, p95=at(math.ceil(.95*count)-1), p99=at(math.ceil(.99*count)-1), max=high)
        else:
            stats = dict.fromkeys(("min", "mean", "median", "p95", "p99", "max"), "")
        output.extend(record(label+"_"+key, value) for key, value in stats.items())
    return output


def save_metadata(root, connection, config, total, status):
    version = config["prompt_schema"].rsplit("_", 1)[1]
    configuration = config.get("runtime_configuration", "legacy")
    outputs = output_paths(version, configuration)
    replace_table(root, outputs[2], common.CONFIG_HEADER, common.config_rows(dict(config, execution_status=status)))
    replace_table(root, outputs[1], SUMMARY_HEADER, summary_rows(root, connection, total, status, version, configuration))
    coverage = "One validated response for each of 1200 manifest queries; not full-population coverage" if configuration in runtime.SUBSET_CONFIGURATIONS else "One validated response for every canonical query required"
    row = common.corpus.check("constrained_llm_" + version, "complete_validated_coverage", "PASS" if status == "PASS" else "INCOMPLETE", coverage)
    replace_table(root, outputs[4], common.corpus.foundation.CHECKLIST_HEADER, [row])


def export_records(root, connection, version="v3", configuration="legacy"):
    destination = output_paths(version, configuration)[0].replace("llm_responses.jsonl", "llm_responses_for_import.jsonl")
    existing = common.corpus.local_path(root, destination, must_exist=False)
    if existing.exists():
        with existing.open(encoding="utf-8") as stream:
            for (payload,) in connection.execute("SELECT done.payload FROM done JOIN prompts USING(id) ORDER BY prompts.ordinal"):
                row = common.parse_json(payload)
                require(common.parse_json(stream.readline()) == {key: row[key] for key in runtime.FIELDS}, "Existing export differs from validated records")
            require(not stream.read(1), "Unexpected export rows")
        return
    common.fresh_outputs(root, [destination])
    with tempfile.TemporaryDirectory(prefix=".rag_response_export_", dir=root) as directory:
        path = Path(directory)/"responses.jsonl"
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            for (payload,) in connection.execute("SELECT done.payload FROM done JOIN prompts USING(id) ORDER BY prompts.ordinal"):
                row = common.parse_json(payload)
                stream.write(common.json_text({key: row[key] for key in runtime.FIELDS})+"\n")
        prompts.publish_staged(root, [(destination, path)])


def continuation_jsonl(root, relative):
    path = common.corpus.local_path(root, relative, must_exist=False)
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                require(line.endswith("\n") and line.strip(), "Incomplete/corrupt durable JSONL: " + relative)
                yield common.parse_json(line)


def continuation_queue(connection, excluded):
    rng, buckets = random.Random(42), []
    for dataset, prediction in CONTINUATION_STRATA:
        ids = [row[0] for row in connection.execute("SELECT id FROM population WHERE dataset=? AND prediction=? ORDER BY id", (dataset, prediction))]
        rng.shuffle(ids)
        buckets.append(iter(identity for identity in ids if identity not in excluded))
    queue = []
    while buckets:
        active = []
        for bucket in buckets:
            identity = next(bucket, None)
            if identity is not None:
                queue.append(identity)
                active.append(bucket)
        buckets = active
    require(len(queue) == len(set(queue)), "Duplicate continuation queue IDs")
    return queue


def continuation_prepare(root, connection, args):
    """Index CSV byte positions, not duplicate multi-gigabyte prompt strings."""
    common.completed(root, prompts.V4_OUTPUTS[1], prompts.V4_OUTPUTS[2])
    require(list(common.rows(root, prompts.V4_OUTPUTS[3], prompts.POLICY_HEADER)) == prompts.policy_rows("v4"), "V4 policy differs")
    paths = [common.QUERY_PATH, *prompts.V4_OUTPUTS]
    before = common.snapshot(root, paths)
    connection.execute("CREATE TABLE population (id TEXT PRIMARY KEY, dataset TEXT, prediction TEXT, query TEXT, position TEXT, tokens INTEGER)")
    counts = Counter()
    for row in common.rows(root, common.QUERY_PATH, prompts.QUERY_FIELDS):
        common.validate_query(row)  # Rejects TEST before any request.
        require(row["source_split"] == "validation" and row["experiment_id"] == "phase2_dev_retrieval_v3", "Validation-only continuation")
        query = {key: row[key] for key in prompts.QUERY_FIELDS}
        connection.execute("INSERT INTO population VALUES (?, ?, ?, ?, NULL, NULL)",
                           (row["query_id"], row["source_dataset"], row["predicted_class_label"], common.json_text(query)))
        counts[row["source_dataset"]] += 1
    require(dict(counts) == common.QUERY_COUNTS, "Incomplete validation population")
    canonical = common.corpus.local_path(root, prompts.V4_OUTPUTS[0])
    with canonical.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(iter(stream.readline, ""))
        require(tuple(reader.fieldnames) == prompts.HEADER, "Canonical prompt schema differs")
        while True:
            position = stream.tell()
            row = next(reader, None)
            if row is None:
                break
            found = connection.execute("SELECT query, position FROM population WHERE id=?", (row["query_id"],)).fetchone()
            require(found is not None and found[1] is None, "Unknown/duplicate prompt ID")
            audit.validate_prompt(row, common.parse_json(found[0]), "v4")
            connection.execute("UPDATE population SET position=? WHERE id=?", (str(position), row["query_id"]))
    require(connection.execute("SELECT id FROM population WHERE position IS NULL LIMIT 1").fetchone() is None, "Missing validation prompts")
    # Revalidate the existing exact-token audit against this complete population.
    # No tokenizer/model is loaded by status-only, and no frozen audit is changed.
    audit_paths = audit.output_paths(args.max_model_len, args.max_completion_tokens, "v4")
    summaries = list(common.rows(root, audit_paths[1], audit.SUMMARY_HEADER))
    require(len(summaries) == 1, "Missing token audit summary")
    summary = summaries[0]
    require(summary["gate_status"] == "PASS" and int(summary["over_limit_with_reservation"]) == 0 and
            int(summary["rows"]) == sum(counts.values()) and common.parse_json(summary["dataset_counts_json"]) == dict(counts), "Token audit not PASS for validation population")
    require(summary["input_metadata_json"] == common.json_text(before) and summary["tokenizer_metadata_json"] == common.json_text(audit.tokenizer_files(root)), "Stale token audit")
    require(summary["configuration_key"] == audit.configuration_key(args.max_model_len, args.max_completion_tokens) and
            int(summary["max_model_len"]) == args.max_model_len and int(summary["reserved_completion_tokens"]) == args.max_completion_tokens, "Token envelope mismatch")
    for row in common.rows(root, audit_paths[0], audit.HEADER):
        found = connection.execute("SELECT dataset, prediction, tokens FROM population WHERE id=?", (row["query_id"],)).fetchone()
        tokens = int(row["qwen_prompt_tokens"])
        require(found is not None and found[2] is None and found[:2] == (row["source_dataset"], row["classifier_prediction"]), "Unknown/duplicate audit ID")
        require(tokens > 0 and row["within_limit"] == "YES" and tokens+args.max_completion_tokens <= args.max_model_len and
                int(row["reserved_total_tokens"]) == tokens+args.max_completion_tokens and int(row["max_model_len"]) == args.max_model_len and
                int(row["reserved_completion_tokens"]) == args.max_completion_tokens and int(row["safe_prompt_limit"]) == args.max_model_len-args.max_completion_tokens, "Unsafe token audit row")
        connection.execute("UPDATE population SET tokens=? WHERE id=?", (tokens, row["query_id"]))
    require(connection.execute("SELECT id FROM population WHERE tokens IS NULL LIMIT 1").fetchone() is None, "Incomplete token audit")
    prior, prior_test, prior_paths = set(), set(), []
    names = {"llm_responses.jsonl", "llm_responses_for_import.jsonl", "validated_llm_responses.csv", "six_class_grounding_diagnostic.jsonl"}
    for path in sorted((root/"results/rag/llm").rglob("*")):
        if path.name not in names or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        prior_paths.append(relative)
        rows = common.rows(root, relative) if path.suffix == ".csv" else continuation_jsonl(root, relative)
        seen = set()
        for row in rows:
            identity = row["query_id"]
            require(identity not in seen, "Duplicate existing IDs in " + relative)
            seen.add(identity)
            if connection.execute("SELECT 1 FROM population WHERE id=?", (identity,)).fetchone():
                require(str(row.get("source_split", "validation")).lower() == "validation", "Prior response/query split differs")
                with canonical.open(encoding="utf-8", newline="") as stream:
                    prompt, _ = continuation_prompt(stream, connection, identity)
                logical = {key: row[key] for key in runtime.FIELDS}
                if path.suffix == ".csv":
                    logical["cited_chunk_ids"] = common.parse_json(logical["cited_chunk_ids"])
                    logical["supported_class"] = logical["supported_class"] or None
                runtime.validate_logical(logical, prompt)
                prior.add(identity)
            else:
                require(str(row.get("source_split", "")).lower() == "test", "Prior response has unknown split/identity")
                prior_test.add(identity)
    require(before == common.snapshot(root, paths), "Canonical inputs changed during preflight")
    connection.commit()
    return prior, prior_test, paths+list(audit_paths), prior_paths


def continuation_prompt(stream, connection, identity):
    position, tokens = connection.execute("SELECT position, tokens FROM population WHERE id=?", (identity,)).fetchone()
    stream.seek(int(position))
    values = next(csv.reader(iter(stream.readline, "")))
    require(len(values) in {len(prompts.HEADER), len(prompts.HEADER)+1}, "Unexpected prompt fields")
    prompt = dict(zip(prompts.HEADER, values))  # TEST adds the audited token count.
    require(prompt["query_id"] == identity, "Canonical prompt position changed")
    return prompt, tokens


def continuation_retry_record(first, correction, result, tokens=0, skip_reason=""):
    attempted = correction is not None
    retry_status = ""
    if attempted and correction["api_response"] is not None:
        content = correction["api_response"]["choices"][0]["message"]["content"]
        try:
            parsed = common.parse_json(content) if isinstance(content, str) else None
            if isinstance(parsed, dict) and isinstance(parsed.get("evidence_status"), str):
                retry_status = parsed["evidence_status"]
        except ValueError:
            pass  # Raw malformed output is already preserved in the ledger.
    accepted = attempted and result.get("validator_fallback_applied") == "NO" and result.get("runtime_fallback_applied") != "YES"
    return dict(query_id=first["query_id"], retry_attempted="YES" if attempted else "NO",
                initial_evidence_status=common.parse_json(first["raw_model_response"])["evidence_status"],
                initial_validator_error=first["grounding_gate_error_message"], retry_evidence_status=retry_status,
                retry_validator_pass="YES" if accepted else "NO", final_evidence_status=result["evidence_status"],
                correction_prompt_tokens=correction["prompt_tokens"] if attempted else tokens, skip_reason=skip_reason)


def run_validation_continuation(args):
    require(not args.export and not args.repair_trailing_partial, "Unsupported continuation option")
    require(not (args.status_only and args.execute), "Status-only cannot execute")
    require(common.finite(args.time_budget_minutes) > common.finite(args.reserve_minutes) >= 0, "Time budget must exceed shutdown reserve")
    root = args.project_root.resolve(strict=True)
    final_test = getattr(args, "final_frozen_test", False)
    namespace, split = CONTINUATION, "validation"
    canonical_prompt = prompts.V4_OUTPUTS[0]
    snapshot = common.snapshot
    if final_test:
        from src.rag.final_test import freeze
        namespace, split, canonical_prompt = freeze.OUTPUT, "test", freeze.TEST_PROMPTS
        snapshot = freeze.c.snapshot
    with tempfile.TemporaryDirectory(prefix=".rag_validation_", dir=runtime_temp_root(root)) as directory:
        connection = sqlite3.connect(str(Path(directory)/"state.sqlite"))
        try:
            prior, prior_test, inputs, prior_paths = (freeze.prepare(root, connection, args) if final_test else continuation_prepare(root, connection, args))
            input_identity = snapshot(root, inputs)
            prior_identity = snapshot(root, prior_paths)
            config = dict(runtime_configuration=runtime.configuration_name(args), source_split=split, sampling_seed="42", concurrency="1",
                          max_model_len=str(args.max_model_len), max_completion_tokens=str(args.max_completion_tokens), model=args.model,
                          endpoint=args.endpoint, timeout=str(float(args.timeout)), retry_limit=str(args.retry_limit), temperature="0",
                          input_metadata_json=common.json_text(input_identity), resume_policy="exact_query_id; raw_first_fsync; one_positive_correction")
            config_path = namespace+"run_configuration.csv"
            if (root/config_path).exists():
                config_rows = freeze.c.rows(root, config_path) if final_test else common.rows(root, config_path, common.CONFIG_HEADER)
                saved = {row["field"]: row["value"] for row in config_rows}
                require(saved == config, "Continuation configuration/input identity differs")
            raw, validated = {}, {}
            raw_path, valid_path = namespace+"raw_responses.jsonl", namespace+"validated_responses.jsonl"
            for event in continuation_jsonl(root, raw_path):
                key = (event["query_id"], event["phase"])
                require(key not in raw and connection.execute("SELECT 1 FROM population WHERE id=?", (key[0],)).fetchone(), "Duplicate/unknown continuation raw ID")
                require(event["phase"] == "initial" or (key[0], "initial") in raw, "Correction precedes initial response")
                raw[key] = event
            for row in continuation_jsonl(root, valid_path):
                require(row["query_id"] not in validated and (row["query_id"], "initial") in raw, "Duplicate/orphan validated continuation response")
                validated[row["query_id"]] = row
            require(not raw or (root/config_path).exists(), "Durable continuation lacks configuration")
            if args.execute and (raw or validated):
                require(args.resume, "Existing continuation requires --resume")
            canonical = common.corpus.local_path(root, canonical_prompt)
            with canonical.open(encoding="utf-8", newline="") as prompt_stream:
                # Reconstruct from raw without another request, including crashes
                # between raw fsync and validation or correction fsync and validation.
                recovered, retry_recovery, saved_retries = {}, {}, {}
                for row in continuation_jsonl(root, namespace+"positive_retry_audit.jsonl"):
                    require(row["query_id"] not in saved_retries and (row["query_id"], "initial") in raw, "Duplicate/orphan correction audit")
                    saved_retries[row["query_id"]] = row
                correction_tokenizer = None
                for identity in sorted({key[0] for key in raw}):
                    prompt, tokens = continuation_prompt(prompt_stream, connection, identity)
                    require(raw[(identity, "initial")]["prompt_tokens"] == tokens, "Saved initial token audit differs")
                    first = runtime.continuation_result(prompt, raw[(identity, "initial")], args)
                    event = raw.get((identity, "correction"))
                    if event:
                        require(first.get("validator_fallback_reason") == "POSITIVE_EVIDENCE", "Unjustified correction")
                        if correction_tokenizer is None:
                            correction_tokenizer = audit.load_tokenizer(root)
                        correction = runtime.correction_prompt(prompt, first["grounding_gate_error_message"])
                        require(event["prompt_tokens"] == audit.prompt_tokens(correction_tokenizer, correction["prompt_text"]), "Saved correction token count differs")
                    result = runtime.continuation_result(prompt, event, args) if event else first
                    if first.get("validator_fallback_reason") == "POSITIVE_EVIDENCE":
                        retry_record = continuation_retry_record(first, event, result, skip_reason="" if event else "RECOVERED_RAW_NO_REISSUE")
                        if identity in saved_retries:
                            require(all(saved_retries[identity].get(key) == value for key, value in retry_record.items()
                                        if key not in {"skip_reason", "correction_prompt_tokens"}), "Correction audit differs from durable raw")
                        else:
                            retry_recovery[identity] = retry_record
                    else:
                        require(identity not in saved_retries, "Unexpected positive correction audit")
                    if identity in validated:
                        require(validated[identity] == result, "Saved continuation validation differs from durable raw")
                    else:
                        recovered[identity] = result
                current_ids = {key[0] for key in raw} | set(validated)
                excluded = prior | prior_test | current_ids
                queue = continuation_queue(connection, excluded)
                population = connection.execute("SELECT COUNT(*) FROM population").fetchone()[0]
                stats = dict(completed_global=len(excluded), completed_validation=len(prior|current_ids), completed_test=len(prior_test),
                             validation_population=population, remaining_validation=len(queue), duplicate_existing_ids=0, duplicate_queue_ids=0,
                             test_rows_in_queue=0, prompt_token_audit="PASS", raw_pending_validation=len(recovered),
                             continuation_output_state="RESUMABLE" if current_ids else "READY" if (root/config_path).exists() else "ABSENT")
                if final_test:
                    stats = dict(test_subset_rows=population, test_subset_unique_ids=population, validation_rows_in_test_queue=0,
                        test_rows=population, ground_truth_in_prompt=0, duplicate_test_ids=0, overlap_with_validation_ids=0,
                        prompt_token_audit="PASS", frozen_validator_hash="MATCH", frozen_runtime_configuration="MATCH",
                        existing_test_outputs_not_overwritten="PASS", completed_current_test=len(current_ids),
                        remaining_test=len(queue), raw_pending_validation=len(recovered))
                for key, value in stats.items():
                    print(f"{key}={value}")
                if not args.execute:
                    print("FINAL_FROZEN_TEST_PREFLIGHT=PASS" if final_test else "VALIDATION_CONTINUATION_PREFLIGHT=PASS")
                    return 0
                # Lock the one canonical raw ledger throughout execution/recovery.
                with response_file(common.corpus.local_path(root, raw_path, must_exist=False), True) as raw_stream:
                    raw_stream.seek(0)
                    require(sum(1 for _ in raw_stream) == len(raw), "Concurrent continuation writer detected")
                    replace_table(root, config_path, common.CONFIG_HEADER, common.config_rows(config))
                    queue_header = ("query_id", "source_dataset", "classifier_prediction", "source_split", "selection_ordinal", "sampling_seed")
                    def queue_rows():
                        for ordinal, identity in enumerate(queue, 1):
                            dataset, prediction = connection.execute("SELECT dataset, prediction FROM population WHERE id=?", (identity,)).fetchone()
                            yield dict(query_id=identity, source_dataset=dataset, classifier_prediction=prediction, source_split=split, selection_ordinal=ordinal, sampling_seed=42)
                    replace_table(root, namespace+"continuation_queue.csv", queue_header, queue_rows())
                    replace_table(root, namespace+"continuation_manifest.csv", common.CONFIG_HEADER, common.config_rows(dict(
                        authoritative_inputs=common.json_text(prior_identity), prior_completed_validation=len(prior), population=population,
                        time_budget_minutes=args.time_budget_minutes, reserve_minutes=args.reserve_minutes, sampling_seed=42)))
                    valid_file = common.corpus.local_path(root, valid_path, must_exist=False)
                    retry_file = common.corpus.local_path(root, namespace+"positive_retry_audit.jsonl", must_exist=False)
                    errors_file = common.corpus.local_path(root, namespace+"execution_errors.csv", must_exist=False)
                    if not errors_file.exists():
                        common.corpus.write_table(errors_file, ERROR_HEADER, [], mode="x")
                    started = time.monotonic()
                    deadline = started+(args.time_budget_minutes-args.reserve_minutes)*60
                    metrics = Counter(new_attempted=0, new_raw_durable=0, new_validated=0, positive_retry_attempted=0, positive_retry_accepted=0, positive_retry_failed=0, execution_errors=0)
                    tokenizer, streak, state = correction_tokenizer, 0, "INCOMPLETE"
                    if not args.reset_transport_circuit:
                        for identity in reversed([key[0] for key in raw if key[1] == "initial"]):
                            if (validated.get(identity) or recovered[identity]).get("runtime_fallback_reason") != "TRANSPORT_RETRIES_EXHAUSTED":
                                break
                            streak += 1
                    def on_error(error):
                        metrics["execution_errors"] += 1
                        with errors_file.open("a", encoding="utf-8", newline="") as stream:
                            csv.DictWriter(stream, fieldnames=ERROR_HEADER).writerow(error)
                            stream.flush()
                            os.fsync(stream.fileno())
                    def persist(event):
                        append_response(raw_stream, event)
                        raw[(event["query_id"], event["phase"])] = event
                        if event["phase"] == "initial":
                            metrics["new_raw_durable"] += 1
                    with valid_file.open("a", encoding="utf-8", newline="\n") as valid_stream, retry_file.open("a", encoding="utf-8", newline="\n") as retry_stream:
                        def finish(result):
                            runtime.flush_record(valid_stream, result)
                            validated[result["query_id"]] = result
                            metrics["new_validated"] += 1
                            metrics[result["evidence_status"]] += 1
                            metrics[result["reasoning_decision"]] += 1
                        try:
                            for row in retry_recovery.values():
                                runtime.flush_record(retry_stream, row)
                            for result in recovered.values():
                                finish(result)
                            require(streak < 5, "Continuation transport circuit open; repair server then use --reset-transport-circuit")
                            for identity in queue:
                                if time.monotonic() >= deadline:
                                    state = "TIME_BUDGET_REACHED"
                                    break
                                prompt, tokens = continuation_prompt(prompt_stream, connection, identity)
                                metrics["new_attempted"] += 1
                                event = runtime.continuation_request(prompt, tokens, args, deadline, "initial", persist, on_error)
                                result = runtime.continuation_result(prompt, event, args)
                                if result.get("validator_fallback_reason") == "POSITIVE_EVIDENCE":
                                    first = result
                                    error = result["grounding_gate_error_message"]
                                    corrected = runtime.correction_prompt(prompt, error)
                                    if tokenizer is None:
                                        tokenizer = audit.load_tokenizer(root)
                                    correction_tokens = audit.prompt_tokens(tokenizer, corrected["prompt_text"])
                                    attempted = correction_tokens+args.max_completion_tokens <= args.max_model_len and time.monotonic() < deadline
                                    correction = None
                                    if attempted:
                                        metrics["positive_retry_attempted"] += 1
                                        correction = runtime.continuation_request(corrected, correction_tokens, args, deadline, "correction", persist, on_error)
                                        result = runtime.continuation_result(prompt, correction, args)
                                    accepted = attempted and result.get("validator_fallback_applied") == "NO" and result.get("runtime_fallback_applied") != "YES"
                                    metrics["positive_retry_accepted" if accepted else "positive_retry_failed"] += int(attempted)
                                    runtime.flush_record(retry_stream, continuation_retry_record(first, correction, result, correction_tokens,
                                        "" if attempted else "TOKEN_BUDGET_OR_DEADLINE"))
                                finish(result)
                                streak = streak+1 if result.get("runtime_fallback_reason") == "TRANSPORT_RETRIES_EXHAUSTED" else 0
                                require(streak < 5, "Continuation transport circuit open after five exhausted queries")
                            else:
                                state = "COMPLETE"
                        except runtime.ContinuationDeadline:
                            state = "TIME_BUDGET_REACHED"
                        finally:
                            elapsed = time.monotonic()-started
                            require(input_identity == snapshot(root, inputs) and prior_identity == snapshot(root, prior_paths), "Frozen inputs/results changed")
                            summary = dict(metrics, execution_status=state, elapsed_seconds=elapsed, queries_per_hour=metrics["new_validated"]*3600/elapsed if elapsed else 0,
                                prior_completed_validation=len(prior), new_validation_completed=len(validated), cumulative_validation_completed=len(prior|set(validated)),
                                remaining_validation=population-len(prior|set(validated)), evaluation_scope="validation_continuation", source_split="validation",
                                invocation_metrics="new_attempted/new_raw_durable/new_validated/status/decision/retry counts describe this invocation",
                                cumulative_metrics="prior_completed_validation is pre-continuation; new_validation_completed covers all continuation invocations")
                            if final_test:
                                for key in ("prior_completed_validation", "new_validation_completed", "cumulative_validation_completed", "remaining_validation"):
                                    summary.pop(key)
                                summary.update(source_split="test", evaluation_scope="stratified_held_out_test_subset",
                                    completed_test=len(validated), remaining_test=population-len(validated), population_test_queries=129978,
                                    selected_test_queries=1200, cumulative_metrics="Current frozen TEST namespace only; historical TEST excluded")
                            for key in (*common.STATUSES, "SUPPORT_CLASS", "ABSTAIN"):
                                summary.setdefault(key, 0)
                            replace_table(root, namespace+"execution_summary.csv", common.CONFIG_HEADER, common.config_rows(summary))
                            replace_table(root, namespace+"validation_checklist.csv", common.corpus.foundation.CHECKLIST_HEADER,
                                          common.checks("validation_continuation", stats) + [common.corpus.check("validation_continuation", "execution", "PASS" if state == "COMPLETE" else "INCOMPLETE", state)])
                    print(("FINAL_FROZEN_TEST=" if final_test else "VALIDATION_CONTINUATION=")+state)
                    return 0
        finally:
            connection.close()


def main(argv=None):
    parser = runtime.add_arguments(common.parser(__doc__))
    parser.add_argument("--repair-trailing-partial", action="store_true")
    parser.add_argument("--export", action="store_true", help="Validate complete raw records and export six fields; never send requests")
    parser.add_argument("--preflight", action="store_true", help="No-network validation of inputs and saved responses")
    parser.add_argument("--reset-transport-circuit", action="store_true", help="Explicitly retry remaining queries after repairing the server; completed records are retained")
    parser.add_argument("--validation-continuation", action="store_true")
    parser.add_argument("--final-frozen-test", action="store_true")
    parser.add_argument("--freeze-validation", action="store_true", help="Create the exclusive freeze record; never infer")
    parser.add_argument("--time-budget-minutes", type=float, default=390)
    parser.add_argument("--reserve-minutes", type=float, default=15)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        runtime.validate_options(args)
        if args.freeze_validation:
            require(args.final_frozen_test and not args.execute, "Freeze creation requires --final-frozen-test without --execute")
            from src.rag.final_test import freeze
            freeze.create(args.project_root.resolve(strict=True))
            return 0
        if args.validation_continuation or args.final_frozen_test:
            return run_validation_continuation(args)
        require(not args.status_only and not args.resume, "Continuation flags require --validation-continuation")
        require(not (args.preflight and (args.execute or args.export)), "Preflight cannot execute or export")
        root = args.project_root.resolve(strict=True)
        runtime_configuration = runtime.configuration_name(args)
        outputs = output_paths(args.version, runtime_configuration)
        with tempfile.TemporaryDirectory(prefix=".rag_v3_execution_", dir=runtime_temp_root(root)) as directory:
            connection = sqlite3.connect(str(Path(directory)/"state.sqlite"))
            try:
                optimized = runtime_configuration == runtime.SEMANTIC_STRATIFIED_1200
                if optimized:
                    total, audit_paths, diagnostic_paths = llm_subset_preflight.prepare(root, connection, args, validate_saved)
                else:
                    total = runtime.prepare_population(root, connection) if args.version == "v3" else runtime.prepare_population(root, connection, args.version)
                    audit_paths = audit_gate(root, connection, args)
                    diagnostic_paths = diagnostic_gate(root, connection, args) if args.version == "v4" else ()
                paths = [common.QUERY_PATH, *prompts.version_settings(args.version)[0], *audit_paths, *diagnostic_paths]
                if runtime_configuration in runtime.SUBSET_CONFIGURATIONS:
                    paths.append(subset.MANIFEST)
                before = common.snapshot(root, paths)
                if runtime_configuration in runtime.SUBSET_CONFIGURATIONS and not optimized:
                    total = subset.restrict_population(root, connection)
                    require(before == common.snapshot(root, paths), "Subset inputs changed")
                config = configuration(args, before, audit_paths, total)
                raw = common.corpus.local_path(root, outputs[0], must_exist=False)
                with response_file(raw, args.execute and not args.export) as stream:
                    check_configuration(root, config)
                    on_error = error_writer(root, args.version, runtime_configuration) if args.execute and not args.export else lambda row: None
                    if args.execute and not args.export:
                        replace_table(root, outputs[2], common.CONFIG_HEADER, common.config_rows(dict(config, execution_status="INCOMPLETE")))
                        replace_table(root, outputs[4], common.corpus.foundation.CHECKLIST_HEADER,
                                      [common.corpus.check("constrained_llm_" + args.version, "complete_validated_coverage", "INCOMPLETE", "Validating saved JSONL")])
                    try:
                        completed = scan_saved(stream, connection, args, on_error)
                    except BaseException:
                        if args.execute and not args.export:
                            save_metadata(root, connection, config, total, "INCOMPLETE")
                        raise
                    require(before == common.snapshot(root, paths), "Execution inputs changed during resume validation")
                    if args.export:
                        require(completed == total, "Export requires complete validated responses")
                        if args.execute:
                            export_records(root, connection, args.version, runtime_configuration)
                        print("CONSTRAINED_LLM_" + args.version.upper() + "_EXPORT=" + ("PASS" if args.execute else "READY"))
                        return 0
                    if not args.execute:
                        if args.preflight:
                            require(runtime_configuration == runtime.SEMANTIC_STRATIFIED_1200 and total == 1200, "Preflight requires the canonical 1200 runtime")
                            print("V4_STRATIFIED_1200_PREFLIGHT=PASS\nSELECTED=1200\nFOUND_PROMPTS=1200\nDUPLICATE_QUERY_IDS=0\nTEST_ROWS=0")
                        state = "PASS" if completed == total else "INCOMPLETE"
                        print(f"CONSTRAINED_LLM_{args.version.upper()}_EXECUTION={state}\nPREFLIGHT=PASS\nCOMPLETED_QUERIES={completed}\nREMAINING_QUERIES={total-completed}")
                        return 0
                    save_metadata(root, connection, config, total, "INCOMPLETE")
                    try:
                        streak = 0 if args.reset_transport_circuit else transport_streak(connection)
                        require(streak < 5, "Transport circuit open; repair server then explicitly use --reset-transport-circuit")
                        for payload, tokens in connection.execute("SELECT payload, tokens FROM prompts WHERE ordinal>=? ORDER BY ordinal", (completed,)):
                            prompt = common.parse_json(payload)
                            row = runtime.generate(prompt, tokens, args, on_error)
                            validate_saved(row, prompt, tokens, args)
                            append_response(stream, row)
                            remember(connection, row)
                            completed += 1
                            streak = streak+1 if row.get("runtime_fallback_reason") == "TRANSPORT_RETRIES_EXHAUSTED" else 0
                            require(streak < 5, "Transport circuit open after five consecutive exhausted queries; completed records retained")
                        connection.commit()
                        require(before == common.snapshot(root, paths), "Execution inputs changed")
                    except BaseException:
                        save_metadata(root, connection, config, total, "INCOMPLETE")
                        raise
                    # Re-scan persisted JSONL, not transient API state, for the final gate/statistics.
                    connection.execute("DROP TABLE done")
                    completed = scan_saved(stream, connection, args)
                    require(completed == total, "Missing responses")
                    save_metadata(root, connection, config, total, "PASS")
                print("CONSTRAINED_LLM_" + args.version.upper() + "_EXECUTION=PASS")
                return 0
            finally:
                connection.close()
    except KeyboardInterrupt:
        print("CONSTRAINED_LLM_" + args.version.upper() + "_EXECUTION=INCOMPLETE\nInterrupted; validated completed lines are resumable")
        return 130
    except (OSError, ValueError, TypeError, KeyError, ImportError, csv.Error, sqlite3.Error) as exc:
        print("CONSTRAINED_LLM_" + args.version.upper() + "_EXECUTION=INCOMPLETE\n" + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
