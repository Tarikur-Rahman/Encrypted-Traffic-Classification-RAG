"""Read-only streaming preflight; retain 1200 prompts, never a population spool."""
import random
from collections import Counter
from types import SimpleNamespace

from src.rag import llm_v3_runtime as runtime
from src.rag import materialize_llm_evaluation_subset as subset
from src.rag import run_v3_grounding_diagnostic as diagnostic

common, prompts, audit = runtime.common, runtime.prompts, runtime.audit
require = common.require


def load_manifest(root):
    selected, counts, previous = {}, Counter(), None
    for ordinal, row in enumerate(common.rows(root, subset.MANIFEST, subset.HEADER), 1):
        require(ordinal <= subset.TOTAL and set(row) == set(subset.HEADER), "Invalid subset size/schema")
        identity = row["query_id"]
        pair = (row["source_dataset"], row["classifier_prediction"])
        require(pair in subset.STRATA and row["stratum"] == "/".join(pair), "Invalid subset stratum")
        require(identity and (previous is None or identity > previous), "Duplicate/out-of-order subset ID")
        require(row["selection_ordinal"] == str(ordinal) and row["sampling_seed"] == "42", "Invalid subset ordinal/seed")
        selected[identity] = row
        counts[pair] += 1
        previous = identity
    require(len(selected) == subset.TOTAL and counts == dict.fromkeys(subset.STRATA, subset.PER_STRATUM), "Expected 200 per stratum and 1200 IDs")
    return selected


def prepare(root, connection, args, validate_saved):
    selected = load_manifest(root)  # Must precede any prompt population read.
    diag_paths = diagnostic.output_paths("v4", runtime.FAIL_CLOSED_5120_1024)
    audit_paths = audit.output_paths(5120, 1024, "v4")
    paths = [common.QUERY_PATH, *prompts.V4_OUTPUTS, *audit_paths, *diag_paths, subset.MANIFEST]
    config = common.completed(root, prompts.V4_OUTPUTS[1], prompts.V4_OUTPUTS[2])
    before = common.snapshot(root, paths)
    require(config.get("prompt_schema_version") == "constrained_evidence_v4", "Completed V4 materialization required")
    require(list(common.rows(root, prompts.V4_OUTPUTS[3], prompts.POLICY_HEADER)) == prompts.policy_rows("v4"), "V4 policy differs")
    prompt_identity = common.json_text(common.snapshot(root, [common.QUERY_PATH, *prompts.V4_OUTPUTS]))
    tokenizer_identity = common.json_text(audit.tokenizer_files(root))
    summaries = list(common.rows(root, audit_paths[1], audit.SUMMARY_HEADER))
    require(len(summaries) == 1, "Exactly one audit summary required")
    summary = summaries[0]
    require(summary["gate_status"] == "PASS" and int(summary["over_limit_with_reservation"]) == 0, "Token audit PASS prerequisite failed")
    require(summary["configuration_key"] == "maxlen_5120_completion_1024" and int(summary["max_model_len"]) == 5120 and int(summary["reserved_completion_tokens"]) == 1024 and int(summary["safe_prompt_limit"]) == 4096, "Audit configuration differs")
    require(int(summary["rows"]) == sum(common.QUERY_COUNTS.values()) and common.parse_json(summary["dataset_counts_json"]) == common.QUERY_COUNTS, "Audit population differs")
    require(summary["input_metadata_json"] == prompt_identity and summary["tokenizer_metadata_json"] == tokenizer_identity, "Audit input identity differs")

    # Six small durable diagnostic responses, not copies of diagnostic prompts.
    prior_args = SimpleNamespace(**vars(args))
    prior_args.runtime_configuration = runtime.FAIL_CLOSED_5120_1024
    header = diagnostic.summary_header(runtime.FAIL_CLOSED_5120_1024)
    diag_summaries = iter(common.rows(root, diag_paths[1], header))
    diagnostics = {}
    with common.corpus.local_path(root, diag_paths[0]).open(encoding="utf-8") as stream:
        for pair in diagnostic.PAIRS:
            row = common.parse_json(stream.readline())
            require((row["source_dataset"], row["classifier_prediction"]) == pair and row["query_id"] not in diagnostics, "Diagnostic class/order differs")
            require(row["input_metadata_json"] == prompt_identity and row["tokenizer_metadata_json"] == tokenizer_identity, "Diagnostic input identity differs")
            require(row["human_grounding_status"] == "REVIEW_REQUIRED", "Human acceptance cannot be inferred")
            expected = dict({k: row[k] for k in header if k in row}, citation_count=len(row["cited_chunk_ids"]), structural_status="PASS", human_grounding_status="REVIEW_REQUIRED")
            require(next(diag_summaries, None) == {k: "" if v is None else str(v) for k, v in expected.items()}, "Diagnostic summary differs")
            diagnostics[row["query_id"]] = row
        require(not stream.read(1) and next(diag_summaries, None) is None, "Exactly six diagnostics required")

    needed = set(selected) | set(diagnostics)
    queries = {}
    counts = Counter()
    # Stream once; retain only projected development rows for selected/diagnostic IDs.
    for row in common.rows(root, common.QUERY_PATH, prompts.QUERY_FIELDS):
        common.validate_query(row)
        require(row["experiment_id"] == "phase2_dev_retrieval_v3", "Wrong development experiment")
        counts[row["source_dataset"]] += 1
        if row["query_id"] in needed:
            require(row["query_id"] not in queries, "Duplicate selected query")
            queries[row["query_id"]] = {key: row[key] for key in prompts.QUERY_FIELDS}
    require(counts == common.QUERY_COUNTS and set(queries) == needed, "Incomplete development query population")
    connection.execute("PRAGMA cache_size=-2048")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("CREATE TABLE prompts (id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE, payload TEXT NOT NULL, tokens INTEGER)")
    connection.execute("CREATE TABLE queries (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    connection.executemany("INSERT INTO queries VALUES (?, ?)", [(identity, common.json_text(queries[identity])) for identity in selected])
    found, diag_found, first = set(), set(), set()
    counts, reservoirs = Counter(), {pair: [] for pair in subset.STRATA}
    rng, previous = random.Random(42), None
    for row in common.rows(root, prompts.V4_OUTPUTS[0], prompts.HEADER):
        identity = row["query_id"]
        pair = (row["source_dataset"], row["classifier_prediction"])
        require(previous is None or identity > previous, "Duplicate/out-of-order prompt ID")
        require(pair in reservoirs, "Invalid prompt stratum")
        previous = identity
        counts[pair] += 1
        bucket = reservoirs[pair]
        if len(bucket) < 200:
            bucket.append(identity)
        else:
            position = rng.randrange(counts[pair])
            if position < 200:
                bucket[position] = identity
        if pair not in first:
            require(identity in diagnostics, "Diagnostic is not the first canonical query in stratum")
            first.add(pair)
        if identity not in needed:
            continue
        audit.validate_prompt(row, queries[identity], "v4")
        if identity in diagnostics:
            saved = diagnostics[identity]
            value = {k: v for k, v in saved.items() if k not in {"input_metadata_json", "tokenizer_metadata_json", "api_prompt_tokens", "retry_errors", "human_grounding_status"}}
            validate_saved(value, row, saved["prompt_tokens"], prior_args)
            require(saved["api_prompt_tokens"] == saved["prompt_tokens"], "Diagnostic token counts differ")
            diag_found.add(identity)
        if identity in selected:
            manifest = selected[identity]
            require(pair == (manifest["source_dataset"], manifest["classifier_prediction"]), "Manifest/prompt stratum differs")
            connection.execute("INSERT INTO prompts VALUES (?, ?, ?, NULL)", (identity, int(manifest["selection_ordinal"])-1, common.json_text(row)))
            found.add(identity)
    require(found == set(selected) and diag_found == set(diagnostics), "Missing selected/diagnostic prompts")
    require({dataset: sum(n for (d, _), n in counts.items() if d == dataset) for dataset in common.QUERY_COUNTS} == common.QUERY_COUNTS, "Prompt population differs")
    require({identity for bucket in reservoirs.values() for identity in bucket} == set(selected), "Manifest differs from seed-42 sampling")
    seen = set()
    for row in common.rows(root, audit_paths[0], audit.HEADER):
        identity = row["query_id"]
        if identity not in needed:
            continue
        require(identity not in seen, "Duplicate selected audit row")
        seen.add(identity)
        count = int(row["qwen_prompt_tokens"])
        query = queries[identity]
        require(row["source_dataset"] == query["source_dataset"] and row["classifier_prediction"] == query["predicted_class_label"], "Audit/query mismatch")
        require(count > 0 and count+1024 <= 5120 and row["within_limit"] == "YES", "Unsafe selected prompt")
        require(int(row["max_model_len"]) == 5120 and int(row["reserved_completion_tokens"]) == 1024 and int(row["safe_prompt_limit"]) == 4096 and int(row["reserved_total_tokens"]) == count+1024, "Audit row limits differ")
        if identity in diagnostics:
            require(count == diagnostics[identity]["prompt_tokens"], "Diagnostic/audit tokens differ")
        if identity in selected:
            connection.execute("UPDATE prompts SET tokens=? WHERE id=?", (count, identity))
    require(seen == needed, "Missing selected audit rows")
    require(before == common.snapshot(root, paths), "Preflight inputs changed")
    connection.commit()
    return subset.TOTAL, audit_paths, diag_paths
