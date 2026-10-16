"""Stream the selected V3/V4 population through the exact local Qwen chat tokenizer.

No model weights, inference, network, GPU, or automatic prompt truncation.
Percentiles use nearest rank; median is the mean of the two middle values.
"""
import csv
import math
import sqlite3
import tempfile
from collections import Counter
from collections.abc import Mapping
from contextlib import nullcontext
from numbers import Integral
from pathlib import Path

from src.rag import phase2_common as common
from src.rag import materialize_constrained_llm_experiment as prompts

MODEL = "models/rag/llm/Qwen3-4B-Instruct-2507-FP8"
MAX_MODEL_LEN = 4096
RESERVED_COMPLETION_TOKENS = 512
SAFE_MAX_PROMPT_TOKENS = MAX_MODEL_LEN - RESERVED_COMPLETION_TOKENS
OUTPUTS = ("results/rag/llm/v3/qwen_token_audit.csv", "results/rag/llm/v3/qwen_token_audit_summary.csv")
LEGACY_OUTPUTS = OUTPUTS  # Readable historical names; never written by new audits.
HEADER = ("query_id", "source_dataset", "classifier_prediction", "qwen_prompt_tokens", "max_model_len", "reserved_completion_tokens", "safe_prompt_limit", "reserved_total_tokens", "within_limit")
SUMMARY_HEADER = ("rows", "min_prompt_tokens", "mean_prompt_tokens", "median_prompt_tokens", "p95_prompt_tokens", "p99_prompt_tokens", "max_prompt_tokens", "max_query_id", "max_model_len", "reserved_completion_tokens", "safe_prompt_limit", "over_limit_with_reservation", "dataset_counts_json", "dataset_over_limit_json", "prediction_counts_json", "percentile_method", "tokenizer_path", "chat_template_policy", "configuration_key", "input_metadata_json", "tokenizer_metadata_json", "gate_status")
require = common.require


def safe_prompt_limit(max_model_len, reserved_completion_tokens):
    require(type(max_model_len) is int and max_model_len > 0, "max_model_len must be a positive integer")
    require(type(reserved_completion_tokens) is int and 0 <= reserved_completion_tokens < max_model_len,
            "reserved_completion_tokens must be a nonnegative integer below max_model_len")
    return max_model_len - reserved_completion_tokens


def configuration_key(max_model_len, reserved_completion_tokens):
    safe_prompt_limit(max_model_len, reserved_completion_tokens)
    return f"maxlen_{max_model_len}_completion_{reserved_completion_tokens}"


def output_paths(max_model_len=MAX_MODEL_LEN, reserved_completion_tokens=RESERVED_COMPLETION_TOKENS, version="v3"):
    require(version in {"v3", "v4"}, "Unknown audit version")
    base = "results/rag/llm/" + version + "/token_audits/" + configuration_key(max_model_len, reserved_completion_tokens) + "/"
    return (base + "qwen_token_audit.csv", base + "qwen_token_audit_summary.csv")


def load_tokenizer(root):
    # Validate the local path before invoking a library that also accepts hub IDs.
    common.corpus.local_path(root, MODEL + "/tokenizer_config.json")
    prompts.contexts.index.v1.offline_mode()
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str((root/MODEL).resolve()), local_files_only=True, trust_remote_code=False)


def prompt_tokens(tokenizer, text):
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=True,
        add_generation_prompt=True, return_dict=True, return_tensors=None,
        truncation=False, padding=False,
    )
    require(isinstance(encoded, Mapping) and "input_ids" in encoded, "Tokenizer must return input_ids")
    ids = encoded["input_ids"]
    require(isinstance(ids, (list, tuple)), "Expected list input_ids without tensors")
    if ids and isinstance(ids[0], (list, tuple)):
        require(len(ids) == 1, "Unexpected multi-prompt token batch")
        ids = ids[0]
    require(ids and all(isinstance(value, Integral) and not isinstance(value, bool) for value in ids), "Invalid input token IDs")
    return len(ids)  # Never len(encoded): BatchEncoding counts fields, not tokens.


def audit_row(row, tokenizer, max_model_len=MAX_MODEL_LEN, reserved_completion_tokens=RESERVED_COMPLETION_TOKENS):
    limit = safe_prompt_limit(max_model_len, reserved_completion_tokens)
    count = prompt_tokens(tokenizer, row["prompt_text"])
    total = count + reserved_completion_tokens
    return dict(query_id=row["query_id"], source_dataset=row["source_dataset"], classifier_prediction=row["classifier_prediction"],
                qwen_prompt_tokens=count, max_model_len=max_model_len, reserved_completion_tokens=reserved_completion_tokens, safe_prompt_limit=limit,
                reserved_total_tokens=total, within_limit="YES" if total <= max_model_len else "NO")


def populate_queries(root, connection):
    connection.execute("PRAGMA cache_size=-2048")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("CREATE TABLE queries (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    connection.execute("CREATE TABLE audit (id TEXT PRIMARY KEY, tokens INTEGER NOT NULL)")
    counts = Counter()
    for row in common.rows(root, common.QUERY_PATH, prompts.QUERY_FIELDS):
        common.validate_query(row)
        require(row["query_id"].strip() and row["experiment_id"] == "phase2_dev_retrieval_v3", "Invalid development query")
        connection.execute("INSERT INTO queries VALUES (?, ?)", (row["query_id"], common.json_text({key: row[key] for key in prompts.QUERY_FIELDS})))
        counts[row["source_dataset"]] += 1
    require(dict(counts) == common.QUERY_COUNTS, "Development query population differs")
    connection.commit()


def validate_prompt(row, query, version="v3"):
    prompts.validate_grounding(row, version)
    require(row["query_id"] == query["query_id"] and row["source_dataset"] == query["source_dataset"] and row["classifier_prediction"] == query["predicted_class_label"] and row["classifier_confidence"] == query["confidence"], "Prompt/query metadata differs")
    payload = common.parse_json(row["prompt_text"].partition("INPUT_JSON:\n")[2])
    required = {"query_id", "source_dataset", "classifier_prediction", "classifier_confidence", "traffic_profile", "trusted_evidence", "allowed_chunk_ids", "allowed_classes"}
    require(isinstance(payload, dict) and set(payload) == required, "Unknown/missing prompt payload fields")
    require(payload["query_id"] == query["query_id"] and payload["source_dataset"] == query["source_dataset"] and payload["classifier_prediction"] == query["predicted_class_label"] and payload["classifier_confidence"] == common.finite(query["confidence"]) and payload["traffic_profile"] == query["retrieval_query_text"], "Prompt semantics differ from development query")
    require(payload["allowed_classes"] == list(common.CLASSES[query["source_dataset"]]), "Allowed classes differ")
    evidence = payload["trusted_evidence"]
    require(isinstance(evidence, list), "Invalid supplied evidence")
    common.unique(evidence, "chunk_id")
    for item in evidence:
        require(set(item) == {"chunk_id", "document_id", "source_type", "chunk_text", "rerank_rank", "token_count", "provenance_status"} and item["provenance_status"] == "VERIFIED", "Invalid evidence schema/provenance")
    require(payload["allowed_chunk_ids"] == common.parse_json(row["allowed_chunk_ids"]) == [item["chunk_id"] for item in evidence], "Allowed citations differ")
    require(row["trusted_context"] == prompts.contexts.context_text(evidence), "Trusted context differs")


def summarize(connection, datasets, over, predictions, max_model_len=MAX_MODEL_LEN, reserved_completion_tokens=RESERVED_COMPLETION_TOKENS):
    limit = safe_prompt_limit(max_model_len, reserved_completion_tokens)
    count, minimum, mean, maximum = connection.execute("SELECT COUNT(*), MIN(tokens), AVG(tokens), MAX(tokens) FROM audit").fetchone()
    require(count > 0, "Empty audit")
    connection.execute("CREATE INDEX token_order ON audit(tokens, id)")
    def at(position):
        return connection.execute("SELECT tokens FROM audit ORDER BY tokens, id LIMIT 1 OFFSET ?", (position,)).fetchone()[0]
    median = (at((count-1)//2) + at(count//2))/2
    max_id = connection.execute("SELECT id FROM audit WHERE tokens=? ORDER BY id LIMIT 1", (maximum,)).fetchone()[0]
    return dict(rows=count, min_prompt_tokens=minimum, mean_prompt_tokens=mean, median_prompt_tokens=median,
                p95_prompt_tokens=at(math.ceil(.95*count)-1), p99_prompt_tokens=at(math.ceil(.99*count)-1), max_prompt_tokens=maximum, max_query_id=max_id,
                max_model_len=max_model_len, reserved_completion_tokens=reserved_completion_tokens, safe_prompt_limit=limit, over_limit_with_reservation=sum(over.values()),
                dataset_counts_json=common.json_text(datasets), dataset_over_limit_json=common.json_text({key: over[key] for key in datasets}), prediction_counts_json=common.json_text(predictions),
                percentile_method="nearest_rank; median averages middle pair", tokenizer_path=MODEL,
                chat_template_policy="one user message; tokenize=True; add_generation_prompt=True; return_dict=True; return_tensors=None; no truncation",
                configuration_key=configuration_key(max_model_len, reserved_completion_tokens), input_metadata_json="", tokenizer_metadata_json="",
                gate_status="FAIL" if sum(over.values()) else "PASS")


def audit_population(root, tokenizer, connection, stream=None, max_model_len=MAX_MODEL_LEN, reserved_completion_tokens=RESERVED_COMPLETION_TOKENS, version="v3"):
    safe_prompt_limit(max_model_len, reserved_completion_tokens)
    populate_queries(root, connection)
    writer = csv.DictWriter(stream, fieldnames=HEADER) if stream is not None else None
    if writer:
        writer.writeheader()
    datasets, over, predictions = Counter(), Counter(), Counter()
    previous = None
    for row in common.rows(root, prompts.version_settings(version)[0][0], prompts.HEADER):
        identity = row["query_id"]
        require(previous is None or identity > previous, "Duplicate/out-of-order prompt query_id")
        previous = identity
        query = connection.execute("SELECT payload FROM queries WHERE id=?", (identity,)).fetchone()
        require(query is not None, "Unknown prompt query_id")
        validate_prompt(row, common.parse_json(query[0]), version)
        result = audit_row(row, tokenizer, max_model_len, reserved_completion_tokens)
        connection.execute("INSERT INTO audit VALUES (?, ?)", (identity, result["qwen_prompt_tokens"]))
        datasets[row["source_dataset"]] += 1
        over[row["source_dataset"]] += result["within_limit"] == "NO"
        predictions[row["source_dataset"] + "/" + row["classifier_prediction"]] += 1
        if writer:
            writer.writerow(result)
    require(dict(datasets) == common.QUERY_COUNTS, "Incomplete prompt population")
    require(connection.execute("SELECT id FROM queries EXCEPT SELECT id FROM audit LIMIT 1").fetchone() is None, "Missing prompt queries")
    connection.commit()
    return summarize(connection, datasets, over, predictions, max_model_len, reserved_completion_tokens)


def tokenizer_files(root):
    # Readable tokenizer-only identity; do not inspect or load weight files.
    names = ("tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt", "added_tokens.json", "special_tokens_map.json", "chat_template.jinja", "config.json")
    result = []
    for name in names:
        relative = MODEL + "/" + name
        path = common.corpus.local_path(root, relative, must_exist=False)
        if path.exists():
            stat = path.stat()
            result.append((relative, stat.st_size, stat.st_mtime_ns))
    return result


def main(argv=None):
    parser = common.parser(__doc__)
    parser.add_argument("--version", choices=("v3", "v4"), default="v3")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--reserved-completion-tokens", type=int, default=RESERVED_COMPLETION_TOKENS)
    args = parser.parse_args(argv)
    if args.max_model_len is None:
        args.max_model_len = 4608 if args.version == "v4" else MAX_MODEL_LEN
    try:
        safe_prompt_limit(args.max_model_len, args.reserved_completion_tokens)
        root = args.project_root.resolve(strict=True)
        selected, schema, _ = prompts.version_settings(args.version)
        config = common.completed(root, selected[1], selected[2])
        require(config.get("prompt_schema_version") == schema, "Completed selected materialization required")
        require(list(common.rows(root, selected[3], prompts.POLICY_HEADER)) == prompts.policy_rows(args.version), "Selected policy differs")
        paths = [common.QUERY_PATH, *selected]
        before, tokenizer_before = common.snapshot(root, paths), tokenizer_files(root)
        outputs = output_paths(args.max_model_len, args.reserved_completion_tokens, args.version)
        if args.execute:
            common.fresh_outputs(root, outputs)
        tokenizer = load_tokenizer(root)
        with tempfile.TemporaryDirectory(prefix=".rag_qwen_audit_", dir=root) as directory:
            directory = Path(directory)
            connection = sqlite3.connect(str(directory/"audit.sqlite"))
            try:
                manager = (directory/"audit.csv").open("x", encoding="utf-8", newline="") if args.execute else nullcontext(None)
                with manager as stream:
                    summary = audit_population(root, tokenizer, connection, stream, args.max_model_len, args.reserved_completion_tokens, args.version)
            finally:
                connection.close()
            require(before == common.snapshot(root, paths) and tokenizer_before == tokenizer_files(root), "Audit inputs changed")
            summary.update(input_metadata_json=common.json_text(before), tokenizer_metadata_json=common.json_text(tokenizer_before))
            if args.execute:
                common.corpus.write_table(directory/"summary.csv", SUMMARY_HEADER, [summary], mode="x")
                prompts.publish_staged(root, [(outputs[0], directory/"audit.csv"), (outputs[1], directory/"summary.csv")])
        print("QWEN_TOKEN_AUDIT=" + summary["gate_status"] + "\n" + common.json_text(summary))
        return 2 if summary["gate_status"] == "FAIL" else 0
    except (OSError, ValueError, KeyError, TypeError, ImportError, csv.Error, sqlite3.Error) as exc:
        print("QWEN_TOKEN_AUDIT=FAIL\n" + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
