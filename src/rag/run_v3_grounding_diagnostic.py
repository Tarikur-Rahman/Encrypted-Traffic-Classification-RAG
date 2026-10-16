"""Exactly six selected V3/V4 requests; validation is not human grounding acceptance."""
import csv
import os
import sqlite3
import tempfile
from pathlib import Path

from src.rag import llm_v3_runtime as runtime

common, audit, prompts = runtime.common, runtime.audit, runtime.prompts
BASE = "results/rag/llm/v3/diagnostics/"
OUTPUTS = (BASE + "six_class_grounding_diagnostic.jsonl", BASE + "six_class_grounding_summary.csv")
PAIRS = tuple((dataset, label) for dataset in ("CIC-Darknet2020", "USTC-TFC2016") for label in common.CLASSES[dataset])
HEADER = ("query_id", "source_dataset", "classifier_prediction", "prompt_tokens", "api_prompt_tokens", "completion_tokens", "total_tokens", "finish_reason", "reasoning_decision", "evidence_status", "supported_class", "citation_count", "explanation", "retry_count", "structural_status", "human_grounding_status")


def output_paths(version="v3", configuration="legacy"):
    runtime.require(version in {"v3", "v4"}, "Unknown diagnostic version")
    suffix = runtime.configuration_directory(version, configuration)
    return tuple(path.replace("/v3/", "/" + version + "/").replace("/diagnostics/", "/diagnostics/" + suffix) for path in OUTPUTS)


def summary_header(configuration="legacy"):
    return HEADER + (("runtime_configuration", "grounding_gate_status", "grounding_gate_error_type", "grounding_gate_error_message", "validator_fallback_applied") if configuration in runtime.FAIL_CLOSED_CONFIGURATIONS else ())


def failure_path(version="v3", configuration="legacy"):
    return output_paths(version, configuration)[0].replace("six_class_grounding_diagnostic.jsonl", "six_class_grounding_failures.jsonl")


def publish_failure(root, directory, version, record, configuration="legacy"):
    """Atomically replace the last failed diagnostic, never accepted results."""
    staged = directory/"failures.jsonl"
    with staged.open("x", encoding="utf-8", newline="\n") as stream:
        runtime.flush_record(stream, record)
    target = common.corpus.local_path(root, failure_path(version, configuration), must_exist=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staged, target)


def select_six(rows):
    selected = {}
    for row in rows:
        pair = (row["source_dataset"], row["classifier_prediction"])
        runtime.require(pair in PAIRS, "Unknown dataset/prediction")
        if pair not in selected:
            selected[pair] = row
    runtime.require(set(selected) == set(PAIRS), "All six prediction classes required")
    return [selected[pair] for pair in PAIRS]


def main(argv=None):
    args = runtime.add_arguments(common.parser(__doc__)).parse_args(argv)
    try:
        runtime.validate_options(args)
        root = args.project_root.resolve(strict=True)
        paths = [common.QUERY_PATH, *prompts.version_settings(args.version)[0]]
        configuration = runtime.configuration_name(args)
        outputs = output_paths(args.version, configuration)
        header = summary_header(configuration)
        before = common.snapshot(root, paths)
        tokenizer_before = audit.tokenizer_files(root)
        if args.execute:
            common.fresh_outputs(root, outputs)
        with tempfile.TemporaryDirectory(prefix=".rag_v3_diagnostic_", dir=root) as directory:
            directory = Path(directory)
            connection = sqlite3.connect(str(directory/"population.sqlite"))
            try:
                if args.version == "v3":
                    runtime.prepare_population(root, connection)
                else:
                    runtime.prepare_population(root, connection, args.version)
                selected = select_six(common.parse_json(row[0]) for row in connection.execute("SELECT payload FROM prompts ORDER BY ordinal"))
            finally:
                connection.close()
            tokenizer = audit.load_tokenizer(root)
            counts = [audit.prompt_tokens(tokenizer, row["prompt_text"]) for row in selected]
            runtime.require(all(n + args.max_completion_tokens <= args.max_model_len for n in counts), "Diagnostic prompt exceeds runtime envelope")
            if args.execute:
                summaries = []
                with (directory/"records.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
                    for row, count in zip(selected, counts):
                        errors = []
                        try:
                            result = runtime.generate(row, count, args, errors.append)
                        except runtime.RequestFailure as exc:
                            failure = dict(query_id=row["query_id"], source_dataset=row["source_dataset"],
                                           classifier_prediction=row["classifier_prediction"], prompt_tokens=count,
                                           attempt_errors=errors, final_error_type=type(exc).__name__,
                                           final_error_message=str(exc), completed_before_failure=len(summaries))
                            if configuration != "legacy":
                                failure["runtime_configuration"] = configuration
                            try:
                                publish_failure(root, directory, args.version, failure, configuration)
                            except (OSError, ValueError) as publication_error:
                                print("FAILURE_ARTIFACT_PUBLICATION_FAILED: " + str(publication_error))
                            raise
                        identity = dict(input_metadata_json=common.json_text(before), tokenizer_metadata_json=common.json_text(tokenizer_before)) if args.version == "v4" else {}
                        runtime.flush_record(stream, dict(result, api_prompt_tokens=result["prompt_tokens"], retry_errors=errors, human_grounding_status="REVIEW_REQUIRED", **identity))
                        summaries.append(dict({k: result[k] for k in header if k in result}, api_prompt_tokens=result["prompt_tokens"], citation_count=len(result["cited_chunk_ids"]), structural_status="PASS", human_grounding_status="REVIEW_REQUIRED"))
                common.corpus.write_table(directory/"summary.csv", header, summaries, mode="x")
                runtime.require(before == common.snapshot(root, paths) and tokenizer_before == audit.tokenizer_files(root), "Diagnostic inputs changed")
                prompts.publish_staged(root, [(outputs[0], directory/"records.jsonl"), (outputs[1], directory/"summary.csv")])
        print("SIX_CLASS_DIAGNOSTIC=" + ("STRUCTURAL_PASS\nHUMAN_GROUNDING_ACCEPTANCE=REVIEW_REQUIRED" if args.execute else "READY_FOR_EXECUTION"))
        return 0
    except (OSError, ValueError, TypeError, KeyError, ImportError, csv.Error, sqlite3.Error) as exc:
        print("SIX_CLASS_DIAGNOSTIC=FAIL\n" + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
