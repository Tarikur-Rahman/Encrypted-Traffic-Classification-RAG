"""Canonical Stage 8 validation, retain-classifier fusion, finalization and closure.

No model/network operations. Raw rejected model content never reaches fusion.
Every stage revalidates the durable Stage-8 records and exact six-field export.
"""
import csv
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

from src.rag import run_constrained_llm_v3 as execution
from src.rag import run_evidence_supported_fusion as fusion

runtime, common, prompts, subset = execution.runtime, execution.common, execution.prompts, execution.subset
CONFIGURATION = runtime.SEMANTIC_STRATIFIED_1200
BASE = "results/rag/llm/v4/execution/stratified_1200_semantic_fail_closed_5120_1024/"
EXPORT = BASE + "llm_responses_for_import.jsonl"
FUSION = BASE + "fusion_decisions.csv"
FINAL = BASE + "final_classifications.csv"
CLOSURE = BASE + "phase2_closure_checklist.csv"
SCOPE_HEADER = ("evaluation_scope", "population_queries", "evaluated_queries", "sampling_seed", "sampling_strategy")
FUSION_HEADER = (*fusion.HEADER, "reasoning_decision", "supported_class", "decision_source", "explanation_reference", *SCOPE_HEADER)
FINAL_HEADER = ("query_id", "source_dataset", "final_class", "evidence_status", "provenance", "explanation", "decision_source", "explanation_reference", *SCOPE_HEADER)
CLOSURE_HEADER = ("stage", "status", "details", *SCOPE_HEADER, "input_metadata_json")
STAGES = ("Structured Security Query", "Evidence Corpus", "Evidence Indexing", "RAG Retrieval", "Provenance Verification",
          "Provenance-Aware Evidence Reranking", "Trusted Context Construction", "Constrained LLM Reasoning",
          "Evidence-Supported Decision Fusion", "Final Classification")
# Existing upstream metadata only; no full context/retrieval or TEST reads.
UPSTREAM = (
    (None, "results/rag/interface/interface_validation_checklist.csv"),
    (None, "metadata/rag/evidence_corpus_v2_validation_checklist.csv"),
    ("metadata/rag/evidence_index_v2_configuration.csv", "metadata/rag/evidence_index_v2_validation_checklist.csv"),
    ("results/rag/phase2_dev_retrieval_v3/experiment_configuration.csv", "metadata/rag/rag_retrieval_v3_validation_checklist.csv"),
    ("metadata/rag/provenance_verification_v3_configuration.csv", "metadata/rag/provenance_verification_v3_validation_checklist.csv"),
    ("results/rag/reranking/reranking_configuration.csv", "results/rag/reranking/reranking_validation_checklist.csv"),
    ("results/rag/trusted_contexts/trusted_context_configuration.csv", "results/rag/trusted_contexts/trusted_context_validation_checklist.csv"),
)


def scope(count=1200):
    return dict(evaluation_scope=subset.SCOPE, population_queries=str(sum(common.QUERY_COUNTS.values())), evaluated_queries=str(count),
                sampling_seed="42", sampling_strategy=subset.STRATEGY)


def validated_export(root, connection):
    with common.corpus.local_path(root, EXPORT).open(encoding="utf-8") as stream:
        for saved, prompt in connection.execute("SELECT done.payload, prompts.payload FROM done JOIN prompts USING(id) ORDER BY ordinal"):
            row = common.parse_json(saved)
            value = common.parse_json(stream.readline())
            common.require(isinstance(value, dict) and set(value) == set(runtime.FIELDS), "Export must have exactly six fields")
            common.require(value == {key: row[key] for key in runtime.FIELDS}, "Export differs from validated constrained response")
            runtime.importer.validate_response(value, common.parse_json(prompt))
        common.require(not stream.read(1), "Extra export responses")


def fusion_rows(connection):
    records = []
    for saved, materialized, query in connection.execute("SELECT done.payload, prompts.payload, queries.payload FROM done JOIN prompts USING(id) JOIN queries USING(id) ORDER BY prompts.ordinal"):
        value, prompt, query = common.parse_json(saved), common.parse_json(materialized), common.parse_json(query)
        logical = {key: value[key] for key in runtime.FIELDS}
        validated = runtime.importer.validate_response(logical, prompt)
        payload = common.parse_json(prompt["prompt_text"].partition("INPUT_JSON:\n")[2])
        context = dict(query_id=query["query_id"], source_dataset=query["source_dataset"], source_split=query["source_split"], rag_partition=query["rag_partition"],
            classifier_prediction=query["predicted_class_label"], classifier_confidence=query["confidence"], traffic_profile=query["retrieval_query_text"],
            selected_chunk_ids=prompt["allowed_chunk_ids"], evidence_records_json=common.json_text(payload["trusted_evidence"]), context_text=prompt["trusted_context"])
        # Existing authoritative implementation; no override option is exposed.
        row = fusion.fuse(query, validated, context, policy="retain_classifier")
        common.require(row["final_class"] == row["classifier_prediction"], "Classifier override prohibited")
        row.update(reasoning_decision=logical["reasoning_decision"], supported_class=logical["supported_class"] or "",
                   decision_source="retain_classifier", explanation_reference=EXPORT + "#" + query["query_id"], **scope())
        records.append(row)
    common.require(len(records) == 1200, "Fusion requires all 1200 selected queries")
    return records


def final_rows(fused):
    return [dict(query_id=row["query_id"], source_dataset=row["source_dataset"], final_class=row["final_class"], evidence_status=row["evidence_status"],
                 provenance=row["provenance_summary"], explanation=row["evidence_explanation"], decision_source="retain_classifier",
                 explanation_reference=FUSION + "#" + row["query_id"], **scope()) for row in fused]


def validate_table(root, path, header, expected):
    actual = list(common.rows(root, path, header))
    common.require(actual == [{key: str(row[key]) for key in header} for row in expected], "Canonical output differs: " + path)


def closure_rows(root, count, stage8_pass, fused, finalized, input_paths):
    records = []
    paths = list(input_paths)
    for stage, (config, checklist) in zip(STAGES, UPSTREAM):
        try:
            rows = list(common.rows(root, checklist, common.corpus.foundation.CHECKLIST_HEADER))
            common.require(rows and all(row["status"] == "PASS" for row in rows), "Upstream checklist incomplete")
            if config:
                common.completed(root, config, checklist)
                paths.append(config)
            paths.append(checklist)
            status, detail = "PASS", "Existing canonical upstream checklist validated"
        except (OSError, ValueError, KeyError) as exc:
            status, detail = "INCOMPLETE", str(exc)
        records.append(dict(stage=stage, status=status, details=detail))
    records.append(dict(stage=STAGES[7], status="PASS" if stage8_pass else "INCOMPLETE", details=f"{count}/1200 durable validated responses; technical fallbacks remain separately reported"))
    for stage, path, header, expected in ((STAGES[8], FUSION, FUSION_HEADER, fused), (STAGES[9], FINAL, FINAL_HEADER, finalized)):
        try:
            common.require(stage8_pass and len(fused) == 1200, "Stage 8 or validated export not complete")
            validated_path = FUSION if path == FINAL else path
            if path == FINAL:
                validate_table(root, FUSION, FUSION_HEADER, fused)
            validate_table(root, path, header, expected)
            paths.extend([validated_path, path])
            status, detail = "PASS", "Canonical retain_classifier output validated"
        except (OSError, ValueError, KeyError) as exc:
            status, detail = "INCOMPLETE", str(exc)
        records.append(dict(stage=stage, status=status, details=detail))
    identity = common.json_text(common.snapshot(root, list(dict.fromkeys(paths))))
    return [dict(row, **scope(count), input_metadata_json=identity) for row in records]


def main(argv=None):
    parser = common.parser(__doc__)
    parser.add_argument("--stage", required=True, choices=("validate-export", "fusion", "finalize", "closure", "status"))
    args = parser.parse_args(argv)
    try:
        root = args.project_root.resolve(strict=True)
        if args.stage == "status":
            rows = list(common.rows(root, CLOSURE, CLOSURE_HEADER))
            common.require([row["stage"] for row in rows] == list(STAGES), "Invalid closure stages")
            for row in rows:
                identity = common.parse_json(row["input_metadata_json"])
                common.require(identity == common.snapshot(root, [item["path"] for item in identity]), "Closure is stale; revalidate")
                print(row["stage"] + "=" + row["status"] + ": " + row["details"])
            return 0 if all(row["status"] == "PASS" for row in rows) else 2
        opts = SimpleNamespace(version="v4", runtime_configuration=CONFIGURATION, max_model_len=5120, max_completion_tokens=1024,
                               endpoint=runtime.ENDPOINT, model=runtime.MODEL, retry_limit=3, timeout=120, execute=False, export=False, repair_trailing_partial=False)
        with tempfile.TemporaryDirectory(prefix=".rag_phase2_completion_", dir=execution.runtime_temp_root(root)) as directory:
            with closing(sqlite3.connect(str(Path(directory)/"state.sqlite"))) as connection:
                total, audits, diagnostics = execution.llm_subset_preflight.prepare(root, connection, opts, execution.validate_saved)
                paths = [common.QUERY_PATH, *prompts.V4_OUTPUTS, *audits, *diagnostics, subset.MANIFEST]
                before = common.snapshot(root, paths)
                outputs = execution.output_paths("v4", CONFIGURATION)
                execution.check_configuration(root, execution.configuration(opts, before, audits, total))
                with execution.response_file(common.corpus.local_path(root, outputs[0], must_exist=False), False) as stream:
                    count = execution.scan_saved(stream, connection, opts)
                checks = list(common.rows(root, outputs[4])) if (root/outputs[4]).exists() else []
                config = {row["field"]: row["value"] for row in common.rows(root, outputs[2])} if (root/outputs[2]).exists() else {}
                stage8_pass = count == total == 1200 and config.get("execution_status") == "PASS" and bool(checks) and all(row["status"] == "PASS" for row in checks)
                source_paths = paths + [path for path in (*outputs, EXPORT) if (root/path).exists()]
                source_before = common.snapshot(root, source_paths)
                if args.stage != "closure":
                    common.require(stage8_pass, "Stage 8 requires 1200/1200 validated responses and PASS metadata")
                fused, finalized = [], []
                if stage8_pass:
                    try:
                        validated_export(root, connection)
                    except (OSError, ValueError) as exc:
                        if args.stage != "closure":
                            raise
                        print("STAGE8_EXPORT=INCOMPLETE: " + str(exc))
                    else:
                        fused = fusion_rows(connection)
                        finalized = final_rows(fused)
                common.require(source_before == common.snapshot(root, [item["path"] for item in source_before]), "Stage inputs changed")
                if args.stage == "fusion":
                    if args.execute:
                        execution.replace_table(root, FUSION, FUSION_HEADER, fused)
                    else:
                        validate_table(root, FUSION, FUSION_HEADER, fused)
                elif args.stage == "finalize":
                    validate_table(root, FUSION, FUSION_HEADER, fused)
                    if args.execute:
                        execution.replace_table(root, FINAL, FINAL_HEADER, finalized)
                    else:
                        validate_table(root, FINAL, FINAL_HEADER, finalized)
                elif args.stage == "closure":
                    rows = closure_rows(root, count, stage8_pass, fused, finalized, source_paths)
                    if args.execute:
                        execution.replace_table(root, CLOSURE, CLOSURE_HEADER, rows)
                    else:
                        validate_table(root, CLOSURE, CLOSURE_HEADER, rows)
                    common.require(all(row["status"] == "PASS" for row in rows), "Phase II remains INCOMPLETE; see canonical closure checklist")
        print("PHASE2_" + args.stage.upper().replace("-", "_") + "=PASS")
        return 0
    except (OSError, ValueError, KeyError, TypeError, csv.Error, sqlite3.Error) as exc:
        print("PHASE2_COMPLETION=INCOMPLETE\n" + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
