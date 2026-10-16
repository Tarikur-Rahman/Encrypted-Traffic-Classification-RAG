"""Validate externally generated JSONL/CSV responses; never invoke an LLM."""
from src.rag import phase2_common as common
from src.rag import materialize_constrained_llm_experiment as prompts
from src.rag import build_trusted_contexts as contexts
from src.rag import verify_retrieval_provenance_v3 as provenance

OUTPUTS = (prompts.BASE + "validated_llm_responses.csv", prompts.BASE + "llm_response_validation_summary.csv")
HEADER = (*prompts.REQUIRED_RESPONSE_FIELDS, "source_dataset", "prompt_schema_version", "response_validation_status")
SUMMARY_HEADER = ("source_dataset", "response_count", "evidence_status", "prompt_schema_version", "response_source", "input_metadata_json", "validation_status")
require = common.require


def validate_response(value, prompt):
    require(isinstance(value, dict) and set(value) == set(prompts.REQUIRED_RESPONSE_FIELDS), "Missing/unknown response fields")
    require(value["query_id"] == prompt["query_id"], "Unknown query response")
    status = value["evidence_status"]
    require(isinstance(status, str) and status in common.STATUSES, "Invalid evidence status")
    require(isinstance(value["explanation"], str) and value["explanation"].strip(), "Explanation required")
    citations = value["cited_chunk_ids"]
    require(isinstance(citations, list) and all(isinstance(item, str) for item in citations) and len(set(citations)) == len(citations), "Invalid/duplicate citations")
    require(set(citations) <= set(common.parse_json(prompt["allowed_chunk_ids"])), "Unknown/unsupplied chunk citation")
    supported = value["supported_class"]
    if status in {"SUPPORTED", "CONTRADICTED"}:
        require(value["reasoning_decision"] == "SUPPORT_CLASS" and isinstance(supported, str) and supported in common.CLASSES[prompt["source_dataset"]] and citations, "Invalid supported class/decision/citations")
        require((supported == prompt["classifier_prediction"]) == (status == "SUPPORTED"), "Evidence status/classifier relationship differs")
    else:
        require(value["reasoning_decision"] == "ABSTAIN" and supported is None, "Abstention requires null supported_class")
        require(status != "CONFLICTING" or citations, "Conflicting evidence requires citations")
    require(common.parse_json(prompt["allowed_chunk_ids"]) or status == "INSUFFICIENT", "No evidence requires INSUFFICIENT")
    return dict(value, cited_chunk_ids=common.json_text(citations), supported_class=supported or "", source_dataset=prompt["source_dataset"],
                prompt_schema_version=prompts.SCHEMA_VERSION, response_validation_status="PASS")


def external_rows(root, relative):
    path = common.corpus.local_path(root, common.safe_input(relative))
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                require(line.strip(), "Blank JSONL response")
                yield common.parse_json(line)
    else:
        require(path.suffix.lower() == ".csv", "Response format must be JSONL or CSV")
        for row in common.rows(root, relative, prompts.REQUIRED_RESPONSE_FIELDS):
            require(set(row) == set(prompts.REQUIRED_RESPONSE_FIELDS), "Unexpected CSV response fields")
            row["cited_chunk_ids"] = common.parse_json(row["cited_chunk_ids"])
            row["supported_class"] = None if row["supported_class"] in {"", "null"} else row["supported_class"]
            yield row


def validate_all(values, prompt_records):
    seen, output = set(), []
    for value in values:
        require(isinstance(value, dict) and isinstance(value.get("query_id"), str), "Malformed response")
        identity = value["query_id"]
        require(identity in prompt_records and identity not in seen, "Unknown/duplicate query response")
        seen.add(identity)
        output.append(validate_response(value, prompt_records[identity]))
    require(seen == set(prompt_records), "Missing query responses; complete experiment required")
    return sorted(output, key=lambda row: row["query_id"])


def build(root, args):
    require(args.responses, "--responses project-relative JSONL/CSV required")
    paths = [common.QUERY_PATH, *prompts.OUTPUTS, *contexts.OUTPUTS, *provenance.OUTPUTS, args.responses]
    before = common.snapshot(root, paths)
    queries = common.queries(root)
    trusted = contexts.validated_records(root, queries, provenance.load_verified(root))
    materialized = prompts.validated_prompts(root, queries, trusted)
    records = validate_all(external_rows(root, args.responses), materialized)
    require(before == common.snapshot(root, paths), "Response/source inputs changed")
    from collections import Counter
    counts = Counter((row["source_dataset"], row["evidence_status"]) for row in records)
    summaries = [dict(source_dataset=ds, response_count=count, evidence_status=status, prompt_schema_version=prompts.SCHEMA_VERSION,
                      response_source=args.responses, input_metadata_json=common.json_text(before), validation_status="PASS") for (ds, status), count in sorted(counts.items())]
    return [(OUTPUTS[0], HEADER, records), (OUTPUTS[1], SUMMARY_HEADER, summaries)]


def load_validated(root, prompt_records):
    summaries = list(common.rows(root, OUTPUTS[1], SUMMARY_HEADER))
    require(summaries and all(r["validation_status"] == "PASS" for r in summaries), "Responses not complete")
    for row in summaries:
        identity = common.parse_json(row["input_metadata_json"])
        require(identity == common.snapshot(root, [r["path"] for r in identity]), "Response inputs changed")
    saved = common.unique(common.rows(root, OUTPUTS[0], HEADER), "query_id")
    raw = []
    for row in saved.values():
        value = {key: row[key] for key in prompts.REQUIRED_RESPONSE_FIELDS}
        value["supported_class"] = value["supported_class"] or None
        value["cited_chunk_ids"] = common.parse_json(value["cited_chunk_ids"])
        raw.append(value)
    validated = common.unique(validate_all(raw, prompt_records), "query_id")
    require(saved == validated, "Validated response metadata differs")
    return validated


def main(argv=None):
    parser = common.parser(__doc__)
    parser.add_argument("--responses", required=True, help="Project-relative externally generated JSONL or CSV")
    return common.run_tables(parser.parse_args(argv), "CONSTRAINED_LLM_RESPONSE_IMPORT", build)


if __name__ == "__main__":
    raise SystemExit(main())
