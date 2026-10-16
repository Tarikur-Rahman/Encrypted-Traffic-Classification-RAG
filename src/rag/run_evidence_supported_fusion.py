"""Auditable rule-based fusion; classifier retention is the neutral default."""
from src.rag import phase2_common as common
from src.rag import build_trusted_contexts as contexts
from src.rag import materialize_constrained_llm_experiment as prompts
from src.rag import import_constrained_llm_responses as responses
from src.rag import verify_retrieval_provenance_v3 as provenance

BASE = "results/rag/fusion/"
OUTPUTS = tuple(BASE + name + ".csv" for name in ("final_decisions", "fusion_summary", "fusion_configuration", "fusion_validation_checklist"))
HEADER = tuple("query_id source_dataset classifier_prediction classifier_confidence classifier_probabilities llm_supported_class evidence_status final_class fusion_reason cited_chunk_ids cited_document_ids provenance_summary evidence_explanation".split())
require = common.require


def fuse(query, response, context, policy="retain_classifier", max_override_confidence=0.5, *, source_split="validation", prompt_version="v2"):
    # Explicit projection: evaluation/ground-truth fields are never consulted.
    common.validate_query(query, source_split=source_split)
    if source_split == "test":
        require(policy == "retain_classifier" and prompt_version == "v4", "Final TEST fusion is frozen V4 retain_classifier")
        require(not {"true_class_id", "true_class_label", "classifier_correct", "ground_truth", "target_id", "target_binary"}.intersection(response), "GT prohibited in TEST fusion")
    require(policy in {"retain_classifier", "evidence_override"} and 0 <= common.finite(max_override_confidence) <= 1, "Invalid fusion policy")
    require(query["query_id"] == response["query_id"] == context["query_id"] and query["source_dataset"] == response["source_dataset"] == context["source_dataset"], "Unknown fusion query")
    require(response["response_validation_status"] == "PASS", "Unvalidated response")
    prompt = prompts.prompt_record(context, prompt_version, source_split=source_split)
    raw = {key: response[key] for key in prompts.REQUIRED_RESPONSE_FIELDS}
    raw["cited_chunk_ids"] = common.parse_json(raw["cited_chunk_ids"])
    raw["supported_class"] = raw["supported_class"] or None
    responses.validate_response(raw, prompt)
    citations = raw["cited_chunk_ids"]
    evidence = common.unique(common.parse_json(context["evidence_records_json"]), "chunk_id")
    require(all(identity in evidence and evidence[identity]["provenance_status"] == "VERIFIED" for identity in citations), "Unverified fusion citation")
    predicted, supported, status = query["predicted_class_label"], response["supported_class"], response["evidence_status"]
    final = predicted
    reason = "CLASSIFIER_RETAINED_" + status
    if status == "CONTRADICTED":
        if policy == "evidence_override" and common.finite(query["confidence"]) <= max_override_confidence:
            final, reason = supported, "EXPLICIT_EVIDENCE_OVERRIDE_BELOW_OR_AT_CONFIDENCE_THRESHOLD"
        else:
            reason = "CLASSIFIER_RETAINED_OVERRIDE_DISABLED_OR_CONFIDENCE_ABOVE_THRESHOLD"
    return dict(query_id=query["query_id"], source_dataset=query["source_dataset"], classifier_prediction=predicted, classifier_confidence=query["confidence"],
                classifier_probabilities=query["class_probabilities_json"], llm_supported_class=supported, evidence_status=status, final_class=final, fusion_reason=reason,
                cited_chunk_ids=common.json_text(citations), cited_document_ids=common.json_text(sorted({evidence[c]["document_id"] for c in citations})),
                provenance_summary=common.json_text({c: {key: evidence[c][key] for key in ("document_id", "source_type", "provenance_status")} for c in citations}), evidence_explanation=response["explanation"])


def build(root, args):
    paths = [common.QUERY_PATH, *contexts.OUTPUTS, *prompts.OUTPUTS, *responses.OUTPUTS, *provenance.OUTPUTS]
    before = common.snapshot(root, paths)
    queries = common.queries(root)
    trusted = contexts.validated_records(root, queries, provenance.load_verified(root))
    materialized = prompts.validated_prompts(root, queries, trusted)
    validated = responses.load_validated(root, materialized)
    records = [fuse(query, validated[identity], trusted[identity], args.policy, args.max_override_confidence) for identity, query in sorted(queries.items())]
    require(before == common.snapshot(root, paths), "Inputs changed")
    from collections import Counter
    counts = Counter((r["source_dataset"], r["classifier_prediction"], r["evidence_status"], r["final_class"], r["fusion_reason"]) for r in records)
    summary_header = ("source_dataset", "classifier_prediction", "evidence_status", "final_class", "fusion_reason", "record_count")
    summaries = [dict(zip(summary_header, (*key, count))) for key, count in sorted(counts.items())]
    config = dict(stage="EVIDENCE_SUPPORTED_DECISION_FUSION", policy=args.policy, max_override_confidence=args.max_override_confidence,
                  ground_truth_used="NO", evaluation_metrics_computed="NO", input_metadata_json=common.json_text(before),
                  rule="Only CONTRADICTED + explicit evidence_override + confidence <= threshold changes class; other cases retain classifier")
    return [(OUTPUTS[0], HEADER, records), (OUTPUTS[1], summary_header, summaries), (OUTPUTS[2], common.CONFIG_HEADER, common.config_rows(config)),
            (OUTPUTS[3], common.corpus.foundation.CHECKLIST_HEADER, common.checks("fusion", {"queries": len(records), "valid_citations": True, "ground_truth_isolated": True, "final_class_reconstructible": True}))]


def main(argv=None):
    parser = common.parser(__doc__)
    parser.add_argument("--policy", choices=("retain_classifier", "evidence_override"), default="retain_classifier")
    parser.add_argument("--max-override-confidence", type=float, default=0.5)
    return common.run_tables(parser.parse_args(argv), "EVIDENCE_SUPPORTED_DECISION_FUSION", build)


if __name__ == "__main__":
    raise SystemExit(main())
