"""Independent development retrieval provenance validation; no models."""
from collections import Counter

from src.rag import phase2_common as common
from src.rag import build_evidence_index_v2 as index

BASE = "results/rag/phase2_dev_retrieval_v3/"
RESULTS = BASE + "retrieval_results.csv"
RETRIEVAL_CONFIG = BASE + "experiment_configuration.csv"
RETRIEVAL_CHECKS = "metadata/rag/rag_retrieval_v3_validation_checklist.csv"
OUTPUTS = (BASE + "provenance_chunk_records.csv", BASE + "provenance_verification_summary.csv",
           "metadata/rag/provenance_verification_v3_configuration.csv", "metadata/rag/provenance_verification_v3_validation_checklist.csv")
HEADER = ("chunk_id", "document_id", "source_type", "source_provenance_status", "corpus_partition", "authorized_for_retrieval", "review_unit_id", "quality_status", "source_location", "provenance_status", "authorized_for_downstream", "verification_reason")
SUMMARY_HEADER = ("source_dataset", "retrieval_method", "query_count", "retrieval_rows", "verified_chunks", "unknown_chunks", "unauthorized_rows")
METHODS = ("tfidf", "dense_bge", "hybrid_rrf")
require = common.require


def output_paths(development_id=None):
    if development_id is None:
        return OUTPUTS
    index.corpus_v2.validate_development_id(development_id)
    base = f"results/rag/{development_id}/stage05_provenance/"
    return (base+"provenance_chunk_records.csv", base+"provenance_verification_summary.csv",
            f"metadata/rag/{development_id}/stage05_provenance/configuration.csv",
            f"metadata/rag/{development_id}/stage05_provenance/validation_checklist.csv")


def retrieval_paths(development_id=None):
    return ("results/rag/phase2_dev_retrieval_v3/retrieval_results.csv",
            "results/rag/phase2_dev_retrieval_v3/experiment_configuration.csv",
            "metadata/rag/rag_retrieval_v3_validation_checklist.csv") if development_id is None else (
            f"results/rag/{development_id}/stage04_retrieval/retrieval_results.csv",
            f"results/rag/{development_id}/stage04_retrieval/experiment_configuration.csv",
            f"metadata/rag/{development_id}/stage04_retrieval/validation_checklist.csv")


def verify_chunks(chunks, empirical_documents=None):
    output = []
    for row in common.unique(chunks, "chunk_id").values():
        require(common.corpus.trusted(row) and row["quality_status"] == "KEEP" and row["document_id"] != "TOR-SPEC-001", "Unauthorized/ambiguous chunk")
        empirical = row["source_type"] == "FROZEN_TRAIN_EMPIRICAL_EVIDENCE"
        if empirical:
            document = (empirical_documents or {}).get(row["document_id"])
            require(document is not None and row["chunk_text"] in document["normalized_text"],
                    "Empirical chunk lacks verified derivation")
        else:
            require(row["source_type"] in common.corpus.foundation.SOURCE_CATEGORIES, "Unknown source type")
        output.append(dict({k: row[k] for k in HEADER if k in row}, provenance_status="VERIFIED", authorized_for_downstream="YES", verification_reason="MANIFEST_CORPUS_KEEP_UNIT_CHUNK_LINKAGE_PASS"))
    return output


def empirical_documents(root, development_id):
    if development_id not in {"v5e_validation", "v5f_validation"}:
        return None
    if development_id == "v5f_validation":
        from src.rag.build_semantic_association_v5 import verify_documents, SOURCE_TYPE
    else:
        from src.rag.build_train_evidence_v5 import verify_documents, SOURCE_TYPE
    documents = list(common.rows(root, index.corpus_v2.output_paths(development_id)[0], index.corpus_v2.DOCUMENT_HEADER))
    verify_documents(root, development_id, documents)
    return {r["document_id"]:r for r in documents if r["source_type"] == SOURCE_TYPE}


def validated_retrieval(root, queries, chunks, results_path=None):
    # Integer masks encode per-query method/rank and chunk uniqueness compactly.
    slots, candidates = {}, {}
    positions = {identity: i for i, identity in enumerate(chunks)}
    fields = ("experiment_id", "query_id", "source_dataset", "retrieval_method", "rank", "chunk_id", "document_id", "retrieval_score")
    for row in common.rows(root, results_path or RESULTS, fields):
        query = queries.get(row["query_id"])
        require(query is not None and row["source_dataset"] == query["source_dataset"], "Unknown query/dataset")
        require(row["experiment_id"] == "phase2_dev_retrieval_v3" and row["retrieval_method"] in METHODS, "Wrong experiment/method")
        chunk = chunks.get(row["chunk_id"])
        require(chunk is not None and chunk["document_id"] == row["document_id"], "Unknown chunk/document")
        require(chunk["provenance_status"] == "VERIFIED" and chunk["authorized_for_downstream"] == "YES", "Unauthorized evidence")
        rank = int(row["rank"])
        require(1 <= rank <= 10, "Invalid rank")
        common.finite(row["retrieval_score"])
        key = (row["query_id"], row["retrieval_method"])
        bit, candidate = 1 << (rank-1), 1 << positions[row["chunk_id"]]
        require(not slots.get(key, 0) & bit and not candidates.get(key, 0) & candidate, "Duplicate rank/chunk")
        slots[key] = slots.get(key, 0) | bit
        candidates[key] = candidates.get(key, 0) | candidate
        yield row
    require(len(slots) == len(queries)*3 and all(mask == 1023 for mask in slots.values()), "Incomplete retrieval coverage")


def build(root, args):
    index_inputs, index_outputs = index.input_paths(args.development_id), index.output_paths(args.development_id)
    result_path, retrieval_config, retrieval_checks = retrieval_paths(args.development_id)
    outputs = output_paths(args.development_id)
    query_path = "results/rag/phase2_dev_retrieval_v3/query_records.csv" if args.development_id is None else f"results/rag/{args.development_id}/stage04_retrieval/query_records.csv"
    inputs = (*index_inputs, index_outputs[0], query_path, result_path, retrieval_config, retrieval_checks)
    before = common.snapshot(root, inputs)
    config = common.completed(root, retrieval_config, retrieval_checks)
    require(config.get("experiment_id") == "phase2_dev_retrieval_v3" and config.get("index_revision") == "V2", "Wrong completed retrieval")
    require(config.get("development_id", "") == (args.development_id or ""), "Retrieval development-id mismatch")
    queries = common.queries(root, query_path)
    records = verify_chunks(index.validated_chunks(root, inputs=index_inputs, outputs=index_outputs), empirical_documents(root, args.development_id))
    lookup = common.unique(records, "chunk_id")
    counts = Counter((row["source_dataset"], row["retrieval_method"]) for row in validated_retrieval(root, queries, lookup, result_path))
    summaries = [dict(source_dataset=ds, retrieval_method=method, query_count=count//10, retrieval_rows=count, verified_chunks=len(records), unknown_chunks=0, unauthorized_rows=0) for (ds, method), count in sorted(counts.items())]
    require(before == common.snapshot(root, inputs), "Inputs changed")
    saved = dict(stage="PROVENANCE_VERIFICATION", input_metadata_json=common.json_text(before), query_count=len(queries), chunk_count=len(records), policy="VERIFIED_AUTHORIZED_KEEP_ONLY")
    if args.development_id: saved["development_id"] = args.development_id
    return [(outputs[0], HEADER, records), (outputs[1], SUMMARY_HEADER, summaries),
            (outputs[2], common.CONFIG_HEADER, common.config_rows(saved)),
            (outputs[3], common.corpus.foundation.CHECKLIST_HEADER, common.checks("provenance_v3", {"coverage": sum(counts.values()), "unknown_chunks": 0, "unauthorized_rows": 0, "development_only": True}))]


def load_verified(root, development_id=None):
    # Revalidate intrinsic linkage when later stages consume saved verification.
    index_inputs, index_outputs, outputs = index.input_paths(development_id), index.output_paths(development_id), output_paths(development_id)
    expected = verify_chunks(index.validated_chunks(root, inputs=index_inputs, outputs=index_outputs), empirical_documents(root, development_id))
    saved = common.unique(common.rows(root, outputs[0], HEADER), "chunk_id")
    require(saved == common.unique(expected, "chunk_id"), "Saved provenance differs from current evidence")
    config = common.unique(common.rows(root, outputs[2], common.CONFIG_HEADER), "field")
    require(config.get("development_id", {}).get("value", "") == (development_id or ""), "Provenance development-id mismatch")
    identity = common.parse_json(config["input_metadata_json"]["value"])
    require(identity == common.snapshot(root, [r["path"] for r in identity]), "Provenance inputs changed")
    checklist = list(common.rows(root, outputs[3], common.corpus.foundation.CHECKLIST_HEADER))
    require(checklist and all(r["status"] == "PASS" for r in checklist), "Provenance not complete")
    return saved


def main(argv=None):
    parser = common.parser(__doc__)
    parser.add_argument("--development-id")
    return common.run_tables(parser.parse_args(argv), "PROVENANCE_VERIFICATION_V3", build)


if __name__ == "__main__":
    raise SystemExit(main())
