"""Verify intrinsic chunk provenance and stream development retrieval coverage.

No model dependencies, semantic judgments, reranking, context construction or
traffic access. Canonical outputs are written only after every check succeeds.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.rag import build_evidence_corpus as corpus


EXPERIMENT = "phase2_dev_retrieval_v1"
CONFIGURATION = "lexical_dense_rrf_top10"
BASE = "results/rag/" + EXPERIMENT + "/"
METHODS = ("tfidf", "dense_bge", "hybrid_rrf")
QUERY_COUNTS = {"USTC-TFC2016": 97001, "CIC-Darknet2020": 31723}
CHUNK_COUNT, RETRIEVAL_ROWS, TOP_K = 919, 3861720, 10
CHUNKS = "data/processed/rag/evidence_chunks.csv"
INDEX_CONFIG = "metadata/rag/evidence_index_configuration.csv"
INDEX_CHECKS = "metadata/rag/evidence_index_validation_checklist.csv"
INPUTS = (BASE+"retrieval_results.csv", BASE+"query_records.csv", BASE+"experiment_configuration.csv",
          CHUNKS, *corpus.OUTPUTS, corpus.MANIFEST, corpus.REVIEW, INDEX_CONFIG, INDEX_CHECKS)
OUTPUTS = (BASE+"provenance_chunk_records.csv", BASE+"provenance_verification_summary.csv",
           "metadata/rag/provenance_verification_configuration.csv", "metadata/rag/provenance_verification_validation_checklist.csv")
RECORD_HEADER = tuple("chunk_id document_id title source_type publisher publication_date canonical_url doi local_path corpus_partition manifest_authorized_for_retrieval corpus_authorized_for_retrieval chunk_authorized_for_retrieval source_provenance_status extraction_status extraction_review_status provenance_verification_status authorized_for_downstream verification_reason".split())
COVERAGE_FIELDS = tuple("retrieval_rows unique_retrieved_chunks unknown_retrieved_chunks unknown_provenance_rows unauthorized_retrieval_rows adversarial_retrieval_rows document_link_mismatch_rows".split())
SUMMARY_HEADER = ("summary_level", "source_dataset", "retrieval_method", "canonical_chunks", "verified_chunks", *COVERAGE_FIELDS)
CONFIG_HEADER = ("field", "value")
CHUNK_METADATA = tuple("title source_type publisher publication_date canonical_url doi traffic_topics security_topics corpus_partition source_provenance_status authorized_for_retrieval".split())
require = corpus.require


def stream_rows(root, relative, required):
    corpus.configure_csv_field_limit()
    with corpus.local_path(root, relative).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        require(len(header) == len(set(header)) and set(required) <= set(header), "Missing/duplicate fields: " + relative)
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()), "Malformed CSV: " + relative)
            yield row


def unique_records(rows, key, description):
    result = {}
    for row in rows:
        require(row[key].strip() and row[key] not in result, "Blank/duplicate " + description + ": " + row[key])
        result[row[key]] = row
    return result


def read_config(root, relative):
    return {k: v["value"] for k, v in unique_records(stream_rows(root, relative, CONFIG_HEADER), "field", "configuration field").items()}


def fingerprint_free_snapshot(root):
    result = []
    for name in INPUTS:
        stat = corpus.local_path(root, name).stat()
        result.append(dict(path=name, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return result


def validate_stage_inputs(root):
    for name in (corpus.OUTPUTS[2], INDEX_CHECKS):
        rows = list(stream_rows(root, name, corpus.foundation.CHECKLIST_HEADER))
        require(rows and all(r["status"] in {"PASS", "WARN"} for r in rows), "Blocked/invalid upstream checklist: " + name)
        if name == corpus.OUTPUTS[2]:
            require(any(r["check"] == "nonempty_usable_documents" and r["status"] == "PASS" for r in rows), "Corpus completion not established")
        else:
            require(all(r["status"] == "PASS" for r in rows), "Index validation is not PASS")
    corpus.read_table(root, corpus.OUTPUTS[1], corpus.SUMMARY_HEADER)
    config = read_config(root, BASE+"experiment_configuration.csv")
    expected = dict(experiment_id=EXPERIMENT, configuration_name=CONFIGURATION, stage="RAG_RETRIEVAL",
                    rag_partition="development_validation", retrieval_top_k=str(TOP_K), chunk_count=str(CHUNK_COUNT),
                    authorized_document_count="10", test_access="NO", phase_i_models_loaded="NO",
                    network_used="NO", llm_used="NO", provenance_verification_executed="NO", reranking_executed="NO")
    require(all(config.get(k) == v for k, v in expected.items()), "Retrieval configuration identity/boundary differs")
    indexed = read_config(root, INDEX_CONFIG)
    require(indexed.get("chunk_count") == str(CHUNK_COUNT) and indexed.get("authorized_document_count") == "10" and indexed.get("trusted_only") == "YES" and indexed.get("adversarial_included") == "NO", "Canonical index population/policy differs")
    return indexed


def chunk_records(root, indexed):
    documents = unique_records(stream_rows(root, corpus.OUTPUTS[0], corpus.DOCUMENT_HEADER), "document_id", "corpus document ID")
    manifest = unique_records(stream_rows(root, corpus.MANIFEST, corpus.MANIFEST_HEADER), "document_id", "manifest document ID")
    reviews = corpus.read_reviews(root, list(manifest.values()))
    fields = ("chunk_id", "document_id", "chunk_order", *CHUNK_METADATA, "chunk_word_count", "chunk_character_count", "chunk_token_count", "chunk_text")
    chunks = unique_records(stream_rows(root, CHUNKS, fields), "chunk_id", "chunk ID")
    require(len(chunks) == CHUNK_COUNT, f"Expected {CHUNK_COUNT} canonical chunks; found {len(chunks)}")
    canonical_ids = json.loads(indexed["corpus_document_ids_json"])
    require(isinstance(canonical_ids, list) and len(canonical_ids) == len(set(canonical_ids)) == 10, "Invalid canonical index document set")
    orders = {}
    output = []
    for identity, chunk in chunks.items():
        document_id = chunk["document_id"]
        document, source = documents.get(document_id), manifest.get(document_id)
        rejected, unresolved = [], []
        review = reviews.get(document_id, {}).get("review_status", "MISSING")
        if document is None:
            unresolved.append("MISSING_CORPUS_DOCUMENT")
        if source is None:
            unresolved.append("MISSING_MANIFEST_DOCUMENT")
        elif any(not source[key].strip() for key in corpus.MINIMUM):
            unresolved.append("MISSING_REQUIRED_SOURCE_METADATA")
        if document_id not in canonical_ids:
            unresolved.append("OUTSIDE_CANONICAL_INDEX_DOCUMENT_SET")
        if document_id == "TOR-SPEC-001" or review == "REJECTED":
            rejected.append("REJECTED_SOURCE")
        for label, record in (("CHUNK", chunk), ("CORPUS", document), ("MANIFEST", source)):
            if record is None:
                continue
            if record["corpus_partition"] != "trusted":
                rejected.append(label+"_NOT_TRUSTED")
            if record["source_provenance_status"] != "VERIFIED":
                rejected.append(label+"_PROVENANCE_NOT_VERIFIED")
            if record["authorized_for_retrieval"] != "YES":
                rejected.append(label+"_NOT_AUTHORIZED")
            if record["source_type"] not in corpus.foundation.SOURCE_CATEGORIES:
                rejected.append(label+"_SOURCE_CATEGORY_INVALID")
        if document and source:
            if any(document[k] != source[k] for k in corpus.MANIFEST_HEADER if k != "authorized_for_retrieval"):
                unresolved.append("CORPUS_MANIFEST_METADATA_MISMATCH")
            try:
                corpus.source_path(root, source)  # Existence/path/partition only; no document parsing.
            except ValueError:
                unresolved.append("LOCAL_SOURCE_PATH_INVALID_OR_MISSING")
        if document:
            if any(chunk[k] != document[k] for k in CHUNK_METADATA):
                unresolved.append("CHUNK_DOCUMENT_METADATA_MISMATCH")
            status = document["extraction_status"]
            if status == "REQUIRES_REVIEW":
                if review != "APPROVED" or document["extraction_review_status"] != "APPROVED":
                    rejected.append("EXTRACTION_REVIEW_NOT_APPROVED")
            elif status != "PASS":
                rejected.append("EXTRACTION_NOT_USABLE")
            if document["extraction_review_status"] == "REJECTED":
                rejected.append("CORPUS_EXTRACTION_REVIEW_REJECTED")
            if not document["normalized_text"].strip() or not chunk["chunk_text"].strip() or chunk["chunk_text"] not in document["normalized_text"]:
                unresolved.append("CHUNK_TEXT_NOT_TRACEABLE")
        try:
            order = int(chunk["chunk_order"])
            if order != orders.get(document_id, 0) or identity != f"{document_id}-CHUNK-{order:04d}":
                unresolved.append("CHUNK_ORDER_ID_MISMATCH")
            orders[document_id] = orders.get(document_id, 0)+1
            if not (0 < int(chunk["chunk_word_count"]) == len(chunk["chunk_text"].split()) <= int(indexed["maximum_chunk_words"]) and int(chunk["chunk_character_count"]) == len(chunk["chunk_text"]) and 0 < int(chunk["chunk_token_count"]) <= int(indexed["chunk_token_limit"])):
                unresolved.append("CHUNK_COUNTS_INVALID")
        except ValueError:
            unresolved.append("CHUNK_COUNTS_INVALID")
        state = "REJECTED" if rejected else "UNRESOLVED" if unresolved else "VERIFIED"
        row = {key: chunk.get(key, "") for key in RECORD_HEADER}
        row.update(local_path=source["local_path"] if source else "", manifest_authorized_for_retrieval=source["authorized_for_retrieval"] if source else "",
                   corpus_authorized_for_retrieval=document["authorized_for_retrieval"] if document else "", chunk_authorized_for_retrieval=chunk["authorized_for_retrieval"],
                   extraction_status=document["extraction_status"] if document else "", extraction_review_status=document["extraction_review_status"] if document else "",
                   provenance_verification_status=state, authorized_for_downstream="YES" if state == "VERIFIED" else "NO",
                   verification_reason="ALL_REQUIRED_PROVENANCE_CHECKS_PASS" if state == "VERIFIED" else ";".join(sorted(set(rejected+unresolved))))
        output.append(row)
    require({r["document_id"] for r in output} == set(canonical_ids), "Canonical index document population differs")
    return sorted(output, key=lambda r: r["chunk_id"])


def query_population(root):
    queries, counts = {}, Counter()
    fields = ("experiment_id", "configuration_name", "query_id", "source_dataset", "source_split", "rag_partition")
    for row in stream_rows(root, BASE+"query_records.csv", fields):
        require(row["experiment_id"] == EXPERIMENT and row["configuration_name"] == CONFIGURATION, "Query experiment identity differs")
        require(row["source_split"] == "validation" and row["rag_partition"] == "development_validation", "Non-development or TEST query prohibited")
        identity, dataset = row["query_id"], row["source_dataset"]
        require(identity and identity not in queries and dataset in QUERY_COUNTS, "Invalid/duplicate query or dataset identity")
        queries[identity] = (len(queries), dataset)
        counts[dataset] += 1
    require(dict(counts) == QUERY_COUNTS, "Canonical dataset query populations differ")
    return queries


def coverage(root, records):
    queries = query_population(root)
    lookup = {r["chunk_id"]: (i, r) for i, r in enumerate(records)}
    rank_masks, chunk_masks = [0]*(len(queries)*len(METHODS)), [0]*(len(queries)*len(METHODS))
    groups = {}
    def group(dataset, method):
        return groups.setdefault((dataset, method), dict(counts=Counter(), ids=set(), unknown=set()))
    required = ("experiment_id", "configuration_name", "query_id", "source_dataset", "retrieval_method", "rank", "chunk_id", "document_id")
    for row in stream_rows(root, BASE+"retrieval_results.csv", required):
        require(row["experiment_id"] == EXPERIMENT and row["configuration_name"] == CONFIGURATION, "Retrieval identity differs")
        method, dataset = row["retrieval_method"], row["source_dataset"]
        require(method in METHODS, "Unknown retrieval method")
        require(row["query_id"] in queries, "Retrieval references unknown query")
        query_index, expected_dataset = queries[row["query_id"]]
        require(dataset == expected_dataset, "Retrieval/query dataset identity mismatch")
        rank = int(row["rank"])
        require(1 <= rank <= TOP_K, "Retrieval rank outside canonical top-k")
        slot = query_index*len(METHODS)+METHODS.index(method)
        rank_bit = 1 << (rank-1)
        require(not rank_masks[slot] & rank_bit, "Duplicate query/method/rank")
        rank_masks[slot] |= rank_bit
        found = lookup.get(row["chunk_id"])
        if found:
            position, record = found
            bit = 1 << position
            require(not chunk_masks[slot] & bit, "Duplicate candidate within query/method")
            chunk_masks[slot] |= bit
        for ds, met in (("", ""), ("", method), (dataset, method)):
            stats = group(ds, met)
            stats["counts"]["retrieval_rows"] += 1
            stats["ids"].add(row["chunk_id"])
            if found is None:
                stats["unknown"].add(row["chunk_id"])
                stats["counts"]["unknown_provenance_rows"] += 1
            else:
                stats["counts"]["unauthorized_retrieval_rows"] += record["authorized_for_downstream"] != "YES"
                stats["counts"]["adversarial_retrieval_rows"] += record["corpus_partition"] == "adversarial"
                stats["counts"]["document_link_mismatch_rows"] += row["document_id"] != record["document_id"]
    require(all(mask == (1 << TOP_K)-1 for mask in rank_masks), "Incomplete query/method top-k coverage")
    require(groups.get(("", ""), {}).get("counts", {}).get("retrieval_rows", 0) == RETRIEVAL_ROWS, "Canonical retrieval row count differs")
    summaries = []
    verified = sum(r["provenance_verification_status"] == "VERIFIED" for r in records)
    for (dataset, method), stats in sorted(groups.items()):
        expected = RETRIEVAL_ROWS if not method else (QUERY_COUNTS[dataset] if dataset else sum(QUERY_COUNTS.values()))*TOP_K
        require(stats["counts"]["retrieval_rows"] == expected, "Dataset/method retrieval coverage differs")
        summary = dict(summary_level="dataset_method" if dataset else "method" if method else "overall", source_dataset=dataset, retrieval_method=method, canonical_chunks=len(records), verified_chunks=verified)
        summary.update({k: stats["counts"][k] for k in COVERAGE_FIELDS})
        summary.update(unique_retrieved_chunks=len(stats["ids"]), unknown_retrieved_chunks=len(stats["unknown"]))
        summaries.append(summary)
    return summaries


def validate(root):
    before = fingerprint_free_snapshot(root)
    indexed = validate_stage_inputs(root)
    records = chunk_records(root, indexed)
    summaries = coverage(root, records)
    overall = next(r for r in summaries if r["summary_level"] == "overall")
    print(f"CANONICAL_CHUNKS={len(records)}\nVERIFIED_CHUNKS={overall['verified_chunks']}\nRETRIEVAL_ROWS={overall['retrieval_rows']}\nUNKNOWN_RETRIEVED_CHUNKS={overall['unknown_retrieved_chunks']}\nUNAUTHORIZED_RETRIEVAL_ROWS={overall['unauthorized_retrieval_rows']}\nADVERSARIAL_RETRIEVAL_ROWS={overall['adversarial_retrieval_rows']}")
    require(overall["verified_chunks"] == CHUNK_COUNT, "Not all canonical chunks verified; downstream authorization blocked")
    require(all(overall[k] == 0 for k in ("unknown_provenance_rows", "unauthorized_retrieval_rows", "adversarial_retrieval_rows", "document_link_mismatch_rows")), "Retrieval provenance coverage failed")
    require(before == fingerprint_free_snapshot(root), "Canonical inputs changed during validation")
    checks = [corpus.check("provenance", name, "PASS", detail) for name, detail in {
        "normalized_chunk_records": f"Exactly one VERIFIED record per canonical chunk: {len(records)}",
        "source_linkage": "Manifest/corpus/chunk linkage, metadata, local source path and canonical index population consistent",
        "authorization": "Trusted, VERIFIED, authorized in manifest/corpus/chunk; rejected sources excluded",
        "extraction_review": "PASS extraction needs no manual approval; REQUIRES_REVIEW needs effective and curated APPROVED; REJECTED cannot authorize",
        "query_populations": json.dumps(QUERY_COUNTS, sort_keys=True),
        "test_isolation": "Only saved development_validation query identities; no traffic data access",
        "evaluation_isolation": "Ground-truth columns never consulted for provenance decisions",
        "stage_boundary": "No relevance decisions, screening, models, embeddings, LLM, network, reranking, context or classification",
        "no_hashes_or_figures": "CSV records and readable identities only",
        "canonical_outputs": ";".join(OUTPUTS),
    }.items()]
    for row in summaries:
        checks.append(corpus.check("coverage", (row["source_dataset"] or "all")+"/"+(row["retrieval_method"] or "all"), "PASS", json.dumps({k: row[k] for k in COVERAGE_FIELDS}, sort_keys=True)))
    config = dict(experiment_id=EXPERIMENT, configuration_name=CONFIGURATION, stage="PROVENANCE_VERIFICATION", record_granularity="one_per_canonical_chunk", canonical_chunks=str(CHUNK_COUNT), retrieval_rows=str(RETRIEVAL_ROWS), retrieval_methods=json.dumps(METHODS), top_k=str(TOP_K), query_population_json=json.dumps(QUERY_COUNTS, sort_keys=True), input_file_metadata_json=json.dumps(before, sort_keys=True), test_access="NO", network_used="NO", model_loading="NO", semantic_relevance_assessed="NO", reranking_executed="NO", trusted_context_constructed="NO")
    return records, summaries, checks, config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = args.project_root.resolve(strict=True)
        records, summaries, checks, config = validate(root)
        if not args.execute:
            print("PROVENANCE_VERIFICATION=READY_FOR_EXECUTION\nTEST_ACCESS=NO")
            return 0
        targets = [corpus.local_path(root, name, must_exist=False) for name in OUTPUTS]
        for path, header, rows in zip(targets, (RECORD_HEADER, SUMMARY_HEADER, CONFIG_HEADER, corpus.foundation.CHECKLIST_HEADER), (records, summaries, [dict(field=k, value=v) for k, v in config.items()], checks)):
            corpus.write_table(path, header, rows)
        print("PROVENANCE_VERIFICATION=PASS\nNETWORK_ACCESS=NO\nTEST_ACCESS=NO\nNEXT_STAGE=PROVENANCE_AWARE_RERANKING")
        return 0
    except (OSError, ValueError, KeyError, csv.Error) as exc:
        print("PROVENANCE_VERIFICATION=BLOCKED\n"+str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
