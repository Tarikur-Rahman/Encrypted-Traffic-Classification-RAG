"""KEEP-only Index V2. Local tokenizer preflight; fitting requires --execute."""
from __future__ import annotations

from src.rag import build_evidence_index as v1
from src.rag import build_evidence_corpus_v2 as corpus_v2
from src.rag import phase2_common as common

corpus = common.corpus
require = common.require
INPUTS = (corpus_v2.OUTPUTS[0], corpus_v2.OUTPUTS[1], corpus_v2.OUTPUTS[3])
OUTPUTS = ("data/processed/rag/evidence_chunks_v2.csv", "artifacts/rag/index_v2/tfidf_vectorizer.joblib",
           "artifacts/rag/index_v2/tfidf_matrix.npz", "artifacts/rag/index_v2/dense_embeddings.npy",
           "artifacts/rag/index_v2/dense_chunk_ids.csv", "metadata/rag/evidence_index_v2_configuration.csv",
           "metadata/rag/evidence_index_v2_validation_checklist.csv", "results/rag/index_v2/evidence_index_v2_summary.csv")
CHUNK_HEADER = (*v1.CHUNK_HEADER, "review_unit_id", "quality_status", "source_location", "unit_word_start", "unit_word_end")


def input_paths(development_id=None):
    if development_id is None:
        return INPUTS
    corpus_v2.validate_development_id(development_id)
    outputs = corpus_v2.output_paths(development_id)
    return outputs[0], outputs[1], outputs[3]


def output_paths(development_id=None):
    if development_id is None:
        return OUTPUTS
    corpus_v2.validate_development_id(development_id)
    base = f"rag/{development_id}"
    return (f"data/processed/{base}/evidence_chunks.csv",
            f"artifacts/{base}/index/tfidf_vectorizer.joblib", f"artifacts/{base}/index/tfidf_matrix.npz",
            f"artifacts/{base}/index/dense_embeddings.npy", f"artifacts/{base}/index/dense_chunk_ids.csv",
            f"metadata/{base}/stage03_index/evidence_index_configuration.csv",
            f"metadata/{base}/stage03_index/evidence_index_validation_checklist.csv",
            f"results/{base}/stage03_index/evidence_index_summary.csv")


def eligible_units(root, inputs=None):
    inputs = inputs or INPUTS
    documents = common.unique(common.rows(root, inputs[0], corpus_v2.DOCUMENT_HEADER), "document_id")
    manifest = common.unique(common.rows(root, inputs[1], corpus.MANIFEST_HEADER), "document_id")
    quality_header = corpus_v2.QUALITY_HEADER if inputs[2] == INPUTS[2] else corpus_v2.IMPROVED_QUALITY_HEADER
    quality = common.unique(common.rows(root, inputs[2], quality_header), "review_unit_id")
    profiles = {r.get("quality_profile", "legacy") for r in quality.values()}
    require(len(profiles) == 1 and profiles <= set(corpus_v2.PROFILES), "Mixed/unknown quality profiles")
    profile = next(iter(profiles))
    require((len(documents) == 12 if profile == "legacy" else len(documents) >= 12)
            and "TOR-SPEC-001" not in documents, "Invalid V2 document population")
    require(all(row["document_id"] in documents for row in quality.values()), "Unknown quality document")
    units = []
    for identity, doc in sorted(documents.items()):
        source = manifest.get(identity)
        require(source is not None and corpus.trusted(source) and corpus.usable_document(doc), "Unauthorized V2 document")
        require(doc["extraction_review_status"] != "REJECTED", "Rejected extraction")
        require(all(doc[key] == source[key] for key in corpus.MANIFEST_HEADER), "Manifest/document metadata mismatch")
        parts = corpus_v2.source_units(doc["normalized_text"])
        review = sorted((r for r in quality.values() if r["document_id"] == identity), key=lambda r: int(r["unit_order"]))
        require(len(parts) == len(review), "Quality units cannot be reconstructed; regenerate Corpus V2")
        located = [(r["source_location"], part[1]) for r, part in zip(review, parts)]
        current = corpus_v2.quality_rows(identity, located, profile)
        for row, expected, (_, text) in zip(review, current, located):
            require(all(str(row[key]) == str(expected[key]) for key in corpus_v2.QUALITY_HEADER), "Stale/altered quality review; regenerate Corpus V2")
            if row["quality_decision"] == "KEEP":
                units.append((doc, row, text))
    require(units, "No KEEP evidence")
    return list(documents.values()), units


def chunk_units(units, policy):
    require(policy.limit == 512, "Index V2 requires a 512-token local model")
    chunks, orders = [], {}
    for doc, review, text in units:
        identity = doc["document_id"]
        for offset, (start, end, passage, count) in enumerate(v1.token_windows(text, policy)):
            order = orders.get(identity, 0)
            orders[identity] = order + 1
            row = dict(chunk_id=f"{review['review_unit_id']}-CHUNK-{offset:04d}", document_id=identity, chunk_order=order,
                       **{key: doc[key] for key in v1.METADATA}, chunk_word_count=end-start, chunk_token_count=count,
                       chunk_character_count=len(passage), chunk_text=passage, review_unit_id=review["review_unit_id"],
                       quality_status="KEEP", source_location=review["source_location"], unit_word_start=start, unit_word_end=end)
            chunks.append(row)
    common.unique(chunks, "chunk_id")
    require(chunks and {d["document_id"] for d, _, _ in units} == {c["document_id"] for c in chunks}, "Lost KEEP document")
    v1.validate_token_counts(chunks, policy)
    return chunks


def validated_chunks(root, policy=None, inputs=None, outputs=None):
    inputs, outputs = inputs or INPUTS, outputs or OUTPUTS
    _, units = eligible_units(root, inputs)
    actual = list(common.rows(root, outputs[0], CHUNK_HEADER))
    common.unique(actual, "chunk_id")
    if policy is not None:
        expected = chunk_units(units, policy)
        require(actual == [{k: str(r[k]) for k in CHUNK_HEADER} for r in expected], "Chunk plan differs from KEEP source units")
    else:
        lookup = {review["review_unit_id"]: (doc, review, text) for doc, review, text in units}
        orders, unit_orders, coverage = {}, {}, {}
        for row in actual:
            require(row["review_unit_id"] in lookup, "Chunk is not KEEP-eligible")
            doc, review, text = lookup[row["review_unit_id"]]
            start, end = int(row["unit_word_start"]), int(row["unit_word_end"])
            import re
            words = list(re.finditer(r"\S+", text))
            require(0 <= start < end <= len(words), "Invalid unit offsets")
            require(row["chunk_text"] == text[words[start].start():words[end-1].end()], "Chunk text/source mismatch")
            require(row["document_id"] == doc["document_id"] and all(row[k] == doc[k] for k in v1.METADATA), "Chunk provenance mismatch")
            require(row["quality_status"] == "KEEP" and row["source_location"] == review["source_location"], "Quality/location mismatch")
            unit = row["review_unit_id"]
            ordinal = unit_orders.get(unit, 0)
            require(row["chunk_id"] == f"{unit}-CHUNK-{ordinal:04d}" and int(row["chunk_order"]) == orders.get(doc["document_id"], 0), "Unstable chunk ID/order")
            previous = coverage.get(unit, 0)
            require((start == 0 if not ordinal else 0 <= previous-start <= v1.OVERLAP_WORDS) and end > previous, "Gap/invalid overlap")
            coverage[unit] = end
            unit_orders[unit] = ordinal+1
            orders[doc["document_id"]] = int(row["chunk_order"])+1
            require(int(row["chunk_word_count"]) == end-start <= v1.MAXIMUM_WORDS and int(row["chunk_character_count"]) == len(row["chunk_text"]) and 0 < int(row["chunk_token_count"]) <= 512, "Invalid chunk counts")
        require(set(coverage) == set(lookup) and all(coverage[k] == len(lookup[k][2].split()) for k in lookup), "Incomplete KEEP coverage")
    return actual


def main(argv=None):
    parser = common.parser(__doc__)
    parser.add_argument("--development-id")
    args = parser.parse_args(argv)
    try:
        root = args.project_root.resolve(strict=True)
        corpus_v2.validate_development_id(args.development_id) if args.development_id is not None else None
        inputs, outputs = input_paths(args.development_id), output_paths(args.development_id)
        before = common.snapshot(root, inputs)
        documents, units = eligible_units(root, inputs)
        quality_rows = list(common.rows(root, inputs[2]))
        profiles = {r.get("quality_profile", "legacy") for r in quality_rows}
        require(profiles == ({"evidence_quality"} if args.development_id else {"legacy"}), "Stage 3 development-id/profile mismatch")
        identity = v1.model_identity(root)
        policy = v1.load_token_policy(root)
        chunks = chunk_units(units, policy)
        config = v1.configuration(documents, identity)
        config.update(corpus_path=inputs[0], index_revision="V2", quality_policy="KEEP_ONLY", chunking_method="quality_unit_token_safe_v2",
                      embedding_model="BAAI/bge-small-en-v1.5", input_metadata_json=common.json_text(before))
        if args.development_id:
            config["development_id"] = args.development_id
        if not args.execute:
            print(f"EVIDENCE_INDEX_V2=READY_FOR_CONSTRUCTION\nCHUNKS={len(chunks)}")
            return 0
        targets = common.fresh_outputs(root, outputs)
        vectorizer, lexical = v1.lexical_index(chunks)
        dense = v1.dense_index(root, chunks)
        require(dense.shape == (len(chunks), 384), "Expected BGE dimension 384")
        require(identity == v1.model_identity(root) and before == common.snapshot(root, inputs), "Inputs changed")
        config.update(chunk_count=len(chunks), embedding_dimension=384, lexical_feature_count=lexical.shape[1])
        import joblib
        import numpy as np
        from scipy.sparse import save_npz
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_table(targets[0], CHUNK_HEADER, chunks, mode="x")
        joblib.dump(vectorizer, targets[1])
        save_npz(targets[2], lexical)
        np.save(targets[3], dense, allow_pickle=False)
        corpus.write_table(targets[4], v1.MAPPING_HEADER, [dict(embedding_row=i, chunk_id=r["chunk_id"], document_id=r["document_id"]) for i, r in enumerate(chunks)], mode="x")
        corpus.write_table(targets[5], common.CONFIG_HEADER, common.config_rows(config), mode="x")
        corpus.write_table(targets[7], v1.SUMMARY_HEADER, v1.summary(chunks), mode="x")
        corpus.write_table(targets[6], corpus.foundation.CHECKLIST_HEADER, common.checks("index_v2", {"keep_only": len(chunks), "token_limit": 512, "embedding_dimension": 384, "v1_preserved": True}), mode="x")
        print("EVIDENCE_INDEX_V2=PASS")
        return 0
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print("EVIDENCE_INDEX_V2=BLOCKED\n" + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
