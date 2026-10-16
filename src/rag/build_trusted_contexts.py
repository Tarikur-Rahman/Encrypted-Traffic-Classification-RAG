"""Deterministic, token-bounded trusted evidence packaging; no LLM."""
import csv
import shutil
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from src.rag import phase2_common as common
from src.rag import build_evidence_index_v2 as index
from src.rag import verify_retrieval_provenance_v3 as provenance
from src.rag import run_provenance_aware_reranking as reranking

BASE = "results/rag/trusted_contexts/"
OUTPUTS = tuple(BASE + name + ".csv" for name in ("trusted_context_records", "trusted_context_summary", "trusted_context_configuration", "trusted_context_validation_checklist"))
HEADER = tuple("query_id source_dataset source_split rag_partition classifier_prediction classifier_confidence classifier_probabilities traffic_profile selected_document_ids selected_chunk_ids rerank_ranks evidence_texts source_types provenance_status token_counts evidence_records_json context_text context_token_count context_status".split())
PROFILES = ("legacy", "evidence_diverse")
require = common.require


def paths_for(development_id=None):
    if development_id is None:
        return OUTPUTS, common.QUERY_PATH, index.INPUTS, index.OUTPUTS, provenance.output_paths(), reranking.OUTPUTS
    index.corpus_v2.validate_development_id(development_id)
    base = f"results/rag/{development_id}/stage07_trusted_contexts/"
    outputs = tuple(base+name+".csv" for name in ("trusted_context_records", "trusted_context_summary", "trusted_context_configuration", "trusted_context_validation_checklist"))
    return (outputs, f"results/rag/{development_id}/stage04_retrieval/query_records.csv",
            index.input_paths(development_id), index.output_paths(development_id), provenance.output_paths(development_id),
            reranking.paths_for(development_id)[0])


def reranking_header(profile):
    require(profile in reranking.PROFILES, "Unknown completed reranking profile")
    return reranking.TASK_HEADER if profile == "task_relevance" else reranking.HEADER


def context_text(evidence):
    return "\n\n".join(f"[chunk_id={row['chunk_id']}; document_id={row['document_id']}; source_type={row['source_type']}; provenance=VERIFIED]\n{row['chunk_text']}" for row in evidence)


def construct(query, ranked, chunks, verified, policy, top_n=5, token_budget=2048, max_per_document=0, *, source_split="validation", profile="legacy", diversity_threshold=None):
    common.validate_query(query, source_split=source_split)
    require(top_n > 0 and token_budget > 0 and max_per_document >= 0 and profile in PROFILES, "Invalid context limits/profile")
    if profile == "evidence_diverse":
        require(diversity_threshold is not None, "Explicit diversity threshold in [0,1] required")
        diversity_threshold = common.finite(diversity_threshold)
        require(0 <= diversity_threshold <= 1, "Explicit diversity threshold in [0,1] required")
    else:
        diversity_threshold = 0
    require(len({r["chunk_id"] for r in ranked}) == len(ranked), "Duplicate context candidate")
    ordered = sorted(ranked, key=lambda r: int(r["final_rank"]))
    require([int(r["final_rank"]) for r in ordered] == list(range(1, len(ordered)+1)), "Invalid rerank ranks")
    selected, counts = [], Counter()
    def add(row):
        require(row["query_id"] == query["query_id"] and row["source_dataset"] == query["source_dataset"], "Unknown context query")
        chunk, record = chunks.get(row["chunk_id"]), verified.get(row["chunk_id"])
        require(chunk is not None and record is not None, "Unknown context chunk")
        require(record["provenance_status"] == row["provenance_status"] == "VERIFIED" and record["authorized_for_downstream"] == row["authorized_for_retrieval"] == "YES", "Unverified context evidence")
        require(common.corpus.trusted(chunk) and chunk["quality_status"] == record["quality_status"] == "KEEP", "Unauthorized/ambiguous context evidence")
        require(row["document_id"] == record["document_id"] == chunk["document_id"] and row["source_type"] == record["source_type"] == chunk["source_type"], "Context linkage mismatch")
        components = [("retrieval_rank_component", "retrieval_weight"), ("source_preference_component", "source_preference_weight"), ("task_alignment_component", "task_alignment_weight")]
        if row.get("reranking_profile") == "task_relevance":
            components.append(("relationship_richness_component","relationship_weight"))
        reconstructed = sum(common.finite(row[component])*common.finite(row[weight]) for component, weight in components)
        require(abs(reconstructed-common.finite(row["final_rerank_score"])) <= 1e-12, "Rerank score cannot be reconstructed")
        if len(selected) >= top_n or any(item["chunk_id"] == chunk["chunk_id"] for item in selected):
            return
        if max_per_document and counts[chunk["document_id"]] >= max_per_document:
            return
        value = dict(chunk_id=chunk["chunk_id"], document_id=chunk["document_id"], source_type=chunk["source_type"], chunk_text=chunk["chunk_text"],
                     rerank_rank=int(row["final_rank"]), token_count=policy.count(chunk["chunk_text"]), provenance_status="VERIFIED")
        if row.get("reranking_profile") == "task_relevance":
            for key in ("decision_claim_match_count", "decision_claim_ids_json", "decision_claim_classes_json", "decision_claim_features_json"):
                value[key] = row[key]
        if policy.count(context_text([*selected, value])) > token_budget:
            return
        selected.append(value)
        counts[chunk["document_id"]] += 1
    if profile == "evidence_diverse":
        labels = common.CLASSES[query["source_dataset"]]
        prediction = query["predicted_class_label"]
        def intents_for(row):
            intents = common.parse_json(row.get("retrieval_intents_json", "[]"))
            require(isinstance(intents,list) and set(intents) <= set(labels), "Invalid retrieval-intent audit")
            return intents
        def claim_eligible(row, label):
            count = int(row.get("decision_claim_match_count", 0))
            claim_ids = common.parse_json(row.get("decision_claim_ids_json", "[]"))
            claim_classes = common.parse_json(row.get("decision_claim_classes_json", "[]"))
            claim_features = common.parse_json(row.get("decision_claim_features_json", "[]"))
            require(all(isinstance(values, list) and values == sorted(set(values)) for values in (claim_ids, claim_classes, claim_features)),
                    "Invalid decision-claim audit arrays")
            require(count == len(claim_ids) and set(claim_classes) <= set(labels), "Decision-claim audit differs")
            return count > 0 and label in claim_classes
        def fallback_eligible(row, label):
            intents = intents_for(row)
            overlap = common.finite(row.get("task_alignment_component", 0))
            return label in intents and overlap >= diversity_threshold
        # Discovery intent plus independent text-alignment signals guide
        # exposure only; they never label evidence or determine an outcome.
        for label in (prediction, *(item for item in labels if item != prediction)):
            candidate = next((row for row in ordered if claim_eligible(row,label)),None)
            if candidate is None:
                candidate = next((row for row in ordered if fallback_eligible(row,label)),None)
            if candidate is not None:
                add(candidate)
    for row in ordered:
        add(row)
    selected.sort(key=lambda row: row["rerank_rank"])
    text = context_text(selected)
    total = policy.count(text) if text else 0
    require(total <= token_budget, "Context token budget exceeded")
    return dict(query_id=query["query_id"], source_dataset=query["source_dataset"], source_split=source_split, rag_partition=query["rag_partition"],
                classifier_prediction=query["predicted_class_label"], classifier_confidence=query["confidence"], classifier_probabilities=query["class_probabilities_json"],
                traffic_profile=query["retrieval_query_text"], selected_document_ids=common.json_text([r["document_id"] for r in selected]), selected_chunk_ids=common.json_text([r["chunk_id"] for r in selected]),
                rerank_ranks=common.json_text([r["rerank_rank"] for r in selected]), evidence_texts=common.json_text([r["chunk_text"] for r in selected]), source_types=common.json_text([r["source_type"] for r in selected]),
                provenance_status="VERIFIED" if selected else "NO_EVIDENCE", token_counts=common.json_text([r["token_count"] for r in selected]), evidence_records_json=common.json_text(selected),
                context_text=text, context_token_count=total, context_status="READY" if selected else "INSUFFICIENT")


def validated_records(root, queries, verified, outputs=None, index_inputs=None, index_outputs=None):
    outputs, index_inputs, index_outputs = outputs or OUTPUTS, index_inputs or index.INPUTS, index_outputs or index.OUTPUTS
    config = common.completed(root, outputs[2], outputs[3])
    require(common.parse_json(config["tokenizer_identity_json"]) == index.v1.model_identity(root), "Context tokenizer identity changed")
    policy = index.v1.load_token_policy(root)
    chunks = common.unique(index.validated_chunks(root, inputs=index_inputs, outputs=index_outputs), "chunk_id")
    records = common.unique(common.rows(root, outputs[0], HEADER), "query_id")
    require(set(records) == set(queries), "Unknown/missing context queries")
    for identity, row in records.items():
        query = queries[identity]
        require(row["source_dataset"] == query["source_dataset"] and row["source_split"] == "validation" and row["rag_partition"] == "development_validation", "Context partition/dataset mismatch")
        require(row["classifier_prediction"] == query["predicted_class_label"] and row["classifier_confidence"] == query["confidence"] and row["classifier_probabilities"] == query["class_probabilities_json"] and row["traffic_profile"] == query["retrieval_query_text"], "Altered classifier/profile")
        evidence = common.parse_json(row["evidence_records_json"])
        require(isinstance(evidence, list), "Invalid evidence array")
        common.unique(evidence, "chunk_id")
        for item in evidence:
            chunk, record = chunks.get(item["chunk_id"]), verified.get(item["chunk_id"])
            require(chunk is not None and record is not None and record["provenance_status"] == item["provenance_status"] == "VERIFIED" and record["authorized_for_downstream"] == "YES", "Unauthorized context citation")
            require(all(item[k] == chunk[k] for k in ("document_id", "source_type", "chunk_text")), "Altered evidence text/metadata")
            require(item["token_count"] == policy.count(item["chunk_text"]), "Evidence token count differs")
        require(row["context_text"] == context_text(evidence) and common.parse_json(row["selected_chunk_ids"]) == [r["chunk_id"] for r in evidence], "Altered context text/IDs")
        require(row["provenance_status"] == ("VERIFIED" if evidence else "NO_EVIDENCE"), "Context provenance mismatch")
        for column, key in (("selected_document_ids", "document_id"), ("rerank_ranks", "rerank_rank"), ("evidence_texts", "chunk_text"), ("source_types", "source_type"), ("token_counts", "token_count")):
            require(common.parse_json(row[column]) == [item[key] for item in evidence], "Context reconstruction field differs")
        count = policy.count(row["context_text"]) if evidence else 0
        require(int(row["context_token_count"]) == count <= int(config["token_budget"]) and len(evidence) <= int(config["top_n"]), "Context budget/count differs")
        cap = int(config["max_chunks_per_document"])
        require(not cap or all(n <= cap for n in Counter(item["document_id"] for item in evidence).values()), "Per-document context cap differs")
    return records


def stream_records(root, args, query_path, rerank_path, rerank_header, chunks,
                   verified, policy, paths, before, summary, checks):
    """Disk-backed join: population-sized strings never accumulate in RAM."""
    with TemporaryDirectory(prefix="rag-context-inputs-") as scratch:
        with closing(sqlite3.connect(str(Path(scratch)/"inputs.sqlite"))) as db:
            db.execute("PRAGMA cache_size=-8192")
            db.execute("PRAGMA temp_store=FILE")
            db.execute("CREATE TABLE queries (id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE ranked (id TEXT, ordinal INTEGER, payload TEXT NOT NULL, PRIMARY KEY(id, ordinal))")
            populations = Counter()
            required = ("query_id", "source_dataset", "source_split", "rag_partition",
                        "predicted_class_label", "confidence", "class_probabilities_json", "retrieval_query_text")
            for query in common.rows(root, query_path, required):
                common.validate_query(query)
                require(query["query_id"].strip(), "Blank query_id")
                require(query.get("experiment_id") == "phase2_dev_retrieval_v3", "Wrong query experiment")
                try:
                    db.execute("INSERT INTO queries VALUES (?, ?)", (query["query_id"], common.json_text(query)))
                except sqlite3.IntegrityError as exc:
                    raise ValueError("Duplicate query_id") from exc
                populations[query["source_dataset"]] += 1
            require(dict(populations) == common.QUERY_COUNTS, "Development query population differs")
            for ordinal, row in enumerate(common.rows(root, rerank_path, rerank_header)):
                require(db.execute("SELECT 1 FROM queries WHERE id=?", (row["query_id"],)).fetchone(), "Unknown rerank query")
                db.execute("INSERT INTO ranked VALUES (?, ?, ?)", (row["query_id"], ordinal, common.json_text(row)))
            db.commit()
            require(not db.execute("SELECT 1 FROM queries q WHERE NOT EXISTS (SELECT 1 FROM ranked r WHERE r.id=q.id) LIMIT 1").fetchone(),
                    "Incomplete rerank queries")
            for identity, payload in db.execute("SELECT id, payload FROM queries ORDER BY id"):
                # Stage 6 emits at most ten candidates per query. Bound even
                # malformed input before constructing a candidate list.
                candidates = []
                for (value,) in db.execute("SELECT payload FROM ranked WHERE id=? ORDER BY ordinal", (identity,)):
                    require(len(candidates) < 10, "Too many context candidates")
                    candidates.append(common.parse_json(value))
                yield construct(common.parse_json(payload), candidates, chunks, verified, policy,
                    args.top_n, args.token_budget, args.max_chunks_per_document,
                    profile=args.profile, diversity_threshold=args.diversity_threshold)
            require(before == common.snapshot(root, paths), "Inputs changed")
            summary.extend(dict(source_dataset=dataset, record_count=count) for dataset, count in sorted(populations.items()))
            checks.extend(common.checks("trusted_context", {"queries": sum(populations.values()),
                          "token_budget": args.token_budget, "verified_only": True}))


def build(root, args):
    development_id = getattr(args, "development_id", None)
    outputs, query_path, index_inputs, index_outputs, provenance_outputs, rerank_outputs = paths_for(development_id)
    require((args.profile == "legacy") == (development_id is None), "Improved profile requires --development-id; legacy profile must omit it")
    paths = [query_path, index_outputs[0], *rerank_outputs, *provenance_outputs]
    before = common.snapshot(root, paths)
    rerank_saved = common.completed(root, rerank_outputs[2], rerank_outputs[3])
    rerank_profile = rerank_saved.get("profile", "legacy")
    require(rerank_saved.get("development_id", "") == (development_id or ""), "Reranking development-id mismatch")
    expected_rerank_profile = "task_relevance" if args.profile == "evidence_diverse" else "legacy"
    require(rerank_profile == expected_rerank_profile, "Context profile must match the completed reranking profile")
    verified = provenance.load_verified(root,development_id)
    chunks = common.unique(index.validated_chunks(root, inputs=index_inputs, outputs=index_outputs), "chunk_id")
    rerank_header = reranking_header(rerank_profile)
    policy = index.v1.load_token_policy(root)
    summary, checks = [], []
    records = stream_records(root, args, query_path, rerank_outputs[0], rerank_header,
                             chunks, verified, policy, paths, before, summary, checks)
    config = dict(stage="TRUSTED_CONTEXT_CONSTRUCTION", top_n=args.top_n, token_budget=args.token_budget, max_chunks_per_document=args.max_chunks_per_document,
                  token_accounting="local BGE tokenizer with special tokens; downstream LLM tokenizer budget must be checked separately", tokenizer_identity_json=common.json_text(index.v1.model_identity(root)),
                  selection_policy="rerank order; skip whole chunk if over budget; no truncation", input_metadata_json=common.json_text(before))
    if development_id: config["development_id"] = development_id
    if args.profile == "evidence_diverse":
        config.update(selection_policy="decision-claim-first class-balanced exposure; contextual hypothesis fallback; fill by rerank order; no evidence-status judgment",
            profile=args.profile,diversity_threshold=args.diversity_threshold)
    return [(outputs[0], HEADER, records), (outputs[1], common.SUMMARY_HEADER, summary), (outputs[2], common.CONFIG_HEADER, common.config_rows(config)),
            (outputs[3], common.corpus.foundation.CHECKLIST_HEADER, checks)]


def run(args):
    """Consume lazy tables in dry mode; publish only after full validation."""
    try:
        root = args.project_root.resolve(strict=True)
        if args.execute:
            common.fresh_outputs(root, paths_for(args.development_id)[0])
        tables = build(root, args)
        with closing(tables[0][2]):
            if not args.execute:
                for _, _, records in tables:
                    for _ in records:
                        pass
            else:
                with TemporaryDirectory(prefix="rag-context-outputs-") as scratch:
                    staged = []
                    for ordinal, (_, header, records) in enumerate(tables):
                        path = Path(scratch)/f"{ordinal}.csv"
                        with path.open("x", encoding="utf-8", newline="") as stream:
                            writer = csv.DictWriter(stream, fieldnames=header)
                            writer.writeheader()
                            writer.writerows(records)
                        staged.append(path)
                    targets = common.fresh_outputs(root, [name for name, _, _ in tables])
                    # Checklist is last: interrupted publication remains incomplete,
                    # and every existing file still blocks a subsequent overwrite.
                    for path, target in zip(staged, targets):
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with path.open("rb") as source, target.open("xb") as destination:
                            shutil.copyfileobj(source, destination)
        print("TRUSTED_CONTEXT_CONSTRUCTION=" + ("PASS" if args.execute else "READY_FOR_EXECUTION"))
        return 0
    except (OSError, ValueError, KeyError, TypeError, csv.Error, ImportError, sqlite3.Error) as exc:
        print("TRUSTED_CONTEXT_CONSTRUCTION=BLOCKED\n" + str(exc))
        return 2


def main(argv=None):
    parser = common.parser(__doc__)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--token-budget", type=int, default=2048)
    parser.add_argument("--max-chunks-per-document", type=int, default=0)
    parser.add_argument("--profile", choices=PROFILES, default="legacy")
    parser.add_argument("--diversity-threshold", type=float)
    parser.add_argument("--development-id")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
