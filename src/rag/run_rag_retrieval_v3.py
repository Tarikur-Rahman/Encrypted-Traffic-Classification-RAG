"""Semantic traffic-profile retrieval from Index V2 only; dry by default."""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from contextlib import ExitStack

from src.rag import run_rag_retrieval as v2
from src.rag import build_evidence_index_v2 as index
from src.rag import phase2_common as common

require = common.require
EXPERIMENT = "phase2_dev_retrieval_v3"
BASE = "results/rag/" + EXPERIMENT + "/"
OUTPUTS = tuple(BASE + name + ".csv" for name in ("query_records", "retrieval_results", "retrieval_summary", "retrieval_diagnostics", "experiment_configuration")) + ("metadata/rag/rag_retrieval_v3_validation_checklist.csv",)
DIAGNOSTICS_HEADER = (*v2.DIAGNOSTICS_HEADER, "query_count_by_method", "rank_min", "rank_max", "duplicate_query_method_rank_count", "unknown_query_ids", "test_query_count", "non_development_partition_count", "unique_retrieval_query_texts", "top1_unique_chunks", "top1_dominant_share", "unique_chunks_by_rank_json")
PROFILES = ("legacy", "hypothesis_balanced")
INTENT_HEADER = (*v2.RESULT_HEADER, "retrieval_intents_json", "intent_scores_json")


def output_paths(development_id=None):
    if development_id is None:
        return OUTPUTS
    index.corpus_v2.validate_development_id(development_id)
    base = f"results/rag/{development_id}/stage04_retrieval/"
    return tuple(base + name + ".csv" for name in ("query_records", "retrieval_results", "retrieval_summary", "retrieval_diagnostics", "experiment_configuration")) + (f"metadata/rag/{development_id}/stage04_retrieval/validation_checklist.csv",)


def hypotheses(query, *, source_split="validation"):
    common.validate_query(query, source_split=source_split)
    if source_split == "validation":
        require("-VAL-" in query["query_id"] and "-TEST-" not in query["query_id"], "Non-validation query identity")
    labels = common.CLASSES[query["source_dataset"]]
    predicted = query["predicted_class_label"]
    return [predicted, *[label for label in labels if label != predicted]]


def observation(query):
    text = query["retrieval_query_text"]
    marker = "Observed traffic profile relative to the training distribution: "
    if marker in text:
        return text.split(marker,1)[1].split(". Retrieve",1)[0]
    # Supports already-neutral, explicitly supplied observable profiles.
    return re.sub(r"Predicted traffic class:\s*[^.]+\.", "", text)


def retrieval_intents(query, *, source_split="validation"):
    safe_query = {key:query[key] for key in v2.QUERY_FIELDS if key in query}
    return [(label,dict(safe_query,retrieval_query_text=
        "Encrypted network traffic observation. Observed traffic profile: "+observation(query)+
        ". Investigate the "+label+" hypothesis using authentic traffic evidence; the hypothesis is not an evidence conclusion."))
        for label in hypotheses(query, source_split=source_split)]


def merge_intents(query, discoveries, *, source_split="validation"):
    """Round-robin rank merge; retain original scores separately for every intent."""
    labels = hypotheses(query, source_split=source_split)
    require(set(discoveries) == set(labels), "Incomplete hypothesis retrieval")
    output = []
    for method in v2.METHODS:
        groups, audit, representative = {}, defaultdict(list), {}
        for label in labels:
            rows = sorted((r for r in discoveries[label] if r["retrieval_method"] == method),key=lambda r:(int(r["rank"]),r["chunk_id"]))
            require(len(rows) == len({r["chunk_id"] for r in rows}) == 10 and
                [int(r["rank"]) for r in rows] == list(range(1,11)), "Invalid intent candidates")
            groups[label] = rows
            for row in rows:
                identity = row["chunk_id"]
                require(row["query_id"] == query["query_id"] and row["source_dataset"] == query["source_dataset"], "Intent query differs")
                require(identity not in representative or representative[identity]["document_id"] == row["document_id"], "Intent document mismatch")
                representative.setdefault(identity,row)
                audit[identity].append(dict(hypothesis=label,**{k:row.get(k,"") for k in
                    ("rank","retrieval_score","tfidf_score","dense_score","tfidf_rank","dense_rank","rrf_score")}))
        chosen = []
        for position in range(10):
            for label in labels:
                row = groups[label][position]
                if row["chunk_id"] not in chosen:
                    chosen.append(row["chunk_id"])
        for rank,identity in enumerate(chosen[:10],1):
            output.append(dict(representative[identity],rank=rank,
                retrieval_intents_json=common.json_text([r["hypothesis"] for r in audit[identity]]),
                intent_scores_json=common.json_text(audit[identity])))
    return output


def balanced_batch(evidence, batch, model, policy, *, source_split="validation"):
    intents = {q["query_id"]:retrieval_intents(q, source_split=source_split) for q in batch}
    discovered = {q["query_id"]:{} for q in batch}
    # At most the original batch size is encoded at once, not four batches' text.
    for position in range(max(map(len,intents.values()))):
        rows, labels, counts = [], {}, {}
        for query in batch:
            identity = query["query_id"]
            if position >= len(intents[identity]):
                continue
            label,row = intents[identity][position]
            rows.append(row)
            labels[identity] = label
            counts[identity] = policy.count(v2.dense_text(row["retrieval_query_text"]))
            require(0 < counts[identity] <= v2.TOKEN_LIMIT, "Hypothesis query exceeds local token limit")
        for row,results in v2.retrieve_batch(evidence,rows,model,counts):
            discovered[row["query_id"]][labels[row["query_id"]]] = results
    for query in batch:
        yield query,merge_intents(query,discovered[query["query_id"]], source_split=source_split)


def load_index(root, policy, development_id=None):
    import numpy as np
    import joblib
    from scipy.sparse import load_npz
    from sklearn.feature_extraction.text import TfidfVectorizer
    index_outputs = index.output_paths(development_id)
    index_inputs = index.input_paths(development_id)
    v2.check_pass(root, index_outputs[6])
    config = v2.config_table(root, index_outputs[5])
    require(config.get("development_id", "") == (development_id or ""), "Index development-id mismatch")
    expected = dict(index_revision="V2", quality_policy="KEEP_ONLY", chunk_token_limit="512", embedding_dimension="384", embedding_normalized="YES", trusted_only="YES", token_truncation_allowed="NO")
    require(all(config.get(k) == v for k, v in expected.items()), "Not a validated Index V2")
    require(all(config.get(k) == v for k, v in index.v1.model_identity(root).items()), "Local model identity changed")
    require(common.parse_json(config["input_metadata_json"]) == common.snapshot(root, index_inputs), "Index corpus inputs changed")
    chunks = index.validated_chunks(root, policy, index_inputs, index_outputs)
    require(len(chunks) >= v2.TOP_K and len(chunks) == int(config["chunk_count"]), "Invalid chunk population")
    _, mapping = common.corpus.read_table(root, index_outputs[4], index.v1.MAPPING_HEADER)
    require(mapping == [dict(embedding_row=str(i), chunk_id=r["chunk_id"], document_id=r["document_id"]) for i, r in enumerate(chunks)], "Dense mapping differs")
    vectorizer = joblib.load(common.corpus.local_path(root, index_outputs[1]))
    require(type(vectorizer) is TfidfVectorizer and all(vectorizer.get_params()[k] == v for k, v in index.v1.TFIDF.items()), "Invalid lexical artifact")
    lexical = load_npz(common.corpus.local_path(root, index_outputs[2])).tocsr()
    dense = np.load(common.corpus.local_path(root, index_outputs[3]), mmap_mode="r", allow_pickle=False)
    require(lexical.shape == (len(chunks), int(config["lexical_feature_count"])) and np.isfinite(lexical.data).all(), "Invalid lexical dimensions")
    difference = lexical - vectorizer.transform([r["chunk_text"] for r in chunks])
    require(difference.nnz == 0 or np.max(np.abs(difference.data)) <= 1e-12, "Lexical row/content mismatch")
    require(dense.shape == (len(chunks), 384) and dense.dtype == np.float32 and np.isfinite(dense).all() and np.allclose(np.linalg.norm(dense, axis=1), 1, atol=1e-5), "Invalid dense matrix")
    return v2.EvidenceIndex(chunks, vectorizer, lexical, dense, config)


class Diagnostics:
    def __init__(self):
        self.base = v2.Diagnostics()
        self.groups = {}
        self.seen = set()

    def add(self, query, results):
        identity = query["query_id"]
        require(identity not in self.seen, "Duplicate query results")
        self.seen.add(identity)
        common.validate_query(query)
        for method in v2.METHODS:
            selected = [r for r in results if r["retrieval_method"] == method]
            require(sorted(int(r["rank"]) for r in selected) == list(range(1, 11)), "Duplicate/missing/out-of-range ranks")
            require(len({r["chunk_id"] for r in selected}) == 10, "Duplicate candidate")
        require(len(results) == 30 and all(r["query_id"] == identity and r["source_dataset"] == query["source_dataset"] for r in results), "Unknown query/dataset/method")
        self.base.add(query, results)
        for dataset, method in [("", ""), (query["source_dataset"], ""), *[(d, m) for d in ("", query["source_dataset"]) for m in v2.METHODS]]:
            group = self.groups.setdefault((dataset, method), dict(texts=set(), top=Counter(), ranks=defaultdict(set), query_methods=Counter()))
            group["texts"].add(query["retrieval_query_text"])
            for row in results:
                if method and row["retrieval_method"] != method:
                    continue
                rank = int(row["rank"])
                group["ranks"][rank].add(row["chunk_id"])
                if rank == 1:
                    group["top"][row["chunk_id"]] += 1
                    group["query_methods"][row["retrieval_method"]] += 1

    def rows(self):
        for row in self.base.rows():
            group = self.groups[(row["source_dataset"], row["retrieval_method"])]
            total = sum(group["top"].values())
            yield dict(row, query_count_by_method=common.json_text(group["query_methods"]), rank_min=min(group["ranks"]), rank_max=max(group["ranks"]),
                       duplicate_query_method_rank_count=0, unknown_query_ids=0, test_query_count=0, non_development_partition_count=0,
                       unique_retrieval_query_texts=len(group["texts"]), top1_unique_chunks=len(group["top"]), top1_dominant_share=max(group["top"].values())/total,
                       unique_chunks_by_rank_json=common.json_text({rank: len(ids) for rank, ids in group["ranks"].items()}))


def preflight(root, profile="legacy", development_id=None):
    require(profile in PROFILES, "Unknown retrieval profile")
    require((profile == "legacy") == (development_id is None), "Improved profile requires --development-id; legacy profile must omit it")
    index_inputs, index_outputs = index.input_paths(development_id), index.output_paths(development_id)
    # Preserve the complete legacy snapshot. The common metadata path contract
    # does not include namespaced executable/model artifacts, whose loaded
    # shape/content invariants are checked separately.
    index_snapshot = index_outputs if development_id is None else (index_outputs[0], index_outputs[5], index_outputs[6], index_outputs[7])
    paths = (*index_inputs, *index_snapshot, *v2.QUERY_SOURCES,
             "results/rag/interface/interface_validation_checklist.csv",
             *(f"metadata/traffic/{prefix}_experiment_closure.csv" for _, prefix, _ in v2.DATASETS))
    before = common.snapshot(root, paths)
    v2.protect_phase_i(root)
    policy = index.v1.load_token_policy(root)
    require(policy.limit == 512, "Wrong local tokenizer")
    evidence = load_index(root, policy, development_id)
    counts, evaluation = v2.query_preflight(root, policy)
    populations = Counter()
    for row in v2.query_rows(root):
        common.validate_query(row)
        if profile != "legacy":
            for _,intent in retrieval_intents(row):
                require(0 < policy.count(v2.dense_text(intent["retrieval_query_text"])) <= v2.TOKEN_LIMIT,
                    "Hypothesis query exceeds local token limit: "+row["query_id"])
        populations[row["source_dataset"]] += 1
    require(dict(populations) == common.QUERY_COUNTS, "Expected 128724 development queries")
    require(before == common.snapshot(root, paths), "Inputs changed during preflight")
    config = dict(experiment_id=EXPERIMENT, configuration_name=v2.CONFIGURATION_NAME, stage="RAG_RETRIEVAL", index_revision="V2",
                  query_count=len(counts), query_population_json=common.json_text(populations), retrieval_query_schema=v2.RETRIEVAL_QUERY_SCHEMA,
                  semantic_profile_top_features=12, semantic_bin_thresholds=v2.SEMANTIC_BIN_THRESHOLDS, rag_partition="development_validation",
                  top_k=10, component_depth=50, rrf_k=60, dense_batch_size=256, dense_query_prefix=v2.PREFIX, dense_device=v2.dense_device(),
                  input_metadata_json=common.json_text(before), index_configuration_json=common.json_text(evidence.configuration),
                  relevance_judgments="NOT_AVAILABLE", test_access="NO", network_used="NO", llm_used="NO", next_stage="RETRIEVAL_V3_SEMANTIC_DIAGNOSTIC")
    if profile != "legacy":
        config.update(retrieval_profile=profile,intent_merge="rank round-robin, predicted then canonical alternatives; chunk-ID dedup",
            candidate_semantics="discovery intent only; never evidence status",intent_scores="original per-intent scores preserved in intent_scores_json")
        evaluation = ()  # New development outputs contain no evaluation labels.
        config["development_id"] = development_id
    plan = (evidence, policy, counts, evaluation, config, paths, before)
    return (*plan, development_id) if development_id is not None else plan


def execute(root, evidence, policy, counts, evaluation, config, paths, before, development_id=None):
    targets = common.fresh_outputs(root, output_paths(development_id))
    model = v2.load_query_encoder(root, policy)
    require(str(model.device).split(":")[0] == config["dense_device"], "Device changed")
    summary, diagnostics = v2.Summary(), Diagnostics()
    known = {r["chunk_id"]: r["document_id"] for r in evidence.chunks}
    with ExitStack() as stack:
        writers = []
        improved = config.get("retrieval_profile", "legacy") == "hypothesis_balanced"
        for path, header in zip(targets[:2], ((*v2.QUERY_HEADER, *evaluation), INTENT_HEADER if improved else v2.RESULT_HEADER)):
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = csv.DictWriter(stack.enter_context(path.open("x", encoding="utf-8", newline="")), fieldnames=header)
            writer.writeheader()
            writers.append(writer)
        for batch in v2.batches(v2.query_rows(root), 256):
            safe = [{k: row[k] for k in v2.QUERY_FIELDS} for row in batch]
            annotations = {row["query_id"]: {k: row[k] for k in evaluation if k in row} for row in batch}
            retrieved = balanced_batch(evidence,safe,model,policy) if improved else v2.retrieve_batch(evidence,safe,model,counts)
            for query, results in retrieved:
                require(query["query_id"] in counts, "Unknown query")
                for row in results:
                    require(known.get(row["chunk_id"]) == row["document_id"], "Unknown chunk/document")
                    row["experiment_id"] = EXPERIMENT
                diagnostics.add(query, results)
                summary.add(query, results)
                writers[0].writerow(dict(query, experiment_id=EXPERIMENT, configuration_name=v2.CONFIGURATION_NAME,
                                        dense_query_token_count=counts[query["query_id"]], retrieval_status="PASS", **annotations[query["query_id"]]))
                writers[1].writerows(results)
    require(diagnostics.seen == set(counts) and before == common.snapshot(root, paths), "Incomplete/changed inputs")
    require(all(evidence.configuration.get(k) == v for k, v in index.v1.model_identity(root).items()), "Local encoder changed")
    for target, header, records in ((targets[2], v2.SUMMARY_HEADER, summary.rows()), (targets[3], DIAGNOSTICS_HEADER, diagnostics.rows()),
                                    (targets[4], common.CONFIG_HEADER, common.config_rows(config)),
                                    (targets[5], common.corpus.foundation.CHECKLIST_HEADER, common.checks("retrieval_v3", {"coverage": len(counts), "keep_index_v2": True, "development_only": True, "no_relevance_claims": True}))):
        common.corpus.write_table(target, header, records, mode="x")


def main(argv=None):
    parser = common.parser(__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="legacy")
    parser.add_argument("--development-id")
    args = parser.parse_args(argv)
    try:
        root = args.project_root.resolve(strict=True)
        plan = preflight(root) if args.profile == "legacy" and args.development_id is None else preflight(root,args.profile,args.development_id)
        if args.execute:
            execute(root, *plan)
        print("RAG_RETRIEVAL_V3=" + ("PASS\nNEXT_STAGE=RETRIEVAL_V3_SEMANTIC_DIAGNOSTIC" if args.execute else "READY_FOR_EXECUTION"))
        return 0
    except (OSError, ValueError, KeyError, ImportError, RuntimeError, csv.Error) as exc:
        print("RAG_RETRIEVAL_V3=BLOCKED\n" + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
