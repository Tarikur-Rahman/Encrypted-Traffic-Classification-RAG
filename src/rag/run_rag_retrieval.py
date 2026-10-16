"""Frozen development-validation candidate retrieval; preflight by default.

No relevance judgments, evidence-state decisions, provenance verification,
provenance reranking, context construction, traffic inference or network access.
Only canonical project-generated TF-IDF artifacts are deserialized. CUDA is
used when available; passage embeddings are never regenerated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import Counter
from array import array
from dataclasses import dataclass
from pathlib import Path

from src.rag import build_evidence_index as index
from src.rag import build_evidence_corpus as corpus
from src.rag import phase2_foundation as foundation


EXPERIMENT_ID = "phase2_dev_retrieval_v2"
CONFIGURATION_NAME = "semantic_traffic_profile_lexical_dense_rrf_top10"
RETRIEVAL_QUERY_SCHEMA = "semantic_traffic_profile_v2"
SEMANTIC_PROFILE_TOP_FEATURES = 12
NONORDINAL = ("protocol", "ip_version", "ip version")
SEMANTIC_BIN_THRESHOLDS = "z>=3: extremely high; 2<=z<3: very high; 1<=z<2: high; 0.5<=z<1: moderately high; -0.5<z<0.5: typical; -1<z<=-0.5: moderately low; -2<z<=-1: low; -3<z<=-2: very low; z<=-3: extremely low"
METHODS = ("tfidf", "dense_bge", "hybrid_rrf")
PREFIX = "Represent this sentence for searching relevant passages:"
TOP_K, DEPTH, RRF_K = 10, 50, 60
QUERY_BATCH_SIZE, DENSE_BATCH_SIZE = 256, 256
EXPECTED_CHUNKS, DIMENSION, TOKEN_LIMIT = 919, 384, 512
QUERY_SOURCES = ("results/rag/interface/ustc_validation_interface.csv", "results/rag/interface/cic_validation_interface.csv")
DATASETS = (("USTC-TFC2016", "ustc", "20260919T183945Z"), ("CIC-Darknet2020", "cic", "20260920T145831Z"))
BASE = "results/rag/" + EXPERIMENT_ID + "/"
OUTPUTS = (BASE + "query_records.csv", BASE + "retrieval_results.csv", BASE + "retrieval_summary.csv", BASE + "experiment_configuration.csv", "metadata/rag/rag_retrieval_v2_validation_checklist.csv", BASE + "retrieval_diagnostics.csv")
EVALUATION = ("true_class_id", "true_class_label", "classifier_correct")
OPTIONAL_AUDIT_FIELDS = (*EVALUATION, "prob_class_0", "prob_class_1", "prob_class_2", "prob_class_3")
SOURCE_FIELDS = tuple("query_id source_dataset source_split rag_partition classifier_model predicted_class_id predicted_class_label confidence class_probabilities_json query_text".split())
PROFILE_FIELDS = ("retrieval_query_text", "semantic_profile_top_features", "semantic_profile_feature_names_json", "semantic_profile_bins_json")
QUERY_FIELDS = (*SOURCE_FIELDS, *PROFILE_FIELDS)
QUERY_HEADER = ("experiment_id", "configuration_name", *QUERY_FIELDS, "dense_query_token_count", "retrieval_status")
RESULT_HEADER = tuple("experiment_id configuration_name retrieval_method query_id source_dataset rank chunk_id document_id retrieval_score tfidf_score dense_score tfidf_rank dense_rank rrf_score".split())
SUMMARY_HEADER = tuple("summary_level source_dataset retrieval_method query_count queries_with_results zero_result_queries returned_rows mean_top1_score median_top1_score mean_candidate_score median_candidate_score".split())
DIAGNOSTICS_HEADER = tuple("summary_level source_dataset retrieval_method query_count retrieval_rows unique_retrieved_chunks unique_retrieved_documents unique_top10_patterns most_common_top10_pattern_count most_common_top10_pattern_share most_frequent_chunk_id most_frequent_chunk_count most_frequent_chunk_share".split())
require = corpus.require


class QueryTokenViolation(ValueError):
    pass


class QueryTokenCounts(dict):
    unique_retrieval_query_texts = 0


def readable_feature(name):
    aliases = {"src2dst": "source to destination", "dst2src": "destination to source", "fwd": "forward", "bwd": "backward", "piat": "packet inter-arrival time", "iat": "inter-arrival time", "stddev": "standard deviation", "std": "standard deviation", "avg": "average", "pkts": "packets", "pkt": "packet", "ps": "packet size"}
    aliases.update({flag.lower(): flag for flag in ("SYN", "ACK", "PSH", "RST", "FIN")})
    text = re.sub(r"\b[a-z0-9]+\b", lambda match: aliases.get(match.group(), match.group()), name.replace("_", " ").lower())
    return " ".join(text.split())


def semantic_bin(z):
    require(isinstance(z, (int, float)) and not isinstance(z, bool) and math.isfinite(z), "Nonfinite/nonnumeric standardized value")
    if z >= 3: return "extremely high"
    if z >= 2: return "very high"
    if z >= 1: return "high"
    if z >= 0.5: return "moderately high"
    if z > -0.5: return "typical"
    if z > -1: return "moderately low"
    if z > -2: return "low"
    if z > -3: return "very low"
    return "extremely low"


def unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate traffic representation field")
        result[key] = value
    return result


def semantic_profile(source_dataset, predicted_class_label, representation_json):
    """No ground truth, query identity, confidence or source query is accepted."""
    value = json.loads(representation_json, object_pairs_hook=unique_json_object)
    require(isinstance(value, dict) and isinstance(value.get("features"), dict), "Traffic representation features required")
    features = value["features"]
    expected = {"USTC-TFC2016": 50, "CIC-Darknet2020": 62}.get(source_dataset)
    require(expected is not None and type(value.get("predictor_count")) is int and value["predictor_count"] == len(features) == expected, "Predictor count differs from canonical interface")
    forbidden = {re.sub(r"[^a-z0-9]", "", name.lower()) for name in (*foundation.FORBIDDEN_IDENTIFIERS, *EVALUATION, "query_id", "source_row_index", "source_dataset", "classifier_run_id")}
    eligible = []
    for name, z in features.items():
        require(isinstance(name, str) and name.strip() and re.fullmatch(r"[A-Za-z0-9_ /-]+", name) is not None, "Invalid predictor name")
        require(re.sub(r"[^a-z0-9]", "", name.lower()) not in forbidden, "Forbidden predictor identifier")
        bin_name = semantic_bin(z)
        if " ".join(name.lower().replace("_", " ").split()) in {"protocol", "ip version"}:
            continue
        eligible.append((name, z, readable_feature(name), bin_name))
    eligible.sort(key=lambda item: (-abs(item[1]), item[2].casefold(), item[0].casefold(), item[0]))
    selected = eligible[:SEMANTIC_PROFILE_TOP_FEATURES]
    require(selected, "No ordinal traffic features available")
    classes = {"benign": "benign", "malware": "malware", "Tor": "Tor", "VPN": "VPN", "Non-Tor": "non-Tor", "NonVPN": "non-VPN"}
    require(predicted_class_label in classes, "Unknown canonical predicted class")
    profile = "; ".join(f"{readable} is {bin_name}" for _, _, readable, bin_name in selected)
    text = ("Encrypted network traffic observation. Predicted traffic class: " + classes[predicted_class_label] +
            ". Observed traffic profile relative to the training distribution: " + profile +
            ". Retrieve trusted cybersecurity evidence relevant to the predicted class and these observed traffic characteristics.")
    return dict(retrieval_query_text=text, semantic_profile_top_features=SEMANTIC_PROFILE_TOP_FEATURES,
                semantic_profile_feature_names_json=json.dumps([item[0] for item in selected], ensure_ascii=False),
                semantic_profile_bins_json=json.dumps([item[3] for item in selected]))


def config_table(root, path):
    _, rows = corpus.read_table(root, path, index.CONFIG_HEADER)
    require(rows and len({r["field"] for r in rows}) == len(rows), "Invalid configuration keys")
    return {r["field"]: r["value"] for r in rows}


def check_pass(root, path):
    _, rows = corpus.read_table(root, path, foundation.CHECKLIST_HEADER)
    require(rows and all(r["status"] == "PASS" for r in rows), "Required checklist is not PASS: " + path)


def protect_phase_i(root):
    for dataset, prefix, run in DATASETS:
        path = f"metadata/traffic/{prefix}_experiment_closure.csv"
        _, rows = corpus.read_table(root, path)
        require(len(rows) == 1, "Ambiguous Phase-I closure")
        expected = dict(dataset=dataset, authoritative_run=run, stage_status="CLOSED_FROZEN", integrity_audit_status="PASS", finalization_status="PASS", frozen_for_retraining="YES", frozen_for_test_reuse="YES")
        if prefix == "cic":
            expected["phase_i_status"] = "COMPLETE"
        require(all(rows[0].get(k) == v for k, v in expected.items()), "Phase-I protection failed: " + dataset)
    check_pass(root, "results/rag/interface/interface_validation_checklist.csv")


def dense_text(query_text):
    require(not query_text.startswith(PREFIX), "Canonical query must not already contain the dense prefix")
    return PREFIX + " " + query_text


def query_rows(root):
    """Project only canonical query/evaluation fields; never read traffic data."""
    corpus.configure_csv_field_limit()
    for path, (dataset, _, run) in zip(QUERY_SOURCES, DATASETS):
        with corpus.local_path(root, path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            header = reader.fieldnames or []
            require(len(header) == len(set(header)) and set(SOURCE_FIELDS) | {"classifier_run_id", "traffic_representation_json"} <= set(header), "Invalid interface schema")
            require(not {v.lower() for v in foundation.FORBIDDEN_IDENTIFIERS} & {v.lower() for v in header}, "Forbidden endpoint column in interface")
            count = 0
            for row in reader:
                require(None not in row and all(v is not None for v in row.values()), "Malformed interface record")
                require(row["source_dataset"] == dataset and row["classifier_run_id"] == run and row["classifier_model"] == "lightgbm", "Frozen interface classifier identity differs")
                require(row["source_split"] == "validation" and row["rag_partition"] == "development_validation", "Only development_validation/validation queries allowed")
                require(row["query_id"].strip() and row["query_text"].strip(), "Blank query identity/text")
                lower = row["query_text"].lower()
                require(not any(k.lower() in lower for k in (*foundation.FORBIDDEN_IDENTIFIERS, *EVALUATION)), "Forbidden identifier or evaluation field in query text")
                profile = semantic_profile(row["source_dataset"], row["predicted_class_label"], row["traffic_representation_json"])
                count += 1
                yield dict({key: row[key] for key in (*SOURCE_FIELDS, *OPTIONAL_AUDIT_FIELDS) if key in row}, **profile)
            require(count > 0, "Empty development interface: " + dataset)


def query_preflight(root, policy):
    seen, token_counts, evaluation, offending = set(), QueryTokenCounts(), set(), []
    unique_texts = set()
    for row in query_rows(root):
        identity = row["query_id"]
        require(identity not in seen, "Global query_id collision: " + identity)
        seen.add(identity)
        evaluation.update(k for k in OPTIONAL_AUDIT_FIELDS if k in row)
        unique_texts.add(row["retrieval_query_text"])
        count = policy.count(dense_text(row["retrieval_query_text"]))
        token_counts[identity] = count
        if not 0 < count <= TOKEN_LIMIT:
            offending.append(identity)
    if offending:
        raise QueryTokenViolation("Offending query IDs: " + ", ".join(offending))
    token_counts.unique_retrieval_query_texts = len(unique_texts)
    return token_counts, tuple(k for k in OPTIONAL_AUDIT_FIELDS if k in evaluation)


@dataclass
class EvidenceIndex:
    chunks: list
    vectorizer: object
    lexical: object
    dense: object
    configuration: dict
    passage_tensor: object = None
    passage_order: object = None


def load_index(root):
    import numpy as np
    import joblib
    from scipy.sparse import load_npz
    from sklearn.feature_extraction.text import TfidfVectorizer
    check_pass(root, index.OUTPUTS[6])
    config = config_table(root, index.OUTPUTS[5])
    expected = dict(authorized_document_count="10", chunk_count=str(EXPECTED_CHUNKS), embedding_dimension=str(DIMENSION), embedding_model_max_seq_length=str(TOKEN_LIMIT), embedding_normalized="YES", dense_dtype="float32", trusted_only="YES", adversarial_included="NO", token_safety_enforced="YES", token_truncation_allowed="NO")
    require(all(config.get(k) == v for k, v in expected.items()), "Canonical index configuration differs")
    model = index.model_identity(root)
    require(all(config.get(k) == v for k, v in model.items()), "Local encoder identity differs from frozen index")
    documents = {r["document_id"]: r for r in index.load_corpus(root)}
    _, chunks = corpus.read_table(root, index.OUTPUTS[0], index.CHUNK_HEADER)
    _, mapping = corpus.read_table(root, index.OUTPUTS[4], index.MAPPING_HEADER)
    require(len(chunks) == EXPECTED_CHUNKS and len({r["chunk_id"] for r in chunks}) == len(chunks), "Invalid canonical chunk identity/count")
    require(mapping == [dict(embedding_row=str(i), chunk_id=r["chunk_id"], document_id=r["document_id"]) for i, r in enumerate(chunks)], "Dense row mapping differs from chunk CSV")
    require(set(documents) == {r["document_id"] for r in chunks}, "Indexed document set differs")
    orders = {}
    for row in chunks:
        doc = documents.get(row["document_id"])
        require(doc is not None and all(row[k] == doc[k] for k in index.METADATA), "Rejected/adversarial or altered chunk metadata")
        order = orders.get(row["document_id"], 0)
        require(row["chunk_order"] == str(order) and row["chunk_id"] == f"{row['document_id']}-CHUNK-{order:04d}", "Invalid contiguous chunk order")
        orders[row["document_id"]] = order + 1
        require(row["chunk_text"].strip() and row["chunk_text"] in doc["normalized_text"] and 0 < int(row["chunk_token_count"]) <= TOKEN_LIMIT, "Invalid canonical chunk text/token count")
    # Only this fixed, locally generated artifact is ever deserialized.
    vectorizer = joblib.load(corpus.local_path(root, index.OUTPUTS[1]))
    require(type(vectorizer) is TfidfVectorizer, "Expected project-generated TF-IDF vectorizer")
    require(all(vectorizer.get_params()[k] == v for k, v in index.TFIDF.items()), "Frozen lexical parameters differ")
    lexical = load_npz(corpus.local_path(root, index.OUTPUTS[2])).tocsr()
    dense = np.load(corpus.local_path(root, index.OUTPUTS[3]), mmap_mode="r", allow_pickle=False)
    require(lexical.shape == (len(chunks), int(config["lexical_feature_count"])) and np.isfinite(lexical.data).all(), "Lexical dimensions/values invalid")
    # Transform is read-only: independently establish lexical row order using
    # the saved vocabulary, without fitting or performing query retrieval.
    expected_lexical = vectorizer.transform([r["chunk_text"] for r in chunks]).tocsr()
    difference = lexical - expected_lexical
    require(difference.nnz == 0 or np.max(np.abs(difference.data)) <= 1e-12, "TF-IDF row mapping/content differs")
    require(dense.shape == (len(chunks), DIMENSION) and dense.dtype == np.float32 and np.isfinite(dense).all(), "Invalid dense matrix")
    require(np.allclose(np.linalg.norm(dense, axis=1), 1, rtol=0, atol=1e-5), "Passage vectors are not unit normalized")
    return EvidenceIndex(chunks, vectorizer, lexical, dense, config)


def batches(rows, size=QUERY_BATCH_SIZE):
    require(isinstance(size, int) and size > 0, "Invalid batch size")
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def ranked(scores, chunks, depth=DEPTH):
    require(len(scores) == len(chunks) and all(math.isfinite(float(s)) for s in scores), "Invalid similarity scores")
    order = sorted(range(len(chunks)), key=lambda i: (-float(scores[i]), chunks[i]["chunk_id"]))[:depth]
    return [(i, float(scores[i]), rank) for rank, i in enumerate(order, 1)]


def rrf(lexical, dense, chunks):
    first = {i: rank for i, _, rank in lexical}
    second = {i: rank for i, _, rank in dense}
    union = set(first) | set(second)
    scores = {i: (1/(RRF_K+first[i]) if i in first else 0) + (1/(RRF_K+second[i]) if i in second else 0) for i in union}
    order = sorted(union, key=lambda i: (-scores[i], min(first.get(i, math.inf), second.get(i, math.inf)), chunks[i]["chunk_id"]))[:TOP_K]
    return [(i, scores[i], rank) for rank, i in enumerate(order, 1)]


def result_rows(query, lexical, dense, chunks):
    lookup_l = {i: (score, rank) for i, score, rank in lexical}
    lookup_d = {i: (score, rank) for i, score, rank in dense}
    output = []
    for method, candidates in (("tfidf", lexical[:TOP_K]), ("dense_bge", dense[:TOP_K]), ("hybrid_rrf", rrf(lexical, dense, chunks))):
        for i, score, rank in candidates:
            row = dict.fromkeys(RESULT_HEADER, "")
            row.update(experiment_id=EXPERIMENT_ID, configuration_name=CONFIGURATION_NAME, retrieval_method=method,
                       query_id=query["query_id"], source_dataset=query["source_dataset"], rank=rank,
                       chunk_id=chunks[i]["chunk_id"], document_id=chunks[i]["document_id"], retrieval_score=score)
            if method in {"tfidf", "hybrid_rrf"} and i in lookup_l:
                row.update(tfidf_score=lookup_l[i][0], tfidf_rank=lookup_l[i][1])
            if method in {"dense_bge", "hybrid_rrf"} and i in lookup_d:
                row.update(dense_score=lookup_d[i][0], dense_rank=lookup_d[i][1])
            if method == "hybrid_rrf":
                row["rrf_score"] = score
            output.append(row)
    return output


def dense_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_dense_passages(evidence, device):
    """Cache one float32 passage tensor in chunk-ID order on the chosen device."""
    import torch
    if evidence.passage_tensor is not None:
        require(evidence.passage_tensor.device.type == device, "Cached passage device differs")
        return
    order = sorted(range(len(evidence.chunks)), key=lambda i: evidence.chunks[i]["chunk_id"])
    evidence.passage_order = order
    evidence.passage_tensor = torch.as_tensor(evidence.dense[order].copy(), dtype=torch.float32, device=device)
    require(bool(torch.isfinite(evidence.passage_tensor).all()), "Invalid passage tensor")


def load_query_encoder(root, policy):
    index.offline_mode()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = dense_device()
    model = SentenceTransformer(str((root/index.MODEL_DIRECTORY).resolve()), device=device, local_files_only=True, trust_remote_code=False)
    model.to(device)
    model.eval()
    require(str(model.device).split(":")[0] == device, "Actual dense device differs from selected device")
    require(model.max_seq_length == TOKEN_LIMIT and model.get_sentence_embedding_dimension() == DIMENSION, "Local query encoder configuration differs")
    return model


def encode_queries(model, rows, token_counts):
    import torch
    require(0 < len(rows) <= DENSE_BATCH_SIZE, "Dense query batch exceeds configured limit")
    texts = [dense_text(row["retrieval_query_text"]) for row in rows]
    policy = index.TokenPolicy(model.tokenizer, model.max_seq_length, getattr(model[0], "do_lower_case", False))
    offenders = [row["query_id"] for row, text in zip(rows, texts) if not (0 < policy.count(text) == token_counts[row["query_id"]] <= TOKEN_LIMIT)]
    if offenders:
        raise QueryTokenViolation("Offending query IDs: " + ", ".join(offenders))
    with torch.inference_mode():
        values = model.encode(texts, batch_size=DENSE_BATCH_SIZE, show_progress_bar=False, convert_to_tensor=True, normalize_embeddings=False, prompt="")
    matrix = torch.as_tensor(values, dtype=torch.float32, device=model.device)
    require(matrix.ndim == 2 and tuple(matrix.shape) == (len(rows), DIMENSION), "Dense query dimension differs")
    require(bool(torch.isfinite(matrix).all()), "Nonfinite dense query embeddings")
    norms = torch.linalg.vector_norm(matrix, dim=1, keepdim=True)
    require(bool(torch.isfinite(norms).all()) and bool((norms > 0).all()), "Invalid dense query norms")
    matrix = matrix / norms
    require(bool(torch.isfinite(matrix).all()) and bool(torch.allclose(torch.linalg.vector_norm(matrix, dim=1), torch.ones(len(rows), device=matrix.device), atol=1e-5, rtol=0)), "Query normalization failed")
    return matrix


def retrieve_batch(evidence, rows, model, counts):
    import torch
    require(0 < len(rows) <= min(QUERY_BATCH_SIZE, DENSE_BATCH_SIZE), "Retrieval batch exceeds configured limit")
    lexical_queries = evidence.vectorizer.transform([row["retrieval_query_text"] for row in rows])
    dense_queries = encode_queries(model, rows, counts)
    device = str(model.device).split(":")[0]
    prepare_dense_passages(evidence, device)
    dense_queries = torch.as_tensor(dense_queries, dtype=torch.float32, device=device)
    require(dense_queries.device == evidence.passage_tensor.device, "Dense similarity device mismatch")
    with torch.inference_mode():
        scores = torch.matmul(dense_queries, evidence.passage_tensor.T)
        require(bool(torch.isfinite(scores).all()), "Nonfinite dense similarity")
        # Columns were sorted by chunk_id once. Stable descending score sort
        # therefore implements exact score/chunk_id ties, including at depth 50.
        selected = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :min(DEPTH, len(evidence.chunks))]
        candidate_scores = torch.gather(scores, 1, selected).cpu().tolist()
        candidate_indices = selected.cpu().tolist()
    for position, query in enumerate(rows):
        lexical_scores = (lexical_queries[position] @ evidence.lexical.T).toarray().ravel()
        dense = [(evidence.passage_order[i], score, rank) for rank, (i, score) in enumerate(zip(candidate_indices[position], candidate_scores[position]), 1)]
        yield query, result_rows(query, ranked(lexical_scores, evidence.chunks), dense, evidence.chunks)


class Summary:
    def __init__(self):
        self.groups = {}
        self.union_counts = {"": 0}

    def add(self, query, results):
        if results:
            self.union_counts[""] += 1
            dataset = query["source_dataset"]
            self.union_counts[dataset] = self.union_counts.get(dataset, 0)+1
        for method in METHODS:
            rows = [r for r in results if r["retrieval_method"] == method]
            group = self.groups.setdefault((query["source_dataset"], method), dict(queries=0, with_results=0, scores=array("d"), top=array("d")))
            group["queries"] += 1
            group["with_results"] += bool(rows)
            group["scores"].extend(r["retrieval_score"] for r in rows)
            if rows:
                group["top"].append(rows[0]["retrieval_score"])

    def rows(self):
        datasets = sorted({d for d, _ in self.groups})
        scopes = [("overall", "", ""), *[("dataset", d, "") for d in datasets], *[("method", "", m) for m in METHODS], *[("dataset_method", d, m) for d in datasets for m in METHODS]]
        for level, dataset, method in scopes:
            selected = [g for (d, m), g in self.groups.items() if (not dataset or d == dataset) and (not method or m == method)]
            # Query counts are unique queries, not query-method pairs.
            queries = sum(g["queries"] for g in selected) // (1 if method else len(METHODS))
            top, scores = array("d"), array("d")
            for g in selected:
                top.extend(g["top"])
                scores.extend(g["scores"])
            # Every method ranks the same non-empty frozen index. Preserve
            # per-method zero counts; overall union counts are tracked below.
            with_results = sum(g["with_results"] for g in selected) if method else self.union_counts.get(dataset, 0)
            yield dict(zip(SUMMARY_HEADER, (level, dataset, method, queries, with_results, queries-with_results, len(scores), statistics.mean(top) if top else "", statistics.median(top) if top else "", statistics.mean(scores) if scores else "", statistics.median(scores) if scores else "")))

class Diagnostics:
    """Counts use only saved query identities and retrieval result columns.

    Overall pattern share is per query-method observation; chunk share is per
    returned row. Method/dataset-method pattern share is per eligible query.
    Empty results contribute to query denominators but not ranking patterns.
    """
    def __init__(self):
        self.groups = {}

    def add(self, query, results):
        for method in METHODS:
            group = self.groups.setdefault((query["source_dataset"], method), dict(queries=0, chunks=Counter(), documents=set(), patterns=Counter()))
            group["queries"] += 1
            rows = sorted((r for r in results if r["retrieval_method"] == method), key=lambda r: int(r["rank"]))
            group["chunks"].update(r["chunk_id"] for r in rows)
            group["documents"].update(r["document_id"] for r in rows)
            if rows:
                group["patterns"][tuple(r["chunk_id"] for r in rows)] += 1

    def rows(self):
        datasets = sorted({d for d, _ in self.groups})
        scopes = [("overall", "", ""), *[("method", "", m) for m in METHODS], *[("dataset_method", d, m) for d in datasets for m in METHODS]]
        for level, dataset, method in scopes:
            groups = [g for (d, m), g in self.groups.items() if (not dataset or d == dataset) and (not method or m == method)]
            chunks, patterns, documents = Counter(), Counter(), set()
            observations = sum(g["queries"] for g in groups)
            for group in groups:
                chunks.update(group["chunks"])
                patterns.update(group["patterns"])
                documents.update(group["documents"])
            query_count = observations if method else observations//len(METHODS)
            returned = sum(chunks.values())
            pattern_count = max(patterns.values(), default=0)
            frequent = min(chunks, key=lambda chunk: (-chunks[chunk], chunk)) if chunks else ""
            frequent_count = chunks[frequent] if frequent else 0
            yield dict(zip(DIAGNOSTICS_HEADER, (level, dataset, method, query_count, returned, len(chunks), len(documents), len(patterns), pattern_count, pattern_count/observations if observations else "", frequent, frequent_count, frequent_count/returned if returned else "")))


def input_identity(root):
    names = [*QUERY_SOURCES, "results/rag/interface/interface_validation_checklist.csv", *index.OUTPUTS[:7], corpus.MANIFEST, corpus.REVIEW, index.CORPUS_PATH, corpus.OUTPUTS[1], corpus.OUTPUTS[2], *[f"metadata/traffic/{p}_experiment_closure.csv" for _, p, _ in DATASETS]]
    records = []
    for name in names:
        stat = corpus.local_path(root, name).stat()
        records.append(dict(path=name, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return json.dumps(records, sort_keys=True, separators=(",", ":"))


def configuration(root, evidence, count):
    device = dense_device()
    return dict(experiment_id=EXPERIMENT_ID, configuration_name=CONFIGURATION_NAME, stage="RAG_RETRIEVAL",
        retrieval_query_schema=RETRIEVAL_QUERY_SCHEMA, semantic_profile_top_features=str(SEMANTIC_PROFILE_TOP_FEATURES),
        semantic_bin_thresholds=SEMANTIC_BIN_THRESHOLDS, nonordinal_excluded_features=json.dumps(NONORDINAL),
        original_query_text_used_for_retrieval="NO", traffic_representation_used_for_retrieval="YES",
        raw_feature_values_emitted_in_retrieval_text="NO", ground_truth_used_for_retrieval="NO",
        profile_selection_policy="absolute standardized magnitude descending; readable feature name ascending; original name breaks normalized ties",
        profile_reference_distribution="Phase-I TRAIN-derived preprocessing; not absolute network thresholds",
        diagnostics_pattern_denominator="query-method observations overall; eligible queries for method/dataset_method; zero results included in denominator",
        diagnostics_chunk_denominator="returned retrieval rows", relevance_validation="NOT_ESTABLISHED; descriptive diagnostics only; no minimum diversity threshold",
        next_stage="RETRIEVAL_V2_SEMANTIC_DIAGNOSTIC", v2_provenance_coverage_required="YES; do not reuse v1 retrieval coverage",
        query_sources=json.dumps(QUERY_SOURCES), query_count=str(count), rag_partition="development_validation", source_split_policy="validation only; TEST prohibited",
        authorized_document_count="10", chunk_count=str(len(evidence.chunks)), retrieval_top_k=str(TOP_K), component_candidate_depth=str(DEPTH),
        query_batch_size=str(QUERY_BATCH_SIZE), dense_batch_size=str(DENSE_BATCH_SIZE), dense_encoder_microbatch_size=str(DENSE_BATCH_SIZE),
        lexical_method="tfidf", dense_method="bge_cosine", dense_embedding_model_local_path=index.MODEL_DIRECTORY,
        dense_query_prefix=PREFIX, dense_query_prefix_separator=" ", dense_embedding_dimension=str(DIMENSION), dense_normalized="YES", dense_device=device,
        dense_similarity_device=device, torch_cuda_available="YES" if device == "cuda" else "NO", gpu_acceleration_used="YES" if device == "cuda" else "NO",
        embedding_model_max_seq_length=str(TOKEN_LIMIT), query_token_truncation_allowed="NO", hybrid_method="reciprocal_rank_fusion", rrf_k=str(RRF_K),
        network_used="NO", llm_used="NO", phase_i_models_loaded="NO", test_access="NO", provenance_verification_executed="NO", reranking_executed="NO", retrieval_relevance_metrics_computed="NO",
        index_configuration_json=json.dumps(evidence.configuration, sort_keys=True), input_identity_json=input_identity(root),
        tie_policy="score descending; lexical/dense chunk_id ascending; RRF best component rank then chunk_id ascending",
        score_comparability="Scores across methods are not directly comparable; mixed-method summaries are descriptive only")


def overwrite_guard(root, config):
    exists = [corpus.local_path(root, name, must_exist=False).exists() for name in OUTPUTS]
    if any(exists):
        require(all(exists), "Incomplete retrieval output set requires manual review")
        require(config_table(root, OUTPUTS[3]) == config, "Retrieval identity/configuration differs; refusing silent overwrite")


def preflight(root):
    before = input_identity(root)
    protect_phase_i(root)
    evidence = load_index(root)
    policy = index.load_token_policy(root)
    require(policy.limit == TOKEN_LIMIT, "Query tokenizer sequence limit differs")
    counts, evaluation = query_preflight(root, policy)
    config = configuration(root, evidence, len(counts))
    config["unique_retrieval_query_texts"] = str(counts.unique_retrieval_query_texts)
    require(config["input_identity_json"] == before, "Inputs changed during preflight")
    overwrite_guard(root, config)
    return evidence, policy, counts, evaluation, config


def validation_checks(evidence, counts, config):
    facts = {
        "phase_i_closures_protected": "Exact frozen dataset/run identities and freeze flags validated",
        "interface_artifacts_and_checklist": "Canonical interfaces and PASS interface checklist",
        "partition": "validation/development_validation only; global query IDs unique; non-empty canonical query_text",
        "privacy": "Raw endpoint identifiers excluded; evaluation-only fields never passed to ranking/encoding",
        "retrieval_query_schema": RETRIEVAL_QUERY_SCHEMA + "; original query_text preserved; only retrieval_query_text transformed/encoded",
        "semantic_traffic_profile": "12 highest absolute standardized magnitudes; deterministic readable-name ties; non-ordinal codes excluded; exact qualitative bins relative to TRAIN distribution",
        "v1_preservation": "All outputs confined to v2; no v1 deletion or overwrite",
        "semantic_validation_boundary": "Structural PASS only; descriptive diversity has no acceptance threshold; manual semantic diagnostic and new v2 provenance coverage required",
        "index_checklist": "PASS; 10 authorized trusted documents; 919 canonical chunks; rejected/adversarial documents excluded",
        "lexical_row_mapping": "Saved vocabulary transform matches canonical matrix row-by-row; no fitting",
        "dense_row_mapping": "Saved mapping matches chunk CSV order; finite normalized 919x384 matrix",
        "local_only_embedding": "Offline flags; local-only loader; remote code disabled",
        "dense_execution_device": config["dense_device"],
        "dense_similarity_device": config["dense_similarity_device"],
        "torch_cuda_available": config["torch_cuda_available"],
        "dense_batch_size": config["dense_batch_size"],
        "gpu_acceleration_used": config["gpu_acceleration_used"],
        "query_token_safety": f"Fixed prefix exactly once; complete preflight; max={max(counts.values())}; limit=512; no truncation",
        "no_passage_regeneration": "Only saved passage embeddings used",
        "ranking": "Bounded deterministic batches; score/ID ties; RRF k=60, depth=50; contiguous ranks; top-k capped at available chunks",
        "canonical_chunk_ids": "All returned IDs selected by validated matrix row index",
        "no_provenance_verification": "Candidate retrieval only; no downstream evidence decisions",
        "no_reranking": "Fixed rank fusion only; no provenance-aware reranking",
        "no_llm_network_phase_i_test": "No LLM, network, traffic models or traffic datasets",
        "no_relevance_metrics": "No fabricated judgments or relevance metrics",
        "canonical_outputs": ";".join(OUTPUTS), "no_hashes_or_figures": "Readable configuration and CSV statistics only",
    }
    return [corpus.check("retrieval", k, "PASS", v) for k, v in facts.items()]


def execute(root, evidence, policy, counts, evaluation, config):
    from contextlib import ExitStack
    model = load_query_encoder(root, policy)
    require(str(model.device).split(":")[0] == config["dense_device"] == dense_device(), "Device availability changed after preflight")
    prepare_dense_passages(evidence, config["dense_similarity_device"])
    require(all(evidence.configuration.get(k) == v for k, v in index.model_identity(root).items()), "Embedding model identity changed after preflight")
    require(input_identity(root) == config["input_identity_json"], "Inputs changed after preflight")
    overwrite_guard(root, config)
    summary = Summary()
    diagnostics = Diagnostics()
    # A failed overwrite must never leave an old PASS checklist attached to
    # partially replaced result files. Only successful completion recreates it.
    completed = corpus.local_path(root, OUTPUTS[4], must_exist=False)
    if completed.exists():
        completed.unlink()
    seen = set()
    with ExitStack() as stack:
        writers = []
        for name, header in zip(OUTPUTS[:2], ((*QUERY_HEADER, *evaluation), RESULT_HEADER)):
            path = corpus.local_path(root, name, must_exist=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = stack.enter_context(path.open("w", encoding="utf-8", newline=""))
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            writers.append(writer)
        for batch in batches(query_rows(root), min(QUERY_BATCH_SIZE, DENSE_BATCH_SIZE)):
            # Isolate evaluation labels before any numeric retrieval operation.
            safe = [{k: row[k] for k in QUERY_FIELDS} for row in batch]
            annotations = {row["query_id"]: {k: row[k] for k in evaluation if k in row} for row in batch}
            for query, results in retrieve_batch(evidence, safe, model, counts):
                identity = query["query_id"]
                require(identity in counts and identity not in seen, "Query identity changed after preflight")
                seen.add(identity)
                row = dict(query, experiment_id=EXPERIMENT_ID, configuration_name=CONFIGURATION_NAME,
                           dense_query_token_count=counts[identity], retrieval_status="PASS" if results else "ZERO_RESULT")
                row.update(annotations[identity])
                writers[0].writerow(row)
                writers[1].writerows(results)
                summary.add(query, results)
                diagnostics.add(query, results)
    require(seen == set(counts), "Query set changed during retrieval")
    require(input_identity(root) == config["input_identity_json"], "Input files changed during retrieval; outputs must not be treated as complete")
    require(all(evidence.configuration.get(k) == v for k, v in index.model_identity(root).items()), "Embedding model changed during retrieval; outputs must not be treated as complete")
    corpus.write_table(corpus.local_path(root, OUTPUTS[2], must_exist=False), SUMMARY_HEADER, summary.rows())
    corpus.write_table(corpus.local_path(root, OUTPUTS[5], must_exist=False), DIAGNOSTICS_HEADER, diagnostics.rows())
    corpus.write_table(corpus.local_path(root, OUTPUTS[3], must_exist=False), index.CONFIG_HEADER, [dict(field=k, value=v) for k, v in config.items()])
    corpus.write_table(corpus.local_path(root, OUTPUTS[4], must_exist=False), foundation.CHECKLIST_HEADER, validation_checks(evidence, counts, config))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = args.project_root.resolve(strict=True)
        evidence, policy, counts, evaluation, config = preflight(root)
        if not args.execute:
            print(f"RAG_RETRIEVAL_V2=READY_FOR_EXECUTION\nQUERY_COUNT={len(counts)}\nRETRIEVAL_QUERY_SCHEMA={RETRIEVAL_QUERY_SCHEMA}\nSEMANTIC_PROFILE_TOP_FEATURES={SEMANTIC_PROFILE_TOP_FEATURES}\nUNIQUE_RETRIEVAL_QUERY_TEXTS={counts.unique_retrieval_query_texts}\nMAX_QUERY_TOKENS={max(counts.values())}\nDENSE_DEVICE={config['dense_device']}\nDENSE_BATCH_SIZE={DENSE_BATCH_SIZE}\nTEST_ACCESS=NO")
            return 0
        execute(root, evidence, policy, counts, evaluation, config)
        print(f"RAG_RETRIEVAL_V2=PASS\nQUERY_COUNT={len(counts)}\nTFIDF=PASS\nDENSE_BGE=PASS\nHYBRID_RRF=PASS\nTOP_K=10\nRETRIEVAL_QUERY_SCHEMA={RETRIEVAL_QUERY_SCHEMA}\nDENSE_DEVICE={config['dense_device']}\nDENSE_BATCH_SIZE={DENSE_BATCH_SIZE}\nGPU_ACCELERATION={config['gpu_acceleration_used']}\nNETWORK_ACCESS=NO\nTEST_ACCESS=NO\nNEXT_STAGE=RETRIEVAL_V2_SEMANTIC_DIAGNOSTIC")
        return 0
    except QueryTokenViolation as exc:
        print("RAG_RETRIEVAL_V2=BLOCKED_QUERY_TOKEN_LENGTH\n" + str(exc))
        return 2
    except RuntimeError as exc:
        print("RAG_RETRIEVAL_V2=BLOCKED\nDense execution failed; no device or batch-size fallback: " + str(exc))
        return 2
    except (OSError, ValueError, KeyError, ImportError, csv.Error) as exc:
        print("RAG_RETRIEVAL_V2=BLOCKED\n" + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
