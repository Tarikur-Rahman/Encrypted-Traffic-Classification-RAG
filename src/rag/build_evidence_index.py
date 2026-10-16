"""Deterministic trusted-evidence indexing. Dry mode performs no fitting/encoding.

All model loading is local-only, on CPU, with offline flags and remote code
disabled. No queries, retrieval, traffic datasets or Phase-I artifacts are used.
Model file names, sizes and modification times form a readable local identity;
this is a change guard, not cryptographic integrity verification.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from src.rag import build_evidence_corpus as corpus


EXPECTED_DOCUMENTS = 10
TARGET_WORDS, MAXIMUM_WORDS, OVERLAP_WORDS = 250, 320, 50
MODEL_DIRECTORY = "models/rag/embedding_model"
CORPUS_PATH = "data/processed/rag/evidence_documents.csv"
OUTPUTS = (
    "data/processed/rag/evidence_chunks.csv",
    "artifacts/rag/index/tfidf_vectorizer.joblib",
    "artifacts/rag/index/tfidf_matrix.npz",
    "artifacts/rag/index/dense_embeddings.npy",
    "artifacts/rag/index/dense_chunk_ids.csv",
    "metadata/rag/evidence_index_configuration.csv",
    "metadata/rag/evidence_index_validation_checklist.csv",
    "results/rag/index/evidence_index_summary.csv",
)
METADATA = tuple("title source_type publisher publication_date canonical_url doi traffic_topics security_topics corpus_partition source_provenance_status authorized_for_retrieval".split())
CHUNK_HEADER = ("chunk_id", "document_id", "chunk_order", *METADATA, "chunk_word_count", "chunk_token_count", "chunk_character_count", "chunk_text")
MAPPING_HEADER = ("embedding_row", "chunk_id", "document_id")
CONFIG_HEADER = ("field", "value")
SUMMARY_HEADER = tuple("summary_level source_type document_id document_count chunk_count total_chunk_words mean_chunk_words median_chunk_words min_chunk_words max_chunk_words mean_chunk_tokens median_chunk_tokens min_chunk_tokens max_chunk_tokens".split())
TFIDF = dict(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, norm="l2", min_df=1)
require = corpus.require


class MissingLocalModel(ValueError):
    pass


class TokenLengthViolation(ValueError):
    pass


def offline_mode():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


@dataclass
class TokenPolicy:
    tokenizer: object
    limit: int
    do_lower_case: bool = False

    def count(self, text):
        # Match the standard Sentence-Transformers Transformer preprocessing.
        value = text.strip()
        if self.do_lower_case:
            value = value.lower()
        encoded = self.tokenizer(value, add_special_tokens=True, truncation=False, padding=False)
        return len(encoded["input_ids"])


def tokenizer_settings(root):
    modules = json.loads(corpus.local_path(root, MODEL_DIRECTORY + "/modules.json").read_text(encoding="utf-8"))
    transformers = [m for m in modules if m["type"] == "sentence_transformers.models.Transformer"]
    require(len(transformers) == 1 and modules[0] == transformers[0], "A single leading local Transformer module is required for exact token accounting")
    relative = (Path(MODEL_DIRECTORY) / transformers[0]["path"]).as_posix()
    settings = json.loads(corpus.local_path(root, relative + "/sentence_bert_config.json").read_text(encoding="utf-8"))
    limit = settings.get("max_seq_length")
    require(isinstance(limit, int) and not isinstance(limit, bool) and limit > 0, "Explicit local Sentence-Transformers max_seq_length required")
    require(isinstance(settings.get("do_lower_case", False), bool), "Invalid local lowercasing policy")
    require(not settings.get("tokenizer_args") and not settings.get("tokenizer_name_or_path"), "Custom tokenizer settings require explicit token-policy support")
    return relative, limit, settings.get("do_lower_case", False)


def load_token_policy(root):
    offline_mode()
    relative, limit, lower = tokenizer_settings(root)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str((root/relative).resolve()), local_files_only=True, trust_remote_code=False)
    return TokenPolicy(tokenizer, limit, lower)


def token_windows(text, policy):
    """Largest safe prefix of each proposed paragraph-aware source window.

    Descending bounded search avoids assuming token counts are monotonic under
    BPE merges. Reduce overlap only when needed to cover a new source word.
    Single oversized source words block instead of being character-split.
    """
    words = list(re.finditer(r"\S+", text))
    boundaries = {i for i in range(1, len(words)) if re.findall(r"\n\s*\n", text[words[i-1].end():words[i].start()])}
    boundaries.add(len(words))
    start, previous_end, windows = 0, 0, []
    while previous_end < len(words):
        if len(words)-start <= TARGET_WORDS:
            proposed = len(words)
        else:
            earlier = [n for n in boundaries if max(previous_end, start+OVERLAP_WORDS) < n <= start+TARGET_WORDS]
            later = [n for n in boundaries if start+TARGET_WORDS < n <= start+MAXIMUM_WORDS and n > previous_end]
            proposed = max(earlier) if earlier else min(later) if later else min(len(words), start+TARGET_WORDS)
        safe = None
        for end in range(min(proposed, start+MAXIMUM_WORDS), previous_end, -1):
            value = text[words[start].start():words[end-1].end()]
            count = policy.count(value)
            if 0 < count <= policy.limit:
                safe = (start, end, value, count)
                break
        if safe is None:
            if start < previous_end:
                start += 1
                continue
            raise TokenLengthViolation(f"Source word at zero-based position {start} cannot fit within {policy.limit} tokens without truncation")
        windows.append(safe)
        previous_end = safe[1]
        start = max(safe[0]+1, previous_end-OVERLAP_WORDS)
    return windows


def validate_token_counts(chunks, policy):
    for row in chunks:
        actual = policy.count(row["chunk_text"])
        if not (0 < actual <= policy.limit and actual == row["chunk_token_count"]):
            raise TokenLengthViolation("Invalid or oversized chunk: " + row["chunk_id"])


def load_corpus(root):
    _, documents = corpus.read_table(root, CORPUS_PATH, corpus.DOCUMENT_HEADER)
    _, manifest = corpus.read_table(root, corpus.MANIFEST, corpus.MANIFEST_HEADER)
    corpus.local_path(root, corpus.REVIEW)  # Required for this indexing stage.
    reviews = corpus.read_reviews(root, manifest)
    corpus.read_table(root, corpus.OUTPUTS[1], corpus.SUMMARY_HEADER)
    _, checks = corpus.read_table(root, corpus.OUTPUTS[2], corpus.foundation.CHECKLIST_HEADER)
    require(checks and not any(r["status"] == "BLOCKED" for r in checks), "Corpus validation is blocked")
    require(any(r["check"] == "nonempty_usable_documents" and r["status"] == "PASS" for r in checks), "Corpus validation PASS required")
    require(len({r["document_id"] for r in manifest}) == len(manifest), "Duplicate manifest IDs")
    require(len({r["document_id"] for r in documents}) == len(documents), "Duplicate corpus IDs")
    curated = {r["document_id"]: r for r in manifest}
    require(set(curated) == {r["document_id"] for r in documents}, "Corpus/manifest document sets differ")
    selected = []
    for row in documents:
        identity = row["document_id"]
        original = curated[identity]
        require(identity and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", identity), "Invalid curated document ID")
        require(all(row[k] == original[k] for k in corpus.MANIFEST_HEADER if k != "authorized_for_retrieval"), "Corpus bibliographic/partition metadata differs from manifest")
        require(row["authorized_for_retrieval"] in {"YES", "NO"}, "Invalid effective authorization")
        review = reviews.get(identity, {}).get("review_status", "MISSING")
        # Explicit rejection always excludes a record from indexing, even if a
        # prior extraction happened to be PASS. No hard-coded Tor exception.
        if review == "REJECTED" or not corpus.trusted(row):
            continue
        require(corpus.trusted(original), "Corpus authorization exceeds curated provenance")
        require(row["source_type"] in corpus.foundation.SOURCE_CATEGORIES, "Invalid source category")
        require(row["normalized_text"].strip(), "Authorized document has empty text")
        require(row["extraction_status"] == "PASS" or (row["extraction_status"] == "REQUIRES_REVIEW" and review == "APPROVED" and row["extraction_review_status"] == "APPROVED"), "Extraction not approved")
        selected.append(row)
    require(len(selected) == EXPECTED_DOCUMENTS, f"Expected exactly {EXPECTED_DOCUMENTS} eligible trusted documents; found {len(selected)}")
    return sorted(selected, key=lambda r: r["document_id"])


def word_windows(text, target=TARGET_WORDS, maximum=MAXIMUM_WORDS, overlap=OVERLAP_WORDS):
    """Return contiguous source slices, preserving paragraph whitespace verbatim.

    Prefer the last paragraph boundary up to target; if none is usable, accept
    a complete paragraph up to maximum, otherwise split at target words.
    Each successor starts exactly overlap words before the preceding end.
    """
    require(0 <= overlap < target <= maximum, "Invalid chunking configuration")
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []
    boundaries = {i for i in range(1, len(words)) if re.findall(r"\n\s*\n", text[words[i-1].end():words[i].start()])}
    boundaries.add(len(words))
    start, windows = 0, []
    while start < len(words):
        if len(words) - start <= target:
            end = len(words)
        else:
            ends = [n for n in boundaries if start + overlap < n <= start + target]
            later = [n for n in boundaries if start + target < n <= start + maximum]
            end = max(ends) if ends else min(later) if later else start + target
        windows.append((start, end, text[words[start].start():words[end-1].end()]))
        if end == len(words):
            break
        start = end - overlap
    return windows


def chunk_documents(documents, policy):
    chunks = []
    for doc in documents:
        for order, (start, end, text, count) in enumerate(token_windows(doc["normalized_text"], policy)):
            row = dict(chunk_id=f"{doc['document_id']}-CHUNK-{order:04d}", document_id=doc["document_id"], chunk_order=order)
            row.update({key: doc[key] for key in METADATA})
            row.update(chunk_word_count=end-start, chunk_token_count=count, chunk_character_count=len(text), chunk_text=text)
            chunks.append(row)
    validate_chunks(documents, chunks, policy)
    return chunks


def validate_chunks(documents, chunks, policy):
    validate_token_counts(chunks, policy)
    require(len({r["chunk_id"] for r in chunks}) == len(chunks), "Duplicate chunk IDs")
    require({r["document_id"] for r in chunks} == {r["document_id"] for r in documents}, "Document lost in chunking")
    for doc in documents:
        actual = [r for r in chunks if r["document_id"] == doc["document_id"]]
        expected = token_windows(doc["normalized_text"], policy)
        require(len(actual) == len(expected), "Chunk count differs")
        for i, (row, (start, end, text, count)) in enumerate(zip(actual, expected)):
            require(row["chunk_order"] == i and row["chunk_id"] == f"{doc['document_id']}-CHUNK-{i:04d}", "Chunk identity/order differs")
            require(row["chunk_text"] == text and text and len(text.split()) == row["chunk_word_count"] == end-start <= MAXIMUM_WORDS, "Invalid source slice/word count")
            require(all(row[key] == doc[key] for key in METADATA), "Chunk metadata differs")
            if i:
                require(0 <= expected[i-1][1]-start <= OVERLAP_WORDS and end > expected[i-1][1], "Invalid overlap or source progression")
        require(expected and expected[0][0] == 0 and expected[-1][1] == len(doc["normalized_text"].split()), "Source text lost")


def model_identity(root):
    directory = root / MODEL_DIRECTORY
    if not directory.exists():
        raise MissingLocalModel("Local embedding model directory is absent")
    require(directory.is_dir() and directory.resolve().is_relative_to(root), "Invalid local model directory")
    files = []
    for parent, directories, names in os.walk(directory, followlinks=False):
        for child in [Path(parent), *(Path(parent)/d for d in directories)]:
            require(not corpus.foundation.linked_path(child), "Linked embedding model directory prohibited")
        for name in sorted(names):
            path = Path(parent) / name
            corpus.local_path(root, path.relative_to(root).as_posix())
            stat = path.stat()
            files.append(dict(path=path.relative_to(directory).as_posix(), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    require(files, "Local embedding model directory is empty")
    modules_path = corpus.local_path(root, MODEL_DIRECTORY + "/modules.json")
    modules = json.loads(modules_path.read_text(encoding="utf-8"))
    require(isinstance(modules, list) and modules, "Local sentence-transformer modules required")
    for module in modules:
        require(str(module.get("type", "")).startswith("sentence_transformers.models."), "Custom embedding module code prohibited")
        module_path = Path(module.get("path", ""))
        require(not module_path.is_absolute() and ".." not in module_path.parts, "Embedding module path escapes local directory")
    names = []
    for item in files:
        if Path(item["path"]).name == "config.json":
            config = json.loads((directory/item["path"]).read_text(encoding="utf-8"))
            require(not config.get("auto_map"), "Remote/custom model code prohibited")
            if config.get("_name_or_path"):
                names.append(str(config["_name_or_path"]))
    tokenizer_relative, limit, lower = tokenizer_settings(root)
    return dict(embedding_model_local_path=MODEL_DIRECTORY,
                embedding_model_max_seq_length=str(limit), chunk_token_limit=str(limit),
                tokenizer_local_path=tokenizer_relative, tokenizer_do_lower_case=str(lower),
                embedding_model_name_if_available=";".join(sorted(set(names))),
                embedding_model_files_json=json.dumps(sorted(files, key=lambda f: f["path"]), separators=(",", ":")))


def configuration(documents, model):
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "joblib", "torch", "sentence-transformers", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "UNAVAILABLE"
    result = dict(index_stage="EVIDENCE_INDEXING", corpus_path=CORPUS_PATH,
        authorized_document_count=str(len(documents)), corpus_document_ids_json=json.dumps([r["document_id"] for r in documents]),
        target_chunk_words=str(TARGET_WORDS), maximum_chunk_words=str(MAXIMUM_WORDS), overlap_words=str(OVERLAP_WORDS),
        chunking_method="paragraph_aware_word_and_token_bounded_v2", lexical_method="sklearn_tfidf",
        tokenizer_local_only="YES", token_truncation_allowed="NO", token_safety_enforced="YES",
        tfidf_lowercase="True", tfidf_ngram_min="1", tfidf_ngram_max="2", tfidf_sublinear_tf="True", tfidf_norm="l2", tfidf_min_df="1",
        tfidf_token_pattern=r"(?u)\b\w\w+\b", tfidf_dtype="float64", embedding_normalized="YES", dense_dtype="float32",
        embedding_device="cpu", embedding_batch_size="32", embedding_prompt="", random_seed="0", torch_threads="1", token_truncation="PROHIBITED",
        trusted_only="YES", adversarial_included="NO", network_used="NO", llm_used="NO", retrieval_executed="NO",
        package_versions_json=json.dumps(versions, sort_keys=True))
    result.update(model)
    return result


def overwrite_guard(root, config, documents, prospective=None):
    existing = [corpus.local_path(root, name, must_exist=False).exists() for name in OUTPUTS]
    if not any(existing):
        return
    require(all(existing), "Incomplete canonical index requires manual review")
    _, rows = corpus.read_table(root, OUTPUTS[5], CONFIG_HEADER)
    require(len({r["field"] for r in rows}) == len(rows), "Duplicate index configuration keys")
    old = {r["field"]: r["value"] for r in rows}
    require(all(old.get(k) == v for k, v in config.items()), "Index identity/configuration differs; refusing overwrite")
    # Full readable text comparison through existing chunks prevents silent
    # reuse after corpus edits, without any digest or hidden runtime identity.
    _, chunks = corpus.read_table(root, OUTPUTS[0], CHUNK_HEADER)
    if prospective is None:
        prospective = chunk_documents(documents, load_token_policy(root))
    require(chunks == [{k: str(r[k]) for k in CHUNK_HEADER} for r in prospective], "Indexed corpus text/metadata or token-safe chunk plan changed")


def lexical_index(chunks):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(**TFIDF, dtype=np.float64, token_pattern=r"(?u)\b\w\w+\b")
    matrix = vectorizer.fit_transform([r["chunk_text"] for r in chunks]).tocsr()
    require(matrix.shape[0] == len(chunks) and matrix.shape[1] > 0 and np.isfinite(matrix.data).all(), "Invalid lexical matrix")
    return vectorizer, matrix


def normalize_dense(values, count):
    import numpy as np
    values = np.asarray(values, dtype=np.float32)
    require(values.ndim == 2 and values.shape[0] == count and values.shape[1] > 0, "Dense shape differs")
    require(np.isfinite(values).all(), "Nonfinite dense embeddings")
    norms = np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True)
    require((norms > 0).all(), "Zero dense vector")
    values = (values/norms).astype(np.float32)
    require(np.isfinite(values).all() and np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-6), "Dense normalization failed")
    return values


def dense_index(root, chunks):
    # Set offline flags before importing libraries; never resolve a hub name.
    offline_mode()
    policy = load_token_policy(root)
    validate_token_counts(chunks, policy)
    import torch
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = SentenceTransformer(str((root/MODEL_DIRECTORY).resolve()), device="cpu", local_files_only=True, trust_remote_code=False)
    model.eval()
    require(model.max_seq_length == policy.limit, "Loaded embedding model sequence length differs from local configuration")
    model_policy = TokenPolicy(model.tokenizer, model.max_seq_length, getattr(model[0], "do_lower_case", False))
    validate_token_counts(chunks, model_policy)
    texts = [r["chunk_text"] for r in chunks]
    # Whitespace word limits do not imply tokenizer limits. Refuse silent loss.
    with torch.inference_mode():
        raw = model.encode(texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=False, prompt="")
    matrix = normalize_dense(raw, len(chunks))
    require(matrix.shape[1] == model.get_sentence_embedding_dimension(), "Embedding dimension differs from local model")
    return matrix


def summary(chunks):
    groups = [("overall", "", "", chunks)]
    groups += [("source_type", source, "", [r for r in chunks if r["source_type"] == source]) for source in sorted({r["source_type"] for r in chunks})]
    groups += [("document", "", identity, [r for r in chunks if r["document_id"] == identity]) for identity in sorted({r["document_id"] for r in chunks})]
    result = []
    for level, source, identity, rows in groups:
        counts = [r["chunk_word_count"] for r in rows]
        tokens = [r["chunk_token_count"] for r in rows]
        result.append(dict(zip(SUMMARY_HEADER, (level, source, identity, len({r["document_id"] for r in rows}), len(rows), sum(counts), statistics.mean(counts), statistics.median(counts), min(counts), max(counts), statistics.mean(tokens), statistics.median(tokens), min(tokens), max(tokens)))))
    return result


def checklist(documents, chunks, lexical, dense, config):
    facts = {
        "corpus_files_exist": "All five canonical corpus inputs validated",
        "corpus_validation_pass": "No BLOCKED checks; usable corpus PASS",
        "authorized_document_count": str(len(documents)), "rejected_documents_excluded": "Current manual REJECTED reviews excluded, including TOR-SPEC-001",
        "adversarial_excluded": "trusted only", "eligibility": "Manifest and corpus VERIFIED/YES/trusted; approved extraction; non-empty text",
        "deterministic_unique_chunk_ids": "Readable parent ID plus zero-based ordinal; globally unique",
        "contiguous_chunk_order": "Validated against source slices per document", "chunk_text_nonempty": "All chunks non-empty",
        "chunk_word_counts": f"All counts verified; maximum={MAXIMUM_WORDS}", "source_order": "Monotonic source word offsets; no rewriting",
        "overlap": f"Up to {OVERLAP_WORDS} words; reduced only for token-safe forward progress", "all_documents_chunked": "At least one chunk per eligible document",
        "local_tokenizer_available": "Loaded local-only; remote code disabled",
        "embedding_maximum_sequence_length": config["embedding_model_max_seq_length"],
        "token_truncation_prohibited": "add_special_tokens=True; truncation=False for all token counts",
        "chunk_token_counts_valid": f"Exact counts; max={max(r['chunk_token_count'] for r in chunks)}; limit={config['chunk_token_limit']}",
        "encode_input_token_safety": "All final texts recounted against loaded model tokenizer before encode",
        "source_text_coverage": "Source slices span first through final source word; no gaps or rewriting",
        "deterministic_token_safe_boundaries": "Bounded descending search on source word boundaries; deterministic IDs and overlap progression",
        "lexical_trusted_only": "Fitted only on selected trusted chunk_text", "lexical_shape_finite": str(lexical.shape),
        "local_embedding_model": MODEL_DIRECTORY, "no_download": "Local-only loader; offline flags; remote code disabled",
        "dense_shape_finite_normalized": str(dense.shape) + "; float32, unit L2; zero vectors rejected",
        "dense_mapping_order": "embedding_row exactly matches chunk CSV row order",
        "no_retrieval": "No query input or search operation", "no_llm": "Local sentence encoder only",
        "no_phase_i_models": "Only models/rag/embedding_model accessed", "no_traffic_test_access": "No traffic data paths accessed",
        "no_hashes": "Readable identities and direct text comparisons", "no_figures": "CSV statistics only",
        "canonical_output_paths": ";".join(OUTPUTS),
    }
    return [corpus.check("index", key, "PASS", value) for key, value in facts.items()]


def execute(root, documents, config, chunks=None, policy=None):
    import numpy as np
    import joblib
    from scipy.sparse import save_npz
    policy = policy if policy is not None else load_token_policy(root)
    require(str(policy.limit) == config["chunk_token_limit"], "Token policy differs from configuration")
    chunks = chunks if chunks is not None else chunk_documents(documents, policy)
    validate_chunks(documents, chunks, policy)
    vectorizer, lexical = lexical_index(chunks)
    dense = dense_index(root, chunks)
    current_model = model_identity(root)
    require(current_model == {k: config[k] for k in current_model}, "Local embedding files changed during indexing")
    config = dict(config, chunk_count=str(len(chunks)), embedding_dimension=str(dense.shape[1]), lexical_feature_count=str(lexical.shape[1]))
    overwrite_guard(root, config, documents, chunks)
    mapping = [dict(embedding_row=i, chunk_id=r["chunk_id"], document_id=r["document_id"]) for i, r in enumerate(chunks)]
    targets = [corpus.local_path(root, name, must_exist=False) for name in OUTPUTS]
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
    corpus.write_table(targets[0], CHUNK_HEADER, chunks)
    joblib.dump(vectorizer, targets[1])
    save_npz(targets[2], lexical)
    np.save(targets[3], dense, allow_pickle=False)
    corpus.write_table(targets[4], MAPPING_HEADER, mapping)
    corpus.write_table(targets[5], CONFIG_HEADER, [dict(field=k, value=v) for k, v in config.items()])
    corpus.write_table(targets[6], corpus.foundation.CHECKLIST_HEADER, checklist(documents, chunks, lexical, dense, config))
    corpus.write_table(targets[7], SUMMARY_HEADER, summary(chunks))
    return len(chunks)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    try:
        root = args.project_root.resolve(strict=True)
        documents = load_corpus(root)
        config = configuration(documents, model_identity(root))
        policy = load_token_policy(root)
        require(str(policy.limit) == config["chunk_token_limit"], "Tokenizer/model limit differs")
        chunks = chunk_documents(documents, policy)
        overwrite_guard(root, config, documents, chunks)
        print(f"TOKEN_LENGTH_PREFLIGHT=PASS\nMAX_CHUNK_TOKENS={max(r['chunk_token_count'] for r in chunks)}")
        if not args.execute:
            print("EVIDENCE_INDEX=READY_FOR_CONSTRUCTION")
            return 0
        count = execute(root, documents, config, chunks, policy)
        print(f"EVIDENCE_INDEX=PASS\nAUTHORIZED_DOCUMENTS={len(documents)}\nCHUNKS={count}\nLEXICAL_INDEX=PASS\nDENSE_INDEX=PASS\nNETWORK_ACCESS=NO\nTEST_ACCESS=NO\nNEXT_STAGE=RAG_RETRIEVAL")
        return 0
    except TokenLengthViolation as exc:
        print("EVIDENCE_INDEX=BLOCKED_TOKEN_LENGTH_VIOLATION\n" + str(exc))
        return 2
    except MissingLocalModel as exc:
        print("EVIDENCE_INDEX=BLOCKED_NEEDS_LOCAL_EMBEDDING_MODEL\n" + str(exc))
        return 2
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print("EVIDENCE_INDEX=BLOCKED\n" + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
