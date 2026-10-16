"""Write-only synthetic index tests; no real embedding model or downloads."""
import ast
import json
from pathlib import Path

import pytest

from src.rag import build_evidence_index as index
from src.rag import build_evidence_corpus as corpus


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


class SyntheticTokenizer:
    def __init__(self, pieces_per_word=1):
        self.pieces_per_word = pieces_per_word
        self.calls = []

    def __call__(self, text, **kwargs):
        assert kwargs == dict(add_special_tokens=True, truncation=False, padding=False)
        self.calls.append(text)
        return {"input_ids": [0] * (2 + len(text.split()) * self.pieces_per_word)}


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(index, "load_token_policy", lambda root: index.TokenPolicy(SyntheticTokenizer(), 512))
    manifest, documents, reviews = [], [], []
    for i in range(11):
        identity = f"RFC-{9000+i}" if i < 10 else "TOR-SPEC-001"
        row = dict.fromkeys(corpus.MANIFEST_HEADER, "")
        row.update(document_id=identity, title="Synthetic evidence", source_type="standards_rfc", publisher="Synthetic publisher",
                   local_path=f"data/rag/evidence_sources/trusted/{identity}.txt", corpus_partition="trusted",
                   source_provenance_status="VERIFIED", authorized_for_retrieval="YES")
        manifest.append(row)
        text = "\n\n".join(" ".join(f"term{j}" for j in range(start, start+100)) for start in range(0, 600, 100))
        document = dict(row, extraction_status="REQUIRES_REVIEW", extraction_review_status="APPROVED" if i < 10 else "REJECTED",
                        normalized_text=text, character_count=len(text), word_count=len(text.split()), extraction_notes="Synthetic")
        document["authorized_for_retrieval"] = "YES" if i < 10 else "NO"
        documents.append(document)
        reviews.append(dict(document_id=identity, review_status="APPROVED" if i < 10 else "REJECTED", review_notes="Synthetic manual review"))
    corpus.write_table(tmp_path/corpus.MANIFEST, corpus.MANIFEST_HEADER, manifest)
    corpus.write_table(tmp_path/corpus.REVIEW, corpus.REVIEW_HEADER, reviews)
    corpus.write_table(tmp_path/index.CORPUS_PATH, corpus.DOCUMENT_HEADER, documents)
    corpus.write_table(tmp_path/corpus.OUTPUTS[1], corpus.SUMMARY_HEADER, corpus.summarize(documents))
    corpus.write_table(tmp_path/corpus.OUTPUTS[2], corpus.foundation.CHECKLIST_HEADER,
                       [corpus.check("corpus", "nonempty_usable_documents", "PASS", "10 authorized documents")])
    model = tmp_path/index.MODEL_DIRECTORY
    model.mkdir(parents=True)
    (model/"modules.json").write_text(json.dumps([dict(idx=0, name="0", path="", type="sentence_transformers.models.Transformer")]), encoding="utf-8")
    (model/"config.json").write_text(json.dumps(dict(_name_or_path="synthetic-local-encoder")), encoding="utf-8")
    (model/"sentence_bert_config.json").write_text(json.dumps(dict(max_seq_length=512, do_lower_case=False)), encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("path", [index.CORPUS_PATH, corpus.MANIFEST, corpus.REVIEW, corpus.OUTPUTS[1], corpus.OUTPUTS[2]])
def test_required_corpus_files(project, path):
    (project/path).unlink()
    with pytest.raises(ValueError):
        index.load_corpus(project)


def test_exact_ten_trusted_and_rejected_tor_exclusion(project):
    rows = index.load_corpus(project)
    assert len(rows) == 10
    assert all(corpus.trusted(r) for r in rows)
    assert "TOR-SPEC-001" not in {r["document_id"] for r in rows}
    assert index.EXPECTED_DOCUMENTS == 10


@pytest.mark.parametrize("change", ["adversarial", "non_authorized", "unverified", "empty", "failed"])
def test_ineligible_corpus_cannot_reach_expected_count(project, change):
    _, rows = corpus.read_table(project, index.CORPUS_PATH)
    _, manifest = corpus.read_table(project, corpus.MANIFEST)
    if change == "adversarial":
        rows[0]["corpus_partition"] = manifest[0]["corpus_partition"] = "adversarial"
    elif change == "non_authorized":
        rows[0]["authorized_for_retrieval"] = "NO"
    elif change == "unverified":
        rows[0]["source_provenance_status"] = manifest[0]["source_provenance_status"] = "REQUIRES_REVIEW"
    elif change == "empty":
        rows[0]["normalized_text"] = ""
    else:
        rows[0]["extraction_status"] = "FAILED"
    corpus.write_table(project/index.CORPUS_PATH, corpus.DOCUMENT_HEADER, rows)
    corpus.write_table(project/corpus.MANIFEST, corpus.MANIFEST_HEADER, manifest)
    with pytest.raises(ValueError):
        index.load_corpus(project)


def test_paragraph_chunking_deterministic_and_ordered():
    text = "\n\n".join(" ".join(f"word{i}" for i in range(start, start+100)) for start in (0, 100, 200, 300, 400, 500))
    windows = index.word_windows(text)
    assert windows == index.word_windows(text)
    assert windows[0][:2] == (0, 200)
    words = text.split()
    reconstructed = []
    for i, (start, end, chunk) in enumerate(windows):
        assert chunk.split() == words[start:end]
        assert end-start <= 320
        if i:
            assert start == windows[i-1][1]-50
            reconstructed.extend(chunk.split()[50:])
        else:
            reconstructed.extend(chunk.split())
    assert reconstructed == words


def test_long_paragraph_and_small_paragraph_progress():
    for text in (" ".join(f"w{i}" for i in range(1000)), "First tiny paragraph.\n\n" + " ".join(f"w{i}" for i in range(1000))):
        windows = index.word_windows(text)
        assert windows[-1][1] == len(text.split())
        assert all(end-start <= 320 for start, end, _ in windows)
        assert all(windows[i][1] > windows[i-1][1] for i in range(1, len(windows)))
    assert index.word_windows("  ") == []
    with pytest.raises(ValueError):
        index.word_windows("text", overlap=250)


def test_chunk_ids_schema_and_metadata(project):
    documents = index.load_corpus(project)
    chunks = index.chunk_documents(documents, index.load_token_policy(project))
    assert len({r["chunk_id"] for r in chunks}) == len(chunks)
    assert all(set(r) == set(index.CHUNK_HEADER) for r in chunks)
    for doc in documents:
        subset = [r for r in chunks if r["document_id"] == doc["document_id"]]
        assert [r["chunk_order"] for r in subset] == list(range(len(subset)))
        assert subset[0]["chunk_id"] == doc["document_id"] + "-CHUNK-0000"
        assert all(all(r[k] == doc[k] for k in index.METADATA) for r in subset)
    assert not {"query_id", "embedding", "retrieval_score", "reranking_score", "llm_output"} & set(index.CHUNK_HEADER)


def test_tfidf_configuration_and_rows(project):
    pytest.importorskip("sklearn")
    np = pytest.importorskip("numpy")
    chunks = index.chunk_documents(index.load_corpus(project), index.load_token_policy(project))
    vectorizer, matrix = index.lexical_index(chunks)
    for key, value in index.TFIDF.items():
        assert vectorizer.get_params()[key] == value
    assert matrix.shape[0] == len(chunks)
    assert np.isfinite(matrix.data).all()
    assert vectorizer.vocabulary_


def test_dense_normalization_and_rejections():
    np = pytest.importorskip("numpy")
    values = index.normalize_dense([[3, 4], [4, 3]], 2)
    assert values.dtype == np.float32 and values.shape == (2, 2)
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1, atol=1e-6)
    for values, count in [([[0, 0]], 1), ([[float("nan"), 1]], 1), ([[float("inf"), 1]], 1), ([[1, 2]], 2), ([1, 2], 2)]:
        with pytest.raises(ValueError):
            index.normalize_dense(values, count)


def test_missing_local_model_blocks(project, capsys):
    # Move fixture directory outside canonical location; no real model involved.
    (project/index.MODEL_DIRECTORY).rename(project/"synthetic_model")
    before = snapshot(project)
    assert index.main(["--project-root", str(project)]) == 2
    assert "BLOCKED_NEEDS_LOCAL_EMBEDDING_MODEL" in capsys.readouterr().out
    assert snapshot(project) == before


def test_dry_mode_token_preflight_without_fitting_or_encoding(project, monkeypatch, capsys):
    def forbidden(*args):
        raise AssertionError("Dry mode executed indexing")
    for name in ("lexical_index", "dense_index"):
        monkeypatch.setattr(index, name, forbidden)
    before = snapshot(project)
    assert index.main(["--project-root", str(project)]) == 0
    output = capsys.readouterr().out
    assert "READY_FOR_CONSTRUCTION" in output
    assert "TOKEN_LENGTH_PREFLIGHT=PASS" in output
    assert snapshot(project) == before


def test_custom_model_code_rejected(project):
    path = project/index.MODEL_DIRECTORY/"modules.json"
    path.write_text('[{"type":"external.CustomModel","path":""}]', encoding="utf-8")
    with pytest.raises(ValueError, match="Custom"):
        index.model_identity(project)


def test_configuration_and_summary(project):
    docs = index.load_corpus(project)
    config = index.configuration(docs, index.model_identity(project))
    assert config["embedding_model_name_if_available"] == "synthetic-local-encoder"
    for name in ("network_used", "llm_used", "retrieval_executed", "adversarial_included"):
        assert config[name] == "NO"
    assert config["trusted_only"] == "YES"
    assert (config["target_chunk_words"], config["maximum_chunk_words"], config["overlap_words"]) == ("250", "320", "50")
    rows = index.summary(index.chunk_documents(docs, index.load_token_policy(project)))
    assert all(set(row) == set(index.SUMMARY_HEADER) for row in rows)
    assert {r["summary_level"] for r in rows} == {"overall", "source_type", "document"}
    overall = next(r for r in rows if r["summary_level"] == "overall")
    assert overall["document_count"] == 10 and overall["chunk_count"] > 10


def test_execute_outputs_mapping_overwrite_and_corpus_untouched(project, monkeypatch):
    np = pytest.importorskip("numpy")
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    pytest.importorskip("joblib")
    monkeypatch.setattr(index, "dense_index", lambda root, chunks: index.normalize_dense(np.tile([3, 4], (len(chunks), 1)), len(chunks)))
    before = snapshot(project)
    assert index.main(["--project-root", str(project), "--execute"]) == 0
    after = snapshot(project)
    assert set(after)-set(before) == set(index.OUTPUTS)
    assert all(after[name] == value for name, value in before.items())
    _, chunks = corpus.read_table(project, index.OUTPUTS[0], index.CHUNK_HEADER)
    _, mapping = corpus.read_table(project, index.OUTPUTS[4], index.MAPPING_HEADER)
    assert [r["chunk_id"] for r in mapping] == [r["chunk_id"] for r in chunks]
    assert [r["embedding_row"] for r in mapping] == list(map(str, range(len(chunks))))
    dense = np.load(project/index.OUTPUTS[3], allow_pickle=False)
    assert dense.shape == (len(chunks), 2)
    assert index.main(["--project-root", str(project), "--execute"]) == 0
    _, config = corpus.read_table(project, index.OUTPUTS[5], index.CONFIG_HEADER)
    assert dict((r["field"], r["value"]) for r in config)["embedding_dimension"] == "2"
    _, checks = corpus.read_table(project, index.OUTPUTS[6], corpus.foundation.CHECKLIST_HEADER)
    assert all(r["status"] == "PASS" for r in checks)
    docs = index.load_corpus(project)
    identity = index.configuration(docs, index.model_identity(project))
    for key in ("overlap_words", "tfidf_norm", "embedding_model_name_if_available", "corpus_document_ids_json", "chunk_token_limit", "embedding_model_max_seq_length", "chunking_method"):
        with pytest.raises(ValueError, match="identity"):
            index.overwrite_guard(project, dict(identity, **{key: "changed"}), docs)
    docs[0]["normalized_text"] += " changed"
    with pytest.raises(ValueError):
        index.overwrite_guard(project, identity, docs)


def test_no_retrieval_network_phase_i_or_figures_and_local_loader():
    source = Path(index.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = set()
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(n.name.split(".")[0] for n in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module.split(".")[0])
        elif isinstance(node, ast.Call):
            calls.append(node)
    assert not imports & {"requests", "httpx", "urllib", "openai", "lightgbm", "xgboost", "matplotlib", "hashlib", "faiss"}
    loader = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "SentenceTransformer")
    keywords = {k.arg: k.value for k in loader.keywords}
    assert isinstance(keywords["local_files_only"], ast.Constant) and keywords["local_files_only"].value is True
    assert keywords["trust_remote_code"].value is False
    assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in source
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in {"retrieve", "search", "similarity", "predict", "savefig", "sha256", "md5"} for n in calls)
    assert not any(isinstance(n, ast.Constant) and isinstance(n.value, str) and any(path in n.value for path in ("test.csv", "train.csv", "validation.csv", "results/rag/interface")) for n in ast.walk(tree))


def test_technical_word_windows_are_split_without_loss_or_truncation(project):
    document = index.load_corpus(project)[0]
    document["normalized_text"] = "\n\n".join(" ".join(f"technical_{i}_field" for i in range(start, start+80)) for start in (0, 80, 160))
    tokenizer = SyntheticTokenizer(pieces_per_word=7)
    policy = index.TokenPolicy(tokenizer, 512)
    assert len(document["normalized_text"].split()) <= 320
    assert policy.count(document["normalized_text"]) > 512
    chunks = index.chunk_documents([document], policy)
    assert chunks == index.chunk_documents([document], policy)
    assert len(chunks) > 1
    windows = index.token_windows(document["normalized_text"], policy)
    source = document["normalized_text"].split()
    recovered, previous_end = [], 0
    for i, (start, end, text, tokens) in enumerate(windows):
        assert 0 < tokens == policy.count(text) <= 512
        assert 0 < end-start <= 320
        assert text.split() == source[start:end]
        assert start <= previous_end and end > previous_end
        recovered.extend(text.split()[previous_end-start:])
        previous_end = end
        assert chunks[i]["chunk_token_count"] == tokens
        assert chunks[i]["chunk_id"] == document["document_id"] + f"-CHUNK-{i:04d}"
        assert chunks[i]["chunk_order"] == i
    assert recovered == source
    assert previous_end == len(source)
    assert tokenizer.calls


def test_short_safe_spans_reduce_overlap_and_progress():
    text = " ".join(f"word{i}" for i in range(30))
    policy = index.TokenPolicy(SyntheticTokenizer(pieces_per_word=100), 512)
    windows = index.token_windows(text, policy)
    assert windows == index.token_windows(text, policy)
    assert windows[-1][1] == 30
    assert all(end-start <= 5 for start, end, _, _ in windows)
    assert all(windows[i][1] > windows[i-1][1] and windows[i][0] <= windows[i-1][1] for i in range(1, len(windows)))


def test_indivisible_oversized_word_blocks():
    policy = index.TokenPolicy(SyntheticTokenizer(pieces_per_word=600), 512)
    with pytest.raises(index.TokenLengthViolation, match="Source word"):
        index.token_windows("indivisible", policy)


def test_dry_mode_reports_token_length_violation(project, monkeypatch, capsys):
    monkeypatch.setattr(index, "load_token_policy", lambda root: index.TokenPolicy(SyntheticTokenizer(600), 512))
    before = snapshot(project)
    assert index.main(["--project-root", str(project)]) == 2
    assert "EVIDENCE_INDEX=BLOCKED_TOKEN_LENGTH_VIOLATION" in capsys.readouterr().out
    assert snapshot(project) == before


def test_token_configuration_and_summary_statistics(project):
    docs = index.load_corpus(project)
    config = index.configuration(docs, index.model_identity(project))
    assert config["embedding_model_max_seq_length"] == config["chunk_token_limit"] == "512"
    assert config["tokenizer_local_only"] == config["token_safety_enforced"] == "YES"
    assert config["token_truncation_allowed"] == "NO"
    assert config["chunking_method"] == "paragraph_aware_word_and_token_bounded_v2"
    chunks = index.chunk_documents(docs, index.load_token_policy(project))
    summary = index.summary(chunks)
    for row in summary:
        assert 0 < row["min_chunk_tokens"] <= row["median_chunk_tokens"] <= row["max_chunk_tokens"] <= 512
        assert row["min_chunk_tokens"] <= row["mean_chunk_tokens"] <= row["max_chunk_tokens"]


def test_corrupted_final_token_count_blocks_before_any_indexing(project, monkeypatch):
    pytest.importorskip("numpy")
    pytest.importorskip("joblib")
    pytest.importorskip("scipy")
    docs = index.load_corpus(project)
    policy = index.load_token_policy(project)
    config = index.configuration(docs, index.model_identity(project))
    chunks = index.chunk_documents(docs, policy)
    chunks[0]["chunk_token_count"] = 513
    def forbidden(*args):
        raise AssertionError("Invalid chunk reached indexing")
    monkeypatch.setattr(index, "lexical_index", forbidden)
    monkeypatch.setattr(index, "dense_index", forbidden)
    before = snapshot(project)
    with pytest.raises(index.TokenLengthViolation):
        index.execute(project, docs, config, chunks, policy)
    assert snapshot(project) == before


def test_dense_encode_receives_validated_chunks_in_canonical_order(project, monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace
    np = pytest.importorskip("numpy")
    policy = index.load_token_policy(project)
    chunks = index.chunk_documents(index.load_corpus(project), policy)
    received = []
    class LocalModel:
        max_seq_length = 512
        tokenizer = policy.tokenizer
        def __init__(self, path, **kwargs):
            assert path == str((project/index.MODEL_DIRECTORY).resolve())
            assert kwargs == dict(device="cpu", local_files_only=True, trust_remote_code=False)
        def __getitem__(self, item):
            return SimpleNamespace(do_lower_case=False)
        def eval(self):
            return self
        def encode(self, texts, **kwargs):
            assert all(policy.count(text) <= 512 for text in texts)
            assert kwargs["prompt"] == ""
            received.extend(texts)
            return np.tile([3.0, 4.0], (len(texts), 1))
        def get_sentence_embedding_dimension(self):
            return 2
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=LocalModel))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda seed: None, set_num_threads=lambda count: None, use_deterministic_algorithms=lambda flag: None, inference_mode=nullcontext))
    values = index.dense_index(project, chunks)
    assert received == [r["chunk_text"] for r in chunks]
    assert values.shape == (len(chunks), 2)
    received.clear()
    chunks[0]["chunk_text"] = "oversized " * 600
    with pytest.raises(index.TokenLengthViolation):
        index.dense_index(project, chunks)
    assert received == []


def test_local_tokenizer_flags_and_no_truncation():
    source = Path(index.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    loader = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "from_pretrained")
    options = {k.arg: k.value for k in loader.keywords}
    assert options["local_files_only"].value is True
    assert options["trust_remote_code"].value is False
    tokenizer = SyntheticTokenizer()
    policy = index.TokenPolicy(tokenizer, 512, do_lower_case=True)
    assert policy.count("  MiXeD Case  ") == 4
    assert tokenizer.calls == ["mixed case"]
