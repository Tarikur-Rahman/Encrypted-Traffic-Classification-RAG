"""Synthetic retrieval tests; never load real encoders or traffic datasets."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.rag import run_rag_retrieval as retrieval
from src.rag import build_evidence_index as index
from src.rag import build_evidence_corpus as corpus


class Tokenizer:
    def __call__(self, text, **options):
        assert options == dict(add_special_tokens=True, truncation=False, padding=False)
        return {"input_ids": [0] * (len(text.split())+2)}


def traffic_representation(count=50, **features):
    values = {f"feature_{i:02d}": 0.1 for i in range(count-len(features))}
    values.update(features)
    return json.dumps(dict(predictor_count=count, features=values))


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def project(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setattr(retrieval, "dense_device", lambda: "cpu")
    np = pytest.importorskip("numpy")
    joblib = pytest.importorskip("joblib")
    sparse = pytest.importorskip("scipy.sparse")
    sklearn = pytest.importorskip("sklearn.feature_extraction.text")
    monkeypatch.setattr(retrieval, "EXPECTED_CHUNKS", 12)
    monkeypatch.setattr(retrieval, "DIMENSION", 3)
    documents, chunks = [], []
    for i in range(10):
        document = dict(document_id=f"RFC-{i:04d}", normalized_text="encrypted transport protocol evidence")
        document.update(dict.fromkeys(index.METADATA, ""))
        document.update(title="Synthetic evidence", source_type="standards_rfc", corpus_partition="trusted", source_provenance_status="VERIFIED", authorized_for_retrieval="YES")
        documents.append(document)
        for order in range(3 if i == 0 else 1):
            row = {k: document[k] for k in index.METADATA}
            row.update(document_id=document["document_id"], chunk_id=document["document_id"]+f"-CHUNK-{order:04d}", chunk_order=order,
                       chunk_word_count=4, chunk_token_count=6, chunk_character_count=len(document["normalized_text"]), chunk_text=document["normalized_text"])
            chunks.append(row)
    monkeypatch.setattr(index, "load_corpus", lambda root: documents)
    model_identity = dict(embedding_model_local_path=index.MODEL_DIRECTORY, embedding_model_max_seq_length="512", embedding_model_files_json="synthetic-local-model")
    monkeypatch.setattr(index, "model_identity", lambda root: model_identity)
    monkeypatch.setattr(index, "load_token_policy", lambda root: index.TokenPolicy(Tokenizer(), 512))
    vectorizer = sklearn.TfidfVectorizer(**index.TFIDF)
    lexical = vectorizer.fit_transform([r["chunk_text"] for r in chunks])
    dense = index.normalize_dense(np.tile([1, 2, 3], (len(chunks), 1)), len(chunks))
    for name in index.OUTPUTS:
        (tmp_path/name).parent.mkdir(parents=True, exist_ok=True)
    corpus.write_table(tmp_path/index.OUTPUTS[0], index.CHUNK_HEADER, chunks)
    joblib.dump(vectorizer, tmp_path/index.OUTPUTS[1])
    sparse.save_npz(tmp_path/index.OUTPUTS[2], lexical)
    np.save(tmp_path/index.OUTPUTS[3], dense, allow_pickle=False)
    corpus.write_table(tmp_path/index.OUTPUTS[4], index.MAPPING_HEADER, [dict(embedding_row=i, chunk_id=r["chunk_id"], document_id=r["document_id"]) for i, r in enumerate(chunks)])
    config = dict(model_identity, authorized_document_count="10", chunk_count="12", embedding_dimension="3", embedding_normalized="YES", dense_dtype="float32", trusted_only="YES", adversarial_included="NO", token_safety_enforced="YES", token_truncation_allowed="NO", lexical_feature_count=str(lexical.shape[1]))
    corpus.write_table(tmp_path/index.OUTPUTS[5], index.CONFIG_HEADER, [dict(field=k, value=v) for k, v in config.items()])
    check = [corpus.check("synthetic", "fixture", "PASS", "Synthetic artifact")]
    corpus.write_table(tmp_path/index.OUTPUTS[6], corpus.foundation.CHECKLIST_HEADER, check)
    # Presence-only files included in readable input identity; eligibility is
    # provided by the bounded synthetic corpus fixture above.
    for name in (corpus.MANIFEST, corpus.REVIEW, index.CORPUS_PATH, corpus.OUTPUTS[1], corpus.OUTPUTS[2]):
        corpus.write_table(tmp_path/name, ("fixture",), [dict(fixture="synthetic")])
    corpus.write_table(tmp_path/"results/rag/interface/interface_validation_checklist.csv", corpus.foundation.CHECKLIST_HEADER, check)
    for path, (dataset, prefix, run) in zip(retrieval.QUERY_SOURCES, retrieval.DATASETS):
        row = dict(query_id=prefix+"-VAL-000000", source_dataset=dataset, source_split="validation", rag_partition="development_validation",
                   classifier_model="lightgbm", classifier_run_id=run, predicted_class_id="0", predicted_class_label="benign" if prefix == "ustc" else "Non-Tor", confidence="0.9",
                   class_probabilities_json='{"class_0":0.9,"class_1":0.1}', traffic_representation_json=traffic_representation(50 if prefix == "ustc" else 62),
                   query_text="encrypted transport protocol", true_class_id="1", true_class_label="evaluation only", classifier_correct="False")
        corpus.write_table(tmp_path/path, tuple(row), [row])
        closure = dict(dataset=dataset, authoritative_run=run, stage_status="CLOSED_FROZEN", integrity_audit_status="PASS", finalization_status="PASS", frozen_for_retraining="YES", frozen_for_test_reuse="YES")
        if prefix == "cic":
            closure["phase_i_status"] = "COMPLETE"
        corpus.write_table(tmp_path/f"metadata/traffic/{prefix}_experiment_closure.csv", tuple(closure), [closure])
    return tmp_path


def change_query(root, **changes):
    path = retrieval.QUERY_SOURCES[0]
    header, rows = corpus.read_table(root, path)
    rows[0].update(changes)
    corpus.write_table(root/path, tuple(dict.fromkeys([*header, *changes])), rows)


def test_frozen_retrieval_constants():
    assert retrieval.METHODS == ("tfidf", "dense_bge", "hybrid_rrf")
    assert (retrieval.TOP_K, retrieval.DEPTH, retrieval.RRF_K) == (10, 50, 60)
    assert (retrieval.EXPECTED_CHUNKS, retrieval.DIMENSION, retrieval.TOKEN_LIMIT) == (919, 384, 512)


@pytest.mark.parametrize("path", [*retrieval.QUERY_SOURCES, *index.OUTPUTS[:7]])
def test_canonical_inputs_required(project, path):
    (project/path).unlink()
    with pytest.raises((ValueError, OSError)):
        retrieval.preflight(project)


@pytest.mark.parametrize("changes", [dict(source_split="TEST"), dict(source_split="train"), dict(rag_partition="test"), dict(query_text=""), dict(query_id=""), {"Flow ID": "forbidden"}, dict(query_text="true_class_label is benign"), dict(query_text="Src IP 127.0.0.1")])
def test_invalid_queries_rejected(project, changes):
    change_query(project, **changes)
    with pytest.raises(ValueError):
        retrieval.query_preflight(project, index.TokenPolicy(Tokenizer(), 512))


def test_global_query_identity_collision_blocks(project):
    change_query(project, query_id="cic-VAL-000000")
    with pytest.raises(ValueError, match="collision"):
        retrieval.query_preflight(project, index.TokenPolicy(Tokenizer(), 512))


def test_complete_token_preflight_reports_ids(project):
    change_query(project, traffic_representation_json=traffic_representation(**{"long " * 600 + "feature": 9.0}))
    with pytest.raises(retrieval.QueryTokenViolation, match="ustc-VAL-000000"):
        retrieval.query_preflight(project, index.TokenPolicy(Tokenizer(), 512))
    assert retrieval.main(["--project-root", str(project)]) == 2


def test_prefix_exactly_once():
    assert retrieval.dense_text("evidence") == "Represent this sentence for searching relevant passages: evidence"
    with pytest.raises(ValueError):
        retrieval.dense_text(retrieval.dense_text("evidence"))


@pytest.mark.parametrize("kind", ["mapping", "adversarial", "rejected", "dense", "lexical"])
def test_index_alignment_and_eligibility(project, kind):
    np = pytest.importorskip("numpy")
    if kind == "mapping":
        header, rows = corpus.read_table(project, index.OUTPUTS[4])
        rows[0]["chunk_id"] = "WRONG"
        corpus.write_table(project/index.OUTPUTS[4], header, rows)
    elif kind in {"adversarial", "rejected"}:
        header, rows = corpus.read_table(project, index.OUTPUTS[0])
        rows[0]["corpus_partition" if kind == "adversarial" else "document_id"] = "adversarial" if kind == "adversarial" else "TOR-SPEC-001"
        corpus.write_table(project/index.OUTPUTS[0], header, rows)
    elif kind == "dense":
        np.save(project/index.OUTPUTS[3], np.zeros((12, 3), dtype=np.float32), allow_pickle=False)
    else:
        from scipy.sparse import load_npz, save_npz
        matrix = load_npz(project/index.OUTPUTS[2]).tocsr()
        matrix.data[0] += 0.1
        save_npz(project/index.OUTPUTS[2], matrix)
    with pytest.raises(ValueError):
        retrieval.load_index(project)


def test_rank_ties_depth_and_topk():
    chunks = [dict(chunk_id=f"CHUNK-{i:03d}", document_id="DOC") for i in reversed(range(100))]
    ranking = retrieval.ranked([0.5]*100, chunks)
    assert len(ranking) == 50
    assert [chunks[i]["chunk_id"] for i, _, _ in ranking] == [f"CHUNK-{i:03d}" for i in range(50)]
    query = dict(query_id="query", source_dataset="dataset")
    rows = retrieval.result_rows(query, ranking, ranking, chunks)
    for method in retrieval.METHODS:
        selected = [r for r in rows if r["retrieval_method"] == method]
        assert len(selected) == 10
        assert [r["rank"] for r in selected] == list(range(1, 11))
        assert all(set(r) == set(retrieval.RESULT_HEADER) for r in selected)
    assert rows[0]["dense_score"] == "" and rows[0]["retrieval_score"] == rows[0]["tfidf_score"]


def test_rrf_union_missing_contribution_and_ties():
    chunks = [dict(chunk_id=c, document_id="D") for c in ("B", "A", "C")]
    lexical = [(0, 0.8, 1), (2, 0.2, 2)]
    dense = [(1, 0.9, 1)]
    rows = retrieval.rrf(lexical, dense, chunks)
    assert [i for i, _, _ in rows] == [1, 0, 2]
    assert rows[0][1] == pytest.approx(1/61)
    combined = retrieval.rrf([(0, 0.8, 1)], [(0, 0.7, 2)], chunks)
    assert combined[0][1] == pytest.approx(1/61 + 1/62)


def fake_query_embeddings(model, rows, counts):
    import numpy as np
    assert all(not set(retrieval.EVALUATION) & set(row) for row in rows)
    return index.normalize_dense(np.tile([1, 2, 3], (len(rows), 1)), len(rows))


def test_batch_boundaries_and_ground_truth_do_not_affect_rankings(project, monkeypatch):
    monkeypatch.setattr(retrieval, "encode_queries", fake_query_embeddings)
    evidence = retrieval.load_index(project)
    rows = [{k: row[k] for k in retrieval.QUERY_FIELDS} for row in retrieval.query_rows(project)]
    def collect(size):
        return [result for batch in retrieval.batches(rows, size) for _, results in retrieval.retrieve_batch(evidence, batch, SimpleNamespace(device="cpu"), {}) for result in results]
    singleton, batched = collect(1), collect(256)
    assert len(singleton) == len(batched)
    numeric = {"retrieval_score", "tfidf_score", "dense_score", "rrf_score"}
    for left, right in zip(singleton, batched):
        assert {k: v for k, v in left.items() if k not in numeric} == {k: v for k, v in right.items() if k not in numeric}
        for key in numeric:
            if left[key] == "":
                assert right[key] == ""
            else:
                assert left[key] == pytest.approx(right[key], abs=1e-6)
    original = collect(1)
    change_query(project, true_class_id="99", true_class_label="changed evaluation", classifier_correct="True")
    rows = [{k: row[k] for k in retrieval.QUERY_FIELDS} for row in retrieval.query_rows(project)]
    assert collect(1) == original
    assert [len(b) for b in retrieval.batches(range(513), 256)] == [256, 256, 1]


def test_dry_mode_write_free_and_no_encoding(project, monkeypatch, capsys):
    def forbidden(*args):
        raise AssertionError("Dry mode ran retrieval or loaded encoder")
    for name in ("load_query_encoder", "encode_queries", "retrieve_batch"):
        monkeypatch.setattr(retrieval, name, forbidden)
    before = snapshot(project)
    assert retrieval.main(["--project-root", str(project)]) == 0
    output = capsys.readouterr().out
    assert "READY_FOR_EXECUTION" in output and "QUERY_COUNT=2" in output
    assert snapshot(project) == before


def test_execute_canonical_outputs_and_identity_protection(project, monkeypatch):
    monkeypatch.setattr(retrieval, "load_query_encoder", lambda root, policy: SimpleNamespace(device="cpu"))
    monkeypatch.setattr(retrieval, "encode_queries", fake_query_embeddings)
    before = snapshot(project)
    assert retrieval.main(["--project-root", str(project), "--execute"]) == 0
    after = snapshot(project)
    assert set(after)-set(before) == set(retrieval.OUTPUTS)
    assert all(after[name] == value for name, value in before.items())
    header, records = corpus.read_table(project, retrieval.OUTPUTS[0])
    assert header == list((*retrieval.QUERY_HEADER, *retrieval.EVALUATION))
    assert len(records) == 2 and all(r["retrieval_status"] == "PASS" for r in records)
    assert records[0]["true_class_id"] == "1"
    for name, expected in ((retrieval.OUTPUTS[1], retrieval.RESULT_HEADER), (retrieval.OUTPUTS[2], retrieval.SUMMARY_HEADER), (retrieval.OUTPUTS[3], index.CONFIG_HEADER), (retrieval.OUTPUTS[4], corpus.foundation.CHECKLIST_HEADER)):
        assert corpus.read_table(project, name)[0] == list(expected)
    assert retrieval.main(["--project-root", str(project), "--execute"]) == 0
    assert snapshot(project) == after
    config = retrieval.config_table(project, retrieval.OUTPUTS[3])
    for key in ("retrieval_top_k", "component_candidate_depth", "dense_query_prefix", "rrf_k", "index_configuration_json", "query_sources"):
        with pytest.raises(ValueError, match="identity"):
            retrieval.overwrite_guard(project, dict(config, **{key: "different"}))


def test_zero_result_queries_preserved(project, monkeypatch):
    monkeypatch.setattr(retrieval, "load_query_encoder", lambda root, policy: SimpleNamespace(device="cpu"))
    monkeypatch.setattr(retrieval, "retrieve_batch", lambda evidence, rows, model, counts: ((r, []) for r in rows))
    assert retrieval.main(["--project-root", str(project), "--execute"]) == 0
    _, records = corpus.read_table(project, retrieval.OUTPUTS[0])
    assert len(records) == 2 and all(r["retrieval_status"] == "ZERO_RESULT" for r in records)
    _, summaries = corpus.read_table(project, retrieval.OUTPUTS[2])
    overall = next(r for r in summaries if r["summary_level"] == "overall")
    assert overall["zero_result_queries"] == "2" and overall["mean_top1_score"] == ""


def test_dense_query_encoder_prefix_normalization_and_token_guard(project, monkeypatch):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    captured = []
    class Encoder:
        device = torch.device("cpu")
        tokenizer = Tokenizer()
        max_seq_length = 512
        def __getitem__(self, key):
            return SimpleNamespace(do_lower_case=False)
        def encode(self, texts, **kwargs):
            captured.extend(texts)
            assert kwargs["batch_size"] == 256 and kwargs["prompt"] == ""
            assert kwargs["convert_to_tensor"] is True
            assert kwargs["show_progress_bar"] is False
            return np.tile([3, 4, 0], (len(texts), 1))
    rows = list(retrieval.query_rows(project))
    counts, _ = retrieval.query_preflight(project, index.TokenPolicy(Tokenizer(), 512))
    matrix = retrieval.encode_queries(Encoder(), rows, counts)
    assert captured == [retrieval.dense_text(r["retrieval_query_text"]) for r in rows]
    assert matrix.dtype == torch.float32
    np.testing.assert_allclose(np.linalg.norm(matrix.cpu().numpy(), axis=1), 1)
    captured.clear()
    rows[0]["retrieval_query_text"] = "oversized "*600
    with pytest.raises(retrieval.QueryTokenViolation):
        retrieval.encode_queries(Encoder(), rows, counts)
    assert captured == []


def test_phase_i_closures_are_required(project):
    path = project/"metadata/traffic/ustc_experiment_closure.csv"
    header, rows = corpus.read_table(project, path.relative_to(project).as_posix())
    rows[0]["frozen_for_test_reuse"] = "NO"
    corpus.write_table(path, header, rows)
    with pytest.raises(ValueError, match="protection"):
        retrieval.protect_phase_i(project)


def test_static_stage_boundaries_and_local_loader():
    source = Path(retrieval.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    attrs = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not attrs & {"fit", "fit_transform", "predict", "search", "savefig", "sha256", "md5"}
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not imports & {"lightgbm", "xgboost", "openai", "requests", "matplotlib"}
    loader = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "SentenceTransformer")
    kwargs = {k.arg: k.value for k in loader.keywords}
    assert kwargs["local_files_only"].value is True and kwargs["trust_remote_code"].value is False
    assert "test.csv" not in source
    assert not {"evidence_status", "relevant_if_known", "precision", "recall", "ndcg", "mrr"} & set(retrieval.RESULT_HEADER)


@pytest.mark.parametrize("available,expected", [(True, "cuda"), (False, "cpu")])
def test_explicit_device_selection(available, expected, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    assert retrieval.dense_device() == expected


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_encoder_explicit_placement_and_local_only(project, monkeypatch, device):
    import sys
    torch = pytest.importorskip("torch")
    calls = []
    monkeypatch.setattr(retrieval, "dense_device", lambda: device)
    monkeypatch.setattr(torch, "manual_seed", lambda value: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda value: None)
    class Encoder:
        max_seq_length = 512
        def __init__(self, path, **kwargs):
            assert path == str((project/index.MODEL_DIRECTORY).resolve())
            assert kwargs == dict(device=device, local_files_only=True, trust_remote_code=False)
            self.device = "unplaced"
        def to(self, selected):
            calls.append(selected)
            self.device = selected
            return self
        def eval(self):
            return self
        def get_sentence_embedding_dimension(self):
            return 3
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Encoder))
    model = retrieval.load_query_encoder(project, index.TokenPolicy(Tokenizer(), 512))
    assert model.device == device and calls == [device]


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_gpu_configuration_fields(project, monkeypatch, capsys, device):
    monkeypatch.setattr(retrieval, "dense_device", lambda: device)
    evidence = retrieval.load_index(project)
    config = retrieval.configuration(project, evidence, 2)
    assert config["dense_device"] == config["dense_similarity_device"] == device
    assert config["torch_cuda_available"] == config["gpu_acceleration_used"] == ("YES" if device == "cuda" else "NO")
    assert config["dense_batch_size"] == config["dense_encoder_microbatch_size"] == "256"
    assert retrieval.main(["--project-root", str(project)]) == 0
    output = capsys.readouterr().out
    assert "DENSE_DEVICE=" + device in output and "DENSE_BATCH_SIZE=256" in output


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_passages_cached_once_and_exact_ties(project, monkeypatch, device):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA hardware unavailable")
    evidence = retrieval.load_index(project)
    monkeypatch.setattr(retrieval, "encode_queries", fake_query_embeddings)
    rows = [{k: r[k] for k in retrieval.QUERY_FIELDS} for r in retrieval.query_rows(project)]
    model = SimpleNamespace(device=device)
    first = list(retrieval.retrieve_batch(evidence, rows, model, {}))
    pointer = evidence.passage_tensor.data_ptr()
    assert evidence.passage_tensor.device.type == device
    assert evidence.passage_tensor.dtype == torch.float32
    second = list(retrieval.retrieve_batch(evidence, rows, model, {}))
    assert evidence.passage_tensor.data_ptr() == pointer
    assert first == second
    expected_ids = sorted(r["chunk_id"] for r in evidence.chunks)[:10]
    for query, results in first:
        dense = [r for r in results if r["retrieval_method"] == "dense_bge"]
        assert [r["chunk_id"] for r in dense] == expected_ids


def test_oom_fails_without_fallback_or_batch_reduction(project, monkeypatch, capsys):
    calls = []
    def out_of_memory(root, policy):
        calls.append("attempt")
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(retrieval, "load_query_encoder", out_of_memory)
    before = snapshot(project)
    assert retrieval.main(["--project-root", str(project), "--execute"]) == 2
    assert calls == ["attempt"]
    assert retrieval.DENSE_BATCH_SIZE == 256
    assert snapshot(project) == before
    output = capsys.readouterr().out
    assert "CUDA out of memory" in output and "no device or batch-size fallback" in output


def test_one_encode_call_for_full_dense_batch(project):
    torch = pytest.importorskip("torch")
    calls = []
    rows = [dict(query_id=str(i), retrieval_query_text="encrypted protocol") for i in range(256)]
    counts = {r["query_id"]: index.TokenPolicy(Tokenizer(), 512).count(retrieval.dense_text(r["retrieval_query_text"])) for r in rows}
    class Encoder:
        device = torch.device("cpu")
        max_seq_length = 512
        tokenizer = Tokenizer()
        def __getitem__(self, key):
            return SimpleNamespace(do_lower_case=False)
        def encode(self, texts, **kwargs):
            calls.append((len(texts), kwargs["batch_size"]))
            return torch.ones((len(texts), 3))
    values = retrieval.encode_queries(Encoder(), rows, counts)
    assert calls == [(256, 256)] and values.shape == (256, 3)
    with pytest.raises(ValueError, match="batch"):
        retrieval.encode_queries(Encoder(), rows + [rows[0]], counts)
    assert len(calls) == 1


@pytest.mark.parametrize("z,expected", [(3.0, "extremely high"), (2.999, "very high"), (2.0, "very high"), (1.999, "high"), (1.0, "high"), (0.999, "moderately high"), (0.5, "moderately high"), (0.499, "typical"), (0, "typical"), (-0.499, "typical"), (-0.5, "moderately low"), (-0.999, "moderately low"), (-1.0, "low"), (-1.999, "low"), (-2.0, "very low"), (-2.999, "very low"), (-3.0, "extremely low")])
def test_exact_standardized_semantic_bins(z, expected):
    assert retrieval.semantic_bin(z) == expected


@pytest.mark.parametrize("name,expected", [
    ("src2dst_stddev_piat_ms", "source to destination standard deviation packet inter-arrival time ms"),
    ("dst2src_mean_ps", "destination to source mean packet size"),
    ("bidirectional_syn_packets", "bidirectional SYN packets"),
    ("Fwd IAT Std", "forward inter-arrival time standard deviation"),
    ("Bwd Packet/Bulk Avg", "backward packet/bulk average"),
    ("Fwd Act Data Pkts", "forward act data packets"),
    ("ACK PSH RST FIN Flag Count", "ACK PSH RST FIN flag count"),
])
def test_readable_ustc_and_cic_features(name, expected):
    assert retrieval.readable_feature(name) == expected


def test_top12_magnitude_ties_and_nonordinal_exclusion():
    features = {f"feature_{i:02d}": float(i) for i in range(48)}
    features.update(Protocol=9999.0, ip_version=-9999.0)
    profile = retrieval.semantic_profile("USTC-TFC2016", "malware", json.dumps(dict(predictor_count=50, features=features)))
    selected = json.loads(profile["semantic_profile_feature_names_json"])
    assert selected == [f"feature_{i:02d}" for i in range(47, 35, -1)]
    assert "protocol is" not in profile["retrieval_query_text"].lower()
    assert "ip version is" not in profile["retrieval_query_text"].lower()
    features = {f"feature_{i:02d}": (-2.0 if i % 2 else 2.0) for i in reversed(range(50))}
    value = json.dumps(dict(predictor_count=50, features=features))
    selected = json.loads(retrieval.semantic_profile("USTC-TFC2016", "benign", value)["semantic_profile_feature_names_json"])
    assert selected == [f"feature_{i:02d}" for i in range(12)]


@pytest.mark.parametrize("code", ["protocol", "PROTOCOL", "ip_version", "IP_VERSION", "ip version", "Ip Version"])
def test_nonordinal_codes_never_get_direction(code):
    profile = retrieval.semantic_profile("USTC-TFC2016", "benign", traffic_representation(**{code: 999.0}))
    assert code not in json.loads(profile["semantic_profile_feature_names_json"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "1.0", None])
def test_invalid_standardized_features_block(value):
    with pytest.raises(ValueError):
        retrieval.semantic_profile("USTC-TFC2016", "benign", traffic_representation(invalid=value))


@pytest.mark.parametrize("value", [{}, {"predictor_count": 50}, {"predictor_count": 50, "features": []}, {"predictor_count": 50, "features": {"duration": 2.0}}, {"predictor_count": True, "features": {}}])
def test_missing_or_inconsistent_representation_blocks(value):
    with pytest.raises(ValueError):
        retrieval.semantic_profile("USTC-TFC2016", "benign", json.dumps(value))


@pytest.mark.parametrize("label,expected", [("benign", "benign"), ("malware", "malware"), ("Tor", "Tor"), ("VPN", "VPN"), ("Non-Tor", "non-Tor"), ("NonVPN", "non-VPN")])
def test_predicted_class_meaning_preserved(label, expected):
    profile = retrieval.semantic_profile("USTC-TFC2016", label, traffic_representation())
    assert f"Predicted traffic class: {expected}." in profile["retrieval_query_text"]


def test_separate_semantic_query_preserves_audit_and_excludes_metadata(project):
    original = "Original classifier observation; traffic_representation_reference retained for audit."
    change_query(project, query_text=original, traffic_representation_json=traffic_representation(src2dst_stddev_piat_ms=2.123456789))
    first = list(retrieval.query_rows(project))[0]
    assert first["query_text"] == original
    assert first["confidence"] == "0.9" and first["class_probabilities_json"]
    text = first["retrieval_query_text"]
    assert "source to destination standard deviation packet inter-arrival time ms is very high" in text
    for forbidden in ("2.123456789", "0.9", "USTC-TFC2016", "ustc-VAL-000000", "20260919T183945Z", "lightgbm", "traffic_representation_reference", "true_class", "classifier_correct", "SUPPORTING", "CONTRADICTING", "Future checks:", "{", "}"):
        assert forbidden not in text
    change_query(project, true_class_id="99", true_class_label="unrelated ground truth", classifier_correct="True", confidence="0.8", class_probabilities_json='{"x":0.8}')
    assert list(retrieval.query_rows(project))[0]["retrieval_query_text"] == text


def test_original_query_length_does_not_drive_v2_token_preflight(project):
    change_query(project, query_text="audit only " * 1000)
    counts, _ = retrieval.query_preflight(project, index.TokenPolicy(Tokenizer(), 512))
    assert max(counts.values()) <= 512
    assert 0 < counts.unique_retrieval_query_texts <= len(counts)


def test_tfidf_consumes_only_retrieval_query_text(project, monkeypatch):
    evidence = retrieval.load_index(project)
    captured = []
    original = evidence.vectorizer.transform
    def transform(texts):
        captured.extend(texts)
        return original(texts)
    monkeypatch.setattr(evidence.vectorizer, "transform", transform)
    monkeypatch.setattr(retrieval, "encode_queries", fake_query_embeddings)
    rows = [{key: row[key] for key in retrieval.QUERY_FIELDS} for row in retrieval.query_rows(project)]
    list(retrieval.retrieve_batch(evidence, rows, SimpleNamespace(device="cpu"), {}))
    assert captured == [r["retrieval_query_text"] for r in rows]
    assert captured != [r["query_text"] for r in rows]


def test_v1_preserved_v2_outputs_and_diagnostics(project, monkeypatch):
    prior = project/"results/rag/phase2_dev_retrieval_v1"
    prior.mkdir(parents=True)
    (prior/"query_records.csv").write_text("Historical v1 audit evidence", encoding="utf-8")
    (prior/"experiment_configuration.csv").write_text("Historical v1 identity", encoding="utf-8")
    before = snapshot(prior)
    monkeypatch.setattr(retrieval, "load_query_encoder", lambda root, policy: SimpleNamespace(device="cpu"))
    monkeypatch.setattr(retrieval, "encode_queries", fake_query_embeddings)
    assert retrieval.EXPERIMENT_ID == "phase2_dev_retrieval_v2"
    assert retrieval.CONFIGURATION_NAME == "semantic_traffic_profile_lexical_dense_rrf_top10"
    assert len(retrieval.OUTPUTS) == 6 and all("v2" in name for name in retrieval.OUTPUTS)
    assert retrieval.main(["--project-root", str(project)]) == 0
    assert snapshot(prior) == before
    assert retrieval.main(["--project-root", str(project), "--execute"]) == 0
    assert snapshot(prior) == before
    header, diagnostics = corpus.read_table(project, retrieval.OUTPUTS[5])
    assert header == list(retrieval.DIAGNOSTICS_HEADER)
    assert {r["summary_level"] for r in diagnostics} == {"overall", "method", "dataset_method"}
    _, rows = corpus.read_table(project, retrieval.OUTPUTS[0])
    assert rows[0]["query_text"] == "encrypted transport protocol"
    assert rows[0]["retrieval_query_text"] != rows[0]["query_text"]
    assert len(json.loads(rows[0]["semantic_profile_feature_names_json"])) == 12
    config = retrieval.config_table(project, retrieval.OUTPUTS[3])
    for key in ("original_query_text_used_for_retrieval", "raw_feature_values_emitted_in_retrieval_text", "ground_truth_used_for_retrieval"):
        assert config[key] == "NO"
    assert config["traffic_representation_used_for_retrieval"] == "YES"
    assert config["semantic_profile_top_features"] == "12"
    assert config["retrieval_query_schema"] == "semantic_traffic_profile_v2"


def test_diagnostics_patterns_chunks_documents_and_denominators():
    diagnostics = retrieval.Diagnostics()
    for i, pattern in enumerate((("A", "B"), ("A", "B"), ("B", "C"))):
        query = dict(query_id=str(i), source_dataset="USTC-TFC2016")
        rows = [dict(retrieval_method=method, rank=rank, chunk_id=chunk, document_id="DOC-1" if chunk in {"A", "B"} else "DOC-2") for method in retrieval.METHODS for rank, chunk in enumerate(pattern, 1)]
        diagnostics.add(query, rows)
    summaries = list(diagnostics.rows())
    method = next(r for r in summaries if r["summary_level"] == "method" and r["retrieval_method"] == "tfidf")
    assert method["query_count"] == 3 and method["retrieval_rows"] == 6
    assert method["unique_retrieved_chunks"] == 3 and method["unique_retrieved_documents"] == 2
    assert method["unique_top10_patterns"] == 2
    assert method["most_common_top10_pattern_count"] == 2
    assert method["most_common_top10_pattern_share"] == pytest.approx(2/3)
    assert method["most_frequent_chunk_id"] == "B"
    assert method["most_frequent_chunk_count"] == 3
    assert method["most_frequent_chunk_share"] == 0.5
    overall = next(r for r in summaries if r["summary_level"] == "overall")
    assert overall["query_count"] == 3 and overall["retrieval_rows"] == 18
    assert overall["most_common_top10_pattern_count"] == 6
    assert overall["most_common_top10_pattern_share"] == pytest.approx(2/3)
    assert all(set(row) == set(retrieval.DIAGNOSTICS_HEADER) for row in summaries)


def test_available_numeric_probability_columns_remain_audit_only(project):
    change_query(project, prob_class_0="0.9", prob_class_1="0.1")
    row = list(retrieval.query_rows(project))[0]
    counts, optional = retrieval.query_preflight(project, index.TokenPolicy(Tokenizer(), 512))
    assert {"prob_class_0", "prob_class_1"} <= set(optional)
    assert row["prob_class_0"] == "0.9" and row["prob_class_1"] == "0.1"
    assert "0.9" not in row["retrieval_query_text"] and "0.1" not in row["retrieval_query_text"]
    assert max(counts.values()) <= 512
