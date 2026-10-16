"""Synthetic tokenizer tests; no Qwen weights, vLLM, GPU, or network."""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.rag import audit_qwen_prompt_lengths as audit
from src.rag import phase2_common as common
from src.rag import materialize_constrained_llm_experiment as prompts
from src.rag import build_trusted_contexts as contexts


class FakeTokenizer:
    def __init__(self, count=2195, nested=False):
        self.count, self.nested = count, nested
        self.calls = 0

    def apply_chat_template(self, messages, **kwargs):
        self.calls += 1
        assert len(messages) == 1 and messages[0]["role"] == "user"
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors=None, truncation=False, padding=False)
        ids = list(range(self.count))
        return {"input_ids": [ids] if self.nested else ids, "attention_mask": [1]*self.count}


@pytest.mark.parametrize("nested", [False, True])
def test_counts_input_ids_not_batchencoding_keys(nested):
    tokenizer = FakeTokenizer(2999, nested)
    assert audit.prompt_tokens(tokenizer, "synthetic prompt") == 2999
    assert tokenizer.calls == 1


@pytest.mark.parametrize("count,within", [(3584, "YES"), (3585, "NO")])
def test_exact_completion_reservation_boundary(count, within):
    row = dict(query_id="Q", source_dataset="USTC-TFC2016", classifier_prediction="benign", prompt_text="synthetic")
    result = audit.audit_row(row, FakeTokenizer(count))
    assert set(result) == set(audit.HEADER)
    assert result["qwen_prompt_tokens"] == count
    assert result["reserved_completion_tokens"] == 512
    assert result["reserved_total_tokens"] == count + 512
    assert result["within_limit"] == within
    assert result["max_model_len"] == 4096 and result["safe_prompt_limit"] == 3584
    assert audit.SAFE_MAX_PROMPT_TOKENS == 3584


@pytest.mark.parametrize("count,within", [(4096, "YES"), (4097, "NO")])
def test_4608_completion_reservation_boundary(count, within):
    row = dict(query_id="Q", source_dataset="USTC-TFC2016", classifier_prediction="benign", prompt_text="synthetic")
    result = audit.audit_row(row, FakeTokenizer(count), max_model_len=4608, reserved_completion_tokens=512)
    assert result["within_limit"] == within
    assert result["max_model_len"] == 4608
    assert result["safe_prompt_limit"] == 4096
    assert result["reserved_total_tokens"] == count + 512
    assert "within_4096" not in result


@pytest.mark.parametrize("maximum,reserve", [(0, 512), (-1, 512), (4096, -1), (512, 512), (512, 513), (True, 0), (4096, 1.5)])
def test_invalid_runtime_limits_rejected(maximum, reserve):
    with pytest.raises(ValueError):
        audit.safe_prompt_limit(maximum, reserve)


@pytest.mark.parametrize("ids", [[], [[]], [[1], [2]], [True], ["1"]])
def test_invalid_token_shapes_rejected(ids):
    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: {"input_ids": ids})
    with pytest.raises(ValueError):
        audit.prompt_tokens(tokenizer, "synthetic")


def test_tokenizer_load_is_local_only(tmp_path, monkeypatch):
    path = tmp_path/audit.MODEL/"tokenizer_config.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    calls = []
    def loader(path, **kwargs):
        calls.append((path, kwargs))
        return FakeTokenizer()
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=loader)))
    monkeypatch.setattr(audit.prompts.contexts.index.v1, "offline_mode", lambda: None)
    assert isinstance(audit.load_tokenizer(tmp_path), FakeTokenizer)
    assert calls == [(str((tmp_path/audit.MODEL).resolve()), dict(local_files_only=True, trust_remote_code=False))]


def project(root, case, monkeypatch, count=3):
    query_rows, prompt_rows = [], []
    for i in range(count):
        query = dict(case.query, query_id=f"Q-{i:05d}")
        context = contexts.construct(query, [], case.chunks, case.verified, case.policy)
        query_rows.append({k: query[k] for k in prompts.QUERY_FIELDS})
        prompt_rows.append(prompts.prompt_record(context, "v3"))
    common.corpus.write_table(root/common.QUERY_PATH, prompts.QUERY_FIELDS, query_rows)
    common.corpus.write_table(root/prompts.V3_OUTPUTS[0], prompts.HEADER, prompt_rows)
    for name in prompts.V3_OUTPUTS[1:3]:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic completed metadata", encoding="utf-8")
    common.corpus.write_table(root/prompts.V3_OUTPUTS[3], prompts.POLICY_HEADER, prompts.policy_rows("v3"))
    monkeypatch.setattr(common, "completed", lambda *args: {"prompt_schema_version": "constrained_evidence_v3"})
    monkeypatch.setattr(common, "QUERY_COUNTS", {case.query["source_dataset"]: count})
    return query_rows, prompt_rows


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("tokens,code", [(3584, 0), (3585, 2)])
def test_complete_dry_and_execute_gate(tmp_path, phase2_case, monkeypatch, tokens, code):
    project(tmp_path, phase2_case, monkeypatch)
    tokenizer = FakeTokenizer(tokens)
    monkeypatch.setattr(audit, "load_tokenizer", lambda root: tokenizer)
    before = snapshot(tmp_path)
    argv = ["--project-root", str(tmp_path)]
    assert audit.main(argv) == code
    assert snapshot(tmp_path) == before
    assert tokenizer.calls == 3
    assert audit.main([*argv, "--execute"]) == code
    assert tokenizer.calls == 6
    after = snapshot(tmp_path)
    outputs = audit.output_paths()
    assert set(after)-set(before) == set(outputs)
    assert all(after[k] == v for k, v in before.items())
    rows = list(common.rows(tmp_path, outputs[0]))
    summary = list(common.rows(tmp_path, outputs[1]))[0]
    assert len(rows) == 3 and all(int(row["qwen_prompt_tokens"]) == tokens for row in rows)
    assert set(summary) == set(audit.SUMMARY_HEADER)
    assert int(summary["over_limit_with_reservation"]) == (3 if code else 0)
    assert summary["max_model_len"] == "4096"
    assert summary["gate_status"] == ("FAIL" if code else "PASS")
    assert summary["max_query_id"] == "Q-00000"
    assert not list(tmp_path.glob(".rag_*"))
    assert audit.main([*argv, "--execute"]) == 2
    assert snapshot(tmp_path) == after


@pytest.mark.parametrize("maximum,reserve,tokens,code", [
    (4096, 512, 3585, 2),
    (4608, 512, 3585, 0),
    (4608, 512, 4096, 0),
    (4608, 512, 4097, 2),
    (4608, 256, 4097, 0),
])
def test_cli_runtime_limits_control_gate_and_saved_metadata(tmp_path, phase2_case, monkeypatch, maximum, reserve, tokens, code):
    project(tmp_path, phase2_case, monkeypatch)
    monkeypatch.setattr(audit, "load_tokenizer", lambda root: FakeTokenizer(tokens))
    argv = ["--project-root", str(tmp_path), "--max-model-len", str(maximum), "--reserved-completion-tokens", str(reserve)]
    before = snapshot(tmp_path)
    assert audit.main(argv) == code
    assert snapshot(tmp_path) == before
    assert audit.main([*argv, "--execute"]) == code
    outputs = audit.output_paths(maximum, reserve)
    rows = list(common.rows(tmp_path, outputs[0]))
    summary = list(common.rows(tmp_path, outputs[1]))[0]
    for record in [*rows, summary]:
        assert record["max_model_len"] == str(maximum)
        assert record["reserved_completion_tokens"] == str(reserve)
        assert record["safe_prompt_limit"] == str(maximum-reserve)
        assert "within_4096" not in record and "over_4096_with_reservation" not in record
    assert all(row["within_limit"] == ("NO" if code else "YES") for row in rows)
    assert all(int(row["reserved_total_tokens"]) == tokens+reserve for row in rows)
    assert int(summary["over_limit_with_reservation"]) == (len(rows) if code else 0)
    assert common.parse_json(summary["dataset_over_limit_json"]) == {"USTC-TFC2016": len(rows) if code else 0}
    assert summary["gate_status"] == ("FAIL" if code else "PASS")


@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "missing", "test", "bad_grounding", "ground_truth"])
def test_invalid_population_never_published(tmp_path, phase2_case, monkeypatch, mutation):
    queries, records = project(tmp_path, phase2_case, monkeypatch)
    if mutation == "duplicate": records[-1] = records[0]
    if mutation == "unknown": records[-1]["query_id"] = "UNKNOWN"
    if mutation == "missing": records.pop()
    if mutation == "test":
        queries[-1]["source_split"] = "test"
        common.corpus.write_table(tmp_path/common.QUERY_PATH, prompts.QUERY_FIELDS, queries)
    if mutation == "bad_grounding": records[-1]["prompt_text"] = "No grounded instructions"
    if mutation == "ground_truth":
        instructions, _, payload = records[-1]["prompt_text"].partition("INPUT_JSON:\n")
        value = common.parse_json(payload)
        value["true_class_label"] = "forbidden"
        records[-1]["prompt_text"] = instructions + "INPUT_JSON:\n" + common.json_text(value)
    common.corpus.write_table(tmp_path/prompts.V3_OUTPUTS[0], prompts.HEADER, records)
    monkeypatch.setattr(audit, "load_tokenizer", lambda root: FakeTokenizer())
    before = snapshot(tmp_path)
    assert audit.main(["--project-root", str(tmp_path), "--execute"]) == 2
    assert snapshot(tmp_path) == before
    assert not any((tmp_path/name).exists() for name in audit.output_paths())


def test_sql_statistics_and_streaming_release(tmp_path, phase2_case, monkeypatch):
    project(tmp_path, phase2_case, monkeypatch, count=41)
    live = peak = calls = 0
    class TokenIds(list):
        def __init__(self, count):
            nonlocal live, peak
            super().__init__(range(count))
            live += 1
            peak = max(peak, live)
        def __del__(self):
            nonlocal live
            live -= 1
    def encode(*args, **kwargs):
        nonlocal calls
        calls += 1
        return {"input_ids": TokenIds(calls), "attention_mask": []}
    monkeypatch.setattr(audit, "load_tokenizer", lambda root: SimpleNamespace(apply_chat_template=encode))
    assert audit.main(["--project-root", str(tmp_path), "--execute"]) == 0
    assert calls == 41 and peak == 1 and live == 0
    summary = list(common.rows(tmp_path, audit.output_paths()[1]))[0]
    assert float(summary["mean_prompt_tokens"]) == float(summary["median_prompt_tokens"]) == 21
    assert int(summary["p95_prompt_tokens"]) == 39 and int(summary["p99_prompt_tokens"]) == 41


def test_static_tokenizer_only_no_inference():
    source = Path(audit.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not imports & {"torch", "vllm", "requests", "httpx", "openai", "pandas"}
    assert not names & {"AutoModel", "AutoModelForCausalLM", "LLM", "SamplingParams"}
    assert "len(encoded)" not in source.split("# Never len(encoded)")[0]


def test_config_scoped_paths_and_legacy_preservation(tmp_path, phase2_case, monkeypatch):
    project(tmp_path, phase2_case, monkeypatch)
    for name in audit.LEGACY_OUTPUTS:
        path = tmp_path/name
        path.write_text("historical failed audit", encoding="utf-8")
    before = snapshot(tmp_path)
    monkeypatch.setattr(audit, "load_tokenizer", lambda root: FakeTokenizer(3700))
    for maximum, code in ((4096, 2), (4608, 0)):
        paths = audit.output_paths(maximum, 512)
        assert paths == audit.output_paths(maximum, 512)
        assert all(f"token_audits/maxlen_{maximum}_completion_512/" in name for name in paths)
        argv = ["--project-root", str(tmp_path), "--max-model-len", str(maximum)]
        saved = snapshot(tmp_path)
        assert audit.main(argv) == code
        assert snapshot(tmp_path) == saved
        assert audit.main([*argv, "--execute"]) == code
        summary = list(common.rows(tmp_path, paths[1]))[0]
        assert summary["configuration_key"] == f"maxlen_{maximum}_completion_512"
    assert set(audit.output_paths(4096, 512)).isdisjoint(audit.output_paths(4608, 512))
    assert all((tmp_path/name).read_bytes() == value for name, value in before.items())
