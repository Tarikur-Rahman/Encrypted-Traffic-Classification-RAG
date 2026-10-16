import pytest
from types import SimpleNamespace

from src.rag import run_v3_grounding_diagnostic as diagnostic
from src.rag import llm_v3_runtime as runtime

common, prompts = runtime.common, runtime.prompts


def selected_prompts(case, version="v3"):
    result = []
    for i, (dataset, label) in enumerate(diagnostic.PAIRS):
        context = prompts.contexts.construct(case.query, [], case.chunks, case.verified, case.policy)
        context.update(query_id=f"Q-{i:03d}", source_dataset=dataset, classifier_prediction=label)
        result.append(prompts.prompt_record(context, version))
    return result


def api_response(payload, tokens=100):
    identity = payload["response_format"]["json_schema"]["schema"]["properties"]["query_id"]["enum"][0]
    value = dict(query_id=identity, reasoning_decision="ABSTAIN", evidence_status="INSUFFICIENT", supported_class=None, explanation="Supplied evidence is insufficient.", cited_chunk_ids=[])
    return dict(choices=[dict(finish_reason="stop", message=dict(content=common.json_text(value), reasoning_content="DO NOT SAVE HIDDEN CONTENT"))], usage=dict(prompt_tokens=tokens, completion_tokens=1, total_tokens=tokens+1))


def test_six_selection_first_occurrence(phase2_case):
    rows = selected_prompts(phase2_case)
    assert diagnostic.select_six(iter(rows + [dict(r, query_id="LATER-"+r["query_id"]) for r in rows])) == rows
    assert len(rows) == 6
    with pytest.raises(ValueError):
        diagnostic.select_six(rows[:-1])


def test_schema_compatible_without_unique_items(phase2_case):
    prompt = selected_prompts(phase2_case)[0]
    schema = runtime.response_schema(prompt)
    assert set(schema["required"]) == set(runtime.FIELDS) == set(schema["properties"])
    assert "uniqueItems" not in common.json_text(schema)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["cited_chunk_ids"]["maxItems"] == 0


@pytest.mark.parametrize("mutation", ["wrong_id", "unknown_citation", "duplicate_citation", "unsupported_class", "contradicted_same", "supported_different", "abstain_with_class", "conflicting_no_citations", "extra_field"])
def test_semantic_cross_field_rejection(phase2_case, mutation):
    prompt = dict(selected_prompts(phase2_case)[-2], allowed_chunk_ids='["C1"]')
    value = dict(query_id=prompt["query_id"], reasoning_decision="SUPPORT_CLASS", evidence_status="SUPPORTED", supported_class="benign", explanation="Synthetic support.", cited_chunk_ids=["C1"])
    if mutation == "wrong_id": value["query_id"] = "UNKNOWN"
    if mutation == "unknown_citation": value["cited_chunk_ids"] = ["C2"]
    if mutation == "duplicate_citation": value["cited_chunk_ids"] = ["C1", "C1"]
    if mutation == "unsupported_class": value["supported_class"] = "Tor"
    if mutation == "contradicted_same": value["evidence_status"] = "CONTRADICTED"
    if mutation == "supported_different": value["supported_class"] = "malware"
    if mutation == "abstain_with_class": value.update(reasoning_decision="ABSTAIN", evidence_status="INSUFFICIENT")
    if mutation == "conflicting_no_citations": value.update(reasoning_decision="ABSTAIN", evidence_status="CONFLICTING", supported_class=None, cited_chunk_ids=[])
    if mutation == "extra_field": value["hidden_reasoning"] = "not allowed"
    with pytest.raises(ValueError):
        runtime.validate_logical(value, prompt)


@pytest.mark.parametrize("token_count,code", [(100, 0), (4097, 2)])
def test_diagnostic_mocked_http_and_preflight(tmp_path, phase2_case, monkeypatch, token_count, code, capsys):
    rows = selected_prompts(phase2_case)
    for name in [common.QUERY_PATH, *prompts.V3_OUTPUTS]:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic input", encoding="utf-8")
    def prepare(root, connection):
        connection.execute("CREATE TABLE prompts (ordinal INTEGER, payload TEXT)")
        connection.executemany("INSERT INTO prompts VALUES (?, ?)", [(i, common.json_text(row)) for i, row in enumerate(rows)])
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] and kwargs["add_generation_prompt"]
            return {"input_ids": list(range(token_count)), "attention_mask": []}
    monkeypatch.setattr(runtime, "prepare_population", prepare)
    monkeypatch.setattr(diagnostic.audit, "load_tokenizer", lambda root: Tokenizer())
    requests = []
    def post(endpoint, payload, timeout):
        requests.append(payload)
        return api_response(payload, token_count)
    monkeypatch.setattr(runtime, "post_json", post)
    argv = ["--project-root", str(tmp_path)]
    assert diagnostic.main(argv) == code
    assert requests == []
    assert not any((tmp_path/name).exists() for name in diagnostic.OUTPUTS)
    assert diagnostic.main([*argv, "--execute"]) == code
    if code:
        assert requests == []
    else:
        assert len(requests) == 6
        assert all(request["messages"][0]["content"] == row["prompt_text"] for request, row in zip(requests, rows))
        records = [common.parse_json(line) for line in (tmp_path/diagnostic.OUTPUTS[0]).read_text().splitlines()]
        assert len(records) == 6 and all(record["api_prompt_tokens"] == token_count for record in records)
        assert "DO NOT SAVE HIDDEN CONTENT" not in (tmp_path/diagnostic.OUTPUTS[0]).read_text()
        assert "SIX_CLASS_DIAGNOSTIC=STRUCTURAL_PASS" in capsys.readouterr().out
        assert not (tmp_path/"results/rag/llm/v3/execution").exists()


@pytest.mark.parametrize("endpoint", ["https://api.example.com/v1/chat/completions", "http://10.0.0.1/v1/chat/completions", "http://127.0.0.1:8000/other"])
def test_external_endpoints_rejected(endpoint):
    with pytest.raises(ValueError):
        runtime.endpoint_url(endpoint)


def test_retry_exhaustion_preserves_v4_validation_reason_and_chain(phase2_case, monkeypatch):
    prompt = dict(selected_prompts(phase2_case, "v4")[0], allowed_chunk_ids='["C1"]')
    value = dict(query_id=prompt["query_id"], reasoning_decision="SUPPORT_CLASS", evidence_status="SUPPORTED",
                 supported_class=prompt["classifier_prediction"], explanation="No evidence quote.", cited_chunk_ids=["C1"])
    original = common.json_text(value)
    monkeypatch.setattr(runtime.urllib.request, "build_opener", lambda *a, **k: pytest.fail("Network prohibited"))
    monkeypatch.setattr(runtime, "post_json", lambda *a: dict(
        choices=[dict(finish_reason="stop", message=dict(content=original))],
        usage=dict(prompt_tokens=100, completion_tokens=1, total_tokens=101)))
    args = SimpleNamespace(endpoint=runtime.ENDPOINT, model=runtime.MODEL, max_model_len=4608,
                           max_completion_tokens=512, retry_limit=1, timeout=1)
    errors = []
    with pytest.raises(runtime.RequestFailure, match="ValueError: V4 positive response requires nonempty EVIDENCE_QUOTE") as caught:
        runtime.generate(prompt, 100, args, errors.append)
    assert isinstance(caught.value.__cause__, ValueError)
    assert str(caught.value.__cause__) == errors[-1]["error_message"]
    assert [row["will_retry"] for row in errors] == ["YES", "NO"]
    assert common.json_text(value) == original


@pytest.mark.parametrize("version", ["v3", "v4"])
def test_failed_diagnostic_retains_attempts_atomically_without_success_outputs(tmp_path, phase2_case, monkeypatch, capsys, version):
    rows = selected_prompts(phase2_case, version)
    for name in [common.QUERY_PATH, *prompts.version_settings(version)[0]]:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic development input", encoding="utf-8")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    def prepare(root, connection, selected_version="v3"):
        assert selected_version == version
        connection.execute("CREATE TABLE prompts (ordinal INTEGER, payload TEXT)")
        connection.executemany("INSERT INTO prompts VALUES (?, ?)", [(i, common.json_text(row)) for i, row in enumerate(rows)])
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [1]*100}
    monkeypatch.setattr(runtime, "prepare_population", prepare)
    monkeypatch.setattr(diagnostic.audit, "load_tokenizer", lambda root: Tokenizer())
    monkeypatch.setattr(runtime.urllib.request, "build_opener", lambda *a, **k: pytest.fail("Network prohibited"))
    requests = []
    def post(endpoint, payload, timeout):
        requests.append(payload)
        result = api_response(payload)
        query_id = payload["response_format"]["json_schema"]["schema"]["properties"]["query_id"]["enum"][0]
        if query_id == rows[1]["query_id"]:
            result["choices"][0]["finish_reason"] = "length"
        return result
    monkeypatch.setattr(runtime, "post_json", post)
    destination = tmp_path/diagnostic.failure_path(version)
    replace = diagnostic.os.replace
    publications = []
    def atomic_replace(source, target):
        assert target == destination
        record = common.parse_json(source.read_text(encoding="utf-8"))
        assert len(record["attempt_errors"]) == 2
        assert not any((tmp_path/name).exists() for name in diagnostic.output_paths(version))
        publications.append(target)
        replace(source, target)
    monkeypatch.setattr(diagnostic.os, "replace", atomic_replace)
    argv = ["--project-root", str(tmp_path), "--version", version, "--retry-limit", "1"]
    assert diagnostic.main(argv) == 0
    assert requests == [] and not destination.exists()
    assert diagnostic.main([*argv, "--execute"]) == 2
    output = capsys.readouterr().out
    assert "SIX_CLASS_DIAGNOSTIC=FAIL" in output
    assert "ValueError: Truncated or incomplete finish_reason" in output
    assert len(requests) == 3 and publications == [destination]
    failures = [common.parse_json(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert len(failures) == 1
    failure = failures[0]
    assert set(failure) == {"query_id", "source_dataset", "classifier_prediction", "prompt_tokens", "attempt_errors", "final_error_type", "final_error_message", "completed_before_failure"}
    assert failure["query_id"] == rows[1]["query_id"]
    assert failure["source_dataset"] == rows[1]["source_dataset"]
    assert failure["classifier_prediction"] == rows[1]["classifier_prediction"]
    assert failure["completed_before_failure"] == 1 and failure["prompt_tokens"] == 100
    assert failure["final_error_type"] == "RequestFailure"
    assert failure["final_error_message"].endswith("ValueError: Truncated or incomplete finish_reason")
    assert [row["attempt"] for row in failure["attempt_errors"]] == [1, 2]
    assert all(row["error_message"] == "Truncated or incomplete finish_reason" for row in failure["attempt_errors"])
    assert "DO NOT SAVE HIDDEN CONTENT" not in destination.read_text(encoding="utf-8")
    assert all(p.read_bytes() == content for p, content in before.items())
    assert not any((tmp_path/name).exists() for name in diagnostic.output_paths(version))
    assert not list(tmp_path.glob(".rag_v3_diagnostic_*"))


def test_v4_insufficient_bypasses_positive_quote_qualification(phase2_case, monkeypatch):
    prompt = selected_prompts(phase2_case, "v4")[0]
    value = dict(query_id=prompt["query_id"], reasoning_decision="ABSTAIN", evidence_status="INSUFFICIENT",
                 supported_class=None, explanation="No qualifying evidence.", cited_chunk_ids=[])
    monkeypatch.setattr(prompts.positive_evidence_v4.re, "fullmatch", lambda *a, **k: pytest.fail("Positive quote qualification invoked"))
    assert runtime.validate_logical(value, prompt) is value
