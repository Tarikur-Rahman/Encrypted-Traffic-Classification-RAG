import pytest

from src.rag import run_constrained_llm_v3 as execution

runtime = execution.runtime
common, prompts, audit = runtime.common, runtime.prompts, runtime.audit


@pytest.fixture
def execution_project(tmp_path, phase2_case, monkeypatch):
    case = phase2_case
    queries, records = [], []
    for i in range(3):
        query = dict(case.query, query_id=f"Q-{i:03d}")
        context = prompts.contexts.construct(query, [], case.chunks, case.verified, case.policy)
        queries.append({key: query[key] for key in prompts.QUERY_FIELDS})
        records.append(prompts.prompt_record(context, "v3"))
    common.corpus.write_table(tmp_path/common.QUERY_PATH, prompts.QUERY_FIELDS, queries)
    common.corpus.write_table(tmp_path/prompts.V3_OUTPUTS[0], prompts.HEADER, records)
    for name in prompts.V3_OUTPUTS[1:3]:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic metadata", encoding="utf-8")
    common.corpus.write_table(tmp_path/prompts.V3_OUTPUTS[3], prompts.POLICY_HEADER, prompts.policy_rows("v3"))
    monkeypatch.setattr(common, "QUERY_COUNTS", {"USTC-TFC2016": 3})
    monkeypatch.setattr(common, "completed", lambda *args: {"prompt_schema_version": "constrained_evidence_v3"})
    paths = audit.output_paths(4608, 512)
    rows = [dict(query_id=row["query_id"], source_dataset=row["source_dataset"], classifier_prediction=row["classifier_prediction"], qwen_prompt_tokens=100,
                 max_model_len=4608, reserved_completion_tokens=512, safe_prompt_limit=4096, reserved_total_tokens=612, within_limit="YES") for row in records]
    common.corpus.write_table(tmp_path/paths[0], audit.HEADER, rows)
    summary = dict.fromkeys(audit.SUMMARY_HEADER, "")
    summary.update(rows=3, gate_status="PASS", over_limit_with_reservation=0, configuration_key="maxlen_4608_completion_512", max_model_len=4608, reserved_completion_tokens=512,
                   safe_prompt_limit=4096, dataset_counts_json=common.json_text(common.QUERY_COUNTS), input_metadata_json=common.json_text(common.snapshot(tmp_path, [common.QUERY_PATH, *prompts.V3_OUTPUTS])), tokenizer_metadata_json="[]")
    common.corpus.write_table(tmp_path/paths[1], audit.SUMMARY_HEADER, [summary])
    return tmp_path


def answer(payload, finish="stop", mutation=None):
    identity = payload["response_format"]["json_schema"]["schema"]["properties"]["query_id"]["enum"][0]
    value = dict(query_id=identity, reasoning_decision="ABSTAIN", evidence_status="INSUFFICIENT", supported_class=None, explanation="No supplied evidence supports a class.", cited_chunk_ids=[])
    if mutation:
        value.update(mutation)
    return dict(choices=[dict(finish_reason=finish, message=dict(content=common.json_text(value)))], usage=dict(prompt_tokens=100, completion_tokens=1, total_tokens=101))


def saved(root):
    return [common.parse_json(line) for line in (root/execution.OUTPUTS[0]).read_text().splitlines()]


def test_dry_no_network_then_complete_resume_and_export(execution_project, monkeypatch):
    root = execution_project
    calls = []
    def post(endpoint, payload, timeout):
        calls.append(payload)
        return answer(payload)
    monkeypatch.setattr(runtime, "post_json", post)
    argv = ["--project-root", str(root)]
    assert execution.main(argv) == 0
    assert calls == [] and not any((root/name).exists() for name in execution.OUTPUTS)
    assert execution.main([*argv, "--execute"]) == 0
    assert [row["query_id"] for row in saved(root)] == ["Q-000", "Q-001", "Q-002"]
    assert len(calls) == 3
    original = (root/execution.OUTPUTS[0]).read_bytes()
    assert execution.main([*argv, "--execute"]) == 0
    assert len(calls) == 3 and (root/execution.OUTPUTS[0]).read_bytes() == original
    assert execution.main([*argv, "--export", "--execute"]) == 0
    exported = [common.parse_json(line) for line in (root/execution.EXPORT).read_text().splitlines()]
    assert all(set(row) == set(runtime.FIELDS) for row in exported)
    assert [row["query_id"] for row in exported] == ["Q-000", "Q-001", "Q-002"]
    config = {r["field"]: r["value"] for r in common.rows(root, execution.OUTPUTS[2])}
    assert config["execution_status"] == "PASS" and config["seed_expectation"] == "42"
    assert (root/execution.OUTPUTS[0]).read_bytes() == original


def test_partial_failure_resume_skips_completed(execution_project, monkeypatch, capsys):
    root = execution_project
    calls = []
    failing = True
    def post(endpoint, payload, timeout):
        identity = payload["response_format"]["json_schema"]["schema"]["properties"]["query_id"]["enum"][0]
        calls.append(identity)
        return answer(payload, finish="length" if failing and identity == "Q-001" else "stop")
    monkeypatch.setattr(runtime, "post_json", post)
    argv = ["--project-root", str(root), "--retry-limit", "1", "--execute"]
    assert execution.main(argv) == 2
    assert "CONSTRAINED_LLM_V3_EXECUTION=INCOMPLETE" in capsys.readouterr().out
    assert [r["query_id"] for r in saved(root)] == ["Q-000"]
    prefix = (root/execution.OUTPUTS[0]).read_bytes()
    failing = False
    assert execution.main(argv) == 0
    assert calls == ["Q-000", "Q-001", "Q-001", "Q-001", "Q-002"]
    assert (root/execution.OUTPUTS[0]).read_bytes().startswith(prefix)
    assert len(list(common.rows(root, execution.OUTPUTS[3]))) == 2


@pytest.mark.parametrize("corruption", ["duplicate", "unknown", "malformed_complete", "semantic", "citation"])
def test_existing_invalid_jsonl_rejected(execution_project, monkeypatch, corruption):
    root = execution_project
    monkeypatch.setattr(runtime, "post_json", lambda endpoint, payload, timeout: answer(payload))
    argv = ["--project-root", str(root), "--execute"]
    assert execution.main(argv) == 0
    rows = saved(root)
    if corruption == "duplicate": rows.append(rows[0])
    if corruption == "unknown": rows[-1]["query_id"] = "UNKNOWN"
    if corruption == "semantic": rows[-1]["supported_class"] = "malware"
    if corruption == "citation": rows[-1]["cited_chunk_ids"] = ["UNKNOWN"]
    path = root/execution.OUTPUTS[0]
    path.write_text("\n".join(common.json_text(row) for row in rows)+"\n" + ("{bad}\n" if corruption == "malformed_complete" else ""), encoding="utf-8")
    before = path.read_bytes()
    monkeypatch.setattr(runtime, "post_json", lambda *args: pytest.fail("Invalid resume must not send requests"))
    assert execution.main(argv) == 2
    assert path.read_bytes() == before


def test_explicit_partial_line_repair(execution_project, monkeypatch):
    root = execution_project
    monkeypatch.setattr(runtime, "post_json", lambda endpoint, payload, timeout: answer(payload))
    argv = ["--project-root", str(root), "--execute"]
    assert execution.main(argv) == 0
    path = root/execution.OUTPUTS[0]
    original = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b'{"query_id":')
    assert execution.main(argv) == 2
    assert path.read_bytes() == original + b'{"query_id":'
    assert execution.main([*argv, "--repair-trailing-partial"]) == 0
    assert path.read_bytes() == original
    assert any(row["error_type"] == "TRAILING_PARTIAL_DISCARDED" for row in common.rows(root, execution.OUTPUTS[3]))


def test_retry_invalid_content_without_rewriting(execution_project, monkeypatch):
    attempts = 0
    def post(endpoint, payload, timeout):
        nonlocal attempts
        attempts += 1
        return answer(payload, mutation={"cited_chunk_ids": ["NOT_SUPPLIED"]} if attempts == 1 else None)
    monkeypatch.setattr(runtime, "post_json", post)
    assert execution.main(["--project-root", str(execution_project), "--execute"]) == 0
    rows = saved(execution_project)
    assert rows[0]["retry_count"] == 1 and attempts == 4
    assert rows[0]["cited_chunk_ids"] == []
    assert list(common.rows(execution_project, execution.OUTPUTS[3]))[0]["will_retry"] == "YES"


def test_audit_pass_prerequisite(execution_project, monkeypatch):
    root = execution_project
    path = audit.output_paths(4608, 512)[1]
    rows = list(common.rows(root, path))
    rows[0]["gate_status"] = "FAIL"
    common.corpus.write_table(root/path, audit.SUMMARY_HEADER, rows)
    monkeypatch.setattr(runtime, "post_json", lambda *args: pytest.fail("No request without PASS"))
    assert execution.main(["--project-root", str(root), "--execute"]) == 2
    assert not (root/execution.OUTPUTS[0]).exists()


def test_interrupt_then_resume_preserves_completed_bytes(execution_project, monkeypatch):
    root = execution_project
    calls = 0
    def interrupted(endpoint, payload, timeout):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return answer(payload)
    monkeypatch.setattr(runtime, "post_json", interrupted)
    argv = ["--project-root", str(root), "--execute"]
    assert execution.main(argv) == 130
    prefix = (root/execution.OUTPUTS[0]).read_bytes()
    assert len(saved(root)) == 1
    monkeypatch.setattr(runtime, "post_json", lambda endpoint, payload, timeout: answer(payload))
    assert execution.main(argv) == 0
    assert (root/execution.OUTPUTS[0]).read_bytes().startswith(prefix)


def test_non_development_query_blocks_before_http(execution_project, monkeypatch):
    root = execution_project
    rows = list(common.rows(root, common.QUERY_PATH))
    rows[-1]["source_split"] = "test"
    common.corpus.write_table(root/common.QUERY_PATH, prompts.QUERY_FIELDS, rows)
    monkeypatch.setattr(runtime, "post_json", lambda *args: pytest.fail("TEST query sent"))
    assert execution.main(["--project-root", str(root), "--execute"]) == 2


def test_no_ground_truth_in_request(execution_project):
    row = next(common.rows(execution_project, prompts.V3_OUTPUTS[0]))
    expected = runtime.request_payload(row, runtime.MODEL, 512)
    altered = runtime.request_payload(dict(row, true_class_label="FORBIDDEN_GROUND_TRUTH", true_class_id=99), runtime.MODEL, 512)
    assert altered == expected
    assert "FORBIDDEN_GROUND_TRUTH" not in common.json_text(altered)


def test_temporary_failure_retried_and_permanent_not_retried(execution_project, monkeypatch):
    from types import SimpleNamespace
    prompt = next(common.rows(execution_project, prompts.V3_OUTPUTS[0]))
    args = SimpleNamespace(max_completion_tokens=512, max_model_len=4608, retry_limit=3, endpoint=runtime.ENDPOINT, model=runtime.MODEL, timeout=120)
    attempts = []
    def transient(endpoint, payload, timeout):
        attempts.append(1)
        if len(attempts) == 1:
            raise runtime.RequestFailure("temporary", retryable=True)
        return answer(payload)
    monkeypatch.setattr(runtime, "post_json", transient)
    assert runtime.generate(prompt, 100, args)["retry_count"] == 1
    errors = []
    def permanent(*args):
        raise runtime.RequestFailure("permanent", retryable=False)
    monkeypatch.setattr(runtime, "post_json", permanent)
    with pytest.raises(runtime.RequestFailure):
        runtime.generate(prompt, 100, args, errors.append)
    assert len(errors) == 1 and errors[0]["will_retry"] == "NO"


def test_resume_configuration_cannot_silently_change(execution_project, monkeypatch):
    root = execution_project
    monkeypatch.setattr(runtime, "post_json", lambda endpoint, payload, timeout: answer(payload))
    assert execution.main(["--project-root", str(root), "--execute"]) == 0
    before = (root/execution.OUTPUTS[0]).read_bytes()
    monkeypatch.setattr(runtime, "post_json", lambda *args: pytest.fail("Changed configuration sent request"))
    assert execution.main(["--project-root", str(root), "--model", "different-model", "--execute"]) == 2
    assert (root/execution.OUTPUTS[0]).read_bytes() == before


def test_export_refuses_incomplete_run(execution_project, monkeypatch):
    monkeypatch.setattr(runtime, "post_json", lambda *args: pytest.fail("Export called HTTP"))
    assert execution.main(["--project-root", str(execution_project), "--export", "--execute"]) == 2
    assert not (execution_project/execution.EXPORT).exists()
