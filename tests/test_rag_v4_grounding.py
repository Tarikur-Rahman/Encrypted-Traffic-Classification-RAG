"""Synthetic local fixtures only; tokenizer and HTTP are always replaced."""
import sqlite3
from types import SimpleNamespace

import pytest

from src.rag import llm_v3_runtime as runtime
from src.rag import run_v3_grounding_diagnostic as diagnostic
from src.rag import run_constrained_llm_v3 as execution

common, prompts, audit = runtime.common, runtime.prompts, runtime.audit
gate = prompts.positive_evidence_v4
QUOTE = "High flow duration relative to the training distribution indicates benign traffic."
PROFILE = ("Encrypted network traffic observation. Predicted traffic class: benign" + gate.PROFILE_START +
           "flow duration is high; packet length is low" + gate.PROFILE_END)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(runtime, "post_json", lambda *a, **k: pytest.fail("Unexpected HTTP request"))
    monkeypatch.setattr(runtime.urllib.request, "build_opener", lambda *a, **k: pytest.fail("Network prohibited in tests"))
    monkeypatch.setattr(audit, "load_tokenizer", lambda *a: FakeTokenizer())


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert len(messages) == 1 and kwargs["truncation"] is False
        return {"input_ids": list(range(100)), "attention_mask": [1]*100}


def positive_case(case, quote=QUOTE):
    context = prompts.contexts.construct(case.query, [], case.chunks, case.verified, case.policy)
    evidence = [dict(chunk_id="C1", document_id="D1", source_type="standards_rfc", chunk_text=quote,
                     rerank_rank=1, token_count=20, provenance_status="VERIFIED"),
                dict(chunk_id="C2", document_id="D2", source_type="standards_rfc", chunk_text="Unrelated evidence.",
                     rerank_rank=2, token_count=3, provenance_status="VERIFIED")]
    context.update(traffic_profile=PROFILE, selected_chunk_ids='["C1","C2"]', evidence_records_json=common.json_text(evidence), context_text=prompts.contexts.context_text(evidence))
    prompt = prompts.prompt_record(context, "v4")
    value = dict(query_id=prompt["query_id"], reasoning_decision="SUPPORT_CLASS", evidence_status="SUPPORTED",
                 supported_class="benign", explanation=f'EVIDENCE_QUOTE: "{quote}"\nEVIDENCE_LINK: The observation flow duration is high matches this relationship.', cited_chunk_ids=["C1"])
    return prompt, value


def test_positive_exact_quote_deterministic_schema_unchanged(phase2_case):
    prompt, value = positive_case(phase2_case)
    original = common.json_text(value)
    assert runtime.validate_logical(value, prompt) is value
    assert runtime.validate_logical(value, prompt) is value
    assert common.json_text(value) == original
    schema = runtime.response_schema(prompt)
    assert set(schema["properties"]) == set(schema["required"]) == set(value) == set(runtime.FIELDS)
    assert len(value) == 6


@pytest.mark.parametrize("mutation", ["missing", "empty", "fabricated", "paraphrased", "uncited", "unknown", "duplicate", "wrong_direction", "wrong_class", "negated", "missing_link", "invented_observation"])
def test_invalid_positive_quotes_rejected_without_rewriting(phase2_case, mutation):
    prompt, value = positive_case(phase2_case)
    if mutation == "missing": value["explanation"] = "Independent support exists."
    if mutation == "empty": value["explanation"] = 'EVIDENCE_QUOTE: ""\nEVIDENCE_LINK: flow duration is high'
    if mutation == "fabricated": value["explanation"] = value["explanation"].replace("High", "Very high")
    if mutation == "paraphrased": value["explanation"] = value["explanation"].replace("indicates", "implies")
    if mutation == "uncited": value["cited_chunk_ids"] = ["C2"]
    if mutation == "unknown": value["cited_chunk_ids"] = ["UNKNOWN"]
    if mutation == "duplicate": value["cited_chunk_ids"] = ["C1", "C1"]
    if mutation == "wrong_direction": prompt, value = positive_case(phase2_case, QUOTE.replace("High", "Low"))
    if mutation == "wrong_class": prompt, value = positive_case(phase2_case, QUOTE.replace("benign", "malware"))
    if mutation == "negated": prompt, value = positive_case(phase2_case, "It is false that " + QUOTE)
    if mutation == "missing_link": value["explanation"] = value["explanation"].split("\n")[0]
    if mutation == "invented_observation": value["explanation"] = value["explanation"].replace("observation flow duration is high", "observation flow duration is low")
    before = common.json_text(value)
    with pytest.raises(ValueError):
        runtime.validate_logical(value, prompt)
    assert common.json_text(value) == before


@pytest.mark.parametrize("quote", [
    "The dataset contains benign traffic.", "The dataset contains Tor and VPN traffic.",
    "Malware may use encryption.", "Tor/VPN traffic is encrypted.",
    "Packet size is useful for classification.", "Flow duration is important.",
    "Machine learning can distinguish classes.", "Source IP was used to assign labels.",
    "No evidence supports malware, therefore benign is supported.",
    "High flow duration indicates benign traffic.",  # No training-relative reference frame.
])
def test_generic_evidence_never_qualifies(phase2_case, quote):
    prompt, value = positive_case(phase2_case, quote)
    with pytest.raises(ValueError):
        runtime.validate_logical(value, prompt)


def test_contradicted_requires_qualifying_other_class_quote(phase2_case):
    prompt, value = positive_case(phase2_case, QUOTE.replace("benign", "malware"))
    value.update(evidence_status="CONTRADICTED", supported_class="malware")
    assert runtime.validate_logical(value, prompt) is value
    value["explanation"] = "The dataset contains malware."
    with pytest.raises(ValueError):
        runtime.validate_logical(value, prompt)


@pytest.mark.parametrize("status", ["INSUFFICIENT", "CONFLICTING"])
def test_abstention_needs_no_quote(phase2_case, status):
    prompt, value = positive_case(phase2_case)
    value.update(evidence_status=status, supported_class=None, reasoning_decision="ABSTAIN", explanation="Insufficient affirmative evidence." if status == "INSUFFICIENT" else "Supplied evidence conflicts.")
    assert runtime.validate_logical(value, prompt) is value


@pytest.mark.parametrize("rule", list(prompts.version_settings("v4")[2]))
def test_v4_rules_required_old_payload_unchanged(phase2_case, rule):
    case = phase2_case
    context = prompts.contexts.construct(case.query, [], case.chunks, case.verified, case.policy)
    old = [prompts.prompt_record(context, version) for version in ("v2", "v3")]
    record = prompts.prompt_record(context, "v4")
    assert all(row["prompt_text"].partition("INPUT_JSON:\n")[2] == record["prompt_text"].partition("INPUT_JSON:\n")[2] for row in old)
    for version, row in zip(("v2", "v3"), old):
        assert row == prompts.prompt_record(context, version)
    with pytest.raises(ValueError, match=rule):
        prompts.validate_grounding(dict(record, prompt_text=record["prompt_text"].replace(prompts.version_settings("v4")[2][rule], "")), "v4")
    policy = {row["policy_id"]: row["value"] for row in prompts.policy_rows("v4")}
    assert all(policy[version+"_status"] == "SUPERSEDED_NOT_EXECUTABLE" and policy[version+"_execution_allowed"] == "NO" for version in ("v1", "v2", "v3"))


def test_rejection_uses_retry_not_rewrite(phase2_case, monkeypatch):
    prompt, positive = positive_case(phase2_case)
    invalid = dict(positive, explanation="Generic support.")
    responses = iter((invalid, positive))
    def post(*args):
        return dict(choices=[dict(finish_reason="stop", message=dict(content=common.json_text(next(responses))))], usage=dict(prompt_tokens=100, completion_tokens=1, total_tokens=101))
    monkeypatch.setattr(runtime, "post_json", post)
    args = SimpleNamespace(endpoint=runtime.ENDPOINT, model=runtime.MODEL, max_model_len=4608, max_completion_tokens=512, timeout=1, retry_limit=1)
    errors = []
    result = runtime.generate(prompt, 100, args, errors.append)
    assert result["retry_count"] == 1 and len(errors) == 1
    assert result["explanation"] == positive["explanation"] and result["evidence_status"] == "SUPPORTED"


@pytest.fixture
def v4_project(tmp_path, phase2_case, monkeypatch):
    queries, records = [], []
    for i, (dataset, label) in enumerate(diagnostic.PAIRS):
        classes = common.CLASSES[dataset]
        query = dict(phase2_case.query, query_id=f"Q-{i:03d}", source_dataset=dataset, predicted_class_label=label,
                     confidence=str(1/len(classes)), class_probabilities_json=common.json_text(dict.fromkeys(classes, 1/len(classes))))
        context = prompts.contexts.construct(query, [], phase2_case.chunks, phase2_case.verified, phase2_case.policy)
        queries.append({key: query[key] for key in prompts.QUERY_FIELDS})
        records.append(prompts.prompt_record(context, "v4"))
    common.corpus.write_table(tmp_path/common.QUERY_PATH, prompts.QUERY_FIELDS, queries)
    common.corpus.write_table(tmp_path/prompts.V4_OUTPUTS[0], prompts.HEADER, records)
    for name in prompts.V4_OUTPUTS[1:3]:
        (tmp_path/name).write_text("synthetic completed metadata", encoding="utf-8")
    common.corpus.write_table(tmp_path/prompts.V4_OUTPUTS[3], prompts.POLICY_HEADER, prompts.policy_rows("v4"))
    monkeypatch.setattr(common, "QUERY_COUNTS", {"CIC-Darknet2020": 4, "USTC-TFC2016": 2})
    monkeypatch.setattr(common, "completed", lambda *a: {"prompt_schema_version": "constrained_evidence_v4"})
    for name in (*prompts.V1_OUTPUTS, *prompts.OUTPUTS, *prompts.V3_OUTPUTS, *audit.output_paths(4608, 512), *diagnostic.OUTPUTS, *execution.OUTPUTS):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("immutable previous artifact", encoding="utf-8")
    return tmp_path


def response(payload):
    identity = payload["response_format"]["json_schema"]["schema"]["properties"]["query_id"]["enum"][0]
    value = dict(query_id=identity, reasoning_decision="ABSTAIN", evidence_status="INSUFFICIENT", supported_class=None,
                 explanation="No qualifying supplied evidence.", cited_chunk_ids=[])
    return dict(choices=[dict(finish_reason="stop", message=dict(content=common.json_text(value)))], usage=dict(prompt_tokens=100, completion_tokens=1, total_tokens=101))


def test_v4_paths_audit_diagnostic_executor_gates_resume_and_isolation(v4_project, monkeypatch):
    root = v4_project
    argv = ["--project-root", str(root), "--version", "v4"]
    prior = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert audit.output_paths(4608, 512, "v4")[0] == "results/rag/llm/v4/token_audits/maxlen_4608_completion_512/qwen_token_audit.csv"
    assert diagnostic.output_paths("v4")[0] == "results/rag/llm/v4/diagnostics/six_class_grounding_diagnostic.jsonl"
    assert execution.output_paths("v4")[0] == "results/rag/llm/v4/execution/llm_responses.jsonl"
    assert execution.main([*argv, "--execute"]) == 2  # No audit.
    assert audit.main(argv) == 0
    assert prior == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert audit.main([*argv, "--execute"]) == 0
    summary = next(common.rows(root, audit.output_paths(4608, 512, "v4")[1]))
    assert summary["rows"] == "6" and summary["max_model_len"] == "4608"
    assert execution.main([*argv, "--execute"]) == 2  # No diagnostic.
    assert not any((root/name).exists() for name in execution.output_paths("v4"))
    assert diagnostic.main(argv) == 0  # No HTTP permitted by autouse fixture.
    assert not any((root/name).exists() for name in diagnostic.output_paths("v4"))
    calls = []
    def post(endpoint, payload, timeout):
        calls.append(payload)
        return response(payload)
    monkeypatch.setattr(runtime, "post_json", post)
    assert diagnostic.main([*argv, "--execute"]) == 0
    assert len(calls) == 6
    assert execution.main(argv) == 0 and len(calls) == 6
    assert execution.main([*argv, "--execute"]) == 0 and len(calls) == 12
    raw = root/execution.output_paths("v4")[0]
    saved = raw.read_bytes()
    assert [common.parse_json(line)["query_id"] for line in saved.decode().splitlines()] == [f"Q-{i:03d}" for i in range(6)]
    assert execution.main([*argv, "--execute"]) == 0 and len(calls) == 12 and raw.read_bytes() == saved
    assert all(p.read_bytes() == value for p, value in prior.items())


@pytest.mark.parametrize("problem", ["test_partition", "missing_rule", "ground_truth_payload"])
def test_v4_population_fails_before_tokenizer_or_http(v4_project, monkeypatch, problem):
    root = v4_project
    if problem == "test_partition":
        rows = list(common.rows(root, common.QUERY_PATH))
        rows[-1]["source_split"] = "test"
        common.corpus.write_table(root/common.QUERY_PATH, prompts.QUERY_FIELDS, rows)
    else:
        rows = list(common.rows(root, prompts.V4_OUTPUTS[0]))
        if problem == "missing_rule":
            rows[-1]["prompt_text"] = rows[-1]["prompt_text"].replace(prompts.V4_RULES["quote_anchor"], "")
        else:
            instructions, _, payload = rows[-1]["prompt_text"].partition("INPUT_JSON:\n")
            payload = common.parse_json(payload)
            payload["true_class_label"] = "FORBIDDEN"
            rows[-1]["prompt_text"] = instructions + "INPUT_JSON:\n" + common.json_text(payload)
        common.corpus.write_table(root/prompts.V4_OUTPUTS[0], prompts.HEADER, rows)
    with sqlite3.connect(":memory:") as connection, pytest.raises(ValueError):
        runtime.prepare_population(root, connection, "v4")


def test_v3_runtime_superseded_when_v4_policy_present(v4_project):
    with sqlite3.connect(":memory:") as connection, pytest.raises(ValueError, match="SUPERSEDED_NOT_EXECUTABLE"):
        runtime.prepare_population(v4_project, connection, "v3")


def test_v4_overlength_audit_blocks_execution(v4_project, monkeypatch):
    class TooLong:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [1]*4097}
    monkeypatch.setattr(audit, "load_tokenizer", lambda root: TooLong())
    argv = ["--project-root", str(v4_project), "--version", "v4", "--execute"]
    assert audit.main(argv) == 2
    summary = next(common.rows(v4_project, audit.output_paths(4608, 512, "v4")[1]))
    assert summary["gate_status"] == "FAIL" and summary["over_limit_with_reservation"] == "6"
    assert execution.main(argv) == 2
    assert not any((v4_project/name).exists() for name in execution.output_paths("v4"))


@pytest.mark.parametrize("corruption", ["quote", "identity", "missing", "summary", "human_acceptance"])
def test_v4_saved_diagnostic_must_be_current_and_valid(v4_project, monkeypatch, corruption):
    root = v4_project
    argv = ["--project-root", str(root), "--version", "v4", "--execute"]
    assert audit.main(argv) == 0
    monkeypatch.setattr(runtime, "post_json", lambda endpoint, payload, timeout: response(payload))
    assert diagnostic.main(argv) == 0
    raw = root/diagnostic.output_paths("v4")[0]
    rows = [common.parse_json(line) for line in raw.read_text(encoding="utf-8").splitlines()]
    if corruption == "quote":
        rows[0].update(reasoning_decision="SUPPORT_CLASS", evidence_status="SUPPORTED", supported_class=rows[0]["classifier_prediction"], explanation='EVIDENCE_QUOTE: "invented"\nEVIDENCE_LINK: invented', cited_chunk_ids=["UNKNOWN"])
    if corruption == "identity": rows[0]["input_metadata_json"] = "[]"
    if corruption == "missing": rows.pop()
    if corruption == "human_acceptance": rows[0]["human_grounding_status"] = "ACCEPTED"
    raw.write_text("".join(common.json_text(row)+"\n" for row in rows), encoding="utf-8")
    if corruption == "summary":
        path = diagnostic.output_paths("v4")[1]
        summary = list(common.rows(root, path))
        summary[0]["structural_status"] = "FAIL"
        common.corpus.write_table(root/path, diagnostic.HEADER, summary)
    monkeypatch.setattr(runtime, "post_json", lambda *a: pytest.fail("Invalid diagnostic must block HTTP"))
    assert execution.main(argv) == 2
    assert not any((root/name).exists() for name in execution.output_paths("v4"))


def fail_closed_args():
    return SimpleNamespace(version="v4", runtime_configuration=runtime.FAIL_CLOSED, endpoint=runtime.ENDPOINT,
                           model=runtime.MODEL, max_model_len=4608, max_completion_tokens=512, timeout=1, retry_limit=3)


def structured_response(value):
    return dict(choices=[dict(finish_reason="stop", message=dict(content=common.json_text(value), reasoning_content="HIDDEN_COT_NOT_FOR_LOGGING"))],
                usage=dict(prompt_tokens=100, completion_tokens=1, total_tokens=101))


@pytest.mark.parametrize("kind", ["qualifying", "missing_quote", "fabricated_quote", "dataset_quote", "insufficient", "conflicting"])
def test_explicit_fail_closed_policy_preserves_raw_and_never_retries_gate(phase2_case, monkeypatch, kind):
    prompt, value = positive_case(phase2_case)
    if kind == "missing_quote": value["explanation"] = "The classifier is supported."
    if kind == "fabricated_quote": value["explanation"] = value["explanation"].replace("High", "Very high")
    if kind == "dataset_quote": prompt, value = positive_case(phase2_case, "The dataset contains benign traffic.")
    if kind in {"insufficient", "conflicting"}:
        value.update(reasoning_decision="ABSTAIN", evidence_status=kind.upper(), supported_class=None, explanation="Evidence is insufficient or conflicting.")
    raw = common.json_text(value)
    prompt_before = dict(prompt)
    calls, errors = [], []
    def post(*args):
        calls.append(args)
        return structured_response(value)
    monkeypatch.setattr(runtime, "post_json", post)
    fallback = kind in {"missing_quote", "fabricated_quote", "dataset_quote"}
    reason = ""
    if fallback:
        with pytest.raises(ValueError) as caught:
            gate.validate(value, prompt)
        reason = str(caught.value)
    result = runtime.generate(prompt, 100, fail_closed_args(), errors.append)
    assert len(calls) == 1 and errors == [] and result["retry_count"] == 0
    assert result["raw_model_response"] == raw and common.json_text(value) == raw
    assert result["runtime_configuration"] == runtime.FAIL_CLOSED
    assert result["grounding_gate_error_message"] == reason
    assert result["grounding_gate_error_type"] == ("ValueError" if fallback else "")
    assert result["validator_fallback_applied"] == ("YES" if fallback else "NO")
    constrained = {key: result[key] for key in runtime.FIELDS}
    if fallback:
        assert result["grounding_gate_status"] == "REJECTED"
        assert constrained == dict(query_id=value["query_id"], reasoning_decision="ABSTAIN", evidence_status="INSUFFICIENT",
                                   supported_class=None, cited_chunk_ids=[], explanation=runtime.FALLBACK_EXPLANATION)
    else:
        assert constrained == value
        assert result["grounding_gate_status"] == ("PASS" if kind == "qualifying" else "NOT_APPLICABLE")
    assert "HIDDEN_COT_NOT_FOR_LOGGING" not in common.json_text(result)
    assert prompt == prompt_before
    execution.validate_saved(result, prompt, 100, fail_closed_args())


@pytest.mark.parametrize("failure", ["transport", "http", "malformed_api", "malformed_json", "token_mismatch", "wrong_query", "extra_field", "bad_citation", "software"])
def test_fail_closed_never_converts_other_failures(phase2_case, monkeypatch, failure):
    prompt, value = positive_case(phase2_case)
    # This would fail the gate, but schema and token checks must fail first.
    value["explanation"] = "Missing quote."
    if failure == "wrong_query": value["query_id"] = "UNKNOWN"
    if failure == "extra_field": value["extra"] = "invalid"
    if failure == "bad_citation": value["cited_chunk_ids"] = ["UNKNOWN"]
    calls = []
    def post(*args):
        calls.append(args)
        if failure == "transport": raise runtime.RequestFailure("Temporary loopback transport failure")
        if failure == "http": raise runtime.RequestFailure("HTTP status 400", retryable=False)
        if failure == "malformed_api": return {"choices": []}
        data = structured_response(value)
        if failure == "malformed_json": data["choices"][0]["message"]["content"] = "{"
        if failure == "token_mismatch": data["usage"]["prompt_tokens"] = 99
        return data
    monkeypatch.setattr(runtime, "post_json", post)
    if failure == "software":
        def broken_gate(*args):
            raise ValueError("Unexpected internal problem")
        monkeypatch.setattr(gate, "validate", broken_gate)
    errors = []
    with pytest.raises(runtime.RequestFailure):
        runtime.generate(prompt, 100, fail_closed_args(), errors.append)
    assert len(calls) == (4 if failure == "transport" else 1)
    assert len(errors) == len(calls) and errors[-1]["will_retry"] == "NO"


def test_fail_closed_retries_temporary_transport_then_validates(phase2_case, monkeypatch):
    prompt, value = positive_case(phase2_case)
    calls = []
    def post(*args):
        calls.append(args)
        if len(calls) == 1:
            raise runtime.RequestFailure("Temporary loopback transport failure")
        return structured_response(value)
    monkeypatch.setattr(runtime, "post_json", post)
    result = runtime.generate(prompt, 100, fail_closed_args())
    assert len(calls) == 2 and result["retry_count"] == 1 and result["validator_fallback_applied"] == "NO"


@pytest.mark.parametrize("configuration,maximum,completion", [(runtime.FAIL_CLOSED, 4608, 512), (runtime.FAIL_CLOSED_5120_1024, 5120, 1024)])
def test_fail_closed_diagnostic_continues_executor_resumes_exports_and_preserves_inputs(v4_project, monkeypatch, configuration, maximum, completion):
    root = v4_project
    # Supply one synthetic verified chunk per case so logical citation checks
    # succeed before the positive-grounding gate rejects the fourth response.
    records = list(common.rows(root, prompts.V4_OUTPUTS[0]))
    for row in records:
        instructions, _, payload = row["prompt_text"].partition("INPUT_JSON:\n")
        payload = common.parse_json(payload)
        evidence = [dict(chunk_id="C1", document_id="D1", source_type="standards_rfc", chunk_text="The dataset contains VPN traffic.",
                         rerank_rank=1, token_count=7, provenance_status="VERIFIED")]
        payload.update(trusted_evidence=evidence, allowed_chunk_ids=["C1"])
        row.update(prompt_text=instructions+"INPUT_JSON:\n"+common.json_text(payload), allowed_chunk_ids='["C1"]', trusted_context=prompts.contexts.context_text(evidence))
    common.corpus.write_table(root/prompts.V4_OUTPUTS[0], prompts.HEADER, records)
    old_failure = root/diagnostic.failure_path("v4")
    old_failure.parent.mkdir(parents=True, exist_ok=True)
    old_failure.write_text("previous failed diagnostic must survive", encoding="utf-8")
    base_argv = ["--project-root", str(root), "--version", "v4"]
    assert audit.main([*base_argv, "--max-model-len", str(maximum), "--reserved-completion-tokens", str(completion), "--execute"]) == 0
    if configuration == runtime.FAIL_CLOSED_5120_1024:
        # Prior incomplete experiment is deliberately not valid resume input
        # for the new configuration; no read, repair or migration is allowed.
        for path in execution.output_paths("v4", runtime.FAIL_CLOSED):
            target = root/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('{"preserved_response":true}\n'*20 if path.endswith(".jsonl") else "INCOMPLETE: preserved 4608/512 experiment", encoding="utf-8")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    argv = [*base_argv, "--runtime-configuration", configuration, "--max-model-len", str(maximum), "--max-completion-tokens", str(completion)]
    # Previous-policy diagnostics cannot satisfy this configuration's gate.
    assert execution.main([*argv, "--execute"]) == 2
    calls = []
    def post(endpoint, payload, timeout):
        calls.append(payload)
        data = response(payload)
        value = common.parse_json(data["choices"][0]["message"]["content"])
        if value["query_id"] == "Q-003":
            value.update(reasoning_decision="SUPPORT_CLASS", evidence_status="SUPPORTED", supported_class="VPN", explanation="Dataset contains VPN.", cited_chunk_ids=["C1"])
        data["choices"][0]["message"]["content"] = common.json_text(value)
        return data
    monkeypatch.setattr(runtime, "post_json", post)
    assert diagnostic.main([*argv, "--execute"]) == 0 and len(calls) == 6
    paths = diagnostic.output_paths("v4", configuration)
    suffix = configuration.removeprefix("v4_")
    assert paths[0] == f"results/rag/llm/v4/diagnostics/{suffix}/six_class_grounding_diagnostic.jsonl"
    diagnostic_rows = [common.parse_json(line) for line in (root/paths[0]).read_text(encoding="utf-8").splitlines()]
    assert len(diagnostic_rows) == 6 and diagnostic_rows[3]["validator_fallback_applied"] == "YES"
    assert diagnostic_rows[3]["evidence_status"] == "INSUFFICIENT"
    assert all(row["human_grounding_status"] == "REVIEW_REQUIRED" for row in diagnostic_rows)
    assert execution.main(argv) == 0 and len(calls) == 6
    assert execution.main([*argv, "--execute"]) == 0 and len(calls) == 12
    outputs = execution.output_paths("v4", configuration)
    assert outputs[0] == f"results/rag/llm/v4/execution/{suffix}/llm_responses.jsonl"
    raw = (root/outputs[0]).read_bytes()
    assert execution.main([*argv, "--execute"]) == 0 and len(calls) == 12
    assert (root/outputs[0]).read_bytes() == raw
    summary = list(common.rows(root, outputs[1]))
    assert next(row["value"] for row in summary if row["metric"] == "validator_fallback_count") == "1"
    assert next(row["value"] for row in summary if row["metric"] == "dataset_prediction_validator_fallback_count" and row["source_dataset"] == "CIC-Darknet2020" and row["classifier_prediction"] == "VPN") == "1"
    assert execution.main([*argv, "--export", "--execute"]) == 0 and len(calls) == 12
    exported = root/outputs[0].replace("llm_responses.jsonl", "llm_responses_for_import.jsonl")
    values = [common.parse_json(line) for line in exported.read_text(encoding="utf-8").splitlines()]
    assert all(set(value) == set(runtime.FIELDS) for value in values)
    assert values[3]["evidence_status"] == "INSUFFICIENT" and values[3]["explanation"] == runtime.FALLBACK_EXPLANATION
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("field", ["raw_model_response", "grounding_gate_status", "grounding_gate_error_message", "validator_fallback_applied", "explanation", "runtime_configuration"])
def test_saved_fail_closed_audit_metadata_cannot_be_forged(phase2_case, monkeypatch, field):
    prompt, value = positive_case(phase2_case)
    value["explanation"] = "Missing quote."
    monkeypatch.setattr(runtime, "post_json", lambda *args: structured_response(value))
    result = runtime.generate(prompt, 100, fail_closed_args())
    result[field] = "tampered"
    with pytest.raises(ValueError):
        execution.validate_saved(result, prompt, 100, fail_closed_args())


def test_fail_closed_rejects_v3_configuration():
    args = fail_closed_args()
    args.version = "v3"
    with pytest.raises(ValueError, match="requires V4"):
        runtime.validate_options(args)


def test_fail_closed_real_failure_uses_isolated_failure_artifact(v4_project, monkeypatch):
    root = v4_project
    previous = root/diagnostic.failure_path("v4")
    previous.parent.mkdir(parents=True, exist_ok=True)
    previous.write_text("preserved historical V4 rejection", encoding="utf-8")
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    calls = []
    def fail(*args):
        calls.append(args)
        raise runtime.RequestFailure("HTTP status 400", retryable=False)
    monkeypatch.setattr(runtime, "post_json", fail)
    argv = ["--project-root", str(root), "--version", "v4", "--runtime-configuration", runtime.FAIL_CLOSED, "--execute"]
    assert diagnostic.main(argv) == 2 and len(calls) == 1
    failure_path = diagnostic.failure_path("v4", runtime.FAIL_CLOSED)
    assert failure_path == "results/rag/llm/v4/diagnostics/fail_closed_abstention/six_class_grounding_failures.jsonl"
    failure = common.parse_json((root/failure_path).read_text(encoding="utf-8"))
    assert failure["runtime_configuration"] == runtime.FAIL_CLOSED
    assert failure["completed_before_failure"] == 0 and len(failure["attempt_errors"]) == 1
    assert "HTTP status 400" in failure["final_error_message"]
    assert not any((root/path).exists() for path in diagnostic.output_paths("v4", runtime.FAIL_CLOSED))
    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("maximum,completion,accepted", [(5120, 1024, True), (4608, 1024, False), (5120, 512, False), (4608, 512, False)])
def test_5120_configuration_requires_exact_envelope(maximum, completion, accepted):
    args = fail_closed_args()
    args.runtime_configuration = runtime.FAIL_CLOSED_5120_1024
    args.max_model_len, args.max_completion_tokens = maximum, completion
    if accepted:
        runtime.validate_options(args)
    else:
        with pytest.raises(ValueError, match="requires max_model_len=5120 and max_completion_tokens=1024"):
            runtime.validate_options(args)
    runtime.validate_options(fail_closed_args())  # Original envelope unchanged.


def test_runtime_envelope_directories_are_disjoint():
    old, new = runtime.FAIL_CLOSED, runtime.FAIL_CLOSED_5120_1024
    assert runtime.configuration_directory("v4", old) == "fail_closed_abstention/"
    assert runtime.configuration_directory("v4", new) == "fail_closed_abstention_5120_1024/"
    for paths in (execution.output_paths, diagnostic.output_paths):
        assert set(paths("v4", old)).isdisjoint(paths("v4", new))
    assert diagnostic.failure_path("v4", new) == "results/rag/llm/v4/diagnostics/fail_closed_abstention_5120_1024/six_class_grounding_failures.jsonl"
    assert diagnostic.summary_header(old) == diagnostic.summary_header(new)


@pytest.mark.parametrize("qualifying", [True, False])
def test_new_configuration_uses_identical_fail_closed_constraint(phase2_case, qualifying):
    prompt, value = positive_case(phase2_case)
    if not qualifying:
        value["explanation"] = "Missing evidence quote."
    raw = common.json_text(value)
    old_value, old_metadata = runtime.constrain_response(raw, prompt, runtime.FAIL_CLOSED)
    new_value, new_metadata = runtime.constrain_response(raw, prompt, runtime.FAIL_CLOSED_5120_1024)
    assert old_value == new_value
    assert new_metadata == dict(old_metadata, runtime_configuration=runtime.FAIL_CLOSED_5120_1024)


@pytest.mark.parametrize("audit_state", ["old_only", "failed", "wrong_envelope", "pass"])
def test_new_configuration_requires_its_exact_passing_audit(v4_project, audit_state):
    root = v4_project
    argv = ["--project-root", str(root), "--version", "v4", "--execute"]
    assert audit.main([*argv, "--max-model-len", "4608", "--reserved-completion-tokens", "512"]) == 0
    if audit_state != "old_only":
        assert audit.main([*argv, "--max-model-len", "5120", "--reserved-completion-tokens", "1024"]) == 0
        summary_path = audit.output_paths(5120, 1024, "v4")[1]
        rows = list(common.rows(root, summary_path))
        if audit_state == "failed": rows[0]["gate_status"] = "FAIL"
        if audit_state == "wrong_envelope": rows[0]["reserved_completion_tokens"] = "512"
        common.corpus.write_table(root/summary_path, audit.SUMMARY_HEADER, rows)
    args = fail_closed_args()
    args.runtime_configuration = runtime.FAIL_CLOSED_5120_1024
    args.max_model_len, args.max_completion_tokens = 5120, 1024
    with sqlite3.connect(":memory:") as connection:
        runtime.prepare_population(root, connection, "v4")
        if audit_state == "pass":
            paths = execution.audit_gate(root, connection, args)
            assert paths[1] == "results/rag/llm/v4/token_audits/maxlen_5120_completion_1024/qwen_token_audit_summary.csv"
        else:
            with pytest.raises((ValueError, OSError)):
                execution.audit_gate(root, connection, args)


@pytest.mark.parametrize("mutation,reason", [("valid", ""), ("decision", "SEMANTIC_VALIDATION"),
    ("class_relationship", "SEMANTIC_VALIDATION"), ("absent_citation", "SEMANTIC_VALIDATION"),
    ("unknown_citation", "SEMANTIC_VALIDATION"), ("duplicate_citation", "SEMANTIC_VALIDATION"),
    ("missing_quote", "POSITIVE_EVIDENCE")])
def test_semantic_runtime_abstains_auditably_without_retry(phase2_case, monkeypatch, mutation, reason):
    prompt, value = positive_case(phase2_case)
    if mutation == "decision": value["reasoning_decision"] = "ABSTAIN"
    if mutation == "class_relationship": value["evidence_status"] = "CONTRADICTED"
    if mutation == "absent_citation": value["cited_chunk_ids"] = []
    if mutation == "unknown_citation": value["cited_chunk_ids"] = ["UNKNOWN"]
    if mutation == "duplicate_citation": value["cited_chunk_ids"] = ["C1", "C1"]
    if mutation == "missing_quote": value["explanation"] = "No qualifying quote."
    original = common.json_text(value)
    error = ""
    if reason:
        with pytest.raises(ValueError) as caught:
            runtime.validate_logical(value, prompt)
        error = str(caught.value)
    args = fail_closed_args()
    args.runtime_configuration = runtime.SEMANTIC_STRATIFIED_1200
    args.max_model_len, args.max_completion_tokens = 5120, 1024
    calls, errors = [], []
    def post(*a):
        calls.append(a)
        return structured_response(value)
    monkeypatch.setattr(runtime, "post_json", post)
    result = runtime.generate(prompt, 100, args, errors.append)
    assert len(calls) == 1 and not errors and result["retry_count"] == 0
    assert result["raw_model_response"] == original and common.json_text(value) == original
    assert result["validator_fallback_reason"] == reason
    assert result["validator_fallback_applied"] == ("YES" if reason else "NO")
    constrained = {key: result[key] for key in runtime.FIELDS}
    if reason:
        assert constrained == dict(query_id=value["query_id"], reasoning_decision="ABSTAIN", evidence_status="INSUFFICIENT",
                                   supported_class=None, cited_chunk_ids=[], explanation=runtime.SEMANTIC_FALLBACK_EXPLANATION)
        field = "semantic_validation_error_message" if reason == "SEMANTIC_VALIDATION" else "grounding_gate_error_message"
        assert result[field] == error
    else:
        assert constrained == value
    execution.validate_saved(result, prompt, 100, args)
    if reason:
        result["validator_fallback_reason"] = "forged"
        with pytest.raises(ValueError):
            execution.validate_saved(result, prompt, 100, args)


@pytest.mark.parametrize("failure", ["query", "transport", "tokens", "budget", "implementation"])
def test_semantic_runtime_preserves_hard_failures(phase2_case, monkeypatch, failure):
    prompt, value = positive_case(phase2_case)
    value["reasoning_decision"] = "ABSTAIN"  # Otherwise eligible semantic rejection.
    if failure == "missing": value.pop("explanation")
    if failure == "extra": value["extra"] = "invalid"
    if failure == "query": value["query_id"] = "UNKNOWN"
    if failure == "decision_type": value["reasoning_decision"] = None
    if failure == "class_type": value["supported_class"] = []
    if failure == "citation_type": value["cited_chunk_ids"] = "C1"
    if failure == "citation_element_type": value["cited_chunk_ids"] = [1]
    args = fail_closed_args()
    args.runtime_configuration = runtime.SEMANTIC_STRATIFIED_1200
    args.max_model_len, args.max_completion_tokens = 5120, 1024
    calls = []
    def post(*a):
        calls.append(a)
        if failure == "transport": raise runtime.RequestFailure("HTTP status 400", retryable=False)
        data = structured_response(value)
        if failure == "json": data["choices"][0]["message"]["content"] = "{"
        if failure == "tokens": data["usage"]["prompt_tokens"] = 99
        if failure == "budget": data["usage"].update(completion_tokens=1025, total_tokens=1125)
        return data
    monkeypatch.setattr(runtime, "post_json", post)
    if failure == "implementation":
        def unexpected(*a):
            raise ValueError("Unexpected implementation failure")
        monkeypatch.setattr(runtime.importer, "validate_response", unexpected)
    with pytest.raises(runtime.RequestFailure):
        runtime.generate(prompt, 100, args)
    assert len(calls) == 1


@pytest.mark.parametrize("finish", ["length", "incomplete", None])
@pytest.mark.parametrize("partial", [True, False])
def test_truncated_completion_is_distinct_abstention_without_retry(phase2_case, monkeypatch, finish, partial):
    prompt, value = positive_case(phase2_case)
    raw = '{"query_id": ' + common.json_text(prompt["query_id"]) + ', "explanation": "unfinished' if partial else common.json_text(value)
    args = fail_closed_args()
    args.runtime_configuration = runtime.SEMANTIC_STRATIFIED_1200
    args.max_model_len, args.max_completion_tokens = 5120, 1024
    calls, errors = [], []
    def post(*a):
        calls.append(a)
        return dict(choices=[dict(finish_reason=finish, message=dict(content=raw, reasoning_content="HIDDEN_NOT_FOR_LOGGING"))],
                    usage=dict(prompt_tokens=100, completion_tokens=1024, total_tokens=1124))
    monkeypatch.setattr(runtime, "post_json", post)
    result = runtime.generate(prompt, 100, args, errors.append)
    assert len(calls) == 1 and errors == [] and result["retry_count"] == 0
    assert {key: result[key] for key in runtime.FIELDS} == dict(query_id=prompt["query_id"], reasoning_decision="ABSTAIN", evidence_status="INSUFFICIENT",
        supported_class=None, cited_chunk_ids=[], explanation=runtime.TRUNCATION_EXPLANATION)
    assert result["raw_model_response"] == raw
    assert result["original_finish_reason"] == result["finish_reason"] == finish
    assert result["runtime_fallback_applied"] == "YES" and result["runtime_fallback_reason"] == "TRUNCATED_COMPLETION"
    assert result["validator_fallback_applied"] == "NO" and result["validator_fallback_reason"] == ""
    assert result["semantic_validation_status"] == result["grounding_gate_status"] == "NOT_APPLICABLE"
    assert (result["prompt_tokens"], result["completion_tokens"], result["total_tokens"]) == (100, 1024, 1124)
    assert result["latency_seconds"] >= 0 and "HIDDEN_NOT_FOR_LOGGING" not in common.json_text(result)
    execution.validate_saved(result, prompt, 100, args)
    for key in ("runtime_fallback_reason", "original_finish_reason", "explanation"):
        with pytest.raises(ValueError):
            execution.validate_saved(dict(result, **{key: "tampered"}), prompt, 100, args)


@pytest.mark.parametrize("failure", ["wrong_query", "partial_wrong_query", "tokens", "budget"])
def test_truncation_does_not_hide_hard_failures(phase2_case, monkeypatch, failure):
    prompt, value = positive_case(phase2_case)
    args = fail_closed_args()
    args.runtime_configuration = runtime.SEMANTIC_STRATIFIED_1200
    args.max_model_len, args.max_completion_tokens = 5120, 1024
    calls = []
    def post(*a):
        calls.append(a)
        if failure == "http": raise runtime.RequestFailure("HTTP status 503")
        content = '{"query_id":"UNKNOWN"}' if failure == "wrong_query" else '{"query_id":"UNKNOWN",' if failure == "partial_wrong_query" else "{"
        return dict(choices=[dict(finish_reason="length", message=dict(content=content))],
                    usage=dict(prompt_tokens=99 if failure == "tokens" else 100,
                               completion_tokens=1025 if failure == "budget" else 1024,
                               total_tokens=1125 if failure == "budget" else 1124))
    monkeypatch.setattr(runtime, "post_json", post)
    with pytest.raises(runtime.RequestFailure):
        runtime.generate(prompt, 100, args)
    assert len(calls) == 1


@pytest.mark.parametrize("mutation", ["json", "missing", "extra", "wrong_type", "null_content"])
def test_hardened_model_format_fallback_preserves_error(phase2_case, monkeypatch, mutation):
    prompt, value = positive_case(phase2_case)
    args = fail_closed_args()
    args.runtime_configuration = runtime.SEMANTIC_STRATIFIED_1200
    args.max_model_len, args.max_completion_tokens = 5120, 1024
    if mutation == "missing": value.pop("reasoning_decision")
    if mutation == "extra": value["extra"] = "forbidden"
    if mutation == "wrong_type": value["cited_chunk_ids"] = "C1"
    raw = "{" if mutation == "json" else None if mutation == "null_content" else common.json_text(value)
    calls = []
    def post(*a):
        calls.append(a)
        data = structured_response(value)
        data["choices"][0]["message"]["content"] = raw
        return data
    monkeypatch.setattr(runtime, "post_json", post)
    result = runtime.generate(prompt, 100, args)
    assert len(calls) == 1 and result["runtime_fallback_reason"] == "UNPARSEABLE_MODEL_OUTPUT"
    assert result["raw_model_response"] == raw and result["runtime_fallback_error_message"]
    assert result["evidence_status"] == "INSUFFICIENT" and result["supported_class"] is None and result["cited_chunk_ids"] == []
    execution.validate_saved(result, prompt, 100, args)
    with pytest.raises(ValueError):
        execution.validate_saved(dict(result, runtime_fallback_error_message="forged"), prompt, 100, args)


def test_previous_runtime_truncation_remains_failure(phase2_case, monkeypatch):
    prompt, value = positive_case(phase2_case)
    args = fail_closed_args()
    args.runtime_configuration = runtime.STRATIFIED_1200
    args.max_model_len, args.max_completion_tokens = 5120, 1024
    def post(*a):
        data = structured_response(value)
        data["choices"][0]["finish_reason"] = "length"
        return data
    monkeypatch.setattr(runtime, "post_json", post)
    with pytest.raises(runtime.RequestFailure, match="Truncated or incomplete"):
        runtime.generate(prompt, 100, args)
