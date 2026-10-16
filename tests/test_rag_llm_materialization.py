import pytest

from src.rag import phase2_common as common
from src.rag import materialize_constrained_llm_experiment as prompts
from src.rag import build_trusted_contexts as contexts
from src.rag import run_provenance_aware_reranking as rerank


def test_prompt_schema_and_allowed_evidence(phase2_case):
    case = phase2_case
    ranked = rerank.rerank(case.query, [r for r in case.results if r["retrieval_method"] == "hybrid_rrf"], case.verified, case.config)
    context = contexts.construct(case.query, ranked, case.chunks, case.verified, case.policy)
    record = prompts.prompt_record(context)
    assert record == prompts.prompt_record(context)
    assert set(record) == set(prompts.HEADER)
    text = record["prompt_text"]
    assert "Use only the supplied trusted evidence" in text
    assert "Do not cite chunks that were not supplied" in text
    assert "Return INSUFFICIENT" in text
    assert "structured JSON only" in text
    assert "not private reasoning traces" in text
    payload = common.parse_json(text.split("INPUT_JSON:\n")[1])
    assert payload["allowed_chunk_ids"] == common.parse_json(context["selected_chunk_ids"])
    assert payload["query_id"] == case.query["query_id"]
    assert not {"true_class_label", "true_class_id", "classifier_correct"} & set(payload)


def test_empty_evidence_prompt_allows_no_citations(phase2_case):
    case = phase2_case
    value = contexts.construct(case.query, [], case.chunks, case.verified, case.policy)
    record = prompts.prompt_record(value)
    assert record["allowed_chunk_ids"] == "[]"


@pytest.mark.parametrize("rule", list(prompts.GROUNDING_RULES))
def test_each_grounding_rule_required_in_instructions(phase2_case, rule):
    case = phase2_case
    context = contexts.construct(case.query, [], case.chunks, case.verified, case.policy)
    record = prompts.prompt_record(context)
    statement = prompts.GROUNDING_RULES[rule]
    instructions, payload = record["prompt_text"].split("INPUT_JSON:\n", 1)
    assert statement in instructions
    # Evidence text containing a rule cannot substitute for an instruction.
    altered = dict(record, prompt_text=instructions.replace(statement, "") + "INPUT_JSON:\n" + payload + statement)
    with pytest.raises(ValueError, match=rule):
        prompts.validate_grounding(altered)


def test_response_schema_and_supersession_are_explicit(phase2_case):
    assert prompts.REQUIRED_RESPONSE_FIELDS == ("query_id", "reasoning_decision", "evidence_status", "supported_class", "explanation", "cited_chunk_ids")
    assert prompts.SCHEMA_VERSION == "constrained_evidence_v2"
    policy = {r["policy_id"]: r["value"] for r in prompts.policy_rows()}
    assert policy["v1_status"] == "SUPERSEDED_NOT_EXECUTABLE"
    assert policy["v1_execution_allowed"] == "NO"
    assert common.parse_json(policy["reasoning_decisions"]) == ["SUPPORT_CLASS", "ABSTAIN"]
    assert common.parse_json(policy["evidence_statuses"]) == list(common.STATUSES)
    assert set(prompts.OUTPUTS).isdisjoint(prompts.V1_OUTPUTS)
    assert len(prompts.OUTPUTS) == 4
    context = contexts.construct(phase2_case.query, [], phase2_case.chunks, phase2_case.verified, phase2_case.policy)
    record = dict(prompts.prompt_record(context), prompt_schema_version="constrained_evidence_v1")
    with pytest.raises(ValueError, match="SUPERSEDED_NOT_EXECUTABLE"):
        prompts.validate_grounding(record)


@pytest.mark.parametrize("dataset,classes", list(common.CLASSES.items()))
def test_v2_payload_exact_classes_and_no_ground_truth(phase2_case, dataset, classes):
    case = phase2_case
    context = contexts.construct(case.query, [], case.chunks, case.verified, case.policy)
    context.update(source_dataset=dataset, classifier_prediction=classes[0], true_class_label="DO_NOT_EXPOSE", true_class_id=99)
    record = prompts.prompt_record(context)
    payload = common.parse_json(record["prompt_text"].split("INPUT_JSON:\n")[1])
    assert payload["allowed_classes"] == list(classes)
    assert payload["classifier_prediction"] == classes[0]
    assert "DO_NOT_EXPOSE" not in record["prompt_text"]
    assert payload["allowed_chunk_ids"] == []


def test_v2_dry_and_execute_preserve_v1_and_upstream(tmp_path, phase2_case, monkeypatch):
    case = phase2_case
    context = contexts.construct(case.query, [], case.chunks, case.verified, case.policy)
    # Synthetic inputs only; no actual corpus, tokenizer, or provenance execution.
    inputs = [common.QUERY_PATH, *contexts.OUTPUTS, *prompts.provenance.OUTPUTS, *prompts.V1_OUTPUTS]
    for name in inputs:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"preserved synthetic input\r\n")
    common.corpus.write_table(tmp_path/common.QUERY_PATH, prompts.QUERY_FIELDS, [{key: case.query[key] for key in prompts.QUERY_FIELDS}])
    common.corpus.write_table(tmp_path/contexts.OUTPUTS[0], contexts.HEADER, [context])
    before = {name: (tmp_path/name).read_bytes() for name in inputs}
    monkeypatch.setattr(common, "QUERY_COUNTS", {case.query["source_dataset"]: 1})
    monkeypatch.setattr(prompts, "validation_resources", lambda root: (dict(token_budget=2048, top_n=5, max_chunks_per_document=0), case.chunks, case.verified, case.policy.count))
    argv = ["--project-root", str(tmp_path)]
    assert prompts.main(argv) == 0
    assert not any((tmp_path/name).exists() for name in prompts.OUTPUTS)
    assert prompts.main([*argv, "--execute"]) == 0
    assert {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()} == set(inputs) | set(prompts.OUTPUTS)
    assert all((tmp_path/name).read_bytes() == value for name, value in before.items())
    checks = list(common.rows(tmp_path, prompts.OUTPUTS[2]))
    assert set(prompts.GROUNDING_RULES) <= {row["check"] for row in checks}
    assert all(row["status"] == "PASS" for row in checks)
    assert list(common.rows(tmp_path, prompts.OUTPUTS[3])) == prompts.policy_rows()
    records = prompts.validated_prompts(tmp_path, {case.query["query_id"]: case.query}, {case.query["query_id"]: context})
    assert records[case.query["query_id"]]["prompt_schema_version"] == prompts.SCHEMA_VERSION
    saved = {name: (tmp_path/name).read_bytes() for name in prompts.OUTPUTS}
    assert prompts.main([*argv, "--execute"]) == 2
    assert all((tmp_path/name).read_bytes() == value for name, value in saved.items())


@pytest.mark.parametrize("field,value", [("source_split", "test"), ("rag_partition", "test"), ("classifier_prediction", "UNKNOWN"), ("selected_chunk_ids", '["NOT_SUPPLIED"]')])
def test_v2_rejects_invalid_context(phase2_case, field, value):
    case = phase2_case
    context = contexts.construct(case.query, [], case.chunks, case.verified, case.policy)
    with pytest.raises(ValueError):
        prompts.prompt_record(dict(context, **{field: value}))


def streaming_inputs(root, case, monkeypatch, count=9):
    ranked = rerank.rerank(case.query, [r for r in case.results if r["retrieval_method"] == "hybrid_rrf"], case.verified, case.config)
    template = contexts.construct(case.query, ranked, case.chunks, case.verified, case.policy)
    queries = [dict(case.query, query_id=f"Q-{i:04d}") for i in range(count)]
    trusted = [dict(template, query_id=row["query_id"]) for row in queries]
    common.corpus.write_table(root/common.QUERY_PATH, prompts.QUERY_FIELDS, [{k: r[k] for k in prompts.QUERY_FIELDS} for r in reversed(queries)])
    common.corpus.write_table(root/contexts.OUTPUTS[0], contexts.HEADER, trusted[::2] + trusted[1::2])
    monkeypatch.setattr(common, "QUERY_COUNTS", {case.query["source_dataset"]: count})
    monkeypatch.setattr(prompts, "validation_resources", lambda root: (dict(token_budget=2048, top_n=5, max_chunks_per_document=0), case.chunks, case.verified, case.policy.count))
    return queries, trusted


def test_streaming_visits_every_row_without_population_loaders(tmp_path, phase2_case, monkeypatch):
    _, trusted = streaming_inputs(tmp_path, phase2_case, monkeypatch)
    original = prompts.prompt_record
    visited = []
    def track(row):
        visited.append(row["query_id"])
        return original(row)
    monkeypatch.setattr(prompts, "prompt_record", track)
    monkeypatch.setattr(common, "queries", lambda *args: pytest.fail("Full query population loaded"))
    monkeypatch.setattr(contexts, "validated_records", lambda *args: pytest.fail("Full context population loaded"))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert prompts.materialize_rows(tmp_path, False, 2) == len(trusted)
    assert visited == sorted(row["query_id"] for row in trusted)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert not list(tmp_path.glob(".rag_llm_v2_*"))


def test_chunk_sizes_produce_identical_bytes_order_and_semantics(tmp_path, phase2_case, monkeypatch):
    outputs = []
    for size in (1, 2, 32):
        root = tmp_path/str(size)
        root.mkdir()
        _, trusted = streaming_inputs(root, phase2_case, monkeypatch)
        assert prompts.materialize_rows(root, True, size) == len(trusted)
        outputs.append((root/prompts.OUTPUTS[0]).read_bytes())
        rows = list(common.rows(root, prompts.OUTPUTS[0], prompts.HEADER))
        assert rows == [{k: str(v) for k, v in prompts.prompt_record(row).items()} for row in sorted(trusted, key=lambda r: r["query_id"])]
        assert outputs[-1].count((",".join(prompts.HEADER)+"\r\n").encode()) == 1
        assert not list(root.glob(".rag_llm_v2_*"))
    assert outputs[0] == outputs[1] == outputs[2]


@pytest.mark.parametrize("mutation", ["duplicate_query", "duplicate_context", "missing_context", "unknown_context", "late_token_count", "late_test", "late_citation"])
def test_incremental_validation_rejects_invalid_population(tmp_path, phase2_case, monkeypatch, mutation):
    queries, trusted = streaming_inputs(tmp_path, phase2_case, monkeypatch)
    if mutation == "duplicate_query": queries.append(queries[0])
    if mutation == "duplicate_context": trusted.append(trusted[0])
    if mutation == "missing_context": trusted.pop()
    if mutation == "unknown_context": trusted[-1]["query_id"] = "UNKNOWN"
    if mutation == "late_token_count": trusted[-1]["context_token_count"] = 99999
    if mutation == "late_test": trusted[-1]["rag_partition"] = "test"
    if mutation == "late_citation": trusted[-1]["selected_chunk_ids"] = '["UNKNOWN"]'
    common.corpus.write_table(tmp_path/common.QUERY_PATH, prompts.QUERY_FIELDS, [{k: r[k] for k in prompts.QUERY_FIELDS} for r in queries])
    common.corpus.write_table(tmp_path/contexts.OUTPUTS[0], contexts.HEADER, trusted)
    with pytest.raises((ValueError, prompts.sqlite3.IntegrityError)):
        prompts.materialize_rows(tmp_path, False, 2)
    assert not any((tmp_path/name).exists() for name in prompts.OUTPUTS)
    assert not list(tmp_path.glob(".rag_llm_v2_*"))


def test_token_accounting_disables_advisory_without_truncation(phase2_case, monkeypatch):
    from types import SimpleNamespace
    calls = []
    def tokenizer(text, **kwargs):
        calls.append((text, kwargs))
        return {"input_ids": list(range(len(text.split())+2))}
    policy = SimpleNamespace(tokenizer=tokenizer, do_lower_case=True)
    config = dict(tokenizer_identity_json="{}")
    monkeypatch.setattr(common, "completed", lambda *args: config)
    monkeypatch.setattr(contexts.index.v1, "model_identity", lambda *args: {})
    monkeypatch.setattr(contexts.index.v1, "load_token_policy", lambda *args: policy)
    monkeypatch.setattr(contexts.index, "validated_chunks", lambda *args: [])
    monkeypatch.setattr(prompts.provenance, "load_verified", lambda *args: {})
    _, _, _, count = prompts.validation_resources(None)
    assert count(" WORD " * 600) == 602
    assert calls[0][0] == (" WORD " * 600).strip().lower()
    assert calls[0][1] == dict(add_special_tokens=True, truncation=False, padding=False, verbose=False)


def test_execute_retains_only_one_prompt_batch(tmp_path, phase2_case, monkeypatch):
    streaming_inputs(tmp_path, phase2_case, monkeypatch, count=41)
    original = prompts.prompt_record
    live = peak = built = 0
    class TrackedPrompt(dict):
        def __init__(self, value):
            nonlocal live, peak, built
            super().__init__(value)
            live += 1
            built += 1
            peak = max(peak, live)
        def __del__(self):
            nonlocal live
            live -= 1
    monkeypatch.setattr(prompts, "prompt_record", lambda row: TrackedPrompt(original(row)))
    assert prompts.materialize_rows(tmp_path, True, 3) == 41
    assert built == 41 and peak <= 4  # Batch plus at most one previous row.
    assert live == 0


@pytest.mark.parametrize("name", list(prompts.V3_RULES))
def test_v3_rules_and_invalid_examples_mandatory(phase2_case, name):
    case = phase2_case
    context = contexts.construct(case.query, [], case.chunks, case.verified, case.policy)
    v2 = prompts.prompt_record(context, "v2")
    v3 = prompts.prompt_record(context, "v3")
    assert v2 == prompts.prompt_record(context)
    assert v3["prompt_schema_version"] == "constrained_evidence_v3"
    assert len(prompts.GROUNDING_RULES) == len(prompts.V3_RULES) == 15
    for rule in (*prompts.GROUNDING_RULES.values(), *prompts.V3_RULES.values()):
        assert rule in v3["prompt_text"].partition("INPUT_JSON:\n")[0]
    assert v3["prompt_text"].count('INVALID: "') == 4
    assert v3["prompt_text"].partition("INPUT_JSON:\n")[2] == v2["prompt_text"].partition("INPUT_JSON:\n")[2]
    assert v3["allowed_chunk_ids"] == v2["allowed_chunk_ids"]
    with pytest.raises(ValueError, match=name):
        prompts.validate_grounding(dict(v3, prompt_text=v3["prompt_text"].replace(prompts.V3_RULES[name], "")), "v3")


def v3_project(root, case, monkeypatch):
    queries, trusted = streaming_inputs(root, case, monkeypatch)
    for name in (*contexts.OUTPUTS[1:], *prompts.provenance.OUTPUTS, *prompts.V1_OUTPUTS, *prompts.OUTPUTS):
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("protected synthetic file", encoding="utf-8")
    return queries, trusted


@pytest.mark.parametrize("version", ["v3", "v4"])
def test_v3_cli_preserves_prior_versions_and_is_deterministic(tmp_path, phase2_case, monkeypatch, version):
    _, trusted = v3_project(tmp_path, phase2_case, monkeypatch)
    outputs = prompts.version_settings(version)[0]
    if version == "v4":
        for name in prompts.V3_OUTPUTS:
            path = tmp_path/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("preserved V3 artifact", encoding="utf-8")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    argv = ["--project-root", str(tmp_path), "--version", version, "--chunk-size", "2"]
    assert prompts.main(argv) == 0
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert prompts.main([*argv, "--execute"]) == 0
    assert all(p.read_bytes() == value for p, value in before.items())
    assert {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file() and p not in before} == set(outputs)
    rows = list(common.rows(tmp_path, outputs[0]))
    assert rows == [{k: str(v) for k, v in prompts.prompt_record(row, version).items()} for row in sorted(trusted, key=lambda r: r["query_id"])]
    assert list(common.rows(tmp_path, outputs[3])) == prompts.policy_rows(version)
    config = {r["field"]: r["value"] for r in common.rows(tmp_path, outputs[1])}
    assert config["v1_status"] == config["v2_status"] == "SUPERSEDED_NOT_EXECUTABLE"
    if version == "v4":
        assert config["v3_status"] == "SUPERSEDED_NOT_EXECUTABLE"
    saved = (tmp_path/outputs[0]).read_bytes()
    assert prompts.main([*argv, "--execute"]) == 2
    assert (tmp_path/outputs[0]).read_bytes() == saved


@pytest.mark.parametrize("failure", ["late_validation", "publication"])
@pytest.mark.parametrize("version", ["v3", "v4"])
def test_v3_failure_leaves_no_partial_canonical_artifacts(tmp_path, phase2_case, monkeypatch, failure, version):
    _, trusted = v3_project(tmp_path, phase2_case, monkeypatch)
    if failure == "late_validation":
        trusted[-1]["context_token_count"] = 99999
        common.corpus.write_table(tmp_path/contexts.OUTPUTS[0], contexts.HEADER, trusted)
    else:
        original = prompts.os.link
        calls = 0
        def fail_second(source, target):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("Synthetic publication failure")
            original(source, target)
        monkeypatch.setattr(prompts.os, "link", fail_second)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert prompts.main(["--project-root", str(tmp_path), "--version", version, "--chunk-size", "2", "--execute"]) == 2
    assert not any((tmp_path/name).exists() for name in prompts.version_settings(version)[0])
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert not list(tmp_path.glob(".rag_*"))


@pytest.mark.parametrize("dataset,classes", list(common.CLASSES.items()))
@pytest.mark.parametrize("version", ["v3", "v4"])
def test_v3_retains_exact_allowed_classes_and_six_field_contract(phase2_case, dataset, classes, version):
    case = phase2_case
    context = contexts.construct(case.query, [], case.chunks, case.verified, case.policy)
    for label in classes:
        context.update(source_dataset=dataset, classifier_prediction=label, true_class_label="FORBIDDEN_GROUND_TRUTH")
        record = prompts.prompt_record(context, version)
        payload = common.parse_json(record["prompt_text"].partition("INPUT_JSON:\n")[2])
        assert payload["allowed_classes"] == list(classes)
        assert payload["classifier_prediction"] == label
        assert "FORBIDDEN_GROUND_TRUTH" not in record["prompt_text"]
        assert ", ".join(prompts.REQUIRED_RESPONSE_FIELDS) in record["prompt_text"]
        assert len(prompts.REQUIRED_RESPONSE_FIELDS) == 6
        for status in common.STATUSES:
            assert status in record["prompt_text"]


@pytest.mark.parametrize("version", ["v3", "v4"])
def test_v3_chunked_live_prompt_bound(tmp_path, phase2_case, monkeypatch, version):
    streaming_inputs(tmp_path, phase2_case, monkeypatch, count=41)
    original = prompts.prompt_record
    live = peak = built = 0
    class Tracked(dict):
        def __init__(self, value):
            nonlocal live, peak, built
            super().__init__(value)
            live += 1
            built += 1
            peak = max(peak, live)
        def __del__(self):
            nonlocal live
            live -= 1
    monkeypatch.setattr(prompts, "prompt_record", lambda row, version="v2": Tracked(original(row, version)))
    assert prompts.materialize_rows(tmp_path, True, 2, version, tmp_path/"synthetic_staged.csv") == 41
    assert built == 41 and peak <= 3 and live == 0
    assert not any((tmp_path/name).exists() for name in prompts.V3_OUTPUTS)


def test_v4_prompt_bytes_independent_of_chunk_size(tmp_path, phase2_case, monkeypatch):
    results = []
    for size in (1, 2, 32):
        root = tmp_path/str(size)
        root.mkdir()
        _, trusted = streaming_inputs(root, phase2_case, monkeypatch)
        destination = root/"staged.csv"
        assert prompts.materialize_rows(root, True, size, "v4", destination) == len(trusted)
        results.append(destination.read_bytes())
        assert not any((root/name).exists() for name in prompts.V4_OUTPUTS)
    assert results[0] == results[1] == results[2]
