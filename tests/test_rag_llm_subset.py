"""Synthetic development populations; no real model, tokenizer, network or TEST data."""
import sqlite3
from collections import Counter

import pytest

from src.rag import materialize_llm_evaluation_subset as subset
from src.rag import run_constrained_llm_v3 as execution
from src.rag import run_v3_grounding_diagnostic as diagnostic

runtime, common, prompts = subset.runtime, subset.common, subset.prompts
audit = runtime.audit
RETIRED_PROMPTS = ("results/rag/llm/llm_prompt_records.csv", "results/rag/llm/v2/llm_prompt_records.csv", "results/rag/llm/v3/llm_prompt_records.csv")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(runtime.urllib.request, "build_opener", lambda *a, **k: pytest.fail("Network prohibited"))
    monkeypatch.setattr(runtime, "post_json", lambda *a, **k: pytest.fail("Unexpected generation"))
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [1]*100}
    monkeypatch.setattr(audit, "load_tokenizer", lambda *a: Tokenizer())


def small_population(connection, per_stratum=205, reverse=False, label="ignored"):
    connection.execute("CREATE TABLE prompts (id TEXT PRIMARY KEY, payload TEXT)")
    rows = [(f"Q-{s}-{i:04d}", common.json_text(dict(source_dataset=dataset, classifier_prediction=prediction, true_class_label=label)))
            for s, (dataset, prediction) in enumerate(subset.STRATA) for i in range(per_stratum)]
    connection.executemany("INSERT INTO prompts VALUES (?, ?)", reversed(rows) if reverse else rows)


def test_selection_exact_deterministic_insertion_order_and_gt_independent():
    results = []
    for reverse, label in ((False, "label_a"), (True, "label_b")):
        with sqlite3.connect(":memory:") as connection:
            small_population(connection, reverse=reverse, label=label)
            results.append(subset.select_rows(connection))
    assert results[0] == results[1]
    rows = results[0]
    assert len(rows) == len({row["query_id"] for row in rows}) == 1200
    assert Counter((row["source_dataset"], row["classifier_prediction"]) for row in rows) == dict.fromkeys(subset.STRATA, 200)
    assert all(row["sampling_seed"] == "42" and set(row) == set(subset.HEADER) for row in rows)
    assert [int(row["selection_ordinal"]) for row in rows] == list(range(1, 1201))
    assert [row["query_id"] for row in rows] == sorted(row["query_id"] for row in rows)


def test_sampling_fails_for_undersized_stratum():
    with sqlite3.connect(":memory:") as connection:
        small_population(connection, per_stratum=199)
        with pytest.raises(ValueError, match="at least 200"):
            subset.select_rows(connection)


@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "wrong_seed", "wrong_stratum", "missing", "extra", "changed_selection"])
def test_manifest_validates_membership_counts_and_exact_selection(tmp_path, mutation):
    with sqlite3.connect(":memory:") as connection:
        small_population(connection)
        rows = subset.select_rows(connection)
        if mutation == "duplicate": rows[-1] = rows[0]
        if mutation == "unknown": rows[0]["query_id"] = "UNKNOWN"
        if mutation == "wrong_seed": rows[0]["sampling_seed"] = "43"
        if mutation == "wrong_stratum": rows[0]["classifier_prediction"] = "VPN"
        if mutation == "missing": rows.pop()
        if mutation == "extra": rows.append(rows[0])
        if mutation == "changed_selection":
            selected = {row["query_id"] for row in rows}
            alternative = next(identity for (identity,) in connection.execute("SELECT id FROM prompts ORDER BY id") if identity not in selected)
            rows[0]["query_id"] = alternative
        common.corpus.write_table(tmp_path/subset.MANIFEST, subset.HEADER, rows)
        with pytest.raises(ValueError):
            subset.validate_manifest(tmp_path, connection)


@pytest.fixture
def population(tmp_path, phase2_case, monkeypatch):
    queries, records = [], []
    case = phase2_case
    for s, (dataset, prediction) in enumerate(subset.STRATA):
        classes = common.CLASSES[dataset]
        for i in range(205):
            query = dict(case.query, query_id=f"Q-{s}-{i:04d}", source_dataset=dataset, predicted_class_label=prediction,
                         confidence=str(1/len(classes)), class_probabilities_json=common.json_text(dict.fromkeys(classes, 1/len(classes))))
            context = prompts.contexts.construct(query, [], case.chunks, case.verified, case.policy)
            queries.append({key: query[key] for key in prompts.QUERY_FIELDS})
            records.append(prompts.prompt_record(context, "v4"))
    common.corpus.write_table(tmp_path/common.QUERY_PATH, prompts.QUERY_FIELDS, queries)
    common.corpus.write_table(tmp_path/prompts.V4_OUTPUTS[0], prompts.HEADER, records)
    identity = common.snapshot(tmp_path, [common.QUERY_PATH])
    identity.extend(dict(path=path, size_bytes=3000000000, mtime_ns=1) for path in RETIRED_PROMPTS)
    for version in ("v2", "v3"):
        path = f"metadata/rag/constrained_llm_reasoning_{version}_policy.csv"
        common.corpus.write_table(tmp_path/path, prompts.POLICY_HEADER, prompts.policy_rows(version))
        identity.extend(common.snapshot(tmp_path, [path]))
    config = dict(prompt_schema_version="constrained_evidence_v4", active_prompt_path=prompts.V4_OUTPUTS[0], input_metadata_json=common.json_text(identity),
                  v2_artifacts=common.json_text(prompts.OUTPUTS), v3_artifacts=common.json_text(prompts.V3_OUTPUTS))
    common.corpus.write_table(tmp_path/prompts.V4_OUTPUTS[1], common.CONFIG_HEADER, common.config_rows(config))
    common.corpus.write_table(tmp_path/prompts.V4_OUTPUTS[2], common.corpus.foundation.CHECKLIST_HEADER, common.checks("llm_materialization_v4", {"complete": True}))
    common.corpus.write_table(tmp_path/prompts.V4_OUTPUTS[3], prompts.POLICY_HEADER, prompts.policy_rows("v4"))
    monkeypatch.setattr(common, "QUERY_COUNTS", {"CIC-Darknet2020": 820, "USTC-TFC2016": 410})
    return tmp_path


def test_materialization_validation_dry_and_no_overwrite(population):
    root = population
    argv = ["--project-root", str(root)]
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert subset.main(argv) == 0
    assert before == {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert subset.main([*argv, "--execute"]) == 0
    manifest = (root/subset.MANIFEST).read_bytes()
    assert subset.main([*argv, "--validate"]) == 0
    assert subset.main([*argv, "--execute"]) == 2
    assert (root/subset.MANIFEST).read_bytes() == manifest
    assert all(path.read_bytes() == data for path, data in before.items())
    assert not list(root.glob(".rag_llm_subset_*"))


def test_test_partition_is_rejected_before_manifest_creation(population):
    rows = list(common.rows(population, common.QUERY_PATH))
    rows[-1]["source_split"] = "test"
    common.corpus.write_table(population/common.QUERY_PATH, prompts.QUERY_FIELDS, rows)
    assert subset.main(["--project-root", str(population), "--execute"]) == 2
    assert not (population/subset.MANIFEST).exists()


@pytest.mark.parametrize("configuration", [runtime.STRATIFIED_1200, runtime.SEMANTIC_STRATIFIED_1200])
def test_executor_only_manifest_ids_resume_export_and_previous_runs_preserved(population, monkeypatch, configuration):
    root = population
    base = ["--project-root", str(root)]
    assert subset.main([*base, "--execute"]) == 0
    manifest = list(common.rows(root, subset.MANIFEST))
    selected = [row["query_id"] for row in manifest]
    assert audit.main([*base, "--version", "v4", "--max-model-len", "5120", "--reserved-completion-tokens", "1024", "--execute"]) == 0
    calls = []
    failing = None
    evaluating = False
    def post(endpoint, payload, timeout):
        identity = payload["response_format"]["json_schema"]["schema"]["properties"]["query_id"]["enum"][0]
        calls.append(identity)
        if identity == failing:
            raise runtime.RequestFailure("HTTP status 400", retryable=False)
        value = dict(query_id=identity, reasoning_decision="ABSTAIN", evidence_status="INSUFFICIENT", supported_class=None,
                     explanation="No supplied evidence.", cited_chunk_ids=[])
        if evaluating and configuration == runtime.SEMANTIC_STRATIFIED_1200 and identity == selected[0]:
            value["reasoning_decision"] = "SUPPORT_CLASS"
        response = dict(choices=[dict(finish_reason="stop", message=dict(content=common.json_text(value)))],
                        usage=dict(prompt_tokens=100, completion_tokens=1, total_tokens=101))
        if evaluating and configuration == runtime.SEMANTIC_STRATIFIED_1200 and identity == selected[1]:
            response["choices"][0].update(finish_reason="length", message=dict(content='{"query_id":' + common.json_text(identity) + ', "explanation":"unfinished'))
            response["usage"].update(completion_tokens=1024, total_tokens=1124)
        return response
    monkeypatch.setattr(runtime, "post_json", post)
    envelope = ["--version", "v4", "--max-model-len", "5120", "--max-completion-tokens", "1024"]
    assert diagnostic.main([*base, *envelope, "--runtime-configuration", runtime.FAIL_CLOSED_5120_1024, "--execute"]) == 0
    evaluating = True
    previous = [runtime.FAIL_CLOSED, runtime.FAIL_CLOSED_5120_1024]
    if configuration == runtime.SEMANTIC_STRATIFIED_1200:
        previous.append(runtime.STRATIFIED_1200)
    for prior_configuration in previous:
        for name in execution.output_paths("v4", prior_configuration):
            path = root/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("preserved incomplete run", encoding="utf-8")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    calls.clear()
    preflight_counts, reads = [], []
    native = root.parent/(root.name + "_native_runtime")
    monkeypatch.setenv("RAG_RUNTIME_TMPDIR", str(native))
    if configuration == runtime.SEMANTIC_STRATIFIED_1200:
        monkeypatch.setattr(runtime, "prepare_population", lambda *a: pytest.fail("Full population spool prohibited"))
        monkeypatch.setattr(audit, "audit_population", lambda *a, **k: pytest.fail("Full audit reconstruction prohibited"))
        monkeypatch.setattr(prompts.contexts, "validated_records", lambda *a: pytest.fail("Context reconstruction prohibited"))
        original_rows = common.rows
        def streamed(project, relative, *args):
            assert "trusted_context_records.csv" not in str(relative)
            reads.append(relative)
            yield from original_rows(project, relative, *args)
        monkeypatch.setattr(common, "rows", streamed)
        original_prepare = execution.llm_subset_preflight.prepare
        def prepared(project, connection, args, validator):
            start = len(reads)
            result = original_prepare(project, connection, args, validator)
            sequence = reads[start:]
            assert sequence.index(subset.MANIFEST) < sequence.index(prompts.V4_OUTPUTS[0])
            assert sequence.count(prompts.V4_OUTPUTS[0]) == 1
            assert sequence.count(audit.output_paths(5120, 1024, "v4")[0]) == 1
            assert [row[0] for row in connection.execute("SELECT id FROM prompts ORDER BY ordinal")] == selected
            from pathlib import Path
            database = Path(connection.execute("PRAGMA database_list").fetchone()[2])
            assert database.is_relative_to(native) and not database.is_relative_to(root)
            preflight_counts.append(result[0])
            return result
        monkeypatch.setattr(execution.llm_subset_preflight, "prepare", prepared)
    argv = [*base, *envelope, "--runtime-configuration", configuration]
    assert execution.main(argv) == 0 and calls == []
    failing = selected[2]
    assert execution.main([*argv, "--execute"]) == 2
    outputs = execution.output_paths("v4", configuration)
    assert outputs[0] == "results/rag/llm/v4/execution/" + configuration.removeprefix("v4_") + "/llm_responses.jsonl"
    prefix = (root/outputs[0]).read_bytes()
    assert len(prefix.decode().splitlines()) == 2
    failing = None
    assert execution.main([*argv, "--execute"]) == 0
    assert calls == selected[:3] + selected[2:]
    raw = (root/outputs[0]).read_bytes()
    assert raw.startswith(prefix)
    assert [common.parse_json(line)["query_id"] for line in raw.decode().splitlines()] == selected
    assert execution.main([*argv, "--execute"]) == 0 and len(calls) == 1201
    assert execution.main([*argv, "--export", "--execute"]) == 0 and len(calls) == 1201
    exported = root/outputs[0].replace("llm_responses.jsonl", "llm_responses_for_import.jsonl")
    values = [common.parse_json(line) for line in exported.read_text().splitlines()]
    assert [row["query_id"] for row in values] == selected
    assert all(set(row) == set(runtime.FIELDS) for row in values)
    summary = list(common.rows(root, outputs[1]))
    metrics = {row["metric"]: row["value"] for row in summary if not row["source_dataset"]}
    assert metrics["evaluation_scope"] == subset.SCOPE and metrics["evaluated_queries"] == "1200"
    assert metrics["population_queries"] == "1230"  # Synthetic fixture; production is 128724.
    assert metrics["sampling_seed"] == "42" and metrics["sampling_strategy"] == subset.STRATEGY
    if configuration == runtime.SEMANTIC_STRATIFIED_1200:
        assert metrics["semantic_validator_fallback_count"] == "1"
        assert metrics["positive_evidence_fallback_count"] == "0"
        assert metrics["total_validator_fallback_count"] == "1"
        assert metrics["truncated_completion_fallback_count"] == "1"
        assert metrics["total_validator_or_runtime_fallback_count"] == "2"
        assert metrics["model_native_count"] == "1198"
        assert values[0]["explanation"] == runtime.SEMANTIC_FALLBACK_EXPLANATION
        assert values[1]["explanation"] == runtime.TRUNCATION_EXPLANATION
    counts = [row for row in summary if row["metric"] == "dataset_prediction_status_count" and row["evidence_status"] == "INSUFFICIENT"]
    assert len(counts) == 6 and all(row["value"] == "200" for row in counts)
    assert all(path.read_bytes() == data for path, data in before.items())
    assert not list(native.glob(".rag_v3_execution_*"))
    assert not list(root.glob(".rag_v3_execution_*"))
    if configuration == runtime.SEMANTIC_STRATIFIED_1200:
        assert preflight_counts == [1200]*5
        # A missing selected prompt must fail, not silently shrink the run.
        def missing(project, relative, *args):
            for row in original_rows(project, relative, *args):
                if relative == prompts.V4_OUTPUTS[0] and row["query_id"] == selected[-1]:
                    continue
                yield row
        monkeypatch.setattr(common, "rows", missing)
        monkeypatch.setattr(execution.llm_subset_preflight, "prepare", original_prepare)
        assert execution.main(argv) == 2
        assert (root/outputs[0]).read_bytes() == raw and len(calls) == 1201


def test_subset_configuration_preserves_grounding_and_requires_envelope(phase2_case):
    from types import SimpleNamespace
    args = SimpleNamespace(version="v4", runtime_configuration=runtime.STRATIFIED_1200,
                           max_model_len=5120, max_completion_tokens=1024, endpoint=runtime.ENDPOINT, model=runtime.MODEL,
                           retry_limit=3, timeout=120)
    runtime.validate_options(args)
    args.max_completion_tokens = 512
    with pytest.raises(ValueError):
        runtime.validate_options(args)
    context = prompts.contexts.construct(phase2_case.query, [], phase2_case.chunks, phase2_case.verified, phase2_case.policy)
    prompt = prompts.prompt_record(context, "v4")
    value = dict(query_id=prompt["query_id"], reasoning_decision="ABSTAIN", evidence_status="INSUFFICIENT", supported_class=None, explanation="No evidence.", cited_chunk_ids=[])
    old, old_meta = runtime.constrain_response(common.json_text(value), prompt, runtime.FAIL_CLOSED_5120_1024)
    new, new_meta = runtime.constrain_response(common.json_text(value), prompt, runtime.STRATIFIED_1200)
    assert old == new and new_meta == dict(old_meta, runtime_configuration=runtime.STRATIFIED_1200)


def test_v4_reads_only_canonical_prompt_without_root_alias(population, monkeypatch):
    root = population
    assert all(not (root/path).exists() for path in RETIRED_PROMPTS)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    original = common.corpus.local_path
    opened = []
    def checked(project, relative, **kwargs):
        assert str(relative) not in RETIRED_PROMPTS, "Retired prompt path was required"
        opened.append(str(relative))
        return original(project, relative, **kwargs)
    monkeypatch.setattr(common.corpus, "local_path", checked)
    config = common.completed(root, prompts.V4_OUTPUTS[1], prompts.V4_OUTPUTS[2])
    assert common.parse_json(config["v2_artifacts"]) == list(prompts.OUTPUTS)
    assert common.parse_json(config["v3_artifacts"]) == list(prompts.V3_OUTPUTS)
    assert set(RETIRED_PROMPTS) <= {row["path"] for row in common.parse_json(config["input_metadata_json"])}
    with sqlite3.connect(":memory:") as connection:
        assert runtime.prepare_population(root, connection, "v4") == 1230
    assert prompts.V4_OUTPUTS[0] in opened
    assert common.QUERY_PATH in opened
    assert "metadata/rag/constrained_llm_reasoning_v2_policy.csv" in opened
    assert "metadata/rag/constrained_llm_reasoning_v3_policy.csv" in opened
    assert before == {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert execution.output_paths("v4", runtime.SEMANTIC_STRATIFIED_1200)[0] == "results/rag/llm/v4/execution/stratified_1200_semantic_fail_closed_5120_1024/llm_responses.jsonl"


def test_real_cli_initialization_does_not_touch_retired_prompts(population, monkeypatch):
    root = population
    original = common.corpus.local_path
    def checked(project, relative, **kwargs):
        assert str(relative) not in RETIRED_PROMPTS
        return original(project, relative, **kwargs)
    monkeypatch.setattr(common.corpus, "local_path", checked)
    # Build real synthetic audit/manifest prerequisites, then enter the same
    # completed -> prepare_population -> audit_gate path as the executor CLI.
    assert subset.main(["--project-root", str(root), "--execute"]) == 0
    assert audit.main(["--project-root", str(root), "--version", "v4", "--max-model-len", "5120", "--reserved-completion-tokens", "1024", "--execute"]) == 0
    reached = []
    original_completed = common.completed
    def completed(*args):
        config = original_completed(*args)
        reached.append(config["prompt_schema_version"])
        return config
    monkeypatch.setattr(common, "completed", completed)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    # The only missing prerequisite is the independent diagnostic, so no
    # generation or execution outputs are possible in this dry regression.
    assert execution.main(["--project-root", str(root), "--version", "v4", "--runtime-configuration", runtime.SEMANTIC_STRATIFIED_1200,
                           "--max-model-len", "5120", "--max-completion-tokens", "1024"]) == 2
    assert reached == ["constrained_evidence_v4"]
    assert before == {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    assert all(not (root/path).exists() for path in RETIRED_PROMPTS)


def test_missing_canonical_v4_file_physically_fails(population):
    (population/prompts.V4_OUTPUTS[0]).unlink()  # Only this synthetic fixture.
    with pytest.raises(ValueError, match="Missing local file"):
        common.completed(population, prompts.V4_OUTPUTS[1], prompts.V4_OUTPUTS[2])


@pytest.mark.parametrize("unsafe", ["missing", "symlink", "hardlink"])
def test_canonical_prompt_retains_regular_file_safety(population, monkeypatch, unsafe):
    # Model unsafe filesystem metadata; never create a link or prompt copy.
    from pathlib import Path
    import os
    canonical = population/prompts.V4_OUTPUTS[0]
    if unsafe == "missing":
        original = common.corpus.local_path
        def absent(root, relative, **kwargs):
            if str(relative) == prompts.V4_OUTPUTS[0]:
                raise ValueError("Missing local file: " + str(relative))
            return original(root, relative, **kwargs)
        monkeypatch.setattr(common.corpus, "local_path", absent)
    elif unsafe == "symlink":
        original = common.corpus.foundation.linked_path
        monkeypatch.setattr(common.corpus.foundation, "linked_path", lambda path: path == canonical or original(path))
    else:
        original = Path.stat
        def linked_stat(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == canonical:
                values = list(result)
                values[3] = 2
                return os.stat_result(values)
            return result
        monkeypatch.setattr(Path, "stat", linked_stat)
    with pytest.raises(ValueError):
        common.completed(population, prompts.V4_OUTPUTS[1], prompts.V4_OUTPUTS[2])


def test_v4_fix_preserves_older_completion_and_other_input_checks(population):
    root = population
    config = list(common.rows(root, prompts.V4_OUTPUTS[1]))
    # Older versions still require every recorded preservation input.
    common.corpus.write_table(root/prompts.V3_OUTPUTS[1], common.CONFIG_HEADER,
                              [dict(row, value="constrained_evidence_v3") if row["field"] == "prompt_schema_version" else row for row in config])
    common.corpus.write_table(root/prompts.V3_OUTPUTS[2], common.corpus.foundation.CHECKLIST_HEADER, common.checks("v3", {"complete": True}))
    with pytest.raises(ValueError, match="Missing local file"):
        common.completed(root, prompts.V3_OUTPUTS[1], prompts.V3_OUTPUTS[2])
    (root/common.QUERY_PATH).write_text("changed upstream input", encoding="utf-8")
    with pytest.raises(ValueError, match="Upstream inputs changed"):
        common.completed(root, prompts.V4_OUTPUTS[1], prompts.V4_OUTPUTS[2])


def test_native_runtime_default_and_override(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(execution.sys, "platform", "linux")
    monkeypatch.delenv("RAG_RUNTIME_TMPDIR", raising=False)
    # Inspect the Linux default without creating a real system directory.
    with monkeypatch.context() as patch:
        patch.setattr(Path, "mkdir", lambda *a, **k: None)
        assert execution.runtime_temp_root(tmp_path) == Path("/tmp/project2_rag").resolve()
    target = tmp_path.parent/(tmp_path.name + "_runtime_override")
    monkeypatch.setenv("RAG_RUNTIME_TMPDIR", str(target))
    assert execution.runtime_temp_root(tmp_path) == target.resolve()
    monkeypatch.setenv("RAG_RUNTIME_TMPDIR", str(tmp_path/"unsafe"))
    with pytest.raises(ValueError, match="outside project_root"):
        execution.runtime_temp_root(tmp_path)
    monkeypatch.setenv("RAG_RUNTIME_TMPDIR", "/mnt/c/runtime")
    with pytest.raises(ValueError, match="native Linux"):
        execution.runtime_temp_root(tmp_path)


@pytest.fixture
def hardened_ready(population, monkeypatch):
    root = population
    rows = list(common.rows(root, prompts.V4_OUTPUTS[0]))
    for row in rows:
        instructions, _, payload = row["prompt_text"].partition("INPUT_JSON:\n")
        payload = common.parse_json(payload)
        evidence = [dict(chunk_id="C1", document_id="D1", source_type="standards_rfc", chunk_text="Dataset class descriptions alone do not establish instance support.",
                         rerank_rank=1, token_count=10, provenance_status="VERIFIED")]
        payload.update(trusted_evidence=evidence, allowed_chunk_ids=["C1"])
        row.update(prompt_text=instructions+"INPUT_JSON:\n"+common.json_text(payload), allowed_chunk_ids='["C1"]', trusted_context=prompts.contexts.context_text(evidence))
    common.corpus.write_table(root/prompts.V4_OUTPUTS[0], prompts.HEADER, rows)
    base = ["--project-root", str(root)]
    assert subset.main([*base, "--execute"]) == 0
    assert audit.main([*base, "--version", "v4", "--max-model-len", "5120", "--reserved-completion-tokens", "1024", "--execute"]) == 0
    monkeypatch.setattr(runtime, "post_json", lambda endpoint, payload, timeout: mock_answer(payload))
    assert diagnostic.main([*base, "--version", "v4", "--runtime-configuration", runtime.FAIL_CLOSED_5120_1024,
                            "--max-model-len", "5120", "--max-completion-tokens", "1024", "--execute"]) == 0
    monkeypatch.setattr(runtime, "post_json", lambda *a: pytest.fail("Unexpected request"))
    monkeypatch.setattr(runtime.time, "sleep", lambda seconds: None)
    monkeypatch.setenv("RAG_RUNTIME_TMPDIR", str(root.parent/(root.name+"_native")))
    return root, [*base, "--version", "v4", "--runtime-configuration", runtime.SEMANTIC_STRATIFIED_1200,
                  "--max-model-len", "5120", "--max-completion-tokens", "1024"]


def mock_answer(payload):
    identity = payload["response_format"]["json_schema"]["schema"]["properties"]["query_id"]["enum"][0]
    value = dict(query_id=identity, reasoning_decision="ABSTAIN", evidence_status="INSUFFICIENT", supported_class=None,
                 explanation="No qualifying evidence.", cited_chunk_ids=[])
    return dict(choices=[dict(finish_reason="stop", message=dict(content=common.json_text(value)))],
                usage=dict(prompt_tokens=100, completion_tokens=1, total_tokens=101))


def test_real_cli_mixed_fallbacks_export_fusion_finalization_and_closure(hardened_ready, monkeypatch, capsys):
    from src.rag import complete_phase2_subset as completion
    root, argv = hardened_ready
    selected = list(common.rows(root, subset.MANIFEST))
    ids = [row["query_id"] for row in selected]
    assert execution.main([*argv, "--preflight"]) == 0
    assert "V4_STRATIFIED_1200_PREFLIGHT=PASS" in capsys.readouterr().out
    attempts, calls = Counter(), []
    def post(endpoint, payload, timeout):
        answer = mock_answer(payload)
        value = common.parse_json(answer["choices"][0]["message"]["content"])
        identity = value["query_id"]
        index = ids.index(identity)
        attempts[identity] += 1
        calls.append(identity)
        if index == 7 and attempts[identity] == 1 or index == 8:
            raise runtime.RequestFailure("HTTP status 503")
        if index == 1:
            value.update(reasoning_decision="SUPPORT_CLASS", evidence_status="SUPPORTED", supported_class=selected[index]["classifier_prediction"], cited_chunk_ids=["C1"])
        if index == 2: value["reasoning_decision"] = "SUPPORT_CLASS"
        if index == 4: value.pop("evidence_status")
        if index == 5: value["extra"] = "invalid"
        answer["choices"][0]["message"]["content"] = "{" if index == 3 else common.json_text(value)
        if index == 6: answer["choices"][0]["finish_reason"] = "length"
        return answer
    monkeypatch.setattr(runtime, "post_json", post)
    assert execution.main([*argv, "--execute"]) == 0
    assert attempts[ids[7]] == 2 and attempts[ids[8]] == 3 and len(calls) == 1203
    outputs = execution.output_paths("v4", runtime.SEMANTIC_STRATIFIED_1200)
    saved = [common.parse_json(line) for line in (root/outputs[0]).read_text().splitlines()]
    assert len(saved) == 1200 and saved[8]["runtime_fallback_reason"] == "TRANSPORT_RETRIES_EXHAUSTED"
    assert saved[1]["validator_fallback_reason"] == "POSITIVE_EVIDENCE" and saved[2]["validator_fallback_reason"] == "SEMANTIC_VALIDATION"
    assert all(saved[i]["runtime_fallback_reason"] == "UNPARSEABLE_MODEL_OUTPUT" for i in (3, 4, 5))
    summary = {row["metric"]: row["value"] for row in common.rows(root, outputs[1]) if not row["source_dataset"] and not row["evidence_status"]}
    assert summary["model_native_count"] == "1193"
    assert summary["unparseable_model_output_fallback_count"] == "3"
    assert summary["transport_retries_exhausted_count"] == "1"
    assert summary["truncated_completion_fallback_count"] == "1"
    assert summary["total_validator_or_runtime_fallback_count"] == "7"
    assert summary["completion_token_usage_unavailable_count"] == "1"
    assert float(summary["completion_tokens_mean"]) == 1
    assert execution.main([*argv, "--execute"]) == 0 and len(calls) == 1203
    assert execution.main([*argv, "--export", "--execute"]) == 0
    exported = (root/completion.EXPORT).read_bytes()
    assert all(set(common.parse_json(line)) == set(runtime.FIELDS) for line in exported.splitlines())
    assert execution.main([*argv, "--export", "--execute"]) == 0
    assert (root/completion.EXPORT).read_bytes() == exported
    monkeypatch.setattr(runtime, "post_json", lambda *a: pytest.fail("Completion stages must not call the model"))
    base = ["--project-root", str(root)]
    assert completion.main([*base, "--stage", "validate-export"]) == 0
    for config, checklist in completion.UPSTREAM:
        common.corpus.write_table(root/checklist, common.corpus.foundation.CHECKLIST_HEADER, common.checks("synthetic", {"complete": True}))
        if config:
            common.corpus.write_table(root/config, common.CONFIG_HEADER, common.config_rows(dict(input_metadata_json=common.json_text(common.snapshot(root, [common.QUERY_PATH])))))
    assert completion.main([*base, "--stage", "closure", "--execute"]) == 2
    assert completion.main([*base, "--stage", "fusion", "--execute"]) == 0
    fused = list(common.rows(root, completion.FUSION))
    assert len(fused) == 1200 and all(row["final_class"] == row["classifier_prediction"] and row["decision_source"] == "retain_classifier" for row in fused)
    assert all("raw_model_response" not in row for row in fused)
    assert completion.main([*base, "--stage", "finalize", "--execute"]) == 0
    assert completion.main([*base, "--stage", "closure", "--execute"]) == 0
    assert completion.main([*base, "--stage", "status"]) == 0


@pytest.mark.parametrize("failure", ["wrong_query", "circuit", "missing_prompt", "test_query"])
def test_real_cli_integrity_and_transport_circuit(hardened_ready, monkeypatch, failure):
    root, argv = hardened_ready
    selected = list(common.rows(root, subset.MANIFEST))
    calls = []
    def post(endpoint, payload, timeout):
        calls.append(payload)
        if failure == "circuit": raise runtime.RequestFailure("Temporary loopback transport failure")
        answer = mock_answer(payload)
        value = common.parse_json(answer["choices"][0]["message"]["content"])
        value["query_id"] = "WRONG"
        answer["choices"][0]["message"]["content"] = common.json_text(value)
        return answer
    monkeypatch.setattr(runtime, "post_json", post)
    if failure in {"missing_prompt", "test_query"}:
        original = common.rows
        def altered(project, path, *args):
            for row in original(project, path, *args):
                if row.get("query_id") == selected[-1]["query_id"]:
                    if failure == "missing_prompt" and path == prompts.V4_OUTPUTS[0]: continue
                    if failure == "test_query" and path == common.QUERY_PATH: row = dict(row, source_split="test")
                yield row
        monkeypatch.setattr(common, "rows", altered)
    assert execution.main([*argv, "--execute"]) == 2
    if failure == "circuit":
        assert len(calls) == 15
        raw = root/execution.output_paths("v4", runtime.SEMANTIC_STRATIFIED_1200)[0]
        assert len(raw.read_text().splitlines()) == 5
        assert execution.main([*argv, "--execute"]) == 2 and len(calls) == 15
        prefix = raw.read_bytes()
        resumed = []
        def recovered(endpoint, payload, timeout):
            resumed.append(payload)
            return mock_answer(payload)
        monkeypatch.setattr(runtime, "post_json", recovered)
        assert execution.main([*argv, "--execute", "--reset-transport-circuit"]) == 0
        assert len(resumed) == 1195 and raw.read_bytes().startswith(prefix)
        values = [common.parse_json(line) for line in raw.read_text().splitlines()]
        assert len(values) == len({row["query_id"] for row in values}) == 1200
    else:
        assert len(calls) == (1 if failure == "wrong_query" else 0)


def continuation_args(argv):
    return [*argv, "--validation-continuation", "--time-budget-minutes", "1", "--reserve-minutes", "0"]


def test_continuation_exact_ids_round_robin_and_read_only_status(hardened_ready, monkeypatch, capsys):
    root, argv = hardened_ready
    original = {p: p.read_bytes() for p in (root/"results/rag/llm/v4/diagnostics").rglob("*.jsonl")}
    assert execution.main([*continuation_args(argv), "--status-only"]) == 0
    assert "remaining_validation=1224" in capsys.readouterr().out
    assert not (root/execution.CONTINUATION).exists()
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE population (id TEXT, dataset TEXT, prediction TEXT)")
    for dataset, prediction in execution.CONTINUATION_STRATA:
        connection.executemany("INSERT INTO population VALUES (?, ?, ?)", [(f"{dataset}/{prediction}/{i}", dataset, prediction) for i in range(5)])
    first = execution.continuation_queue(connection, set())
    assert first == execution.continuation_queue(connection, set())
    assert len(first) == len(set(first)) == 30
    assert [identity.rsplit("/", 1)[0] for identity in first[:6]] == ["/".join(pair) for pair in execution.CONTINUATION_STRATA]
    excluded = {first[2], first[13], first[21]}
    remaining = execution.continuation_queue(connection, excluded)
    assert set(remaining) == set(first)-excluded
    connection.close()
    assert all(path.read_bytes() == data for path, data in original.items())


@pytest.mark.parametrize("corrected_positive", [False, True])
def test_continuation_raw_first_single_positive_correction(hardened_ready, monkeypatch, corrected_positive):
    root, argv = hardened_ready
    clock, calls, durable = [0.0], [], []
    monkeypatch.setattr(execution.time, "monotonic", lambda: clock[0])
    original_append = execution.append_response
    def append(stream, event):
        original_append(stream, event)
        durable.append((event["query_id"], event["phase"]))
    monkeypatch.setattr(execution, "append_response", append)
    original_validate = runtime.continuation_result
    def validate(prompt, event, args):
        assert (event["query_id"], event["phase"]) in durable
        return original_validate(prompt, event, args)
    monkeypatch.setattr(runtime, "continuation_result", validate)
    def post(endpoint, payload, timeout):
        calls.append(payload)
        text = payload["messages"][0]["content"]
        assert "true_class" not in text and "classifier_correct" not in text
        answer = mock_answer(payload)
        value = common.parse_json(answer["choices"][0]["message"]["content"])
        if len(calls) == 1 or corrected_positive:
            prediction = common.parse_json(text.partition("INPUT_JSON:\n")[2])["classifier_prediction"]
            value.update(reasoning_decision="SUPPORT_CLASS", evidence_status="SUPPORTED", supported_class=prediction, cited_chunk_ids=["C1"])
        answer["choices"][0]["message"]["content"] = common.json_text(value)
        if len(calls) == 2: clock[0] = 61
        return answer
    monkeypatch.setattr(runtime, "post_json", post)
    assert execution.main([*continuation_args(argv), "--execute"]) == 0
    assert len(calls) == 2 and "Exact validator error:" in calls[1]["messages"][0]["content"]
    raw = list(execution.continuation_jsonl(root, execution.CONTINUATION+"raw_responses.jsonl"))
    assert [row["phase"] for row in raw] == ["initial", "correction"]
    assert common.parse_json(raw[0]["api_response"]["choices"][0]["message"]["content"])["evidence_status"] == "SUPPORTED"
    validated = list(execution.continuation_jsonl(root, execution.CONTINUATION+"validated_responses.jsonl"))
    assert len(validated) == 1 and validated[0]["evidence_status"] == "INSUFFICIENT"
    retry = list(execution.continuation_jsonl(root, execution.CONTINUATION+"positive_retry_audit.jsonl"))[0]
    assert retry["retry_attempted"] == "YES" and retry["retry_validator_pass"] == ("NO" if corrected_positive else "YES")


def test_continuation_interruption_recovers_raw_without_duplicate_request(hardened_ready, monkeypatch):
    root, argv = hardened_ready
    clock, called = [0.0], []
    monkeypatch.setattr(execution.time, "monotonic", lambda: clock[0])
    def post(endpoint, payload, timeout):
        answer = mock_answer(payload)
        called.append(common.parse_json(answer["choices"][0]["message"]["content"])["query_id"])
        clock[0] += 61
        return answer
    monkeypatch.setattr(runtime, "post_json", post)
    original = runtime.continuation_result
    with monkeypatch.context() as patch:
        patch.setattr(runtime, "continuation_result", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
        assert execution.main([*continuation_args(argv), "--execute"]) == 130
    raw_path = root/(execution.CONTINUATION+"raw_responses.jsonl")
    first_raw = raw_path.read_bytes()
    assert len(called) == 1
    # Exercise real preparation and raw recovery in both no-execute modes.
    # In particular, this executes the positive-retry ledger path expression;
    # an undefined CONTINUATIO (or similar symbol) must fail this regression.
    inspected = []
    original_jsonl = execution.continuation_jsonl
    def inspect_jsonl(project, relative):
        inspected.append(relative)
        yield from original_jsonl(project, relative)
    with monkeypatch.context() as patch:
        patch.setattr(execution, "continuation_jsonl", inspect_jsonl)
        patch.setattr(runtime, "post_json", lambda *a, **k: pytest.fail("Recovery preflight must not request inference"))
        assert execution.main([*continuation_args(argv), "--resume", "--status-only"]) == 0
        assert execution.main([*continuation_args(argv), "--resume"]) == 0
    assert "results/rag/llm/v4/validation_continuation/positive_retry_audit.jsonl" in inspected
    assert raw_path.read_bytes() == first_raw and len(called) == 1
    assert execution.main([*continuation_args(argv), "--resume", "--execute"]) == 0
    assert len(called) == len(set(called)) == 2 and raw_path.read_bytes().startswith(first_raw)
    rows = list(execution.continuation_jsonl(root, execution.CONTINUATION+"validated_responses.jsonl"))
    assert {row["query_id"] for row in rows} == set(called)
    assert execution.main([*continuation_args(argv), "--status-only"]) == 0
    assert len(called) == 2


@pytest.mark.parametrize("omitted", ["--version", "--runtime-configuration"])
def test_continuation_requires_explicit_v4_runtime(tmp_path, monkeypatch, omitted, capsys):
    argv = ["--project-root", str(tmp_path), "--version", "v4", "--runtime-configuration", runtime.SEMANTIC_STRATIFIED_1200,
            "--max-model-len", "5120", "--max-completion-tokens", "1024", "--validation-continuation", "--status-only"]
    index = argv.index(omitted)
    del argv[index:index+2]
    monkeypatch.setattr(execution, "continuation_prepare", lambda *a: pytest.fail("Invalid CLI must fail before preparation"))
    assert execution.main(argv) == 2
    assert "requires" in capsys.readouterr().out.lower()


def test_continuation_test_split_aborts_before_request(hardened_ready, monkeypatch):
    root, argv = hardened_ready
    original = common.rows
    def altered(project, relative, *args):
        for row in original(project, relative, *args):
            if relative == common.QUERY_PATH:
                row = dict(row, source_split="test")
            yield row
    monkeypatch.setattr(common, "rows", altered)
    assert execution.main([*continuation_args(argv), "--execute"]) == 2
    assert not (root/execution.CONTINUATION).exists()


def test_continuation_deadline_prevents_transport_retry(phase2_case, monkeypatch):
    from types import SimpleNamespace
    clock, calls = [0.0], []
    monkeypatch.setattr(runtime.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(runtime.time, "sleep", lambda *a: None)
    def post(*args):
        calls.append(args)
        clock[0] = 61
        raise runtime.RequestFailure("Temporary loopback transport failure")
    monkeypatch.setattr(runtime, "post_json", post)
    prompt = dict(query_id="synthetic", prompt_text="synthetic", allowed_chunk_ids="[]", source_dataset="USTC-TFC2016")
    args = SimpleNamespace(max_model_len=5120, max_completion_tokens=1024, retry_limit=3, endpoint=runtime.ENDPOINT, model=runtime.MODEL, timeout=120)
    with pytest.raises(runtime.ContinuationDeadline):
        runtime.continuation_request(prompt, 100, args, 60, "initial", lambda row: pytest.fail("No successful raw response"), lambda row: None)
    assert len(calls) == 1 and calls[0][2] == 60


def test_continuation_prior_raw_export_union_and_shutdown_reserve(hardened_ready, monkeypatch):
    root, argv = hardened_ready
    prior_ids = [row["query_id"] for row in list(common.rows(root, subset.MANIFEST))[10:13]]
    base = execution.output_paths("v4", runtime.SEMANTIC_STRATIFIED_1200)[0]
    prior_path = root/base
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    values = [dict(query_id=identity, reasoning_decision="ABSTAIN", evidence_status="INSUFFICIENT", supported_class=None,
                   cited_chunk_ids=[], explanation="Synthetic prior validated result.") for identity in prior_ids]
    prior_path.write_text("".join(common.json_text(row)+"\n" for row in values), encoding="utf-8")
    exported = prior_path.with_name("llm_responses_for_import.jsonl")
    exported.write_bytes(prior_path.read_bytes())
    original = prior_path.read_bytes()
    calls, clock = [], [0.0]
    monkeypatch.setattr(execution.time, "monotonic", lambda: clock[0])
    def post(endpoint, payload, timeout):
        answer = mock_answer(payload)
        identity = common.parse_json(answer["choices"][0]["message"]["content"])["query_id"]
        assert identity not in prior_ids and timeout == 60
        calls.append(identity)
        clock[0] = 61
        return answer
    monkeypatch.setattr(runtime, "post_json", post)
    assert execution.main([*continuation_args(argv), "--time-budget-minutes", "2", "--reserve-minutes", "1", "--execute"]) == 0
    assert len(calls) == 1
    assert prior_path.read_bytes() == exported.read_bytes() == original
    summary = {row["field"]: row["value"] for row in common.rows(root, execution.CONTINUATION+"execution_summary.csv")}
    assert summary["execution_status"] == "TIME_BUDGET_REACHED"
    assert summary["new_validated"] == "1"
