import pytest
from types import SimpleNamespace

from src.rag import phase2_common as common
from src.rag import build_trusted_contexts as contexts
from src.rag import run_provenance_aware_reranking as rerank


def ranked(case):
    return rerank.rerank(case.query, [r for r in case.results if r["retrieval_method"] == "hybrid_rrf"], case.verified, case.config)


def test_context_reconstruction_and_token_budget(phase2_case):
    case = phase2_case
    value = contexts.construct(case.query, ranked(case), case.chunks, case.verified, case.policy, top_n=5, token_budget=45)
    assert value == contexts.construct(case.query, ranked(case), case.chunks, case.verified, case.policy, top_n=5, token_budget=45)
    assert value["context_token_count"] <= 45
    assert value["context_token_count"] == case.policy.count(value["context_text"])
    assert value["context_text"] == contexts.context_text(common.parse_json(value["evidence_records_json"]))
    assert set(value) == set(contexts.HEADER)
    assert len(common.parse_json(value["selected_chunk_ids"])) < 5


def test_evidence_diverse_context_preserves_qualified_prediction_and_alternative(phase2_case):
    case=phase2_case
    rows=ranked(case)[:3]
    chunks={k:dict(v) for k,v in case.chunks.items()}
    verified=dict(case.verified)
    for row in rows:
        row.update(retrieval_intents_json=common.json_text(["benign"]),task_alignment_component=0.0,
                   task_alignment_weight=1.0,relationship_richness_component=0.0,relationship_weight=.5,reranking_profile="task_relevance",
                   decision_claim_match_count=0,decision_claim_ids_json="[]",decision_claim_classes_json="[]",decision_claim_features_json="[]")
        chunks[row["chunk_id"]]["chunk_text"]="Generic protocol overview."
    generic,prediction,alternative=rows[0],rows[1],rows[2]
    prediction.update(retrieval_intents_json='["benign"]',task_alignment_component=.5,relationship_richness_component=1.0)
    alternative.update(retrieval_intents_json='["malware"]',task_alignment_component=.4,relationship_richness_component=1.0)
    generic.update(retrieval_intents_json='["benign"]',final_rerank_score=1/generic["original_rank"])
    prediction.update(final_rerank_score=1/prediction["original_rank"]+.5+.5)
    alternative.update(final_rerank_score=1/alternative["original_rank"]+.4+.5)
    by_score=sorted(rows,key=lambda row:(-row["final_rerank_score"],row["original_rank"],row["chunk_id"]))
    for rank,row in enumerate(by_score,1):
        row["final_rank"]=rank
    # The generic candidate keeps the top original retrieval position; the
    # profile reserves the qualifying prediction and alternative intents.
    for row in (prediction,alternative):
        chunks[row["chunk_id"]]["chunk_text"]="Flow duration is associated with the observed behavior."
    value=contexts.construct(case.query,rows,chunks,verified,case.policy,top_n=2,token_budget=2048,
        profile="evidence_diverse",diversity_threshold=.15)
    ids=common.parse_json(value["selected_chunk_ids"])
    assert prediction["chunk_id"] in ids and alternative["chunk_id"] in ids
    assert len(ids)==2 and value["context_token_count"]<=2048
    evidence=common.parse_json(value["evidence_records_json"])
    assert all(item["provenance_status"]=="VERIFIED" and item["chunk_text"]==chunks[item["chunk_id"]]["chunk_text"] for item in evidence)
    assert not any("supporting" in item or "contradicting" in item for item in evidence)


def test_evidence_diverse_context_does_not_force_unqualified_alternative_or_use_ground_truth(phase2_case):
    case=phase2_case
    rows=ranked(case)
    legacy=contexts.construct(case.query,rows,case.chunks,case.verified,case.policy,top_n=5)
    improved=contexts.construct(dict(case.query,true_class_label="malware",true_class_id=1),
        rows,case.chunks,case.verified,case.policy,top_n=5,profile="evidence_diverse",diversity_threshold=.2)
    assert common.parse_json(improved["selected_chunk_ids"]) == common.parse_json(legacy["selected_chunk_ids"])
    assert "true_class_label" not in improved


def test_diversity_requires_explicit_threshold_and_reads_profile_header(phase2_case):
    case = phase2_case
    rows = ranked(case)
    with pytest.raises(ValueError, match="Explicit diversity threshold"):
        contexts.construct(case.query, rows, case.chunks, case.verified, case.policy, profile="evidence_diverse")
    from src.rag import run_provenance_aware_reranking as rerank
    assert contexts.reranking_header("legacy") == rerank.HEADER
    assert contexts.reranking_header("task_relevance") == rerank.TASK_HEADER
    assert contexts.paths_for()[0] == contexts.OUTPUTS
    outputs = contexts.paths_for("v5_validation")[0]
    assert outputs[0] == "results/rag/v5_validation/stage07_trusted_contexts/trusted_context_records.csv"
    assert all("/test/" not in path.lower() for path in outputs)


def _task_rows(case):
    rows = ranked(case)
    for row in rows:
        row.update(reranking_profile="task_relevance", relationship_richness_component=0.0, relationship_weight=1.0,
                   decision_claim_match_count=0, decision_claim_ids_json="[]", decision_claim_classes_json="[]", decision_claim_features_json="[]",
                   retrieval_intents_json="[]")
    return rows


def test_decision_claim_candidate_precedes_same_class_context_fallback_and_audits_survive(phase2_case):
    case = phase2_case
    rows = _task_rows(case)
    label = case.query["predicted_class_label"]
    generic, curated = rows[0], rows[1]
    generic.update(retrieval_intents_json=common.json_text([label]), task_alignment_component=1.0)
    curated.update(decision_claim_match_count=1, decision_claim_ids_json='["CLAIM-01"]',
                   decision_claim_classes_json=common.json_text([label]), decision_claim_features_json='["Packet length"]',
                   relationship_richness_component=1.0, relationship_weight=0.0)
    value = contexts.construct(case.query, rows, case.chunks, case.verified, case.policy,
        top_n=1, token_budget=2048, profile="evidence_diverse", diversity_threshold=0.2)
    evidence = common.parse_json(value["evidence_records_json"])
    assert [item["chunk_id"] for item in evidence] == [curated["chunk_id"]]
    assert evidence[0]["decision_claim_match_count"] == 1
    assert common.parse_json(evidence[0]["decision_claim_ids_json"]) == ["CLAIM-01"]
    assert common.parse_json(evidence[0]["decision_claim_classes_json"]) == [label]
    assert common.parse_json(evidence[0]["decision_claim_features_json"]) == ["Packet length"]
    assert not {"evidence_status", "supported_class", "reasoning_decision"}.intersection(evidence[0])


def test_alternative_claim_exposure_and_hypothesis_fallback_without_claim(phase2_case):
    case = phase2_case
    rows = _task_rows(case)
    labels = common.CLASSES[case.query["source_dataset"]]
    prediction = case.query["predicted_class_label"]
    alternative = next(label for label in labels if label != prediction)
    rows[0].update(retrieval_intents_json=common.json_text([prediction]), task_alignment_component=0.8)
    rows[1].update(decision_claim_match_count=1, decision_claim_ids_json='["CLAIM-ALT"]',
                   decision_claim_classes_json=common.json_text([alternative]), decision_claim_features_json='["Flow Duration"]',
                   relationship_richness_component=1.0, relationship_weight=0.0)
    rows[2].update(retrieval_intents_json=common.json_text([alternative]), task_alignment_component=0.8)
    value = contexts.construct(case.query, rows, case.chunks, case.verified, case.policy,
        top_n=2, token_budget=2048, profile="evidence_diverse", diversity_threshold=0.2)
    selected = common.parse_json(value["selected_chunk_ids"])
    assert rows[1]["chunk_id"] in selected

    no_claim_rows = _task_rows(case)
    no_claim_rows[0].update(retrieval_intents_json=common.json_text([prediction]), task_alignment_component=0.8)
    fallback = contexts.construct(case.query, no_claim_rows, case.chunks, case.verified, case.policy,
        top_n=1, token_budget=2048, profile="evidence_diverse", diversity_threshold=0.2)
    assert common.parse_json(fallback["selected_chunk_ids"]) == [no_claim_rows[0]["chunk_id"]]


def test_generic_relationship_word_does_not_receive_claim_priority(phase2_case):
    case = phase2_case
    rows = _task_rows(case)
    label = case.query["predicted_class_label"]
    generic, qualified = rows[0], rows[1]
    generic.update(retrieval_intents_json=common.json_text([label]), task_alignment_component=0.0,
                   relationship_richness_component=1.0, relationship_weight=0.0)
    qualified.update(retrieval_intents_json=common.json_text([label]), task_alignment_component=0.5)
    value = contexts.construct(case.query, rows, case.chunks, case.verified, case.policy,
        top_n=1, token_budget=2048, profile="evidence_diverse", diversity_threshold=0.2)
    assert common.parse_json(value["selected_chunk_ids"]) == [qualified["chunk_id"]]


def test_contrasted_class_does_not_receive_claim_priority(phase2_case):
    case = phase2_case
    rows = _task_rows(case)
    labels = common.CLASSES[case.query["source_dataset"]]
    predicted = case.query["predicted_class_label"]
    supported = next(label for label in labels if label != predicted)
    contrast_only = rows[0]
    fallback = rows[1]
    contrast_only.update(
        decision_claim_match_count=1,
        decision_claim_ids_json='["CLAIM-SUPPORTS-ALTERNATIVE"]',
        decision_claim_classes_json=common.json_text([supported]),
        retrieval_intents_json=common.json_text([predicted]),
        task_alignment_component=0.0,
    )
    fallback.update(
        retrieval_intents_json=common.json_text([predicted]),
        task_alignment_component=0.5,
    )

    value = contexts.construct(
        case.query, rows, case.chunks, case.verified, case.policy,
        top_n=1, token_budget=2048, profile="evidence_diverse", diversity_threshold=0.2,
    )

    assert common.parse_json(value["selected_chunk_ids"]) == [fallback["chunk_id"]]


def test_hypothesis_fallback_obeys_explicit_diversity_threshold(phase2_case):
    case = phase2_case
    rows = _task_rows(case)
    predicted = case.query["predicted_class_label"]
    below_threshold, at_threshold = rows[0], rows[1]
    below_threshold.update(
        retrieval_intents_json=common.json_text([predicted]),
        task_alignment_component=0.19,
    )
    at_threshold.update(
        retrieval_intents_json=common.json_text([predicted]),
        task_alignment_component=0.2,
    )

    value = contexts.construct(
        case.query, rows, case.chunks, case.verified, case.policy,
        top_n=1, token_budget=2048, profile="evidence_diverse", diversity_threshold=0.2,
    )

    assert common.parse_json(value["selected_chunk_ids"]) == [at_threshold["chunk_id"]]


def test_improved_context_token_budget_and_document_cap(phase2_case):
    case = phase2_case
    rows = _task_rows(case)
    too_small = contexts.construct(case.query, rows, case.chunks, case.verified, case.policy,
        top_n=5, token_budget=1, profile="evidence_diverse", diversity_threshold=0.2)
    assert too_small["context_token_count"] == 0

    shared_document = case.chunks[rows[0]["chunk_id"]]["document_id"]
    chunks = {identity: dict(row, document_id=shared_document) for identity, row in case.chunks.items()}
    verified = {identity: dict(row, document_id=shared_document) for identity, row in case.verified.items()}
    shared_rows = [dict(row, document_id=shared_document) for row in rows]
    capped = contexts.construct(case.query, shared_rows, chunks, verified, case.policy,
        top_n=5, token_budget=2048, max_per_document=1, profile="evidence_diverse", diversity_threshold=0.2)
    evidence = common.parse_json(capped["evidence_records_json"])
    assert len(evidence) <= 1
    assert capped["context_token_count"] <= 2048


def test_empty_context_is_insufficient_not_untrusted(phase2_case):
    case = phase2_case
    value = contexts.construct(case.query, ranked(case), case.chunks, case.verified, case.policy, token_budget=1)
    assert value["context_text"] == "" and value["context_token_count"] == 0
    assert value["context_status"] == "INSUFFICIENT" and value["provenance_status"] == "NO_EVIDENCE"


@pytest.mark.parametrize("mutation", ["unverified", "unknown", "ambiguous", "duplicate", "score", "partition"])
def test_context_rejects_invalid_evidence(phase2_case, mutation):
    case = phase2_case
    rows = ranked(case)
    query = dict(case.query)
    chunks = {k: dict(v) for k, v in case.chunks.items()}
    verified = {k: dict(v) for k, v in case.verified.items()}
    identity = rows[0]["chunk_id"]
    if mutation == "unverified": verified[identity]["provenance_status"] = "UNRESOLVED"
    if mutation == "unknown": del chunks[identity]
    if mutation == "ambiguous": chunks[identity]["quality_status"] = "REQUIRES_REVIEW"
    if mutation == "duplicate": rows.append(rows[0])
    if mutation == "score": rows[0]["final_rerank_score"] = 100
    if mutation == "partition": query["source_split"] = "test"
    with pytest.raises(ValueError):
        contexts.construct(query, rows, chunks, verified, case.policy)


def test_per_document_limit(phase2_case):
    case = phase2_case
    rows = ranked(case)
    chunks = {k: dict(v, document_id="SHARED") for k, v in case.chunks.items()}
    verified = {k: dict(v, document_id="SHARED") for k, v in case.verified.items()}
    rows = [dict(r, document_id="SHARED") for r in rows]
    value = contexts.construct(case.query, rows, chunks, verified, case.policy, max_per_document=1)
    assert len(common.parse_json(value["selected_chunk_ids"])) == 1


def _stream_fixture(monkeypatch, tmp_path, case, count=23):
    args = SimpleNamespace(project_root=tmp_path, development_id="v5_validation",
        execute=False, profile="evidence_diverse", diversity_threshold=1.0,
        top_n=5, token_budget=2048, max_chunks_per_document=0)
    queries = [dict(case.query, query_id=f"Q-{i:04d}", experiment_id="phase2_dev_retrieval_v3") for i in range(count)]
    candidates = [dict(row, query_id=q["query_id"]) for q in queries for row in _task_rows(case)]
    inputs = {"queries": list(reversed(queries)), "ranked": list(reversed(candidates))}
    monkeypatch.setattr(common, "QUERY_COUNTS", {case.query["source_dataset"]: count})
    monkeypatch.setattr(common, "rows", lambda root, path, required=(): iter(inputs[path]))
    monkeypatch.setattr(common, "snapshot", lambda *a: [])
    outputs = contexts.paths_for("v5_validation")[0]
    def build(root, options):
        summary, checks = [], []
        records = contexts.stream_records(root, options, "queries", "ranked", rerank.TASK_HEADER,
            case.chunks, case.verified, case.policy, [], [], summary, checks)
        return [(outputs[0], contexts.HEADER, records),
                (outputs[1], common.SUMMARY_HEADER, summary),
                (outputs[2], common.CONFIG_HEADER, []),
                (outputs[3], common.corpus.foundation.CHECKLIST_HEADER, checks)]
    monkeypatch.setattr(contexts, "build", build)
    return args, queries, inputs, build, outputs


def test_stream_is_lazy_complete_and_equivalent(phase2_case, tmp_path, monkeypatch):
    args, queries, inputs, build, _ = _stream_fixture(monkeypatch, tmp_path, phase2_case)
    original = contexts.construct
    calls = []
    def tracked(query, *a, **kw):
        calls.append(query["query_id"])
        return original(query, *a, **kw)
    monkeypatch.setattr(contexts, "construct", tracked)
    tables = build(tmp_path, args)
    assert calls == []
    stream = tables[0][2]
    first = next(stream)
    assert calls == [queries[0]["query_id"]]
    actual = [first, *stream]
    expected = [original(q, [r for r in inputs["ranked"] if r["query_id"] == q["query_id"]],
        phase2_case.chunks, phase2_case.verified, phase2_case.policy,
        profile="evidence_diverse", diversity_threshold=1.0) for q in queries]
    assert actual == expected
    assert len(calls) == len(queries)
    assert tables[1][2] == [dict(source_dataset=phase2_case.query["source_dataset"], record_count=len(queries))]
    assert tables[3][2] and all(r["status"] == "PASS" for r in tables[3][2])


def test_stream_dry_consumes_all_without_outputs(phase2_case, tmp_path, monkeypatch):
    args, queries, _, _, _ = _stream_fixture(monkeypatch, tmp_path, phase2_case)
    calls = []
    original = contexts.construct
    def tracked(query, *a, **kw):
        calls.append(query["query_id"])
        return original(query, *a, **kw)
    monkeypatch.setattr(contexts, "construct", tracked)
    assert contexts.run(args) == 0
    assert calls == [q["query_id"] for q in queries]
    assert list(tmp_path.iterdir()) == []


def test_stream_execute_deterministic_and_create_only(phase2_case, tmp_path, monkeypatch):
    import csv
    args, queries, _, _, outputs = _stream_fixture(monkeypatch, tmp_path, phase2_case)
    args.execute = True
    args.project_root = tmp_path/"first"
    args.project_root.mkdir()
    assert contexts.run(args) == 0
    first = [(args.project_root/p).read_bytes() for p in outputs]
    with (args.project_root/outputs[0]).open(encoding="utf-8", newline="") as stream:
        records = list(csv.DictReader(stream))
    assert [r["query_id"] for r in records] == [q["query_id"] for q in queries]
    assert all(set(r) == set(contexts.HEADER) for r in records)
    assert contexts.run(args) == 2
    assert first == [(args.project_root/p).read_bytes() for p in outputs]
    args.project_root = tmp_path/"second"
    args.project_root.mkdir()
    assert contexts.run(args) == 0
    assert first == [(args.project_root/p).read_bytes() for p in outputs]


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "unknown", "test", "late_failure"])
def test_stream_failure_never_publishes(phase2_case, tmp_path, monkeypatch, mutation):
    args, queries, inputs, _, _ = _stream_fixture(monkeypatch, tmp_path, phase2_case)
    args.execute = True
    if mutation == "duplicate":
        inputs["queries"].append(dict(queries[0]))
    elif mutation == "missing":
        inputs["ranked"] = [r for r in inputs["ranked"] if r["query_id"] != queries[-1]["query_id"]]
    elif mutation == "unknown":
        inputs["ranked"][0]["query_id"] = "UNKNOWN"
    elif mutation == "test":
        inputs["queries"][0] = dict(inputs["queries"][0], source_split="test")
    else:
        original = contexts.construct
        def fail_last(query, *a, **kw):
            if query["query_id"] == queries[-1]["query_id"]:
                raise ValueError("late validation failure")
            return original(query, *a, **kw)
        monkeypatch.setattr(contexts, "construct", fail_last)
    assert contexts.run(args) == 2
    assert list(tmp_path.iterdir()) == []
