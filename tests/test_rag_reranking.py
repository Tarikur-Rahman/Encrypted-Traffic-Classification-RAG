from types import SimpleNamespace

import pytest

from src.rag import phase2_common as common
from src.rag import run_provenance_aware_reranking as rerank


def candidates(case):
    return [r for r in case.results if r["retrieval_method"] == "hybrid_rrf"]


def test_neutral_scores_reconstruct_and_determinism(phase2_case):
    case = phase2_case
    rows = rerank.rerank(case.query, candidates(case), case.verified, case.config)
    assert rows == rerank.rerank(case.query, list(reversed(candidates(case))), case.verified, case.config)
    assert [r["original_rank"] for r in rows] == list(range(1, 11))
    for row in rows:
        assert set(row) == set(rerank.HEADER)
        assert row["final_rerank_score"] == row["retrieval_weight"]*row["retrieval_rank_component"] + row["source_preference_weight"]*row["source_preference_component"] + row["task_alignment_weight"]*row["task_alignment_component"]
        assert row["source_preference_component"] == row["task_alignment_component"] == 0


def test_explicit_task_component_changes_order(phase2_case):
    case = phase2_case
    values = {(case.query["query_id"], r["chunk_id"]): 0 for r in candidates(case)}
    values[(case.query["query_id"], candidates(case)[-1]["chunk_id"])] = 1
    config = dict(case.config, task_alignment_weight=2)
    rows = rerank.rerank(case.query, candidates(case), case.verified, config, values)
    assert rows[0]["original_rank"] == 10
    with pytest.raises(ValueError, match="Missing explicit"):
        rerank.rerank(case.query, candidates(case), case.verified, config)


def test_task_relevance_can_promote_observed_feature_evidence_without_class_boost(phase2_case):
    case = phase2_case
    query = dict(case.query,retrieval_query_text="Observed traffic profile relative to the training distribution: packet length and flow duration are elevated. Retrieve the predicted class.")
    rows = candidates(case)
    chunks = {k:dict(v) for k,v in case.chunks.items()}
    for row in rows:
        chunks[row["chunk_id"]]["chunk_text"] = "General protocol information without a measured feature relationship."
    target = rows[-1]["chunk_id"]
    chunks[target]["chunk_text"] = "Packet length and flow duration are associated with the observed traffic behavior."
    intents = common.CLASSES[query["source_dataset"]]
    candidates_with_intents = [dict(row,retrieval_intents_json=common.json_text(intents)) for row in rows]
    config = dict(case.config,profile="task_relevance",task_alignment_weight=2,relationship_weight=.5)
    ranked = rerank.rerank(query,candidates_with_intents,case.verified,config,evidence=chunks,decision_claims=[])
    assert ranked[0]["chunk_id"] == target
    assert ranked[0]["original_score"] == float(rows[-1]["retrieval_score"])
    assert ranked[0]["reranking_profile"] == "task_relevance"
    assert common.parse_json(ranked[0]["retrieval_intents_json"]) == list(intents)
    assert rerank.relevance_components(query,"Malware and benign VPN classification.")[1] == 0.0
    assert ranked == rerank.rerank(query,list(reversed(candidates_with_intents)),case.verified,config,evidence=chunks,decision_claims=[])


def _claim_for(query, chunk, **changes):
    classes = common.CLASSES[query["source_dataset"]]
    claim = dict(claim_id="CLAIM-001", document_id=chunk["document_id"], source_dataset=query["source_dataset"],
        source_location="section:2", evidence_quote="Reviewed source sentence.", canonical_feature="Packet length",
        direction="higher", supported_class=classes[0], contrasted_class=classes[-1], review_status="APPROVED",
        authorized_for_decision_evidence="YES")
    claim.update(changes)
    return claim


def _task_rerank(case, claim_changes=None):
    query = dict(case.query, retrieval_query_text="Observed traffic profile relative to the training distribution: Packet length is high; Flow Duration is typical. Retrieve the predicted class.")
    rows = candidates(case)
    target_id = rows[-1]["chunk_id"]
    evidence = {key: dict(value) for key, value in case.chunks.items()}
    evidence[target_id]["chunk_text"] = "Reviewed source sentence. Packet length is associated with the measured observation."
    intents = common.CLASSES[query["source_dataset"]]
    rows = [dict(row, retrieval_intents_json=common.json_text(list(intents))) for row in rows]
    config = dict(case.config, profile="task_relevance", task_alignment_weight=1.0, relationship_weight=1.0)
    claims = [] if claim_changes is None else [_claim_for(query, evidence[target_id], **claim_changes)]
    result = rerank.rerank(query, rows, case.verified, config, evidence=evidence, decision_claims=claims)
    target = next(row for row in result if row["chunk_id"] == target_id)
    return query, target, result, evidence


def test_generic_relationship_words_alone_do_not_create_improved_signal(phase2_case):
    query, target, _, evidence = _task_rerank(phase2_case)
    generic = "Packet length relates to the classification and correlates with behavior."
    assert rerank.relevance_components(query, generic)[1] == 0.0
    assert target["relationship_richness_component"] == 0.0
    assert target["decision_claim_match_count"] == 0


def test_exact_approved_claim_match_sets_signal_and_sorted_audit(phase2_case):
    case = phase2_case
    query = dict(case.query, retrieval_query_text="Observed traffic profile relative to the training distribution: Packet length is high; Flow Duration is typical. Retrieve the predicted class.")
    target_chunk = dict(case.chunks[candidates(case)[-1]["chunk_id"]])
    target_chunk["chunk_text"] = "Reviewed source sentence. Packet length is associated with the measured observation."
    claims = [_claim_for(query, target_chunk, claim_id="CLAIM-B"), _claim_for(query, target_chunk, claim_id="CLAIM-A")]
    claims.append(dict(claims[0]))
    rows = candidates(case)
    rows = [dict(row, retrieval_intents_json=common.json_text(list(common.CLASSES[query["source_dataset"]]))) for row in rows]
    evidence = {key: dict(value) for key, value in case.chunks.items()}
    evidence[target_chunk["chunk_id"]] = target_chunk
    config = dict(case.config, profile="task_relevance", task_alignment_weight=1.0, relationship_weight=1.0)
    ranked = rerank.rerank(query, rows, case.verified, config, evidence=evidence, decision_claims=claims)
    target = next(row for row in ranked if row["chunk_id"] == target_chunk["chunk_id"])
    assert target["relationship_richness_component"] == 1.0
    assert target["decision_claim_match_count"] == 2
    assert common.parse_json(target["decision_claim_ids_json"]) == ["CLAIM-A", "CLAIM-B"]
    assert common.parse_json(target["decision_claim_features_json"]) == ["Packet length"]
    assert common.parse_json(target["decision_claim_classes_json"]) == sorted({claim["supported_class"] for claim in claims})


def test_claim_class_audit_contains_only_supported_class(phase2_case):
    query, _, _, _ = _task_rerank(phase2_case)
    if query["source_dataset"] == "USTC-TFC2016":
        supported, contrasted = "malware", "benign"
    else:
        supported, contrasted = "Tor", "Non-Tor"
    _, target, _, _ = _task_rerank(
        phase2_case,
        {"supported_class": supported, "contrasted_class": contrasted},
    )

    assert target["decision_claim_match_count"] == 1
    classes = common.parse_json(target["decision_claim_classes_json"])
    assert classes == [supported]
    assert contrasted not in classes


@pytest.mark.parametrize("change", [
    {"direction": "lower"}, {"canonical_feature": "Packet length (bytes)"},
    {"document_id": "OTHER-DOCUMENT"}, {"evidence_quote": "Absent quote."},
])
def test_nonmatching_claims_do_not_raise_relationship_signal(phase2_case, change):
    _, target, _, _ = _task_rerank(phase2_case, change)
    assert target["relationship_richness_component"] == 0.0
    assert target["decision_claim_match_count"] == 0


def test_wrong_dataset_claim_does_not_match(phase2_case):
    query, target, _, evidence = _task_rerank(phase2_case)
    other_dataset = "USTC-TFC2016" if query["source_dataset"] == "CIC-Darknet2020" else "CIC-Darknet2020"
    claim = _claim_for(query, evidence[target["chunk_id"]], source_dataset=other_dataset)
    assert rerank.matching_decision_claims(query, evidence[target["chunk_id"]], [claim]) == []


def test_claim_matching_ignores_confidence_and_evaluation_fields(phase2_case):
    case = phase2_case
    query, target, _, evidence = _task_rerank(case, {})
    claim = _claim_for(query, evidence[target["chunk_id"]])
    edited = dict(query, confidence=0.0, class_probabilities_json='{"changed": 1}',
                  true_class_label="NOT_USED", true_class_id="NOT_USED", classifier_correct="NOT_USED")
    assert rerank.matching_decision_claims(query, evidence[target["chunk_id"]], [claim]) == rerank.matching_decision_claims(edited, evidence[target["chunk_id"]], [claim])


def test_task_relevance_requires_claim_artifact_and_legacy_header_is_unchanged(phase2_case):
    case = phase2_case
    query = dict(case.query)
    config = dict(case.config, profile="task_relevance", task_alignment_weight=1.0, relationship_weight=1.0)
    with pytest.raises(ValueError, match="decision-claim artifact"):
        rerank.rerank(query, candidates(case), case.verified, config, evidence=case.chunks)
    legacy_rows = rerank.rerank(case.query, candidates(case), case.verified, case.config)
    assert all(set(row) == set(rerank.HEADER) for row in legacy_rows)


def test_task_relevance_requires_explicit_weights_and_uses_namespaced_paths():
    args = SimpleNamespace(source_preferences_json="{}", retrieval_weight=1, source_preference_weight=0,
        task_alignment_weight=None, relationship_weight=None, top_k=10, method="hybrid_rrf",
        task_alignments=None, profile="task_relevance")
    with pytest.raises(ValueError, match="Explicit"):
        rerank.options(args)
    args.task_alignment_weight, args.relationship_weight = 1.25, 0.75
    config = rerank.options(args)
    assert (config["task_alignment_weight"], config["relationship_weight"]) == (1.25, 0.75)
    assert rerank.paths_for()[0] == rerank.OUTPUTS
    outputs = rerank.paths_for("v5_validation")[0]
    assert outputs[0] == "results/rag/v5_validation/stage06_reranking/reranked_evidence.csv"
    assert all("/test/" not in path.lower() for path in outputs)


@pytest.mark.parametrize("mutation", ["method", "query", "chunk", "duplicate", "unauthorized"])
def test_invalid_candidates_rejected(phase2_case, mutation):
    case = phase2_case
    values = [dict(r) for r in candidates(case)]
    verified = {k: dict(v) for k, v in case.verified.items()}
    if mutation == "method": values[0]["retrieval_method"] = "tfidf"
    if mutation == "query": values[0]["query_id"] = "UNKNOWN"
    if mutation == "chunk": values[0]["chunk_id"] = "UNKNOWN"
    if mutation == "duplicate": values.append(values[0])
    if mutation == "unauthorized": verified[values[0]["chunk_id"]]["authorized_for_downstream"] = "NO"
    with pytest.raises(ValueError):
        rerank.rerank(case.query, values, verified, case.config)


def test_nonzero_preference_requires_explicit_configuration():
    args = SimpleNamespace(source_preferences_json="{}", retrieval_weight=1, source_preference_weight=1, task_alignment_weight=0, top_k=10, method="hybrid_rrf", task_alignments=None)
    with pytest.raises(ValueError, match="Explicit"):
        rerank.options(args)
