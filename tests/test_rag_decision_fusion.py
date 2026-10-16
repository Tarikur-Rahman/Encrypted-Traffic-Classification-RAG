import pytest

from src.rag import phase2_common as common
from src.rag import run_evidence_supported_fusion as fusion
from src.rag import build_trusted_contexts as contexts
from src.rag import run_provenance_aware_reranking as reranking
from src.rag import materialize_constrained_llm_experiment as prompts
from src.rag import import_constrained_llm_responses as responses


def inputs(case, status="CONTRADICTED"):
    ranked = reranking.rerank(case.query, [r for r in case.results if r["retrieval_method"] == "hybrid_rrf"], case.verified, case.config)
    context = contexts.construct(case.query, ranked, case.chunks, case.verified, case.policy)
    raw = dict(query_id=case.query["query_id"], reasoning_decision="SUPPORT_CLASS" if status in {"SUPPORTED", "CONTRADICTED"} else "ABSTAIN",
               evidence_status=status, supported_class="malware" if status == "CONTRADICTED" else "benign" if status == "SUPPORTED" else None,
               explanation="Concise supplied-evidence justification.", cited_chunk_ids=common.parse_json(context["selected_chunk_ids"])[:1])
    return responses.validate_response(raw, prompts.prompt_record(context)), context


def test_fusion_default_and_explicit_override(phase2_case):
    case = phase2_case
    response, context = inputs(case)
    default = fusion.fuse(case.query, response, context)
    assert default["final_class"] == "benign"
    assert default == fusion.fuse(case.query, response, context)
    override = fusion.fuse(case.query, response, context, "evidence_override", 0.6)
    assert override["final_class"] == "malware"
    assert "EXPLICIT_EVIDENCE_OVERRIDE" in override["fusion_reason"]
    assert set(override) == set(fusion.HEADER)
    assert common.parse_json(override["cited_document_ids"])
    assert common.parse_json(override["provenance_summary"])


@pytest.mark.parametrize("status", common.STATUSES)
def test_ground_truth_cannot_change_fusion(phase2_case, status):
    case = phase2_case
    response, context = inputs(case, status)
    first = fusion.fuse(dict(case.query, true_class_label="benign", true_class_id=0, classifier_correct=True), response, context)
    second = fusion.fuse(dict(case.query, true_class_label="malware", true_class_id=1, classifier_correct=False), response, context)
    assert first == second


@pytest.mark.parametrize("status", ["SUPPORTED", "INSUFFICIENT", "CONFLICTING"])
def test_noncontradicted_retains_classifier(phase2_case, status):
    response, context = inputs(phase2_case, status)
    assert fusion.fuse(phase2_case.query, response, context, "evidence_override", 1)["final_class"] == "benign"


def test_fusion_rejects_unknown_query_and_citation(phase2_case):
    response, context = inputs(phase2_case)
    with pytest.raises(ValueError):
        fusion.fuse(phase2_case.query, dict(response, query_id="UNKNOWN"), context)
    with pytest.raises(ValueError):
        fusion.fuse(phase2_case.query, dict(response, cited_chunk_ids='["UNKNOWN"]'), context)
