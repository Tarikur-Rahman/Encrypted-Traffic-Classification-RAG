"""Accepted-only V5 contract-v2 fusion with immutable classifier retention."""
from backend.external_phase2.schemas import validate_query, same_identity, identity
from src.rag import phase2_common as common


def run_external_fusion(query, response, context, verified, registry=None):
    validate_query(query)
    same_identity(query, response, context)
    common.require(response.get('response_validation_status') == 'PASS', 'Unvalidated response')
    validated = response
    evidence = common.unique(common.parse_json(context['evidence_records_json']), 'chunk_id')
    citations = validated['cited_chunk_ids']
    for chunk_id in citations:
        item, record = evidence.get(chunk_id), verified.get(chunk_id)
        common.require(item is not None and record is not None, 'Unknown fusion citation')
        common.require(item['provenance_status'] == record['provenance_status'] == 'VERIFIED'
            and record['authorized_for_downstream'] == record['authorized_for_retrieval'] == 'YES'
            and record['quality_status'] == 'KEEP', 'Unverified/unauthorized fusion citation')
        common.require(item['document_id'] == record['document_id']
            and item['source_type'] == record['source_type'], 'Fusion citation linkage mismatch')
    prediction, status = query['predicted_class_label'], validated['evidence_status']
    reason = 'CLASSIFIER_RETAINED_' + status
    if status == 'CONTRADICTED':
        reason = 'CLASSIFIER_RETAINED_OVERRIDE_DISABLED_OR_CONFIDENCE_ABOVE_THRESHOLD'
    return dict(**identity(query), classifier_prediction=prediction,
        classifier_confidence=query['confidence'], classifier_probabilities=query['class_probabilities_json'],
        llm_supported_class=validated['supported_class'], evidence_status=status, final_class=prediction,
        fusion_reason=reason, cited_chunk_ids=common.json_text(citations),
        cited_document_ids=common.json_text(sorted({evidence[c]['document_id'] for c in citations})),
        provenance_summary=common.json_text({c:{key:evidence[c][key]
            for key in ('document_id','source_type','provenance_status')} for c in citations}),
        evidence_explanation=validated['explanation'],
        fusion_policy='retain_classifier', stage8_validation_status='PASS', response_contract_version='v2',
        reasoning_decision=validated['reasoning_decision'], claim_ids_json=common.json_text(validated['claim_ids']),
        validated_claim_linkage_json=common.json_text(validated['validated_claim_linkage']),
        model_explanation=validated['model_explanation'],
        narrative_validation_status='NOT_SEMANTICALLY_VALIDATED', decision_source='retain_classifier')
