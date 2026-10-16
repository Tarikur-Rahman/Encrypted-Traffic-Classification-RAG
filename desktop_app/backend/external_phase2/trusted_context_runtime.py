"""External adapter for the authoritative V5 evidence-diverse context kernel."""
import json
from backend.external_phase2.v5_contract import kernel_query, context_builder, externalize


def build_external_trusted_context(query, ranked, chunks, verified, policy,
        *, top_n=5, token_budget=2048, max_per_document=0, diversity_threshold=1.0):
    kernel = kernel_query(query)
    local_ranked = [dict(row, source_dataset=kernel['source_dataset']) for row in ranked]
    value = context_builder()(kernel, local_ranked, chunks, verified, policy,
        top_n, token_budget, max_per_document, source_split='external_inference',
        profile='evidence_diverse', diversity_threshold=diversity_threshold)
    value = externalize(value, query)
    evidence = json.loads(value['evidence_records_json'])
    value.update(evidence=evidence, chunk_count=len(evidence),
        distinct_document_count=len(set(json.loads(value['selected_document_ids']))),
        token_budget=token_budget, top_n=top_n, max_chunks_per_document=max_per_document,
        context_profile='evidence_diverse', diversity_threshold=diversity_threshold)
    return value
