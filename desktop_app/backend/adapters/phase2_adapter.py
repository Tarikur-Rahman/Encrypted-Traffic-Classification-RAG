"""Replay authoritative stage artifacts for one selected flow. Never runs an LLM
or claims saved results are newly generated. Missing stages remain unavailable.
"""
from backend.config import ROOT
from backend.adapters.artifacts import select, decode, rows, checklist
from backend.adapters.phase1_adapter import detail

from src.rag.final_test.contracts import BASE as TEST_BASE

TEST = TEST_BASE
PATHS = {
 'query': TEST+'stage01_structured_queries/test_query_records.csv',
 'retrieval': TEST+'stage04_retrieval/retrieval_candidates.csv',
 'provenance': TEST+'stage05_provenance/provenance_verified_candidates.csv',
 'reranking': TEST+'stage06_reranking/reranked_candidates.csv',
 'context': TEST+'stage07_trusted_contexts/trusted_context_records.csv',
 'reasoning': TEST+'stage08_llm/test_llm_responses.jsonl',
 'fusion': TEST+'stage09_fusion/test_fusion_records.csv',
 'decision': TEST+'stage10_final/test_final_classification_records.csv',
}
DEV = dict(query='results/rag/v5f_validation/stage04_retrieval/query_records.csv',
 retrieval='results/rag/v5f_validation/stage04_retrieval/retrieval_results.csv',
 provenance='results/rag/v5f_validation/stage05_provenance/provenance_chunk_records.csv',
 reranking='results/rag/v5f_validation/stage06_reranking/reranked_evidence.csv',
 context='results/rag/v5f_validation/stage07_trusted_contexts/trusted_context_records.csv',
 reasoning='results/rag/v5f_validation/stage08_diagnostic/claim_association_60_request_contract_v2/validated_responses.jsonl',
 fusion='results/rag/v5f_validation/stage09_fusion/v5f_historical4205_comparison_contract_v2/final_decisions.csv',
 decision='results/rag/v5f_validation/stage10_final/v5f_historical4205_comparison_contract_v2/final_classification_records.csv')

def execute(session, flow_id, stage):
    flow = detail(session, flow_id)
    paths = PATHS if flow['source_split'] == 'test' else DEV
    result = dict(flow_id=flow_id, mode='saved_artifact_review', sources={}, unavailable=[], query={}, retrieval=[],
                  provenance=[], reranking=[], context={}, reasoning={}, fusion={}, decision={})
    for key, relative in paths.items():
        if key == 'retrieval':
            for name, artifact in [('corpus','data/processed/rag/v5f_validation/evidence_chunks.csv'),
                                   ('index','artifacts/rag/v5f_validation/index/dense_embeddings.npy')]:
                present = (ROOT / artifact).is_file()
                stage(name, 'reused' if present else 'unavailable', 'Canonical '+name+' artifact present; no regeneration' if present else 'Canonical artifact is missing')
                result['sources'][name] = artifact
        stage(key, 'running', 'Reading authoritative '+key+' artifacts')
        if key == 'provenance' and flow['source_split'] != 'test':
            chunk_ids = {r['chunk_id'] for r in result['retrieval']}
            selected = [r for r in rows(ROOT / relative) if r['chunk_id'] in chunk_ids] if (ROOT / relative).exists() else []
        else:
            selected = select(relative, 'query_id', flow_id)
        if any(r.get('source_dataset', flow['provenance']['source_dataset']) != session['input']['dataset'] for r in selected):
            raise ValueError('Evidence dataset does not match this flow.')
        result['sources'][key] = relative
        if not selected:
            result['unavailable'].append(key)
            stage(key, 'unavailable', 'No saved output for this flow; no result was invented')
            continue
        if key == 'reasoning':
            # V5 stores the public seven-field response under validated_response;
            # never expose raw model content or hidden reasoning in the GUI.
            normalized = []
            for row in selected:
                value = row.get('validated_response') or {}
                normalized.append(dict(query_id=row.get('query_id'),
                    source_dataset=row.get('source_dataset', flow['provenance']['source_dataset']),
                    **{k: value.get(k) for k in ('supported_class','reasoning_decision',
                       'evidence_status','explanation','claim_ids','cited_chunk_ids')},
                    model_name=row.get('server_model', row.get('model_name')),
                    response_valid=row.get('validation_status') == 'PASS',
                    response_contract_version=row.get('response_contract_version','v2'),
                    latency_seconds=row.get('latency_seconds')))
            selected = normalized
        result[key] = selected if key in {'retrieval','provenance','reranking'} else selected[0]
        stage(key, 'reused', f'{len(selected)} saved record(s); not rerun')
    chunks = {r['chunk_id']: r for r in rows(ROOT / 'data/processed/rag/v5f_validation/evidence_chunks.csv')}
    for candidate in result['retrieval']:
        candidate['document'] = chunks.get(candidate['chunk_id'], {})
    result['context']['evidence'] = decode(result['context'].get('evidence_records_json'), [])
    result['decision']['supporting_evidence_ids'] = decode(result['fusion'].get('cited_chunk_ids'), [])
    session.setdefault('phase2', {})[flow_id] = result
    session['status'] = 'complete' if result['decision'].get('final_class') else 'partial'
    return result
