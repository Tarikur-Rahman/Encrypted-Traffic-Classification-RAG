"""One external prompt under the authoritative V5 response contract v2."""
from backend.external_phase2.schemas import identity
from backend.external_phase2 import v5_contract
from src.rag import association_response_v2 as contract


def build_external_prompt(query, context, registry, verified):
    payload = v5_contract.payload(query, context)
    state = v5_contract.admissible(payload, registry, verified)
    return dict(**identity(query), classifier_prediction=query['predicted_class_label'],
        prompt_text=contract.prompt(payload, registry, verified, state),
        response_schema=contract.schema(state, query['query_id']),
        prompt_schema_version=contract.VERSION, response_contract_version='v2',
        payload=payload, admissible_state=state)
