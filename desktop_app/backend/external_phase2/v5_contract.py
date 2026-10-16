"""External-inference binding for the frozen V5 contract-v2 kernels.

Research modules remain unchanged. External records keep EXT identities and
never acquire validation/test membership; only the selected analysis contract
is exposed to kernels that require a canonical class vocabulary.
"""
from functools import lru_cache
from types import FunctionType, SimpleNamespace

from backend.external_phase2.schemas import validate_query, validate_identity, identity
from src.rag import phase2_common as common
from src.rag import positive_evidence_v5 as validator
from src.rag import build_semantic_association_v5 as method
from src.rag import association_response_v2 as contract


def bind(function, namespace):
    value = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    value.__kwdefaults__ = function.__kwdefaults__
    return value


def kernel_query(query):
    validate_query(query)
    return dict(query, source_dataset=query['analysis_contract'])


def validate_kernel_query(query, *, source_split='external_inference'):
    common.require(source_split == 'external_inference', 'External kernel scope required')
    common.require(query.get('source_split') == 'external_inference'
        and query.get('rag_partition') == 'external_inference'
        and query.get('evaluation_scope') == 'external_runtime', 'External kernel identity required')
    common.require(query.get('source_dataset') == query.get('analysis_contract')
        and query['source_dataset'] in common.CLASSES, 'Analysis-contract dataset required')
    common.require(isinstance(query.get('query_id'), str) and query['query_id'].startswith('EXT-')
        and '-VAL-' not in query['query_id'] and '-TEST-' not in query['query_id'], 'External query ID required')
    common.require(query.get('predicted_class_label') in common.CLASSES[query['source_dataset']], 'Invalid external hypothesis')


@lru_cache(maxsize=1)
def retrieval_engine():
    from src.rag import run_rag_retrieval_v3 as frozen
    dependencies = SimpleNamespace(**{**vars(common), 'validate_query': validate_kernel_query})
    namespace = {**vars(frozen), 'common': dependencies}
    for name in ('hypotheses', 'retrieval_intents', 'merge_intents', 'balanced_batch'):
        namespace[name] = bind(getattr(frozen, name), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in
        ('hypotheses', 'retrieval_intents', 'merge_intents', 'balanced_batch')})


@lru_cache(maxsize=1)
def context_builder():
    from src.rag import build_trusted_contexts as frozen
    dependencies = SimpleNamespace(**{**vars(common), 'validate_query': validate_kernel_query})
    return bind(frozen.construct, {**vars(frozen), 'common': dependencies})


def externalize(value, query):
    value = dict(value)
    value.update(identity(query))
    return value


def payload(query, context):
    validate_query(query)
    validate_identity(context)
    common.require(identity(query) == identity(context), 'External query/context identity differs')
    return dict(query_id=query['query_id'], source_dataset=query['analysis_contract'],
        source_split='external_inference', classifier_prediction=query['predicted_class_label'],
        traffic_profile=query['retrieval_query_text'],
        trusted_evidence=common.parse_json(context['evidence_records_json']))


def qualifying(value, registry, verified, *, method=method, source_split='external_inference'):
    """V5 qualifying logic with only the benchmark identity gate replaced."""
    common.require(set(value) == {'query_id','source_dataset','source_split','classifier_prediction','traffic_profile','trusted_evidence'},
        'Unexpected V5 runtime input; labels/confidence prohibited')
    dataset = value['source_dataset']
    common.require(source_split == 'external_inference' and value['source_split'] == source_split,
        'External V5 split required')
    common.require(dataset in common.CLASSES and value['query_id'].startswith('EXT-')
        and '-VAL-' not in value['query_id'] and '-TEST-' not in value['query_id'], 'External V5 identity mismatch')
    common.require(value['classifier_prediction'] in common.CLASSES[dataset], 'Invalid hypothesis')
    observed = validator.observations(value['traffic_profile'], dataset)
    matches, seen = {}, set()
    for item in value['trusted_evidence']:
        chunk_id = item['chunk_id']
        common.require(chunk_id not in seen, 'Duplicate supplied chunk')
        seen.add(chunk_id)
        source = verified.get(chunk_id)
        common.require(source is not None and source['provenance_status'] == 'VERIFIED'
            and source['authorized_for_downstream'] == 'YES' and source['quality_status'] == 'KEEP'
            and source['document_id'] == item['document_id'] and source['chunk_text'] == item['chunk_text'],
            'Unverified/altered supplied evidence')
        for claim_id in common.parse_json(item.get('decision_claim_ids_json', '[]')):
            row = registry.get(claim_id)
            common.require(row is not None and row['decision'] == 'APPROVED'
                and row['source_type'] == method.SOURCE_TYPE, 'Unknown/unapproved claim')
            common.require(row['source_dataset'] == dataset and item['document_id'] == claim_id,
                'Claim dataset/document mismatch')
            common.require(method.render(row) in item['chunk_text'].splitlines(),
                'Claim must be a complete verbatim supplied line')
            common.require(row['canonical_feature'] in method.corpus.CANONICAL_DECISION_FEATURES[dataset]
                and observed.get(row['canonical_feature']) == row['direction'], 'Observed claim direction differs')
            common.require(row['supported_class'] in common.CLASSES[dataset], 'Claim class differs')
            if method.VERSION == 'train-semantic-association-v2':
                common.require(row['evidence_method_version'] == method.VERSION
                    and row['semantic_state'] in method.DIRECTIONS
                    and row['direction'] == method.DIRECTIONS[row['semantic_state']]
                    and row['contrasted_class'] in common.CLASSES[dataset]
                    and row['contrasted_class'] != row['supported_class']
                    and row['reference_classes'] == [row['contrasted_class']], 'Invalid pairwise association')
                if value['classifier_prediction'] not in {row['supported_class'], row['contrasted_class']}:
                    continue
            matches.setdefault(claim_id, set()).add(chunk_id)
    return matches


def admissible(value, registry, verified):
    matches = qualifying(value, registry, verified)
    classes = sorted({registry[key]['supported_class'] for key in matches})
    prediction = value['classifier_prediction']
    predicted, alternative = prediction in classes, bool(set(classes) - {prediction})
    pattern = ('NO_QUALIFYING' if not classes else 'BOTH_DIRECTIONS' if predicted and alternative
        else 'PREDICTED_SUPPORT_ONLY' if predicted else 'ALTERNATIVE_ONLY')
    if not classes:
        status, supported, decision = 'INSUFFICIENT', None, 'ABSTAIN'
    elif len(classes) > 1:
        status, supported, decision = 'CONFLICTING', None, 'ABSTAIN'
    else:
        status, supported, decision = ('SUPPORTED' if predicted else 'CONTRADICTED'), classes[0], 'SUPPORT_CLASS'
    return dict(pattern=pattern, evidence_status=status, supported_class=supported,
        reasoning_decision=decision, classes=classes, claim_ids=sorted(matches),
        claim_chunks={key: sorted(chunks) for key, chunks in sorted(matches.items())})


@lru_cache(maxsize=1)
def response_validator():
    return bind(validator.validate, {**vars(validator), 'qualifying': qualifying})


def validate_response(raw_value, prompt, registry, verified):
    common.require(isinstance(raw_value, dict), 'Structured V5 response object required')
    legacy = contract.to_legacy(raw_value)
    response_validator()(legacy, prompt['payload'], registry, verified,
        method=method, source_split='external_inference')
    return legacy


def claim_linkage(legacy, raw_value, payload_value, registry):
    links = []
    for claim_id in raw_value.get('claim_ids', []):
        claim = registry[claim_id]
        links.append(dict(claim_id=claim_id, source_dataset=claim['source_dataset'],
            canonical_feature=claim['canonical_feature'], semantic_state=claim['semantic_state'],
            supported_class=claim['supported_class'], contrasted_class=claim['contrasted_class'],
            source_type=claim['source_type'], statistics_link=claim['statistics_link'],
            source_artifacts=claim['source_artifacts'], configuration_sha256=claim['configuration_sha256'],
            code_sha256=claim['code_sha256'], evidence_quote=method.render(claim),
            cited_chunk_ids=[item['chunk_id'] for item in payload_value['trusted_evidence']
                if item['document_id'] == claim_id and item['chunk_id'] in legacy['cited_chunk_ids']]))
    return links
