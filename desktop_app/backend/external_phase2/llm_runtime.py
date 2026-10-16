"""Single-request loopback runtime for the authoritative V5 contract v2."""
from functools import lru_cache
from types import SimpleNamespace
import json
import time
import urllib.error
import urllib.request

from backend.config import ROOT
from backend.external_phase2.schemas import identity, validate_identity
from backend.external_phase2 import v5_contract
from src.rag import phase2_common as common
from src.rag import association_response_v2 as contract
from src.rag import positive_evidence_v5 as validator
from src.rag import run_v5_diagnostic as runtime


def runtime_args():
    return SimpleNamespace(endpoint=runtime.ENDPOINT, model=runtime.MODEL, timeout=180,
        retry_limit=0, max_model_len=5120, max_completion_tokens=1024,
        runtime_configuration='v5_contract_v2_no_automatic_retries')


def check_llm_runtime():
    args = runtime_args()
    base = args.endpoint.removesuffix('/v1/chat/completions')
    result = dict(status='OFFLINE', endpoint=args.endpoint, model=args.model,
        max_model_len=args.max_model_len, max_completion_tokens=args.max_completion_tokens,
        retry_limit=0, timeout=args.timeout, temperature=0, seed=42,
        health_verified=False,
        models_verified=False,
        readiness_verified=False,
        # Kept for response compatibility; readiness deliberately does not
        # issue a chat-completion request.
        chat_endpoint_verified=False,
        message=f'Local constrained-LLM runtime is offline. Expected: {base}. Model: {args.model}.')
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            raise ValueError('Redirect prohibited')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(base + '/health', timeout=3) as response:
            common.require(response.status == 200, f'Health endpoint returned HTTP {response.status}')
        result['health_verified'] = True
        with opener.open(base + '/v1/models', timeout=3) as response:
            common.require(response.status == 200, f'Model-list endpoint returned HTTP {response.status}')
            models = common.parse_json(response.read(1024 * 1024).decode('utf-8'))
        result['models_verified'] = True
        common.require(isinstance(models, dict) and isinstance(models.get('data'), list), 'Invalid model-list response')
        model = next((row for row in models['data'] if isinstance(row, dict) and row.get('id') == args.model), None)
        common.require(model is not None, 'Expected local model is not available')
        if model.get('max_model_len') is not None:
            common.require(int(model['max_model_len']) >= args.max_model_len,
                'Local server context capacity is below the frozen runtime envelope')
        result.update(status='READY', readiness_verified=True,
            message='Expected local V5 health endpoint and model-list endpoint are available.')
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        result['error_type'] = type(exc).__name__
    except (ValueError, KeyError, TypeError) as exc:
        result.update(status='FAILED', message=str(exc))
    return result


@lru_cache(maxsize=1)
def tokenizer():
    from src.rag.audit_qwen_prompt_lengths import load_tokenizer
    return load_tokenizer(ROOT)


def token_count(prompt):
    from src.rag.audit_qwen_prompt_lengths import prompt_tokens
    count = prompt_tokens(tokenizer(), prompt['prompt_text'])
    args = runtime_args()
    common.require(count + args.max_completion_tokens <= args.max_model_len,
        'Prompt exceeds the V5 runtime envelope; no truncation is permitted')
    return count


def _raw_fields(response):
    return {key: response[key] for key in contract.FIELDS}


def validate_external_llm_response(prompt, response, registry=None, verified=None):
    validate_identity(prompt)
    common.require(all(response.get(key) == value for key, value in identity(prompt).items()),
        'Response identity/contract mismatch')
    if registry is None or verified is None:
        from backend.adapters.fresh_phase2 import resources
        _, _, verified, _, registry, _ = resources()
    legacy = v5_contract.validate_response(_raw_fields(response), prompt, registry, verified)
    linkage = v5_contract.claim_linkage(legacy, _raw_fields(response), prompt['payload'], registry)
    return dict(legacy, **identity(prompt), response_validation_status='PASS', response_contract_version='v2',
        claim_ids=response['claim_ids'], model_explanation=response['explanation'],
        validated_claim_linkage=linkage, narrative_validation_status='NOT_SEMANTICALLY_VALIDATED')


def validate_validation_llm_response(prompt, response, registry, verified):
    raw = _raw_fields(response)
    legacy = contract.to_legacy(raw)
    validator.validate(legacy, prompt['payload'], registry, verified,
        method=validator.empirical, source_split='validation')
    linkage = v5_contract.claim_linkage(legacy, raw, prompt['payload'], registry)
    return dict(legacy, response_validation_status='PASS', response_contract_version='v2',
        claim_ids=response['claim_ids'], model_explanation=response['explanation'],
        validated_claim_linkage=linkage, narrative_validation_status='NOT_SEMANTICALLY_VALIDATED')


def _run(prompt, count, registry, verified, *, external):
    args = runtime_args()
    body = dict(model=args.model, messages=[dict(role='user', content=prompt['prompt_text'])],
        temperature=0, seed=42, max_tokens=args.max_completion_tokens, stream=False,
        response_format=dict(type='json_schema', json_schema=dict(name=contract.VERSION,
            strict=True, schema=prompt['response_schema'])))
    started = time.monotonic()
    packet = common.parse_json(runtime.post(body).decode('utf-8'))
    elapsed = time.monotonic() - started
    common.require(packet.get('model') == args.model, 'Server model identity differs')
    usage = packet['usage']
    common.require(type(usage.get('prompt_tokens')) is int and usage['prompt_tokens'] == count,
        'API/local prompt-token mismatch')
    common.require(type(usage.get('completion_tokens')) is int
        and 0 <= usage['completion_tokens'] <= args.max_completion_tokens
        and usage.get('total_tokens') == count + usage['completion_tokens'] <= args.max_model_len,
        'Token-budget/usage mismatch')
    common.require(len(packet.get('choices', [])) == 1, 'Expected one response choice')
    choice = packet['choices'][0]
    common.require(choice.get('finish_reason') == 'stop', 'Incomplete V5 generation')
    value = common.parse_json(choice['message']['content'])
    response_identity = identity(prompt) if external else {
        'query_id':prompt['query_id'], 'source_dataset':prompt['source_dataset']}
    output = dict(value, **response_identity, model_name=args.model,
        response_valid=True, prompt_schema_version=contract.VERSION,
        prompt_tokens=count, completion_tokens=usage['completion_tokens'],
        total_tokens=usage['total_tokens'], latency_seconds=elapsed, retry_count=0,
        max_completion_tokens=args.max_completion_tokens, max_model_len=args.max_model_len,
        finish_reason='stop', temperature=0, response_contract_version='v2')
    validated = (validate_external_llm_response(prompt, output, registry, verified) if external
        else validate_validation_llm_response(prompt, output, registry, verified))
    output['validated_response'] = validated
    return output


def run_external_constrained_llm(prompt, count, registry=None, verified=None, on_error=lambda row: None):
    if registry is None or verified is None:
        from backend.adapters.fresh_phase2 import resources
        _, _, verified, _, registry, _ = resources()
    try:
        return _run(prompt, count, registry, verified, external=True)
    except Exception as exc:
        on_error(dict(query_id=prompt['query_id'], error_type=type(exc).__name__,
            error_message=str(exc), retry_count=0, will_retry='NO'))
        raise


def run_validation_constrained_llm(prompt, count, registry, verified):
    return _run(prompt, count, registry, verified, external=False)
