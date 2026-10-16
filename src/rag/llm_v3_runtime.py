"""Shared V3/V4 contracts and opt-in loopback transport; no import side effects."""
import ipaddress
import json
import re
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter

from src.rag import phase2_common as common
from src.rag import audit_qwen_prompt_lengths as audit
from src.rag import materialize_constrained_llm_experiment as prompts
from src.rag import import_constrained_llm_responses as importer

ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "provenance-qwen3-4b"
FIELDS = prompts.REQUIRED_RESPONSE_FIELDS
require = common.require
FAIL_CLOSED = "v4_fail_closed_abstention"
FAIL_CLOSED_5120_1024 = "v4_fail_closed_abstention_5120_1024"
STRATIFIED_1200 = "v4_stratified_200_per_prediction_class_5120_1024"
SEMANTIC_STRATIFIED_1200 = "v4_stratified_1200_semantic_fail_closed_5120_1024"
SUBSET_CONFIGURATIONS = (STRATIFIED_1200, SEMANTIC_STRATIFIED_1200)
FAIL_CLOSED_CONFIGURATIONS = (FAIL_CLOSED, FAIL_CLOSED_5120_1024, *SUBSET_CONFIGURATIONS)
SEMANTIC_FALLBACK_EXPLANATION = "Model output failed the constrained reasoning validation contract; no validated evidence-supported decision was available."
TRUNCATION_EXPLANATION = "Generation did not complete within the bounded reasoning budget; no validated evidence-supported decision was available."
FORMAT_EXPLANATION = "Model output could not be parsed under the response contract; no validated evidence-supported decision was available."
TRANSPORT_EXPLANATION = "Transport retries were exhausted; no model response was available for a validated evidence-supported decision."
FORMAT_ERRORS = {"Missing/unknown response fields", "Invalid response field types", "Invalid supported_class type", "Invalid citation field types", "Missing raw structured response"}
SEMANTIC_REJECTIONS = frozenset({
    "Invalid evidence status", "Explanation required", "Invalid/duplicate citations",
    "Unknown/unsupplied chunk citation", "Invalid supported class/decision/citations",
    "Evidence status/classifier relationship differs", "Abstention requires null supported_class",
    "Conflicting evidence requires citations", "No evidence requires INSUFFICIENT",
})
FALLBACK_EXPLANATION = "No qualifying positive evidence was furnished under the V4 evidence gate."
# Exact intentional rejection messages from the unchanged V4 gate. Other
# ValueErrors (including malformed prompt data) are software/input failures.
GATE_REJECTIONS = frozenset({
    "V4 positive response requires nonempty EVIDENCE_QUOTE and EVIDENCE_LINK",
    "Empty evidence quote/link",
    "Evidence quote must occur verbatim in a cited supplied chunk",
    "Qualifying quote must be a complete standalone evidence line",
    "Unrecognized observed profile; positive evidence cannot qualify",
    "Quote does not establish an exact observed feature/direction to supported-class relationship",
})


def configuration_name(args):
    return getattr(args, "runtime_configuration", "legacy")


def configuration_directory(version, configuration="legacy"):
    require(configuration in ("legacy", *FAIL_CLOSED_CONFIGURATIONS), "Unknown runtime configuration")
    require(configuration not in FAIL_CLOSED_CONFIGURATIONS or version == "v4", "Fail-closed configuration requires V4")
    return configuration.removeprefix("v4_") + "/" if configuration in FAIL_CLOSED_CONFIGURATIONS else ""


def validate_structure(value, prompt):
    """Hard boundary before semantic fallback; enums/relationships checked later."""
    require(isinstance(value, dict) and set(value) == set(FIELDS), "Missing/unknown response fields")
    require(isinstance(value["query_id"], str) and value["query_id"] == prompt["query_id"], "Unknown query response")
    require(all(isinstance(value[key], str) for key in ("reasoning_decision", "evidence_status", "explanation")), "Invalid response field types")
    require(value["supported_class"] is None or isinstance(value["supported_class"], str), "Invalid supported_class type")
    require(isinstance(value["cited_chunk_ids"], list) and all(isinstance(item, str) for item in value["cited_chunk_ids"]), "Invalid citation field types")


def semantic_constrain_response(raw_content, prompt):
    raw = common.parse_json(raw_content)
    validate_structure(raw, prompt)
    try:
        importer.validate_response(raw, prompt)
    except ValueError as exc:
        if type(exc) is not ValueError or str(exc) not in SEMANTIC_REJECTIONS:
            raise
        constrained = dict(query_id=raw["query_id"], reasoning_decision="ABSTAIN", evidence_status="INSUFFICIENT",
                           supported_class=None, explanation=SEMANTIC_FALLBACK_EXPLANATION, cited_chunk_ids=[])
        importer.validate_response(constrained, prompt)
        return constrained, dict(runtime_configuration=SEMANTIC_STRATIFIED_1200, raw_model_response=raw_content,
            validator_fallback_applied="YES", validator_fallback_reason="SEMANTIC_VALIDATION",
            semantic_validation_status="REJECTED", semantic_validation_error_type=type(exc).__name__, semantic_validation_error_message=str(exc),
            grounding_gate_status="NOT_APPLICABLE", grounding_gate_error_type="", grounding_gate_error_message="")
    # Reuse the unchanged positive-evidence constraint, then identify its stage.
    constrained, metadata = constrain_response(raw_content, prompt, STRATIFIED_1200)
    rejected = metadata["validator_fallback_applied"] == "YES"
    if rejected:
        constrained = dict(constrained, explanation=SEMANTIC_FALLBACK_EXPLANATION)
    metadata.update(runtime_configuration=SEMANTIC_STRATIFIED_1200,
                    validator_fallback_reason="POSITIVE_EVIDENCE" if rejected else "",
                    semantic_validation_status="PASS", semantic_validation_error_type="", semantic_validation_error_message="")
    return constrained, metadata


def constrain_response(raw_content, prompt, configuration=FAIL_CLOSED):
    """Explicit V4 generation-plus-constraint output; preserve only visible JSON.

    Called after envelope/token validation during generation, and recomputed on
    resume/export. Never catch normal schema/citation failures as gate failures.
    """
    require(prompt["prompt_schema_version"] == "constrained_evidence_v4", "Fail-closed configuration requires V4")
    require(configuration in FAIL_CLOSED_CONFIGURATIONS, "Unknown fail-closed configuration")
    require(isinstance(raw_content, str), "Missing raw structured response")
    if configuration == SEMANTIC_STRATIFIED_1200:
        return semantic_constrain_response(raw_content, prompt)
    raw = common.parse_json(raw_content)
    importer.validate_response(raw, prompt)
    positive = raw["evidence_status"] in {"SUPPORTED", "CONTRADICTED"}
    metadata = dict(runtime_configuration=configuration, raw_model_response=raw_content,
                    grounding_gate_status="PASS" if positive else "NOT_APPLICABLE",
                    grounding_gate_error_type="", grounding_gate_error_message="", validator_fallback_applied="NO")
    if positive:
        try:
            prompts.positive_evidence_v4.validate(raw, prompt)
        except ValueError as exc:
            if type(exc) is not ValueError or str(exc) not in GATE_REJECTIONS:
                raise
            metadata.update(grounding_gate_status="REJECTED", grounding_gate_error_type=type(exc).__name__,
                            grounding_gate_error_message=str(exc), validator_fallback_applied="YES")
            constrained = dict(query_id=raw["query_id"], reasoning_decision="ABSTAIN", evidence_status="INSUFFICIENT",
                               supported_class=None, explanation=FALLBACK_EXPLANATION, cited_chunk_ids=[])
            importer.validate_response(constrained, prompt)
            return constrained, metadata
    return raw, metadata


class RequestFailure(ValueError):
    def __init__(self, message, retryable=True):
        super().__init__(message)
        self.retryable = retryable


def check_attempt_identity(content, prompt):
    if isinstance(content, dict) and "query_id" in content:
        require(content["query_id"] == prompt["query_id"], "Unknown query response")
    if not isinstance(content, str):
        return
    for marker in re.finditer(r'"query_id"\s*:\s*', content):
        try:
            identity, _ = json.JSONDecoder().raw_decode(content[marker.end():])
        except json.JSONDecodeError:
            continue
        require(identity == prompt["query_id"], "Unknown query response")


def technical_abstention(prompt, category, content, error_type, error_message, finish="stop"):
    explanation = FORMAT_EXPLANATION if category == "UNPARSEABLE_MODEL_OUTPUT" else TRANSPORT_EXPLANATION
    value = dict(query_id=prompt["query_id"], reasoning_decision="ABSTAIN", evidence_status="INSUFFICIENT",
                 supported_class=None, cited_chunk_ids=[], explanation=explanation)
    importer.validate_response(value, prompt)
    return value, dict(runtime_configuration=SEMANTIC_STRATIFIED_1200, raw_model_response=content,
        runtime_fallback_applied="YES", runtime_fallback_reason=category, original_finish_reason=finish,
        runtime_fallback_error_type=error_type, runtime_fallback_error_message=error_message,
        validator_fallback_applied="NO", validator_fallback_reason="", semantic_validation_status="NOT_APPLICABLE",
        semantic_validation_error_type="", semantic_validation_error_message="", grounding_gate_status="NOT_APPLICABLE",
        grounding_gate_error_type="", grounding_gate_error_message="")


def completed_model_response(content, prompt):
    check_attempt_identity(content, prompt)
    try:
        return constrain_response(content, prompt, SEMANTIC_STRATIFIED_1200)
    except ValueError as exc:
        if not (isinstance(exc, json.JSONDecodeError) or type(exc) is ValueError and
                (str(exc) in FORMAT_ERRORS or str(exc).startswith(("Duplicate JSON key:", "Nonfinite JSON:")))):
            raise
        return technical_abstention(prompt, "UNPARSEABLE_MODEL_OUTPUT", content, type(exc).__name__, str(exc))


def transport_abstention(prompt, errors):
    require(1 <= len(errors) <= 3, "Invalid transport attempt count")
    for i, error in enumerate(errors):
        require(set(error) == {"query_id", "attempt", "retry_count", "error_type", "error_message", "will_retry"} and
                error["query_id"] == prompt["query_id"] and error["attempt"] == i+1 and error["retry_count"] == i and
                error["error_type"] == "RequestFailure" and isinstance(error["error_message"], str) and
                error["will_retry"] == ("NO" if i == len(errors)-1 else "YES"), "Invalid transport failure audit")
    last = errors[-1]
    value, metadata = technical_abstention(prompt, "TRANSPORT_RETRIES_EXHAUSTED", None, last["error_type"], last["error_message"], None)
    metadata.update(transport_attempt_errors=errors, token_usage_status="UNAVAILABLE_TRANSPORT")
    return value, metadata


def endpoint_url(value):
    parsed = urllib.parse.urlsplit(value)
    require(parsed.scheme == "http" and not parsed.username and not parsed.password and not parsed.query and not parsed.fragment,
            "Only plain loopback HTTP endpoints without credentials/query are allowed")
    host = parsed.hostname
    require(host == "localhost" or (host is not None and ipaddress.ip_address(host).is_loopback), "External API endpoint prohibited")
    require(parsed.path == "/v1/chat/completions", "Expected chat-completions endpoint")
    return value


def response_schema(prompt):
    allowed = common.parse_json(prompt["allowed_chunk_ids"])
    citations = dict(type="array", items=dict(type="string"))
    if allowed:
        citations["items"]["enum"] = allowed
    else:
        citations["maxItems"] = 0
    return dict(type="object", additionalProperties=False, required=list(FIELDS), properties=dict(
        query_id=dict(type="string", enum=[prompt["query_id"]]),
        reasoning_decision=dict(type="string", enum=["SUPPORT_CLASS", "ABSTAIN"]),
        evidence_status=dict(type="string", enum=list(common.STATUSES)),
        supported_class=dict(type=["string", "null"], enum=[*common.CLASSES[prompt["source_dataset"]], None]),
        explanation=dict(type="string"), cited_chunk_ids=citations))


def request_payload(prompt, model, completion):
    return dict(model=model, messages=[dict(role="user", content=prompt["prompt_text"])], temperature=0, seed=42,
                max_tokens=completion, stream=False,
                response_format=dict(type="json_schema", json_schema=dict(name="grounded_response", strict=True, schema=response_schema(prompt))))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RequestFailure("HTTP redirects prohibited", retryable=False)


def post_json(endpoint, payload, timeout):
    endpoint_url(endpoint)
    request = urllib.request.Request(endpoint, data=common.json_text(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(request, timeout=timeout) as response:
            return common.parse_json(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        exc.close()
        raise RequestFailure(f"HTTP status {exc.code}", exc.code in {408, 429} or 500 <= exc.code <= 599) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise RequestFailure("Temporary loopback transport failure") from exc


def validate_logical(value, prompt):
    # Reuse semantic validation, not the importer's V2 metadata/output paths.
    importer.validate_response(value, prompt)
    require(prompt["prompt_schema_version"] in {"constrained_evidence_v3", "constrained_evidence_v4"}, "Unsupported runtime prompt schema")
    if prompt["prompt_schema_version"] == "constrained_evidence_v4":
        prompts.positive_evidence_v4.validate(value, prompt)
    return value


def truncated_response(content, finish_reason, prompt):
    require(prompt["prompt_schema_version"] == "constrained_evidence_v4", "Truncation fallback requires V4")
    require(finish_reason in ("length", "incomplete", None), "Unsupported incomplete finish_reason")
    require(content is None or isinstance(content, str), "Invalid truncated content type")
    # Incomplete JSON (including duplicate keys) cannot defeat the bounded
    # generation fallback, but an already emitted wrong identity remains hard.
    check_attempt_identity(content, prompt)
    value = dict(query_id=prompt["query_id"], reasoning_decision="ABSTAIN", evidence_status="INSUFFICIENT",
                 supported_class=None, cited_chunk_ids=[], explanation=TRUNCATION_EXPLANATION)
    importer.validate_response(value, prompt)
    metadata = dict(runtime_configuration=SEMANTIC_STRATIFIED_1200, raw_model_response=content,
                    runtime_fallback_applied="YES", runtime_fallback_reason="TRUNCATED_COMPLETION", original_finish_reason=finish_reason,
                    validator_fallback_applied="NO", validator_fallback_reason="",
                    semantic_validation_status="NOT_APPLICABLE", semantic_validation_error_type="", semantic_validation_error_message="",
                    grounding_gate_status="NOT_APPLICABLE", grounding_gate_error_type="", grounding_gate_error_message="")
    return value, metadata


def decode_response(data, prompt, expected_tokens, maximum, completion, grounding=True, semantic_fallback=False, allow_truncation=False):
    require(isinstance(data, dict) and isinstance(data.get("choices"), list) and len(data["choices"]) == 1, "Malformed API choices")
    choice = data["choices"][0]
    require(isinstance(choice, dict), "Malformed API choice")
    truncated = allow_truncation and "finish_reason" in choice and choice["finish_reason"] in ("length", "incomplete", None)
    require(choice.get("finish_reason") == "stop" or truncated, "Truncated or incomplete finish_reason")
    message = choice.get("message", {})
    require(isinstance(message, dict), "Malformed API message")
    require("content" in message and (semantic_fallback or isinstance(message["content"], str) or (truncated and message["content"] is None)), "Missing structured content")
    value = None if truncated or semantic_fallback else common.parse_json(message["content"])
    if truncated or semantic_fallback:
        pass  # Validate usage before permitting any controlled abstention.
    elif grounding:
        validate_logical(value, prompt)
    else:
        importer.validate_response(value, prompt)
    usage = data.get("usage", {})
    require(isinstance(usage, dict), "Malformed API usage")
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        require(type(usage.get(field)) is int and usage[field] >= 0, "Invalid API token usage")
    require(usage["prompt_tokens"] == expected_tokens, "API/local chat-template token counts differ")
    require(usage["completion_tokens"] <= completion and usage["total_tokens"] == expected_tokens + usage["completion_tokens"] <= maximum, "API token limits/total differ")
    return value, usage


def generate(prompt, token_count, args, on_error=lambda row: None):
    require(token_count + args.max_completion_tokens <= args.max_model_len, "Prompt exceeds runtime envelope")
    configuration = configuration_name(args)
    configuration_directory(prompt["prompt_schema_version"].rsplit("_", 1)[1], configuration)
    validate_configuration_envelope(args)
    fail_closed = configuration in FAIL_CLOSED_CONFIGURATIONS
    hardened = configuration == SEMANTIC_STRATIFIED_1200
    attempts = min(args.retry_limit+1, 3) if hardened else args.retry_limit+1
    attempt_errors = []
    started = time.perf_counter()
    for attempt in range(attempts):
        try:
            data = post_json(args.endpoint, request_payload(prompt, args.model, args.max_completion_tokens), args.timeout)
            value, usage = decode_response(data, prompt, token_count, args.max_model_len, args.max_completion_tokens,
                                           grounding=not fail_closed, semantic_fallback=configuration == SEMANTIC_STRATIFIED_1200,
                                           allow_truncation=configuration == SEMANTIC_STRATIFIED_1200)
            metadata = {}
            finish_reason = data["choices"][0]["finish_reason"]
            if configuration == SEMANTIC_STRATIFIED_1200 and finish_reason != "stop":
                value, metadata = truncated_response(data["choices"][0]["message"]["content"], finish_reason, prompt)
            elif hardened:
                value, metadata = completed_model_response(data["choices"][0]["message"]["content"], prompt)
            elif fail_closed:
                value, metadata = constrain_response(data["choices"][0]["message"]["content"], prompt, configuration)
            return dict(value, source_dataset=prompt["source_dataset"], classifier_prediction=prompt["classifier_prediction"],
                        prompt_schema_version=prompt["prompt_schema_version"], model_name=args.model, temperature=0,
                        max_completion_tokens=args.max_completion_tokens, max_model_len=args.max_model_len,
                        finish_reason=finish_reason, prompt_tokens=token_count, completion_tokens=usage["completion_tokens"], total_tokens=usage["total_tokens"],
                        latency_seconds=time.perf_counter()-started, retry_count=attempt, response_valid=True, error_type="", error_message="", **metadata)
        except (ValueError, TypeError, KeyError, UnicodeError) as exc:
            retryable = (isinstance(exc, RequestFailure) and exc.retryable) if fail_closed else (not isinstance(exc, RequestFailure) or exc.retryable)
            again = retryable and attempt+1 < attempts
            error = dict(query_id=prompt["query_id"], attempt=attempt+1, retry_count=attempt, error_type=type(exc).__name__, error_message=str(exc), will_retry="YES" if again else "NO")
            attempt_errors.append(error)
            on_error(error)
            if hardened and retryable and not again:
                value, metadata = transport_abstention(prompt, attempt_errors)
                # Zero completion tokens is an explicit unavailable-usage sentinel,
                # never a claim that the remote server generated zero tokens.
                return dict(value, **metadata, source_dataset=prompt["source_dataset"], classifier_prediction=prompt["classifier_prediction"],
                    prompt_schema_version=prompt["prompt_schema_version"], model_name=args.model, temperature=0,
                    max_completion_tokens=args.max_completion_tokens, max_model_len=args.max_model_len, finish_reason=None,
                    prompt_tokens=token_count, completion_tokens=0, total_tokens=token_count, latency_seconds=time.perf_counter()-started,
                    retry_count=attempt, response_valid=True, error_type="", error_message="")
            if hardened and again:
                time.sleep(min(0.1 * (attempt+1), 0.3))
            if not again:
                raise RequestFailure(
                    f"Response invalid or request failed after permitted attempts: {type(exc).__name__}: {exc}",
                    retryable=False,
                ) from exc
    raise AssertionError("Unreachable retry loop")


def prepare_population(root, connection, version="v3"):
    if version == "v3":
        require(not common.corpus.local_path(root, prompts.V4_OUTPUTS[3], must_exist=False).exists(),
                "V3 is SUPERSEDED_NOT_EXECUTABLE when V4 policy is present; select --version v4")
    outputs, schema, _ = prompts.version_settings(version)
    config = common.completed(root, outputs[1], outputs[2])
    require(config.get("prompt_schema_version") == schema, "Completed selected materialization required")
    require(list(common.rows(root, outputs[3], prompts.POLICY_HEADER)) == prompts.policy_rows(version), "Selected policy differs")
    audit.populate_queries(root, connection)
    connection.execute("CREATE TABLE prompts (id TEXT PRIMARY KEY, ordinal INTEGER UNIQUE, payload TEXT NOT NULL, tokens INTEGER)")
    counts, previous, total = Counter(), None, 0
    for row in common.rows(root, outputs[0], prompts.HEADER):
        require(previous is None or row["query_id"] > previous, "Duplicate/out-of-order prompt IDs")
        previous = row["query_id"]
        query = connection.execute("SELECT payload FROM queries WHERE id=?", (previous,)).fetchone()
        require(query is not None, "Unknown prompt query")
        audit.validate_prompt(row, common.parse_json(query[0]), version)
        connection.execute("INSERT INTO prompts VALUES (?, ?, ?, NULL)", (previous, total, common.json_text(row)))
        counts[row["source_dataset"]] += 1
        total += 1
    require(dict(counts) == common.QUERY_COUNTS, "Incomplete development population")
    require(connection.execute("SELECT id FROM queries EXCEPT SELECT id FROM prompts LIMIT 1").fetchone() is None, "Missing prompt queries")
    connection.commit()
    return total


def add_arguments(parser):
    parser.add_argument("--version", choices=("v3", "v4"), default="v3")
    parser.add_argument("--runtime-configuration", choices=("legacy", *FAIL_CLOSED_CONFIGURATIONS), default="legacy")
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-model-len", type=int, default=4608)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--retry-limit", type=int, default=3, help="Additional attempts; hardened stratified V4 caps transport at three total attempts")
    parser.add_argument("--timeout", type=float, default=120)
    return parser


def validate_configuration_envelope(args):
    if getattr(args, "final_frozen_test", False):
        require(not getattr(args, "validation_continuation", False), "TEST and validation modes are exclusive")
        require(getattr(args, "version", None) == "v4" and configuration_name(args) == SEMANTIC_STRATIFIED_1200,
                "Frozen TEST requires the existing V4 semantic runtime")
        require((args.max_model_len, args.max_completion_tokens) == (5120, 1024) and args.model == MODEL,
                "Frozen TEST model/token envelope differs")
        return
    if getattr(args, "validation_continuation", False):
        require(getattr(args, "version", None) == "v4" and configuration_name(args) == SEMANTIC_STRATIFIED_1200,
                "Continuation requires the existing V4 semantic fail-closed runtime")
        require((args.max_model_len, args.max_completion_tokens) in {(4608, 512), (5120, 1024)}, "Unproven continuation envelope")
        return
    if configuration_name(args) in {FAIL_CLOSED_5120_1024, *SUBSET_CONFIGURATIONS}:
        require(args.max_model_len == 5120 and args.max_completion_tokens == 1024,
                configuration_name(args) + " requires max_model_len=5120 and max_completion_tokens=1024")


def validate_options(args):
    configuration_directory(getattr(args, "version", "v3"), configuration_name(args))
    validate_configuration_envelope(args)
    endpoint_url(args.endpoint)
    audit.safe_prompt_limit(args.max_model_len, args.max_completion_tokens)
    require(args.max_completion_tokens > 0 and args.retry_limit >= 0 and common.finite(args.timeout) > 0 and args.model.strip(), "Invalid generation configuration")


def flush_record(stream, value):
    stream.write(common.json_text(value) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


class ContinuationDeadline(Exception):
    """No further requests may begin during the shutdown reserve."""


def correction_prompt(prompt, error):
    require(error in GATE_REJECTIONS, "Unknown positive validator failure")
    instructions, marker, payload = prompt["prompt_text"].partition("INPUT_JSON:\n")
    require(marker, "Missing canonical prompt payload")
    guidance = (
        "VALIDATOR-GUIDED CORRECTION. Exact validator error: " + error + "\n"
        "Regenerate the structured response only. Cite only supplied chunk IDs. "
        "Use exactly one complete standalone single-line verbatim EVIDENCE_QUOTE from a cited supplied chunk "
        "and exactly one single-line EVIDENCE_LINK. Do not invent feature-to-class relationships. "
        "If supplied evidence cannot genuinely support or contradict the prediction, return INSUFFICIENT.\n"
        'EVIDENCE_QUOTE: "<single standalone verbatim evidence line>"\n'
        "EVIDENCE_LINK: <single-line evidence-to-observed-profile connection>\n"
    )
    return dict(prompt, prompt_text=instructions + guidance + marker + payload)


def continuation_request(prompt, tokens, args, deadline, phase, persist, on_error):
    """Persist visible raw output before ANY semantic/grounding validation."""
    require(tokens + args.max_completion_tokens <= args.max_model_len, "Correction/prompt exceeds audited envelope")
    started, errors = time.perf_counter(), []
    for attempt in range(min(3, args.retry_limit+1)):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ContinuationDeadline()
        try:
            data = post_json(args.endpoint, request_payload(prompt, args.model, args.max_completion_tokens), min(args.timeout, remaining))
        except RequestFailure as exc:
            again = exc.retryable and attempt+1 < min(3, args.retry_limit+1)
            error = dict(query_id=prompt["query_id"], attempt=attempt+1, retry_count=attempt,
                         error_type=type(exc).__name__, error_message=str(exc), will_retry="YES" if again else "NO")
            on_error(error)
            errors.append(error)
            if not exc.retryable:
                raise
            if again:
                time.sleep(min(.1*(attempt+1), max(0, deadline-time.monotonic())))
                continue
            data = None
        # Only the visible message content, finish condition and usage are kept;
        # server hidden reasoning is never serialized into a scientific artifact.
        if data is not None:
            require(isinstance(data, dict) and isinstance(data.get("choices"), list) and len(data["choices"]) == 1, "Malformed API choices")
            choice = data["choices"][0]
            require(isinstance(choice, dict) and isinstance(choice.get("message"), dict), "Malformed API message")
            data = dict(choices=[dict(finish_reason=choice.get("finish_reason"), message=dict(content=choice["message"].get("content")))], usage=data.get("usage"))
        event = dict(query_id=prompt["query_id"], phase=phase, source_split="test" if getattr(args, "final_frozen_test", False) else "validation", prompt_tokens=tokens,
                     api_response=data, transport_errors=errors, latency_seconds=time.perf_counter()-started)
        persist(event)  # Caller writes, flushes and fsyncs before returning.
        return event
    raise AssertionError("Unreachable continuation transport loop")


def continuation_result(prompt, event, args):
    require(set(event) == {"query_id", "phase", "source_split", "prompt_tokens", "api_response", "transport_errors", "latency_seconds"}, "Invalid durable raw schema")
    require(type(event["prompt_tokens"]) is int and event["prompt_tokens"] > 0 and common.finite(event["latency_seconds"]) >= 0, "Invalid durable raw counts/latency")
    require(event["prompt_tokens"]+args.max_completion_tokens <= args.max_model_len, "Saved continuation token budget differs")
    split = "test" if getattr(args, "final_frozen_test", False) else "validation"
    require(event["query_id"] == prompt["query_id"] and event["source_split"] == split, "Wrong continuation identity/split")
    require(event["phase"] in {"initial", "correction"}, "Unknown continuation phase")
    data = event["api_response"]
    if data is None:
        value, metadata = transport_abstention(prompt, event["transport_errors"])
    else:
        decode_response(data, prompt, event["prompt_tokens"], args.max_model_len, args.max_completion_tokens,
                        grounding=False, semantic_fallback=True, allow_truncation=True)
        choice = data["choices"][0]
        if choice["finish_reason"] != "stop":
            value, metadata = truncated_response(choice["message"]["content"], choice["finish_reason"], prompt)
        else:
            value, metadata = completed_model_response(choice["message"]["content"], prompt)
    return dict(value, **metadata, source_split=split, source_dataset=prompt["source_dataset"],
                classifier_prediction=prompt["classifier_prediction"], raw_response_phase=event["phase"])
