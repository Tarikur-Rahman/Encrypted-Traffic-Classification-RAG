"""Capture one rejected USTC and one accepted CIC response without running fusion."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import APP, ROOT, RUNTIME
from backend.adapters import fresh_phase2, input_adapter, phase1_adapter
from backend.adapters.artifacts import rows
from backend.external_phase2 import llm_runtime
from backend.external_phase2.prompt_runtime import build_external_prompt
from backend.external_phase2.safety import assert_desktop_write_path
from backend.session import store
from backend.utils.write_guard import install
from src.rag import llm_v3_runtime as frozen_runtime
from src.rag import phase2_common as common

DIAGNOSTICS = RUNTIME / "diagnostics"


def write_json(path, value):
    target = assert_desktop_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_text(path, value):
    target = assert_desktop_write_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(value, encoding="utf-8")


def type_map(value):
    return {key: type(item).__name__ for key, item in value.items()}


def exact_validation(engine, raw, prompt):
    checks = []
    for name, function in (
        ("validate_structure", engine.validate_structure),
        ("validate_response", engine.validate_response),
        ("validate_logical", engine.validate_logical),
    ):
        try:
            function(raw, prompt)
            checks.append({"check": name, "status": "PASS"})
        except Exception as exc:
            checks.append({"check": name, "status": "FAIL", "exception_type": type(exc).__name__, "message": str(exc)})
            return checks, exc
    return checks, None


def prepare_case(label, path, contract, input_stage):
    source = input_adapter.validate(str(path), contract, input_stage)
    if not source["compatible"]:
        raise ValueError(source["reason"])
    session = store.create(source)
    phase1_adapter.execute(session, lambda *args: None)
    flow_id = session["flows"][0]["id"]
    query = fresh_phase2.query_record(session, flow_id)
    result = dict(flow_id=flow_id, mode="fresh_execution", sources={}, unavailable=[], blockers={},
                  query=query, retrieval=[], provenance=[], reranking=[], context={}, reasoning={}, fusion={}, decision={})
    fresh_phase2.deterministic(query, lambda *args: None, result)
    prompt = build_external_prompt(query, result["context"])
    count = llm_runtime.token_count(prompt)
    engine = llm_runtime.external_engine(contract)
    generated = engine.generate(prompt, count, llm_runtime.runtime_args())
    raw_text = generated.get("raw_model_response")
    if not isinstance(raw_text, str):
        raise ValueError("Frozen runtime did not retain the raw model response")
    raw = common.parse_json(raw_text)
    checks, failure = exact_validation(engine, raw, prompt)
    directory = DIAGNOSTICS / ("ustc_llm_validation" if contract == "USTC-TFC2016" else "cic_llm_validation")
    write_json(directory / "prompt.json", prompt)
    write_json(directory / "trusted_context.json", result["context"])
    write_json(directory / "allowed_chunk_ids.json", common.parse_json(prompt["allowed_chunk_ids"]))
    # Preserve the exact model content. It is already a JSON document; do not parse/reformat it for this artifact.
    write_text(directory / "raw_llm_response.json", raw_text)
    diagnostic = {
        "exception_type": type(failure).__name__ if failure else None,
        "message": str(failure) if failure else "PASS",
        "query_id": prompt.get("query_id"),
        "source_dataset": prompt.get("source_dataset"),
        "analysis_contract": prompt.get("analysis_contract"),
        "classifier_prediction": prompt.get("classifier_prediction"),
        "reasoning_decision": raw.get("reasoning_decision"),
        "evidence_status": raw.get("evidence_status"),
        "supported_class": raw.get("supported_class"),
        "cited_chunk_ids": raw.get("cited_chunk_ids"),
        "allowed_chunk_ids": common.parse_json(prompt["allowed_chunk_ids"]),
        "checks": checks,
        "frozen_runtime_metadata": {key: generated.get(key) for key in (
            "validator_fallback_applied", "validator_fallback_reason", "semantic_validation_status",
            "semantic_validation_error_type", "semantic_validation_error_message", "grounding_gate_status",
            "grounding_gate_error_type", "grounding_gate_error_message", "runtime_fallback_applied",
            "runtime_fallback_reason", "finish_reason", "prompt_tokens", "completion_tokens", "total_tokens")},
    }
    write_text(directory / "validator_error.txt", json.dumps(diagnostic, indent=2, ensure_ascii=False))
    write_json(directory / "runtime_config.json", {
        **llm_runtime.check_llm_runtime(),
        "frozen_classes": list(common.CLASSES[contract]),
        "response_schema": engine.response_schema(prompt),
        "input": label,
        "input_stage": input_stage,
        "flow_id": flow_id,
    })
    return {"label": label, "prompt": prompt, "raw": raw, "diagnostic": diagnostic,
            "schema": engine.response_schema(prompt), "validation_passed": failure is None}


def frozen_ustc_semantics():
    prompt = next(row for row in rows(ROOT / "results/rag/test/final_4300_v5_contract_v2/stage08_llm/test_llm_prompts.csv")
                  if row["source_dataset"] == "USTC-TFC2016")
    schema = frozen_runtime.response_schema(prompt)
    return prompt, schema


def comparison(ustc, cic):
    frozen_prompt, frozen_schema = frozen_ustc_semantics()
    fields = ("query_id", "source_dataset", "classifier_prediction", "classifier_confidence", "prompt_schema_version")
    response_fields = ("reasoning_decision", "evidence_status", "supported_class", "cited_chunk_ids")
    records = {}
    for case in (ustc, cic):
        raw = case["raw"]
        prompt = case["prompt"]
        records[case["label"]] = {
            "prompt": {key: {"value": prompt.get(key), "python_type": type(prompt.get(key)).__name__} for key in fields},
            "response": {key: {"value": raw.get(key), "python_type": type(raw.get(key)).__name__,
                               "is_null": raw.get(key) is None, "is_empty_string": raw.get(key) == ""}
                         for key in response_fields},
            "allowed_chunk_ids": common.parse_json(prompt["allowed_chunk_ids"]),
            "allowed_chunk_ids_python_type": type(common.parse_json(prompt["allowed_chunk_ids"])).__name__,
            "raw_value_types": type_map(raw),
            "validation_passed": case["validation_passed"],
            "failed_predicate": case["diagnostic"]["message"],
            "class_vocabulary_check": {
                "frozen_vocabulary": list(common.CLASSES[case["prompt"]["analysis_contract"]]),
                "classifier_exact_match": case["prompt"]["classifier_prediction"] in common.CLASSES[case["prompt"]["analysis_contract"]],
                "supported_class_exact_match": (raw.get("supported_class") is None or
                    raw.get("supported_class") in common.CLASSES[case["prompt"]["analysis_contract"]]),
                "supported_class_spelling": raw.get("supported_class"),
                "supported_class_casefold_only_match": (isinstance(raw.get("supported_class"), str) and
                    raw.get("supported_class") not in common.CLASSES[case["prompt"]["analysis_contract"]] and
                    raw.get("supported_class").casefold() in {item.casefold() for item in common.CLASSES[case["prompt"]["analysis_contract"]]}),
            },
        }
    external_schema = ustc["schema"]["properties"]
    original_schema = frozen_schema["properties"]
    semantics = {}
    for key in ("reasoning_decision", "evidence_status", "supported_class", "cited_chunk_ids"):
        semantics[key] = {"external": external_schema[key], "frozen_ustc": original_schema[key],
                          "equal": external_schema[key] == original_schema[key]}
    records["frozen_ustc_prompt_comparison"] = {
        "external_prompt_schema_version": ustc["prompt"]["prompt_schema_version"],
        "frozen_prompt_schema_version": frozen_prompt["prompt_schema_version"],
        "external_classes": list(common.CLASSES["USTC-TFC2016"]),
        "frozen_classes": list(common.CLASSES[frozen_prompt["source_dataset"]]),
        "response_semantics": semantics,
        "note": "Citation enums differ only because each prompt supplies different evidence chunk IDs.",
    }
    return records


def main():
    install(APP)
    status = llm_runtime.check_llm_runtime()
    if status["status"] != "READY":
        raise RuntimeError(status["message"])
    ustc = prepare_case("USTC CSV", RUNTIME / "regression-inputs/ustc-raw.csv", "USTC-TFC2016", "raw_flows")
    cic = prepare_case("CIC CSV", RUNTIME / "regression-inputs/cic-raw.csv", "CIC-Darknet2020", "raw_flows")
    report = comparison(ustc, cic)
    write_json(DIAGNOSTICS / "llm_validation_comparison.json", report)
    print(json.dumps({"USTC": ustc["diagnostic"], "CIC": cic["diagnostic"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
