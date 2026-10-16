"""Run one previously accepted frozen USTC V5 prompt against the current server."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import APP, ROOT, RUNTIME
from backend.adapters.artifacts import rows
from backend.external_phase2.llm_runtime import runtime_args, tokenizer
from backend.external_phase2.safety import assert_desktop_write_path
from backend.utils.write_guard import install
from src.rag import llm_v3_runtime as frozen
from src.rag.audit_qwen_prompt_lengths import prompt_tokens

QUERY_ID = "USTC-VAL-003622"


def main():
    install(APP)
    prompt = next(row for row in rows(ROOT / "results/rag/test/final_4300_v5_contract_v2/stage08_llm/test_llm_prompts.csv") if row["query_id"] == QUERY_ID)
    generated = frozen.generate(prompt, prompt_tokens(tokenizer(), prompt["prompt_text"]), runtime_args())
    raw = json.loads(generated["raw_model_response"])
    error = None
    try:
        frozen.validate_logical(raw, prompt)
    except Exception as exc:
        error = {"exception_type": type(exc).__name__, "message": str(exc)}
    report = {
        "query_id": QUERY_ID,
        "recorded_research_response_was_accepted": True,
        "current_raw_response": raw,
        "current_validation_error": error,
        "current_validator_fallback_applied": generated.get("validator_fallback_applied"),
        "current_validator_fallback_reason": generated.get("validator_fallback_reason"),
        "grounding_gate_error_message": generated.get("grounding_gate_error_message"),
        "passed": error is None and generated.get("validator_fallback_applied") != "YES",
    }
    target = assert_desktop_write_path(RUNTIME / "diagnostics/frozen_ustc_live.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
