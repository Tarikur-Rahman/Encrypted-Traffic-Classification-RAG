"""One USTC CSV, then one USTC PCAP, then one CIC CSV regression."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config import APP, ROOT, RUNTIME
from backend.adapters import fresh_phase2, input_adapter, phase1_adapter
from backend.adapters.artifacts import digest
from backend.external_phase2.llm_runtime import check_llm_runtime
from backend.external_phase2.safety import assert_desktop_write_path
from backend.session import store
from backend.utils.write_guard import install


def run_case(label, path, contract, input_stage):
    before = digest(path)
    source = input_adapter.validate(str(path), contract, input_stage)
    if not source["compatible"]:
        raise ValueError(source["reason"])
    session = store.create(source)
    stage = lambda key, status, message: store.stage(session, key, status, message)
    phase1_adapter.execute(session, stage)
    flow_id = session["flows"][0]["id"]
    result = fresh_phase2.execute(session, flow_id, stage)
    store.save(session)
    item = {
        "input": label,
        "analysis_id": session["analysis_id"],
        "flow_id": flow_id,
        "runtime_status": result.get("runtime_status"),
        "blockers": result["blockers"],
        "source_unchanged": digest(path) == before,
        "validation_passed": result.get("runtime_status", {}).get("llm") == "COMPLETED",
        "fusion_executed": result.get("runtime_status", {}).get("fusion") == "COMPLETED",
        "final_class": result["decision"].get("final_class"),
        "evidence_status": result["decision"].get("evidence_status"),
        "cited_chunk_ids": result["reasoning"].get("cited_chunk_ids"),
        "allowed_chunk_ids": json.loads(result["context"]["selected_chunk_ids"]),
    }
    item["citations_verified"] = (item["validation_passed"] and
        set(item["cited_chunk_ids"] or []) <= set(item["allowed_chunk_ids"]))
    item["passed"] = all((item["source_unchanged"], item["validation_passed"], item["fusion_executed"],
                          bool(item["final_class"]), bool(item["evidence_status"]), item["citations_verified"]))
    return item


def main():
    install(APP)
    report = {"preflight": check_llm_runtime(), "cases": [], "passed": False}
    target = assert_desktop_write_path(RUNTIME / "diagnostics/ustc_contract_fix_acceptance.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if report["preflight"]["status"] != "READY":
            raise RuntimeError(report["preflight"]["message"])
        csv_case = run_case("USTC CSV", RUNTIME / "regression-inputs/ustc-raw.csv", "USTC-TFC2016", "raw_flows")
        report["cases"].append(csv_case)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not csv_case["passed"]:
            raise ValueError("USTC CSV response did not pass validation; PCAP and CIC were not run")
        pcap_case = run_case("USTC PCAP", ROOT / "data/raw/USTC-TFC2016/Benign/Facetime.pcap", "USTC-TFC2016", "auto")
        report["cases"].append(pcap_case)
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not pcap_case["passed"]:
            raise ValueError("USTC PCAP response did not pass validation; CIC was not run")
        report["cases"].append(run_case("CIC CSV", RUNTIME / "regression-inputs/cic-raw.csv", "CIC-Darknet2020", "raw_flows"))
        report["passed"] = all(item["passed"] for item in report["cases"])
        if not report["passed"]:
            raise ValueError("CIC regression failed")
    except Exception as exc:
        report["failure"] = str(exc)
        raise
    finally:
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
