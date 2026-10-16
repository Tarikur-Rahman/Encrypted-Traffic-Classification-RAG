"""Stage-9 preflight/opt-in fusion for the accepted historical V5-v2 cohort.

Retain-classifier only. Replays frozen V5 validation without tokenization or
inference. No rejected-response mapping, Stage10, TEST, or evaluation labels.
"""
import argparse
from collections import Counter
from pathlib import Path

from src.rag import run_v5_contract_v2 as stage8
from src.rag import run_evidence_supported_fusion as fusion

runtime=stage8.runtime
common=runtime.common
require=common.require
BASE="results/rag/v5f_validation/stage09_fusion/"+stage8.COMPARISON+"/"
STAGE8_BASE=stage8.BASES[stage8.COMPARISON]
EXTRA=("source_split","fusion_policy","stage8_validation_status","response_contract_version",
       "reasoning_decision","claim_ids_json","validated_claim_linkage_json","model_explanation",
       "narrative_validation_status","stage8_response_reference")
HEADER=(*fusion.HEADER,*EXTRA)


def snapshot(root):
    names=[STAGE8_BASE+name for name in (runtime.CONFIG,runtime.JOURNAL,runtime.RAW,runtime.VALIDATED,runtime.SUMMARY)]
    return {name:runtime.method.previous.sha(runtime.method.path(root,name)) for name in names}


def accepted_inputs(root):
    before=snapshot(root)
    saved=common.parse_json(runtime.method.path(root,STAGE8_BASE+runtime.CONFIG).read_text(encoding="utf-8"))
    unsigned=dict(saved); signature=unsigned.pop("configuration_sha256")
    require(runtime.digest(common.json_text(unsigned))==signature,"Stage8 configuration hash mismatch")
    require(saved["diagnostic_run_id"]==stage8.COMPARISON and saved["response_contract_version"]=="v2"
            and saved["source_split"]=="validation" and saved["ground_truth_used"] is False,"Wrong Stage8 contract")
    for category in ("input_sha256","code_sha256","tokenizer_sha256"):
        for name,expected in saved[category].items():
            require(runtime.method.previous.sha(runtime.method.path(root,name))==expected,"Changed Stage8 dependency: "+name)
    rows,payloads,states,registry,verified,inputs=stage8.prepare(root,stage8.COMPARISON)
    ids=[r["query_id"] for r in rows]
    require(len(ids)==len(set(ids))==4205 and ids==saved["selected_query_ids"],"Stage8 cohort/order differs")
    require(inputs==saved["input_sha256"] and runtime.digest(common.json_text(ids))==saved["selection_sha256"],"Stage8 input/cohort hash differs")
    for row in rows:
        q=row["query_id"]
        require(runtime.digest(row["prompt_text"])==saved["prompt_sha256"][q]
                and runtime.digest(common.json_text(row["response_schema"]))==saved["request_schema_sha256"][q],"Stage8 prompt/schema hash differs")
    with stage8.runtime_scope(stage8.COMPARISON,4205):
        completed,state=runtime.recover(root,rows,payloads,registry,verified,saved,inspect_only=True)
        starts,errors=runtime.transport_state(root,rows,payloads,saved)
        require(set(completed)==set(starts)==set(ids) and not errors and all(not v for v in state.values()),"Incomplete/ambiguous Stage8 state")
        summary=common.parse_json(runtime.method.path(root,STAGE8_BASE+runtime.SUMMARY).read_text(encoding="utf-8"))
        require(summary==runtime.summary(completed),"Stage8 summary differs")
    require(all(v["validation_status"]=="PASS" and v["validated_response"] is not None for v in completed.values()),
            "Stage9 requires accepted V5-v2 responses; no rejection coercion")
    contexts={}
    path=runtime.contexts.paths_for("v5f_validation")[0][0]
    for row in common.rows(root,path,runtime.contexts.HEADER):
        q=row["query_id"]
        if q not in completed: continue
        require(q not in contexts,"Duplicate selected context")
        contexts[q]={k:row[k] for k in ("query_id","source_dataset","source_split","rag_partition","classifier_prediction",
                                       "classifier_confidence","classifier_probabilities","evidence_records_json")}
    require(set(contexts)==set(ids),"Missing fusion context")
    # Check prediction parity independently against the frozen historical cohort.
    historical_ids,predictions=stage8.comparison.historical_population(root)
    require(historical_ids==ids,"Historical order differs")
    require(all((contexts[q]["source_dataset"],contexts[q]["classifier_prediction"])==predictions[q] for q in ids),"Classifier prediction changed")
    require(before==snapshot(root) and inputs==stage8.input_hashes(root,stage8.COMPARISON),"Inputs changed during preflight")
    return ids,contexts,completed,before


def fuse_accepted(context,record,*,source_split="validation",response_reference=None):
    """Project an independently revalidated V5 record with fixed retention.

    Same retain_classifier branch and base output schema as fusion.fuse;
    avoid its incompatible legacy prompt reconstruction/importer.
    """
    require(source_split in {"validation","test"}, "Invalid explicit fusion split")
    partition, marker, excluded = ("development_validation","-VAL-","-TEST-") if source_split=="validation" else ("final_test","-TEST-","-VAL-")
    require(context["source_split"]==source_split and context["rag_partition"]==partition
            and marker in context["query_id"] and excluded not in context["query_id"],"Fusion split/ID mismatch")
    require(source_split=="validation" or response_reference is not None,"TEST response reference required")
    require(record["validation_status"]=="PASS" and record["response_contract_version"]=="v2", "Accepted contract-v2 response required")
    value=record["validated_response"]
    require(value==record["derived_validator_payload"] and value["query_id"]==record["query_id"]==context["query_id"],"Validated payload identity differs")
    prediction=context["classifier_prediction"];status=value["evidence_status"]
    require(prediction in common.CLASSES[context["source_dataset"]] and status in common.STATUSES,"Invalid class/status")
    confidence=common.finite(context["classifier_confidence"])
    require(0<=confidence<=1,"Invalid classifier confidence")
    evidence=common.unique(common.parse_json(context["evidence_records_json"]),"chunk_id")
    citations=value["cited_chunk_ids"]
    require(all(c in evidence and evidence[c]["provenance_status"]=="VERIFIED" for c in citations),"Unverified fusion citation")
    reason="CLASSIFIER_RETAINED_"+status
    if status=="CONTRADICTED": reason="CLASSIFIER_RETAINED_OVERRIDE_DISABLED_OR_CONFIDENCE_ABOVE_THRESHOLD"
    return dict(query_id=context["query_id"],source_dataset=context["source_dataset"],classifier_prediction=prediction,
                classifier_confidence=context["classifier_confidence"],classifier_probabilities=context["classifier_probabilities"],
                llm_supported_class=value["supported_class"] or "",evidence_status=status,final_class=prediction,fusion_reason=reason,
                cited_chunk_ids=common.json_text(citations),cited_document_ids=common.json_text(sorted({evidence[c]["document_id"] for c in citations})),
                provenance_summary=common.json_text({c:{k:evidence[c][k] for k in ("document_id","source_type","provenance_status")} for c in citations}),
                evidence_explanation=value["explanation"],source_split=source_split,fusion_policy="retain_classifier",
                stage8_validation_status="PASS",response_contract_version="v2",reasoning_decision=value["reasoning_decision"],
                claim_ids_json=common.json_text(record["raw_v2_model_response"]["claim_ids"]),
                validated_claim_linkage_json=common.json_text(record["validated_claim_linkage"]),model_explanation=record["model_explanation"],
                narrative_validation_status=record["narrative_validation_status"],stage8_response_reference=(response_reference or STAGE8_BASE+runtime.VALIDATED)+"#"+context["query_id"])


def build(root):
    ids,contexts,completed,hashes=accepted_inputs(root)
    records=[fuse_accepted(contexts[q],completed[q]) for q in ids]
    require([r["query_id"] for r in records]==ids and all(r["final_class"]==r["classifier_prediction"] for r in records),"Fusion changed IDs/classifier")
    checks=dict(status="READY_FOR_STAGE9_PREFLIGHT",stage8_rows=4205,stage9_rows=len(records),unique_ids=len(set(ids)),
                validation_rows=len(records),test_rows=0,missing_stage8_ids=0,duplicate_ids=0,classifier_overrides=0,
                policy="retain_classifier",ground_truth_used=0,evidence_status_counts=dict(Counter(r["evidence_status"] for r in records)))
    configuration=dict(stage="EVIDENCE_SUPPORTED_DECISION_FUSION",policy="retain_classifier",response_contract_version="v2",
                       rejection_mapping="NONE_ACCEPTED_ONLY",ground_truth_used=False,development_disclosure=stage8.contract.DISCLOSURE,
                       stage8_sha256=hashes,adapter_sha256=runtime.method.previous.sha(runtime.method.path(root,"src/evaluation/finalize_v5_contract_v2_stage9.py")))
    tables=[(BASE+"final_decisions.csv",HEADER,records),
            (BASE+"fusion_configuration.csv",common.CONFIG_HEADER,common.config_rows(configuration))]
    return tables,checks,hashes


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--project-root",type=Path,default=Path("."))
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--status-only",action="store_true")
    modes.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv);root=args.project_root.resolve(strict=True)
    tables,checks,hashes=build(root)
    require(hashes==snapshot(root),"Stage8 changed before publication")
    if args.execute:
        target=runtime.method.path(root,BASE+"final_decisions.csv",False)
        require(not target.parent.exists(),"Stage9 namespace already exists; overwrite prohibited")
        common.write_bundle(root,tables)
    print(common.json_text(dict(checks,executed=args.execute)))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
