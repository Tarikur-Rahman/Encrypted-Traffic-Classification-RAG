"""Opt-in V5 execution for the exact historical V4 validation cohort.

Not an unseen validation set. No evidence-coverage filtering, reselection,
ground truth, or correctness enters population construction. Offline modes
construct the queue in memory and never write outputs or make requests.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from src.rag import run_v5_diagnostic as runtime
from src.evaluation import finalize_validation_phase2 as historical

common=runtime.common
require=common.require
RUN_ID="v5f_historical4205_comparison"
BASE="results/rag/v5f_validation/stage08_comparison/"+RUN_ID+"/"
SCOPE="fixed_historical_v4_validation_cohort_not_unseen_validation"
COUNT=4205
FREEZE_SHA256="e839350262f6d613d3d55e09b1347ecdf009b9d161c2ccc682f8b9000558dab7"


def historical_population(root):
    require(runtime.method.previous.sha(runtime.method.path(root,historical.FREEZE))==FREEZE_SHA256,"Historical freeze hash differs")
    freeze=common.parse_json(runtime.method.path(root,historical.FREEZE).read_text(encoding="utf-8"))
    ordered=freeze["completed_validation_ids"]
    require(len(ordered)==len(set(ordered))==COUNT,"Exactly 4205 frozen unique IDs required")
    require(all("-VAL-" in q and "-TEST-" not in q for q in ordered),"Historical cohort must be validation only")
    expected=set(ordered); population={}
    for source in historical.SOURCES:
        for row in runtime.jsonl(root,source):
            q=row["query_id"]
            require(q in expected and q not in population,"Unknown/duplicate historical response")
            require(row.get("source_split","validation")=="validation","Historical non-validation response")
            if source!=historical.SOURCES[-1]:
                require(row.get("response_valid") is True,"Unvalidated historical response")
            # Explicit allowlist: no labels, correctness, confidence or V4 status.
            population[q]=(row["source_dataset"],row["classifier_prediction"])
    require(set(population)==expected,"Missing historical Stage-8 responses")
    return ordered,population


def selected_contexts(root,ordered,population):
    found={}
    outputs=runtime.contexts.paths_for("v5f_validation")[0]
    common.completed(root,outputs[2],outputs[3])
    for row in common.rows(root,outputs[0],runtime.contexts.HEADER):
        q=row["query_id"]
        if q not in population: continue
        require(q not in found,"Duplicate selected V5 context")
        require(row["source_split"]=="validation" and row["rag_partition"]=="development_validation","Non-validation context")
        require((row["source_dataset"],row["classifier_prediction"])==population[q],"Historical classifier prediction/dataset differs")
        found[q]=dict(query_id=q,source_dataset=row["source_dataset"],source_split="validation",
                      classifier_prediction=row["classifier_prediction"],traffic_profile=row["traffic_profile"],
                      trusted_evidence=common.parse_json(row["evidence_records_json"]))
    require(set(found)==set(ordered),"Missing V5f contexts")
    return found


def evidence(root):
    records=runtime.method.verify_bundle(root)
    claims=runtime.method.load_decision_claims(root,"v5f_validation")
    require(claims==runtime.method.derived_tables(records)[2],"Changed association registry")
    registry={r["claim_id"]:r for r in records if r["decision"]=="APPROVED"}
    for identity,row in registry.items():
        require(row["statistics_link"]==runtime.method.BASE+"statistics.json#"+identity,"Internal provenance linkage differs")
    chunks=common.unique(runtime.index.validated_chunks(root,inputs=runtime.index.input_paths("v5f_validation"),
                        outputs=runtime.index.output_paths("v5f_validation")),"chunk_id")
    verified=runtime.provenance.load_verified(root,"v5f_validation")
    return registry,{k:dict(v,chunk_text=chunks[k]["chunk_text"]) for k,v in verified.items()}


def input_hashes(root):
    paths=[historical.FREEZE,*historical.SOURCES,runtime.method.BASE+"bundle.json",
           *runtime.contexts.paths_for("v5f_validation")[0]]
    return {p:runtime.method.previous.sha(runtime.method.path(root,p)) for p in paths}


def prepare(root):
    before=input_hashes(root)
    ordered,population=historical_population(root)
    payloads=selected_contexts(root,ordered,population)
    registry,verified=evidence(root)
    rows=[]
    for q in ordered:
        p=payloads[q]
        # Zero-claim contexts remain included. Existing V5 policy handles them.
        rows.append(dict(query_id=q,source_dataset=p["source_dataset"],source_split="validation",
                         classifier_prediction=p["classifier_prediction"],
                         prompt_text=runtime.validator.prompt(p,registry,verified,method=runtime.method)))
    require([r["query_id"] for r in rows]==ordered,"Historical order changed")
    require(before==input_hashes(root),"Historical/context inputs changed during preparation")
    return rows,payloads,registry,verified,before


def configuration(root,rows,payloads,audit,inputs):
    code_paths=("src/rag/run_v5_comparison.py","src/rag/run_v5_diagnostic.py","src/rag/positive_evidence_v5.py",
                "src/rag/audit_qwen_prompt_lengths.py","src/evaluation/finalize_validation_phase2.py")
    value=dict(development_id="v5f_validation",diagnostic_run_id=RUN_ID,evaluation_scope=SCOPE,selected_queries=COUNT,
               historical_population_source=historical.FREEZE,selected_query_ids=[r["query_id"] for r in rows],
               source_split="validation",ground_truth_used=False,model=runtime.MODEL,endpoint=runtime.ENDPOINT,
               max_model_len=5120,max_tokens=1024,temperature=0,seed=42,corrections=0,automatic_transport_retries=0,
               request_contract_version="v2",validator_version=runtime.VERSION,evidence_method=runtime.method.VERSION,
               input_sha256=inputs,code_sha256={p:runtime.method.previous.sha(runtime.method.path(root,p)) for p in code_paths},
               tokenizer_sha256={p:runtime.method.previous.sha(runtime.method.path(root,p)) for p,_,_ in runtime.qwen.tokenizer_files(root)},
               tokenizer_packages={k:runtime.package_version(k) for k in ("transformers","tokenizers")},
               token_audit=audit,prompt_sha256={r["query_id"]:runtime.digest(r["prompt_text"]) for r in rows},
               request_schema_sha256={r["query_id"]:runtime.digest(common.json_text(runtime.schema(r,payloads[r["query_id"]]))) for r in rows})
    value["configuration_sha256"]=runtime.digest(common.json_text(value))
    return value


def summary(completed):
    return dict(evaluation_scope=SCOPE,run_id=RUN_ID,selected_queries=COUNT,completed=len(completed),remaining=COUNT-len(completed),
                accepted=sum(v["validation_status"]=="PASS" for v in completed.values()),
                rejected=sum(v["validation_status"]!="PASS" for v in completed.values()),
                accepted_evidence_status_counts=dict(Counter(v["validated_response"]["evidence_status"] for v in completed.values() if v["validation_status"]=="PASS")),
                diagnostic_only=False,full_validation_results=False,ground_truth_used=False)


@contextmanager
def runtime_scope():
    """Reuse the sequential durable engine without editing its frozen source.

    CLI execution is single-threaded; restore every namespace/summary binding
    even on failure. No scientific validator or request function is replaced.
    """
    replacements=dict(BASE=BASE,NEW_BASE=BASE,RUN_ID=RUN_ID,summary=summary)
    previous={k:getattr(runtime,k) for k in replacements}
    try:
        for k,v in replacements.items(): setattr(runtime,k,v)
        yield
    finally:
        for k,v in previous.items(): setattr(runtime,k,v)


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--project-root",type=Path,default=Path("."))
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--token-audit",action="store_true")
    modes.add_argument("--status-only",action="store_true")
    modes.add_argument("--execute",action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv)
    require(not args.resume or args.execute,"--resume requires --execute")
    root=args.project_root.resolve(strict=True)
    rows,payloads,registry,verified,inputs=prepare(root)
    tokenizer_before=runtime.qwen.tokenizer_files(root)
    audit=runtime.token_audit(rows,runtime.qwen.load_tokenizer(root))
    require(tokenizer_before==runtime.qwen.tokenizer_files(root),"Tokenizer changed during audit")
    print(common.json_text(dict(token_audit=audit)))
    if args.token_audit: return 0 if audit["gate_status"]=="PASS" else 2
    require(audit["gate_status"]=="PASS","Token budget overflow; no inference or truncation permitted")
    config=configuration(root,rows,payloads,audit,inputs)
    require(inputs==input_hashes(root),"Inputs changed during token audit")
    with runtime_scope():
        if args.status_only:
            target=runtime.method.path(root,BASE+runtime.CONFIG,False)
            completed={}; ready=True; state={}; errors={}
            if target.exists():
                saved=common.parse_json(target.read_text(encoding="utf-8"))
                require(saved==config,"Comparison configuration changed")
                completed,state=runtime.recover(root,rows,payloads,registry,verified,config,inspect_only=True)
                _,errors=runtime.transport_state(root,rows,payloads,config)
                ready=not state["ambiguous_or_blocked_ids"]
            else: require(not target.parent.exists(),"Incomplete comparison initialization")
            print(common.json_text(dict(summary(completed),ready_for_execution=ready,configuration_matches=True,
                                        test_rows=0,missing_contexts=0,duplicate_ids=0,transport_failures=len(errors),request_states=state)))
            return 0 if ready else 2
        print(common.json_text(runtime.execute(root,rows,payloads,registry,verified,config,resume=args.resume)))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
