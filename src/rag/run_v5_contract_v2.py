"""Create-only development/compliance runs for structured V5 response contract v2.

Selection reads Stage7/evidence metadata, never previous model responses.
Offline modes write nothing. No tokenizer/model truncation or automatic retries.
"""
import argparse
import base64
from collections import Counter,defaultdict
from contextlib import contextmanager
from pathlib import Path

from src.rag import association_response_v2 as contract
from src.rag import run_v5_comparison as comparison

runtime=contract.runtime
common=runtime.common
require=common.require
DIAGNOSTIC="v5f_stage8_contract_v2_compliance_60"
COMPARISON="v5f_historical4205_comparison_contract_v2"
BASES={DIAGNOSTIC:"results/rag/v5f_validation/stage08_diagnostic/"+DIAGNOSTIC+"/",
       COMPARISON:"results/rag/v5f_validation/stage08_comparison/"+COMPARISON+"/"}
SELECTION_RULE="15 per evidence pattern; round-robin available dataset/prediction strata ordered by SHA256(42|pattern|dataset|prediction); within-stratum SHA256(42|pattern|query_id), then query_id"
LEGACY_OUTCOMES=runtime.outcomes


def payload(row):
    require(row["source_split"]=="validation" and row["rag_partition"]=="development_validation","Validation-only Stage7 input required")
    return dict(query_id=row["query_id"],source_dataset=row["source_dataset"],source_split="validation",
                classifier_prediction=row["classifier_prediction"],traffic_profile=row["traffic_profile"],
                trusted_evidence=common.parse_json(row["evidence_records_json"]))


def select_diagnostic(payloads,registry,verified):
    buckets=defaultdict(list); seen=set(); counts=Counter()
    for p in payloads:
        q=p["query_id"]
        require(q not in seen,"Duplicate Stage7 query")
        seen.add(q); counts[p["source_dataset"]]+=1
        state=contract.admissible(p,registry,verified)
        pattern=state["pattern"]; key=(pattern,p["source_dataset"],p["classifier_prediction"])
        rank=(runtime.digest("42|"+pattern+"|"+q),q)
        bucket=buckets[key]
        bucket.append((rank,p,state)); bucket.sort(key=lambda entry:entry[0])
        if len(bucket)>15: bucket.pop()
    require(dict(counts)==common.QUERY_COUNTS,"Complete validation Stage7 population required")
    selected=[]
    for pattern in contract.PATTERNS:
        keys=sorted((key for key in buckets if key[0]==pattern),key=lambda key:(runtime.digest("42|"+"|".join(key)),key))
        chosen=[]; offset=0
        while len(chosen)<15:
            before=len(chosen)
            for key in keys:
                if offset<len(buckets[key]): chosen.append(buckets[key][offset])
                if len(chosen)==15: break
            require(len(chosen)>before,"Insufficient candidates for evidence pattern "+pattern)
            offset+=1
        selected.extend((p,s) for _,p,s in chosen)
    require(len(selected)==len({p["query_id"] for p,_ in selected})==60,"Diagnostic ID population differs")
    return selected


def frozen_ids(root):
    path=runtime.method.path(root,comparison.historical.FREEZE)
    require(runtime.method.previous.sha(path)==comparison.FREEZE_SHA256,"Historical freeze hash differs")
    ids=common.parse_json(path.read_text(encoding="utf-8"))["completed_validation_ids"]
    require(len(ids)==len(set(ids))==4205 and all("-VAL-" in q and "-TEST-" not in q for q in ids),"Historical validation IDs differ")
    return ids


def input_hashes(root,run_id):
    paths=[runtime.method.BASE+"bundle.json",*runtime.contexts.paths_for("v5f_validation")[0]]
    if run_id==COMPARISON: paths.append(comparison.historical.FREEZE)
    return {p:runtime.method.previous.sha(runtime.method.path(root,p)) for p in paths}


def prepare(root,run_id):
    require(run_id in BASES,"Unknown contract-v2 run")
    require(runtime.method.previous.sha(runtime.method.path(root,"src/rag/positive_evidence_v5.py"))==contract.FROZEN_VALIDATOR_SHA256,
            "Frozen V5 validator source changed")
    before=input_hashes(root,run_id)
    registry,verified=comparison.evidence(root)
    outputs=runtime.contexts.paths_for("v5f_validation")[0]
    common.completed(root,outputs[2],outputs[3])
    stream=(payload(r) for r in common.rows(root,outputs[0],runtime.contexts.HEADER))
    if run_id==DIAGNOSTIC:
        selected=select_diagnostic(stream,registry,verified)
    else:
        ids=frozen_ids(root); expected=set(ids); found={}
        for p in stream:
            q=p["query_id"]
            if q not in expected: continue
            require(q not in found,"Duplicate historical context")
            found[q]=(p,contract.admissible(p,registry,verified))
        require(set(found)==expected,"Missing historical V5 context")
        selected=[found[q] for q in ids]
    rows=[]; payloads={}; states={}
    for p,state in selected:
        q=p["query_id"]; payloads[q]=p; states[q]=state
        rows.append(dict(query_id=q,source_dataset=p["source_dataset"],classifier_prediction=p["classifier_prediction"],
                         prompt_text=contract.prompt(p,registry,verified,state),response_schema=contract.schema(state,q)))
    require(before==input_hashes(root,run_id),"Inputs changed during preparation")
    return rows,payloads,states,registry,verified,before


def request(row,p):
    return dict(model=runtime.MODEL,messages=[dict(role="user",content=row["prompt_text"])],temperature=0,seed=42,
                max_tokens=1024,stream=False,response_format=dict(type="json_schema",json_schema=dict(
                    name=contract.VERSION,strict=True,schema=row["response_schema"])))


def outcomes(raw,payload,registry,verified,expected_tokens,*,source_split="validation"):
    # Raw HTTP bytes have already been fsynced by the unchanged durable engine.
    packet=common.parse_json(base64.b64decode(raw["response_base64"],validate=True).decode("utf-8"))
    content=packet["choices"][0]["message"].get("content")
    value=None; legacy=None; error=None
    try:
        require(isinstance(content,str),"Structured content string required")
        value=common.parse_json(content)
        legacy=contract.to_legacy(value)
    except (ValueError,TypeError) as exc:
        error=type(exc).__name__+": "+str(exc)
    validation_raw=raw
    if legacy is not None:
        # An internal representation for the unchanged validator, never a
        # replacement raw response. Every non-content HTTP field is preserved.
        derived=common.parse_json(common.json_text(packet))
        derived["choices"][0]["message"]["content"]=common.json_text(legacy)
        validation_raw=dict(raw,response_base64=base64.b64encode(common.json_text(derived).encode()).decode())
    if source_split=="validation":
        result=LEGACY_OUTCOMES(validation_raw,payload,registry,verified,expected_tokens)
    else:
        result=LEGACY_OUTCOMES(validation_raw,payload,registry,verified,expected_tokens,source_split=source_split)
    if error:
        # Even an otherwise valid old six-field response is invalid contract v2.
        result.update(validation_status="REJECTED",validated_response=None,validation_error=error,
                      effective_evidence_status="INSUFFICIENT",validator_fallback_applied="YES")
        result.pop("validated_claim_linkage",None)
    result.update(response_contract_version="v2",raw_v2_model_response=value,
                  raw_v2_response_sha256=runtime.digest(common.json_text(value)) if value is not None else None,
                  raw_model_content_sha256=runtime.digest(content) if isinstance(content,str) else None,
                  derived_validator_payload=legacy,
                  derived_validator_payload_sha256=runtime.digest(common.json_text(legacy)) if legacy is not None else None,
                  model_explanation=value.get("explanation") if isinstance(value,dict) else None,
                  narrative_validation_status="NOT_SEMANTICALLY_VALIDATED")
    return result


def configuration(root,run_id,rows,payloads,states,audit,inputs):
    value=comparison.configuration(root,rows,payloads,audit,inputs)
    value.pop("configuration_sha256")
    for path in ("src/rag/association_response_v2.py","src/rag/run_v5_contract_v2.py"):
        value["code_sha256"][path]=runtime.method.previous.sha(runtime.method.path(root,path))
    value.update(diagnostic_run_id=run_id,response_contract_version="v2",response_schema_name=contract.VERSION,
                 selected_queries=len(rows),parent_contract_v1_run="v5f_historical4205_comparison",
                 parent_contract_v1_status="completed_with_model_compliance_failure",repair_reason="model_output_compliance_only",
                 development_disclosure=contract.DISCLOSURE,evaluation_scope="development_compliance_diagnostic" if run_id==DIAGNOSTIC else "development_compliance_fixed_historical_validation_comparison",
                 historical_population_source=comparison.historical.FREEZE if run_id==COMPARISON else None,
                 selection_rule=SELECTION_RULE if run_id==DIAGNOSTIC else "Exact frozen historical 4205 IDs in freeze order; no reselection",
                 selection_sha256=runtime.digest(common.json_text([r["query_id"] for r in rows])),
                 request_schema_sha256={r["query_id"]:runtime.digest(common.json_text(r["response_schema"])) for r in rows},
                 evidence_patterns={q:s["pattern"] for q,s in states.items()})
    value["configuration_sha256"]=runtime.digest(common.json_text(value))
    return value


@contextmanager
def runtime_scope(run_id,count):
    require(run_id in BASES,"Unknown contract-v2 run")
    def summary(completed):
        return dict(run_id=run_id,response_contract_version="v2",selected_queries=count,completed=len(completed),remaining=count-len(completed),
                    accepted=sum(r["validation_status"]=="PASS" for r in completed.values()),rejected=sum(r["validation_status"]!="PASS" for r in completed.values()),
                    accepted_evidence_status_counts=dict(Counter(r["validated_response"]["evidence_status"] for r in completed.values() if r["validation_status"]=="PASS")),
                    diagnostic_only=run_id==DIAGNOSTIC,development_compliance_only=True,full_validation_results=False)
    changes=dict(BASE=BASES[run_id],NEW_BASE=BASES[run_id],RUN_ID=run_id,request=request,outcomes=outcomes,summary=summary)
    previous={k:getattr(runtime,k) for k in changes}
    try:
        for k,v in changes.items(): setattr(runtime,k,v)
        yield
    finally:
        for k,v in previous.items(): setattr(runtime,k,v)


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--project-root",type=Path,default=Path("."))
    parser.add_argument("--run-id",required=True,choices=tuple(BASES))
    modes=parser.add_mutually_exclusive_group(required=True)
    for mode in ("token-audit","status-only","execute"): modes.add_argument("--"+mode,action="store_true")
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args(argv); require(not args.resume or args.execute,"Resume requires execute")
    root=args.project_root.resolve(strict=True)
    rows,payloads,states,registry,verified,inputs=prepare(root,args.run_id)
    distribution=Counter((states[r["query_id"]]["pattern"],r["source_dataset"],r["classifier_prediction"]) for r in rows)
    before=runtime.qwen.tokenizer_files(root)
    audit=runtime.token_audit(rows,runtime.qwen.load_tokenizer(root))
    require(before==runtime.qwen.tokenizer_files(root),"Tokenizer changed")
    require(inputs==input_hashes(root,args.run_id),"Inputs changed during audit")
    print(common.json_text(dict(token_audit=audit,distribution={"/".join(k):v for k,v in sorted(distribution.items())})))
    if args.token_audit: return 0 if audit["gate_status"]=="PASS" else 2
    require(audit["gate_status"]=="PASS","Token overflow; no truncation or inference permitted")
    config=configuration(root,args.run_id,rows,payloads,states,audit,inputs)
    with runtime_scope(args.run_id,len(rows)):
        if args.status_only:
            target=runtime.method.path(root,runtime.BASE+runtime.CONFIG,False); completed={}; ready=True; errors={}; state={}
            if target.exists():
                require(common.parse_json(target.read_text(encoding="utf-8"))==config,"Configuration mismatch")
                completed,state=runtime.recover(root,rows,payloads,registry,verified,config,inspect_only=True)
                _,errors=runtime.transport_state(root,rows,payloads,config)
                ready=not state["ambiguous_or_blocked_ids"]
            else: require(not target.parent.exists(),"Incomplete create-only initialization")
            print(common.json_text(dict(runtime.summary(completed),ready_for_execution=ready,configuration_matches=True,
                                        ambiguous=len(state.get("ambiguous_or_blocked_ids",[])),transport_failures=len(errors))))
            return 0 if ready else 2
        print(common.json_text(runtime.execute(root,rows,payloads,registry,verified,config,args.resume)))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
