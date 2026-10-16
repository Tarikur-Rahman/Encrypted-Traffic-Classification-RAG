"""Dedicated offline preflight / opt-in loopback V5 methodology diagnostic.

Never invokes V4 runtime or expands the frozen 60-query selection. Token audit
and status modes write nothing and make no requests. Execution is sequential,
raw-first, append-only and refuses ambiguous in-flight recovery.
"""
import argparse
import base64
import hashlib
import json
import math
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter
from contextlib import contextmanager
from importlib.metadata import version as package_version

from src.rag import phase2_common as common
from src.rag import build_semantic_association_v5 as method
from src.rag import positive_evidence_v5 as validator
from src.rag import audit_qwen_prompt_lengths as qwen
from src.rag import build_evidence_index_v2 as index
from src.rag import verify_retrieval_provenance_v3 as provenance
from src.rag import build_trusted_contexts as contexts

require=common.require
INPUT="results/rag/v5f_validation/pre_llm/diagnostic_prompts.jsonl"
COVERAGE="results/rag/v5f_validation/pre_llm/coverage_audit.json"
BASE="results/rag/v5f_validation/stage08_diagnostic/claim_association_60/"
PARENT_BASE=BASE
RUN_ID="claim_association_60_request_contract_v2"
NEW_BASE="results/rag/v5f_validation/stage08_diagnostic/"+RUN_ID+"/"
MODEL="provenance-qwen3-4b"
ENDPOINT="http://127.0.0.1:8000/v1/chat/completions"
VERSION="v5f-association-diagnostic-v1"
FIELDS=("query_id","reasoning_decision","evidence_status","supported_class","explanation","cited_chunk_ids")
PROMPT_FIELDS={"query_id","source_dataset","classifier_prediction","source_split","prompt_text","token_audit_status","diagnostic_scope"}
SCOPE="claim-covered_stratified_validation_not_population_estimate"
CONFIG="configuration.json"
JOURNAL="request_journal.jsonl"
RAW="raw_responses.jsonl"
VALIDATED="validated_responses.jsonl"
SUMMARY="diagnostic_summary.json"
ERRORS="transport_errors.jsonl"


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def jsonl(root,relative):
    with method.path(root,relative).open(encoding="utf-8") as stream:
        for line in stream:
            require(line.endswith("\n") and line.strip(),"Incomplete/blank durable JSONL record; manual recovery required")
            value=common.parse_json(line)
            require(isinstance(value,dict),"JSONL object required")
            yield value


def parse_prompt(row):
    require(set(row)==PROMPT_FIELDS,"Unexpected prompt fields; GT/correctness prohibited")
    require(row["source_split"]=="validation" and "-VAL-" in row["query_id"] and "-TEST-" not in row["query_id"],"Validation-only diagnostic")
    require(row["source_dataset"] in common.CLASSES and row["classifier_prediction"] in common.CLASSES[row["source_dataset"]],"Unknown prediction stratum")
    require(row["diagnostic_scope"]==SCOPE and row["token_audit_status"]=="NOT_TOKEN_AUDITED","Wrong diagnostic preparation contract")
    require(row["prompt_text"].count("\nINPUT_JSON:\n")==1,"Ambiguous prompt payload")
    payload=common.parse_json(row["prompt_text"].split("\nINPUT_JSON:\n",1)[1])
    require(all(payload[k]==row[k] for k in ("query_id","source_dataset","source_split","classifier_prediction")),"Prompt identity mismatch")
    return payload


def validate_population(rows):
    require(len(rows)==60 and len({r["query_id"] for r in rows})==60,"Exactly 60 unique diagnostic IDs required")
    counts=Counter((r["source_dataset"],r["classifier_prediction"]) for r in rows)
    require(counts==Counter({(d,c):10 for d in common.CLASSES for c in common.CLASSES[d]}),"Ten queries per prediction stratum required")
    for row in rows: parse_prompt(row)
    # Same deterministic order as coverage preparation: stratum then seed42 hash.
    expected=sorted(rows,key=lambda r:(r["source_dataset"],r["classifier_prediction"],digest("42|"+r["query_id"]),r["query_id"]))
    require(rows==expected,"Prepared selection order differs from deterministic seed42 policy")


def prepare(root):
    report=common.parse_json(method.path(root,COVERAGE).read_text(encoding="utf-8"))
    require(report["status"]=="PASS" and report["diagnostic_prompts"]==60 and report["test_rows"]==report["ground_truth_used"]==0,"Coverage gate not PASS")
    require(report["totals"]["queries"]==sum(common.QUERY_COUNTS.values()),"Incomplete validation coverage audit")
    rows=list(jsonl(root,INPUT)); validate_population(rows)
    records=method.verify_bundle(root)
    claims=method.load_decision_claims(root,method.DEVELOPMENT_ID)
    require(claims==method.derived_tables(records)[2],"Changed association registry")
    registry={r["claim_id"]:r for r in records if r["decision"]=="APPROVED"}
    for identity,record in registry.items():
        require(record["statistics_link"]==method.BASE+"statistics.json#"+identity,
                "Unregistered internal empirical provenance linkage")
    chunks=common.unique(index.validated_chunks(root,inputs=index.input_paths(method.DEVELOPMENT_ID),outputs=index.output_paths(method.DEVELOPMENT_ID)),"chunk_id")
    verified=provenance.load_verified(root,method.DEVELOPMENT_ID)
    verified={k:dict(v,chunk_text=chunks[k]["chunk_text"]) for k,v in verified.items()}
    payloads={r["query_id"]:parse_prompt(r) for r in rows}
    # Recheck the selected contexts against canonical Stage 7, without labels.
    outputs=contexts.paths_for(method.DEVELOPMENT_ID)[0]
    common.completed(root,outputs[2],outputs[3])
    found=set()
    for row in common.rows(root,outputs[0],contexts.HEADER):
        identity=row["query_id"]
        if identity not in payloads: continue
        require(identity not in found,"Duplicate selected Stage-7 query")
        found.add(identity)
        p=payloads[identity]
        require(row["source_split"]=="validation" and row["rag_partition"]=="development_validation"
                and row["source_dataset"]==p["source_dataset"] and row["classifier_prediction"]==p["classifier_prediction"]
                and row["traffic_profile"]==p["traffic_profile"]
                and common.parse_json(row["evidence_records_json"])==p["trusted_evidence"],"Prepared prompt differs from canonical Stage 7")
    require(found==set(payloads),"Missing selected Stage-7 IDs")
    distributions={}
    for row in rows:
        payload=payloads[row["query_id"]]
        require(row["prompt_text"]==validator.prompt(payload,registry,verified,method=method),"Prepared V5 prompt/claim policy altered")
        matches=validator.qualifying(payload,registry,verified,method=method)
        require(matches,"Diagnostic requires qualifying claim-backed context")
        classes={registry[c]["supported_class"] for c in matches}
        predicted=payload["classifier_prediction"] in classes
        alternative=bool(classes-{payload["classifier_prediction"]})
        counts=distributions.setdefault(row["source_dataset"]+"/"+row["classifier_prediction"],Counter())
        counts.update(queries=1,predicted_support=int(predicted),alternative_support=int(alternative),both=int(predicted and alternative),
                      predicted_only=int(predicted and not alternative),alternative_only=int(alternative and not predicted))
    return rows,payloads,registry,verified,{k:dict(v) for k,v in distributions.items()}


def token_audit(rows,tokenizer):
    counts={r["query_id"]:qwen.prompt_tokens(tokenizer,r["prompt_text"]) for r in rows}
    values=sorted(counts.values())
    require(values,"Empty token audit")
    offending=sorted(k for k,v in counts.items() if v+1024>5120)
    valid=[v for v in values if v+1024<=5120]
    return dict(prompt_count=len(rows),min_prompt_tokens=values[0],median_prompt_tokens=statistics.median(values),
                p95_prompt_tokens=values[math.ceil(.95*len(values))-1],max_prompt_tokens=values[-1],
                reserved_completion_tokens=1024,max_model_len=5120,total_maximum_requested_length=values[-1]+1024,
                over_limit=len(offending),offending_query_ids=offending,
                longest_valid_prompt_margin=5120-1024-max(valid) if valid else None,
                gate_status="FAIL" if offending else "PASS",prompt_tokens=counts,
                tokenizer_path=qwen.MODEL,chat_template="one user message; generation prompt; no truncation; no extra system text")


def configuration(root,rows,audit):
    prepared=list(jsonl(root,INPUT))
    require(rows==prepared,"Prepared 60 prompt records/order changed")
    parent=common.parse_json(method.path(root,PARENT_BASE+CONFIG).read_text(encoding="utf-8"))
    parent_dir=method.path(root,PARENT_BASE+CONFIG).parent
    require({p.name for p in parent_dir.iterdir()}=={CONFIG,JOURNAL},"Parent aborted namespace state changed")
    intents=list(jsonl(root,PARENT_BASE+JOURNAL))
    require(len(intents)==1 and intents[0]["query_id"]=="CIC-VAL-012485"
            and intents[0]["configuration_sha256"]==parent["configuration_sha256"],"Parent aborted request differs")
    require({r["query_id"]:digest(r["prompt_text"]) for r in rows}==parent["prompt_sha256"],"Parent/prepared prompt IDs or text differ")
    require(method.previous.sha(method.path(root,INPUT))==parent["input_sha256"][INPUT],"Prepared prompt artifact hash differs from parent")
    tokenizer={name:method.previous.sha(method.path(root,name)) for name,_,_ in qwen.tokenizer_files(root)}
    inputs={p:method.previous.sha(method.path(root,p)) for p in (INPUT,COVERAGE,method.BASE+"bundle.json")}
    code={p:method.previous.sha(method.path(root,p)) for p in (
        "src/rag/run_v5_diagnostic.py","src/rag/positive_evidence_v5.py","src/rag/audit_qwen_prompt_lengths.py")}
    value=dict(runtime_version=VERSION,development_id=method.DEVELOPMENT_ID,evidence_method=method.VERSION,
               model=MODEL,endpoint=ENDPOINT,max_model_len=5120,max_completion_tokens=1024,temperature=0,seed=42,
               corrections=0,automatic_transport_retries=0,scope=SCOPE,validator="positive_evidence_v5:train-semantic-association-v2",
               prompt_sha256={r["query_id"]:digest(r["prompt_text"]) for r in rows},input_sha256=inputs,
               tokenizer_sha256=tokenizer,code_sha256=code,token_audit=audit,
               tokenizer_packages={k:package_version(k) for k in ("transformers","tokenizers")})
    value.update(diagnostic_scope="claim_association_60",diagnostic_run_id=RUN_ID if BASE==NEW_BASE else "claim_association_60",
                 request_contract_version="v2",max_tokens=1024,
                 parent_aborted_namespace=PARENT_BASE,parent_state="aborted_before_durable_response",
                 parent_completed_count=0,parent_ambiguous_query_id="CIC-VAL-012485",
                 parent_configuration_sha256=parent["configuration_sha256"],
                 request_contract_change="Removed JSON-Schema uniqueItems constraint to match the known-working V4 OpenAI/vLLM request schema; original HTTP 400 body was unavailable, therefore causality is unconfirmed.",
                 selected_query_ids=[r["query_id"] for r in rows],
                 request_schema_sha256={r["query_id"]:digest(common.json_text(schema(r,parse_prompt(r)))) for r in rows})
    value["configuration_sha256"]=digest(common.json_text(value))
    return value


def schema(row,payload):
    citations=dict(type="array",items=dict(type="string"))
    allowed=[e["chunk_id"] for e in payload["trusted_evidence"]]
    if allowed: citations["items"]["enum"]=allowed
    else: citations["maxItems"]=0
    return dict(type="object",additionalProperties=False,required=list(FIELDS),properties=dict(
        query_id=dict(type="string",enum=[row["query_id"]]),
        reasoning_decision=dict(type="string",enum=["SUPPORT_CLASS","ABSTAIN"]),
        evidence_status=dict(type="string",enum=list(common.STATUSES)),
        supported_class=dict(type=["string","null"],enum=[*common.CLASSES[row["source_dataset"]],None]),
        explanation=dict(type="string"),cited_chunk_ids=citations))


def request(row,payload):
    return dict(model=MODEL,messages=[dict(role="user",content=row["prompt_text"])],temperature=0,seed=42,max_tokens=1024,stream=False,
                response_format=dict(type="json_schema",json_schema=dict(name="v5_association_response",strict=True,schema=schema(row,payload))))


class TransportFailure(RuntimeError):
    def __init__(self,status,body,content_type=None):
        self.status=status
        self.content_type=content_type
        self.body=body
        self.error_type="HTTPError"
        self.error_message="Server returned HTTP "+str(status)
        try:
            error=json.loads(body.decode("utf-8"))
            if isinstance(error,dict):
                error=error.get("error",error)
                if isinstance(error,dict):
                    self.error_type=str(error.get("type",self.error_type))
                    self.error_message=str(error.get("message",self.error_message))
        except (ValueError,UnicodeError):
            pass
        super().__init__(f"HTTP {status}: {self.error_type}: {self.error_message}")


def post(body):
    # Deliberately independent of V4. No redirects, proxies or external endpoint.
    import urllib.request
    import urllib.error
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs): raise ValueError("Redirect prohibited")
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    req=urllib.request.Request(ENDPOINT,data=common.json_text(body).encode(),headers={"Content-Type":"application/json"},method="POST")
    try:
        with opener.open(req,timeout=180) as response:
            raw=response.read(8*1024*1024+1)
    except urllib.error.HTTPError as exc:
        try: error_body=exc.read()
        finally: exc.close()
        raise TransportFailure(exc.code,error_body,exc.headers.get("Content-Type") if exc.headers else None) from exc
    require(len(raw)<=8*1024*1024,"Unexpected response size")
    return raw  # Persist these exact HTTP response bytes before parsing/validation.


def append(root,name,value):
    target=method.path(root,BASE+name,False)
    with target.open("a",encoding="utf-8",newline="\n") as stream:
        stream.write(common.json_text(value)+"\n"); stream.flush(); os.fsync(stream.fileno())


def create_json(root,name,value):
    with method.path(root,BASE+name,False).open("x",encoding="utf-8",newline="\n") as stream:
        stream.write(common.json_text(value)+"\n"); stream.flush(); os.fsync(stream.fileno())


def outcomes(raw,payload,registry,verified,expected_tokens,*,source_split="validation"):
    """Only validator-passing model output is accepted. Failures remain visible.

    Effective INSUFFICIENT is an explicit diagnostic disposition, never an
    invented accepted six-field response or a model-native evidence status.
    """
    packet=common.parse_json(base64.b64decode(raw["response_base64"],validate=True).decode("utf-8"))
    require(packet.get("model")==MODEL,"Server model identity differs")
    usage=packet["usage"]
    require(type(usage["prompt_tokens"]) is int and usage["prompt_tokens"]==expected_tokens,"API/local prompt-token mismatch")
    require(type(usage["completion_tokens"]) is int and 0<=usage["completion_tokens"]<=1024
            and usage["total_tokens"]==expected_tokens+usage["completion_tokens"]<=5120,"Token-budget/usage mismatch")
    require(len(packet["choices"])==1,"Expected one response choice")
    choice=packet["choices"][0]; content=choice["message"].get("content")
    result=dict(query_id=raw["query_id"],prompt_sha256=raw["prompt_sha256"],validator_version=VERSION,
                finish_reason=choice.get("finish_reason"),usage=usage,latency_seconds=raw["latency_seconds"],
                server_model=packet["model"],server_response_id=packet.get("id"),system_fingerprint=packet.get("system_fingerprint"),
                validated_response=None,validation_status="REJECTED",effective_evidence_status="INSUFFICIENT",
                validator_fallback_applied="YES",validation_error="")
    if choice.get("finish_reason")!="stop":
        result["validation_error"]="Incomplete generation: "+str(choice.get("finish_reason")); return result
    try:
        require(isinstance(content,str),"Structured content string required")
        value=common.parse_json(content)
    except (ValueError,TypeError) as exc:
        result["validation_error"]=type(exc).__name__+": "+str(exc); return result
    require(not isinstance(value,dict) or value.get("query_id")==raw["query_id"],"Wrong response query_id")
    try:
        validator.validate(value,payload,registry,verified,method=method,source_split=source_split)
    except ValueError as exc:
        result["validation_error"]=type(exc).__name__+": "+str(exc); return result
    linkage=[]
    if value["evidence_status"]!="INSUFFICIENT":
        for identity in common.parse_json(value["explanation"][len("CLAIM_IDS: "):]):
            claim=registry[identity]
            linkage.append(dict(claim_id=identity,source_dataset=claim["source_dataset"],canonical_feature=claim["canonical_feature"],
                                semantic_state=claim["semantic_state"],supported_class=claim["supported_class"],contrasted_class=claim["contrasted_class"],
                                source_type=claim["source_type"],statistics_link=claim["statistics_link"],
                                source_artifacts=claim["source_artifacts"],configuration_sha256=claim["configuration_sha256"],
                                code_sha256=claim["code_sha256"],evidence_quote=method.render(claim),
                                cited_chunk_ids=[e["chunk_id"] for e in payload["trusted_evidence"] if e["document_id"]==identity and e["chunk_id"] in value["cited_chunk_ids"]]))
    result.update(validated_response=value,validation_status="PASS",effective_evidence_status=value["evidence_status"],validator_fallback_applied="NO",
                  validated_claim_linkage=linkage)
    return result


def ledger(root,name,ids):
    if not method.path(root,BASE+name,False).exists(): return {}
    result={}
    for row in jsonl(root,BASE+name):
        identity=row.get("query_id")
        require(identity in ids and identity not in result,"Unknown/duplicate durable query ID")
        result[identity]=row
    return result


def transport_state(root,rows,payloads,config):
    ids={r["query_id"] for r in rows}
    errors={}
    if method.path(root,BASE+ERRORS,False).exists():
        for error in jsonl(root,BASE+ERRORS):
            key=(error["query_id"],error["attempt"])
            require(key not in errors,"Duplicate transport error")
            errors[key]=error
    starts={}
    all_keys=set()
    if method.path(root,BASE+JOURNAL,False).exists():
        for item in jsonl(root,BASE+JOURNAL):
            identity=item["query_id"]
            require(identity in ids,"Unknown request query ID")
            attempt=item.get("attempt",1)
            previous=starts.get(identity)
            require(type(attempt) is int and attempt==(previous.get("attempt",1)+1 if previous else 1),"Invalid/duplicate request attempt")
            if previous:
                require(errors.get((identity,attempt-1),{}).get("http_status")==400,"Ambiguous in-flight request; no automatic reissue")
            require(item["prompt_sha256"]==config["prompt_sha256"][identity]
                    and item["configuration_sha256"]==config["configuration_sha256"],"Request journal differs")
            row=next(r for r in rows if r["query_id"]==identity)
            if "request_payload_sha256" in item:
                require(item["request_payload_sha256"]==digest(common.json_text(request(row,payloads[identity]))),"Request payload differs")
            key=(identity,attempt); all_keys.add(key)
            if key in errors:
                error=errors[key]
                require(all(error[k]==item[k] for k in item) and error["endpoint"]==ENDPOINT
                        and error["transport_state"]=="HTTP_ERROR","Transport error identity differs")
                base64.b64decode(error["response_base64"],validate=True)
            starts[identity]=item
    require(set(errors)<=all_keys,"Orphan transport error")
    return starts,errors


def recover(root,rows,payloads,registry,verified,config,*,write=False,inspect_only=False):
    ids={r["query_id"] for r in rows}
    starts,errors=transport_state(root,rows,payloads,config)
    raw=ledger(root,RAW,ids); completed=ledger(root,VALIDATED,ids)
    require(set(completed)<=set(raw)<=set(starts),"Corrupted raw-first ledger ordering")
    for identity,item in raw.items():
        require(item.get("attempt",1)==starts[identity].get("attempt",1)
                and (identity,item.get("attempt",1)) not in errors,"Raw response/transport error conflict")
        require(item["prompt_sha256"]==config["prompt_sha256"][identity]
                and item["configuration_sha256"]==config["configuration_sha256"],"Durable response identity differs")
        value=outcomes(item,payloads[identity],registry,verified,config["token_audit"]["prompt_tokens"][identity])
        if identity in completed: require(completed[identity]==value,"Saved validation differs from independent revalidation")
        else:
            if write: append(root,VALIDATED,value)
            completed[identity]=value
    pending=set(starts)-set(raw)
    safe={q for q in pending if errors.get((q,starts[q].get("attempt",1)),{}).get("http_status")==400}
    blocked=pending-safe
    if inspect_only:
        return completed,dict(never_attempted_ids=sorted(ids-set(starts)),transport_rejected_reissuable_ids=sorted(safe),
                              ambiguous_or_blocked_ids=sorted(blocked),raw_unvalidated_ids=sorted(set(raw)-set(ledger(root,VALIDATED,ids))))
    require(not blocked,"Ambiguous in-flight request; no automatic reissue; manual review required")
    return completed


def summary(completed):
    return dict(diagnostic_run_id=RUN_ID if BASE==NEW_BASE else "claim_association_60",scope=SCOPE,selected_queries=60,completed=len(completed),remaining=60-len(completed),
                accepted=sum(r["validation_status"]=="PASS" for r in completed.values()),
                rejected=sum(r["validation_status"]!="PASS" for r in completed.values()),
                accepted_evidence_status_counts=dict(Counter(r["validated_response"]["evidence_status"] for r in completed.values() if r["validation_status"]=="PASS")),
                diagnostic_only=True,full_validation_results=False)


@contextmanager
def execution_lock(root):
    lock=method.path(root,BASE+"execution.lock",False)
    handle=lock.open("x",encoding="utf-8")
    try:
        with handle:
            yield
    finally:
        lock.unlink()  # Only our lock; handle is closed first, including Windows.


def execute(root,rows,payloads,registry,verified,config,resume=False):
    require(BASE==NEW_BASE,"Old aborted namespace is read-only; select the explicit v2 diagnostic run ID")
    target=method.path(root,BASE+CONFIG,False)
    directory=target.parent
    if not resume:
        directory.mkdir(parents=True,exist_ok=False)
    require(directory.is_dir(),"Missing diagnostic directory for resume")
    with execution_lock(root):
        try:
            if resume:
                require(common.parse_json(target.read_text(encoding="utf-8"))==config,"Resume configuration/input/tokenizer mismatch")
            else: create_json(root,CONFIG,config)
            completed=recover(root,rows,payloads,registry,verified,config,write=True)
            starts,_=transport_state(root,rows,payloads,config)
            for row in rows:
                identity=row["query_id"]
                if identity in completed: continue
                body=request(row,payloads[identity])
                entry=dict(query_id=identity,prompt_sha256=config["prompt_sha256"][identity],configuration_sha256=config["configuration_sha256"],
                           attempt=starts[identity].get("attempt",1)+1 if identity in starts else 1,
                           request_payload_sha256=digest(common.json_text(body)))
                append(root,JOURNAL,entry)
                start=time.monotonic()
                try:
                    response=post(body)
                except TransportFailure as exc:
                    error=dict(entry,transport_state="HTTP_ERROR",http_status=exc.status,endpoint=ENDPOINT,
                               server_content_type=exc.content_type,timestamp=datetime.now(timezone.utc).isoformat(),diagnostic_run_id=RUN_ID,
                               server_error_type=exc.error_type,server_error_message=exc.error_message,
                               response_base64=base64.b64encode(exc.body).decode("ascii"),latency_seconds=time.monotonic()-start)
                    append(root,ERRORS,error)  # Durable before reporting; never sent to the validator.
                    raise RuntimeError(common.json_text({k:v for k,v in error.items() if k!="response_base64"})) from exc
                raw=dict(entry,response_base64=base64.b64encode(response).decode("ascii"),latency_seconds=time.monotonic()-start)
                append(root,RAW,raw)  # flush/fsync BEFORE parsing, token checks or validator
                value=outcomes(raw,payloads[identity],registry,verified,config["token_audit"]["prompt_tokens"][identity])
                append(root,VALIDATED,value); completed[identity]=value
            result=summary(completed)
            saved=method.path(root,BASE+SUMMARY,False)
            if saved.exists(): require(common.parse_json(saved.read_text(encoding="utf-8"))==result,"Saved summary differs")
            else: create_json(root,SUMMARY,result)
            return result
        finally:
            # execution_lock closes and removes only the lock acquired here.
            pass


def main(argv=None):
    global BASE
    # Scope namespace selection to this invocation, including exceptions/tests.
    previous=BASE
    try: return cli(argv)
    finally: BASE=previous


def cli(argv=None):
    global BASE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path("."))
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--token-audit",action="store_true")
    action.add_argument("--status-only",action="store_true")
    action.add_argument("--execute",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--diagnostic-run-id",choices=("claim_association_60",RUN_ID),default="claim_association_60")
    args=parser.parse_args(argv)
    BASE=NEW_BASE if args.diagnostic_run_id==RUN_ID else PARENT_BASE
    require(not args.execute or BASE==NEW_BASE,"Old aborted namespace is read-only; select the explicit v2 diagnostic run ID")
    require(not args.resume or args.execute,"--resume requires --execute")
    root=args.project_root.resolve(strict=True)
    rows,payloads,registry,verified,distribution=prepare(root)
    before=qwen.tokenizer_files(root)
    tokenizer=qwen.load_tokenizer(root)
    audit=token_audit(rows,tokenizer)
    require(before==qwen.tokenizer_files(root),"Tokenizer changed during audit")
    print(common.json_text(dict(token_audit=audit,selection_distribution=distribution)))
    config=configuration(root,rows,audit)
    if args.token_audit: return 0 if audit["gate_status"]=="PASS" else 2
    require(audit["gate_status"]=="PASS","Prompt budget overflow; no execution or truncation allowed")
    if args.status_only:
        target=method.path(root,BASE+CONFIG,False)
        if target.exists():
            saved=common.parse_json(target.read_text(encoding="utf-8"))
            completed,state=recover(root,rows,payloads,registry,verified,saved,inspect_only=True)
            ready=BASE==NEW_BASE and saved==config and not state["ambiguous_or_blocked_ids"]
            _,errors=transport_state(root,rows,payloads,saved)
            print(common.json_text(dict(configuration_matches=saved==config,ready_for_execution=ready,request_states=state,
                                        ambiguous=len(state["ambiguous_or_blocked_ids"]),transport_failures=len(errors))))
            if not ready:
                print(common.json_text(summary(completed))); return 2
        else:
            require(not target.parent.exists(),"Incomplete diagnostic initialization")
            completed={}
            print(common.json_text(dict(configuration_matches=True,ready_for_execution=BASE==NEW_BASE,ambiguous=0,transport_failures=0)))
        print(common.json_text(summary(completed))); return 0
    result=execute(root,rows,payloads,registry,verified,config,args.resume)
    print(common.json_text(result)); return 0


if __name__=="__main__":
    raise SystemExit(main())
