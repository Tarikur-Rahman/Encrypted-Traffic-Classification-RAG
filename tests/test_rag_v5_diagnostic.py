"""Offline synthetic diagnostic tests; no traffic files, server or LLM."""
import base64
import copy
import json
import io
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from src.rag import run_v5_diagnostic as runner
from src.rag import positive_evidence_v5 as gate


@pytest.fixture(autouse=True)
def fresh_run_namespace(monkeypatch):
    monkeypatch.setattr(runner,"BASE",runner.NEW_BASE)


@pytest.fixture
def population():
    rows=[]; payloads={}; registry={}; verified={}; values={}
    for dataset,classes in runner.common.CLASSES.items():
        feature=sorted(runner.method.corpus.CANONICAL_DECISION_FEATURES[dataset])[0]
        for c,prediction in enumerate(classes):
            identity=f"claim-{dataset}-{c}"
            claim=dict(claim_id=identity,evidence_method_version=runner.method.VERSION,source_dataset=dataset,
                       supported_class=prediction,contrasted_class=classes[(c+1)%len(classes)],reference_classes=[classes[(c+1)%len(classes)]],
                       canonical_feature=feature,semantic_state="HIGHER",direction="higher",source_type=runner.method.SOURCE_TYPE,
                       decision="APPROVED",target_prevalence=.9,contrast_prevalence=.1,effect_size=1.,bootstrap_interval=[.8,.8])
            claim.update(statistics_link=runner.method.BASE+"statistics.json#"+identity,source_artifacts={"synthetic/train.csv":"a"*64},
                         configuration_sha256="b"*64,code_sha256="c"*64)
            registry[identity]=claim
            chunk=dict(chunk_id=identity+"-chunk",document_id=identity,chunk_text=runner.method.render(claim),decision_claim_ids_json=json.dumps([identity]))
            verified[chunk["chunk_id"]]=dict(chunk,provenance_status="VERIFIED",authorized_for_downstream="YES",quality_status="KEEP")
            for n in range(10):
                query=f"{dataset}-VAL-{c}-{n}"
                payload=dict(query_id=query,source_dataset=dataset,source_split="validation",classifier_prediction=prediction,
                             traffic_profile=f"Traffic. Observed traffic profile relative to the training distribution: {feature} is high. Retrieve trusted cybersecurity evidence relevant to the predicted class and these observed traffic characteristics.",trusted_evidence=[chunk])
                row=dict(query_id=query,source_dataset=dataset,source_split="validation",classifier_prediction=prediction,
                         diagnostic_scope=runner.SCOPE,token_audit_status="NOT_TOKEN_AUDITED",prompt_text=gate.prompt(payload,registry,verified,method=runner.method))
                rows.append(row); payloads[query]=payload
                values[query]=dict(query_id=query,reasoning_decision="SUPPORT_CLASS",evidence_status="SUPPORTED",supported_class=prediction,
                                   explanation="CLAIM_IDS: "+json.dumps([identity]),cited_chunk_ids=[chunk["chunk_id"]])
    rows.sort(key=lambda r:(r["source_dataset"],r["classifier_prediction"],runner.digest("42|"+r["query_id"]),r["query_id"]))
    config=dict(configuration_sha256="configuration",prompt_sha256={r["query_id"]:runner.digest(r["prompt_text"]) for r in rows},
                token_audit=dict(prompt_tokens={r["query_id"]:200 for r in rows}))
    return rows,payloads,registry,verified,values,config


def packet(value,finish="stop",tokens=200):
    return json.dumps(dict(model=runner.MODEL,id="synthetic",usage=dict(prompt_tokens=tokens,completion_tokens=10,total_tokens=tokens+10),
                           choices=[dict(finish_reason=finish,message=dict(content=json.dumps(value)))])).encode()


def raw_record(row,response,config):
    return dict(query_id=row["query_id"],prompt_sha256=config["prompt_sha256"][row["query_id"]],configuration_sha256=config["configuration_sha256"],
                response_base64=base64.b64encode(response).decode(),latency_seconds=1.)


def test_exact_sixty_stratified_deterministic_selection(population):
    rows,*_=population
    runner.validate_population(rows)
    assert len(rows)==60
    with pytest.raises(ValueError): runner.validate_population(rows[:-1])
    with pytest.raises(ValueError): runner.validate_population(rows[::-1])
    with pytest.raises(ValueError): runner.validate_population(rows[:-1]+[rows[0]])


@pytest.mark.parametrize("field",["ground_truth","true_class_label","classifier_correct"])
def test_no_evaluation_fields(population,field):
    rows,payloads,registry,verified,_,_=population
    row=copy.deepcopy(rows[0]); row[field]="forbidden"
    with pytest.raises(ValueError): runner.parse_prompt(row)
    payload=copy.deepcopy(payloads[rows[0]["query_id"]]); payload[field]="forbidden"
    with pytest.raises(ValueError): gate.qualifying(payload,registry,verified,method=runner.method)


def test_test_scope_rejected(population):
    rows,*_=population
    changed=copy.deepcopy(rows)
    changed[0]["source_split"]="test"
    with pytest.raises(ValueError): runner.validate_population(changed)


def test_exact_template_counting_and_overflow(population):
    rows,*_=population
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            assert messages[0]["role"]=="user" and len(messages)==1
            assert kwargs==dict(tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors=None,truncation=False,padding=False)
            return {"input_ids":[1]*4096,"attention_mask":[1]*4096}
    audit=runner.token_audit(rows,Tokenizer())
    assert audit["gate_status"]=="PASS" and audit["max_prompt_tokens"]==4096 and audit["longest_valid_prompt_margin"]==0
    class Overflow(Tokenizer):
        def apply_chat_template(self,*a,**kw): return {"input_ids":[1]*4097}
    audit=runner.token_audit(rows,Overflow())
    assert audit["gate_status"]=="FAIL" and audit["over_limit"]==60 and len(audit["offending_query_ids"])==60


def test_raw_first_resume_does_not_reissue(tmp_path,monkeypatch,population):
    rows,payloads,registry,verified,values,config=population
    requests=[]
    def post(body):
        query=json.loads(body["messages"][0]["content"].split("\nINPUT_JSON:\n")[1])["query_id"]
        requests.append(query)
        assert body["temperature"]==0 and body["seed"]==42 and body["max_tokens"]==1024
        return packet(values[query])
    monkeypatch.setattr(runner,"post",post)
    real=runner.outcomes
    def interrupted(*args): raise KeyboardInterrupt()
    monkeypatch.setattr(runner,"outcomes",interrupted)
    with pytest.raises(KeyboardInterrupt): runner.execute(tmp_path,rows,payloads,registry,verified,config)
    raw=list(runner.jsonl(tmp_path,runner.BASE+runner.RAW))
    assert len(raw)==1 and base64.b64decode(raw[0]["response_base64"])==packet(values[rows[0]["query_id"]])
    assert not (tmp_path/runner.BASE/runner.VALIDATED).exists()
    assert not (tmp_path/runner.BASE/"execution.lock").exists()
    monkeypatch.setattr(runner,"outcomes",real)
    result=runner.execute(tmp_path,rows,payloads,registry,verified,config,resume=True)
    assert result["completed"]==60 and result["accepted"]==60
    assert len(requests)==len(set(requests))==60
    assert runner.execute(tmp_path,rows,payloads,registry,verified,config,resume=True)==result
    assert len(requests)==60
    with pytest.raises(FileExistsError): runner.execute(tmp_path,rows,payloads,registry,verified,config)


def test_ambiguous_transport_never_automatically_reissued(tmp_path,monkeypatch,population):
    rows,payloads,registry,verified,_,config=population
    def fail(body): raise ConnectionError("synthetic failure")
    monkeypatch.setattr(runner,"post",fail)
    with pytest.raises(ConnectionError): runner.execute(tmp_path,rows,payloads,registry,verified,config)
    with pytest.raises(ValueError,match="Ambiguous in-flight"):
        runner.execute(tmp_path,rows,payloads,registry,verified,config,resume=True)


def test_semantic_failure_is_not_an_accepted_rewrite(population):
    rows,payloads,registry,verified,values,config=population
    row=rows[0]; identity=row["query_id"]
    value=dict(values[identity],explanation='CLAIM_IDS: ["invented"]')
    result=runner.outcomes(raw_record(row,packet(value),config),payloads[identity],registry,verified,200)
    assert result["validation_status"]=="REJECTED" and result["validated_response"] is None
    assert result["effective_evidence_status"]=="INSUFFICIENT" and result["validator_fallback_applied"]=="YES"
    assert result["validation_error"]


def test_pairwise_nonvpn_not_contradicted_by_vpn_vs_nontor(population):
    rows,payloads,registry,verified,values,config=population
    row=next(r for r in rows if r["classifier_prediction"]=="VPN")
    identity=row["query_id"]
    payload=copy.deepcopy(payloads[identity]); payload["classifier_prediction"]="NonVPN"
    # Fixture VPN contrasts Non-Tor, not NonVPN.
    value=dict(values[identity],evidence_status="CONTRADICTED")
    with pytest.raises(ValueError): gate.validate(value,payload,registry,verified,method=runner.method)
    value.update(reasoning_decision="ABSTAIN",evidence_status="INSUFFICIENT",supported_class=None,
                 explanation="No qualifying approved claim.",cited_chunk_ids=[])
    gate.validate(value,payload,registry,verified,method=runner.method)


def test_tor_vs_vpn_not_about_nontor(population):
    rows,payloads,registry,verified,values,_=population
    row=next(r for r in rows if r["classifier_prediction"]=="Tor")
    identity=row["query_id"]
    payload=copy.deepcopy(payloads[identity]); payload["classifier_prediction"]="Non-Tor"
    with pytest.raises(ValueError):
        gate.validate(dict(values[identity],evidence_status="CONTRADICTED"),payload,registry,verified,method=runner.method)


def test_api_token_mismatch_is_hard_failure(population):
    rows,payloads,registry,verified,values,config=population
    row=rows[0]
    with pytest.raises(ValueError,match="prompt-token mismatch"):
        runner.outcomes(raw_record(row,packet(values[row["query_id"]],tokens=201),config),payloads[row["query_id"]],registry,verified,200)


def test_accepted_response_retains_exact_internal_claim_provenance(population):
    rows,payloads,registry,verified,values,config=population
    row=rows[0]; identity=row["query_id"]
    result=runner.outcomes(raw_record(row,packet(values[identity]),config),payloads[identity],registry,verified,200)
    assert result["validated_response"]==values[identity]
    assert result["validation_status"]=="PASS"
    link=result["validated_claim_linkage"][0]
    claim=registry[link["claim_id"]]
    assert link["statistics_link"]==runner.method.BASE+"statistics.json#"+claim["claim_id"]
    assert link["evidence_quote"]==runner.method.render(claim)
    assert link["source_artifacts"]==claim["source_artifacts"]
    assert link["cited_chunk_ids"]==values[identity]["cited_chunk_ids"]


def test_wrong_query_is_hard_failure(population):
    rows,payloads,registry,verified,values,config=population
    row=rows[0]; identity=row["query_id"]
    with pytest.raises(ValueError,match="Wrong response query_id"):
        runner.outcomes(raw_record(row,packet(dict(values[identity],query_id="wrong")),config),payloads[identity],registry,verified,200)


def test_incomplete_generation_is_not_accepted(population):
    rows,payloads,registry,verified,values,config=population
    row=rows[0]; identity=row["query_id"]
    result=runner.outcomes(raw_record(row,packet(values[identity],finish="length"),config),payloads[identity],registry,verified,200)
    assert result["validated_response"] is None
    assert result["validation_error"]=="Incomplete generation: length"
    assert result["effective_evidence_status"]=="INSUFFICIENT"


def test_duplicate_ledger_rejected(tmp_path,population):
    rows,_,_,_,_,config=population
    (tmp_path/runner.BASE).mkdir(parents=True)
    row=dict(query_id=rows[0]["query_id"])
    runner.append(tmp_path,runner.JOURNAL,row); runner.append(tmp_path,runner.JOURNAL,row)
    with pytest.raises(ValueError,match="duplicate"):
        runner.ledger(tmp_path,runner.JOURNAL,{r["query_id"] for r in rows})


@pytest.mark.parametrize("mode",["--status-only","--token-audit"])
def test_offline_modes_no_requests_or_writes(tmp_path,monkeypatch,population,mode):
    rows,payloads,registry,verified,_,config=population
    monkeypatch.setattr(runner,"prepare",lambda root:(rows,payloads,registry,verified,{}))
    monkeypatch.setattr(runner.qwen,"tokenizer_files",lambda root:[])
    monkeypatch.setattr(runner.qwen,"load_tokenizer",lambda root:object())
    monkeypatch.setattr(runner,"token_audit",lambda *args:dict(gate_status="PASS"))
    monkeypatch.setattr(runner,"configuration",lambda *args:config)
    monkeypatch.setattr(runner,"post",lambda *args:pytest.fail("Unexpected request"))
    assert runner.main(["--project-root",str(tmp_path),"--diagnostic-run-id",runner.RUN_ID,mode])==0
    assert list(tmp_path.iterdir())==[]


def test_separate_namespace_and_no_v4_runtime_import():
    assert runner.BASE.startswith("results/rag/v5f_validation/stage08_diagnostic/")
    assert "/v4/" not in runner.BASE and "/test/" not in runner.BASE
    assert "llm_v3_runtime" not in Path(runner.__file__).read_text(encoding="utf-8")


def test_working_local_request_contract(population):
    rows,payloads,*_=population
    row=rows[0]; payload=payloads[row["query_id"]]
    body=runner.request(row,payload)
    assert runner.ENDPOINT=="http://127.0.0.1:8000/v1/chat/completions"
    assert body==dict(model="provenance-qwen3-4b",messages=[dict(role="user",content=row["prompt_text"])],
                     temperature=0,seed=42,max_tokens=1024,stream=False,
                     response_format=dict(type="json_schema",json_schema=dict(name="v5_association_response",strict=True,
                                                                              schema=runner.schema(row,payload))))
    assert body["response_format"]["json_schema"]["schema"]["properties"]["cited_chunk_ids"]==dict(
        type="array",items=dict(type="string",enum=[e["chunk_id"] for e in payload["trusted_evidence"]]))
    empty=dict(payload,trusted_evidence=[])
    assert runner.schema(row,empty)["properties"]["cited_chunk_ids"]==dict(type="array",items=dict(type="string"),maxItems=0)


def test_http_error_body_captured_by_transport(monkeypatch,population):
    rows,payloads,*_=population
    body=b'{"error":{"type":"BadRequestError","message":"synthetic schema rejection"}}'
    class Opener:
        def open(self,request,timeout):
            assert request.full_url==runner.ENDPOINT and timeout==180
            raise urllib.error.HTTPError(request.full_url,400,"Bad Request",{},io.BytesIO(body))
    monkeypatch.setattr(urllib.request,"build_opener",lambda *args:Opener())
    with pytest.raises(runner.TransportFailure) as caught:
        runner.post(runner.request(rows[0],payloads[rows[0]["query_id"]]))
    assert caught.value.body==body and caught.value.status==400
    assert caught.value.error_type=="BadRequestError"
    assert caught.value.error_message=="synthetic schema rejection"


def test_durable_http_400_explicit_resume_not_validator_result(tmp_path,monkeypatch,population):
    rows,payloads,registry,verified,values,config=population
    calls=[]
    error_body=b'{"error":{"type":"BadRequestError","message":"synthetic rejection"}}'
    def rejected(body):
        calls.append(body)
        raise runner.TransportFailure(400,error_body)
    monkeypatch.setattr(runner,"post",rejected)
    with pytest.raises(RuntimeError,match="synthetic rejection"):
        runner.execute(tmp_path,rows,payloads,registry,verified,config)
    assert len(calls)==1
    assert not (tmp_path/runner.BASE/runner.RAW).exists()
    assert not (tmp_path/runner.BASE/runner.VALIDATED).exists()
    error=list(runner.jsonl(tmp_path,runner.BASE+runner.ERRORS))[0]
    assert base64.b64decode(error["response_base64"])==error_body
    assert error["request_payload_sha256"]==runner.digest(runner.common.json_text(calls[0]))
    assert error["diagnostic_run_id"]==runner.RUN_ID
    assert error["timestamp"] and "server_content_type" in error
    completed,state=runner.recover(tmp_path,rows,payloads,registry,verified,config,inspect_only=True)
    assert runner.summary(completed)["completed"]==0
    assert state["transport_rejected_reissuable_ids"]==[rows[0]["query_id"]]
    assert state["ambiguous_or_blocked_ids"]==[] and len(state["never_attempted_ids"])==59
    def success(body):
        calls.append(body)
        identity=json.loads(body["messages"][0]["content"].split("\nINPUT_JSON:\n")[1])["query_id"]
        return packet(values[identity])
    monkeypatch.setattr(runner,"post",success)
    assert runner.execute(tmp_path,rows,payloads,registry,verified,config,resume=True)["completed"]==60
    assert len(calls)==61
    runner.execute(tmp_path,rows,payloads,registry,verified,config,resume=True)
    assert len(calls)==61


def test_http_500_does_not_authorize_reissue(tmp_path,monkeypatch,population):
    rows,payloads,registry,verified,_,config=population
    def fail(body): raise runner.TransportFailure(500,b'{}')
    monkeypatch.setattr(runner,"post",fail)
    with pytest.raises(RuntimeError): runner.execute(tmp_path,rows,payloads,registry,verified,config)
    with pytest.raises(ValueError,match="Ambiguous"):
        runner.execute(tmp_path,rows,payloads,registry,verified,config,resume=True)


def test_legacy_unrecorded_http_failure_stays_ambiguous_read_only(tmp_path,monkeypatch,population,capsys):
    rows,payloads,registry,verified,_,config=population
    (tmp_path/runner.BASE).mkdir(parents=True)
    runner.create_json(tmp_path,runner.CONFIG,config)
    identity=rows[0]["query_id"]
    runner.append(tmp_path,runner.JOURNAL,dict(query_id=identity,prompt_sha256=config["prompt_sha256"][identity],
                                             configuration_sha256=config["configuration_sha256"]))
    before={p.name:p.read_bytes() for p in (tmp_path/runner.BASE).iterdir()}
    monkeypatch.setattr(runner,"prepare",lambda root:(rows,payloads,registry,verified,{}))
    monkeypatch.setattr(runner.qwen,"tokenizer_files",lambda root:[])
    monkeypatch.setattr(runner.qwen,"load_tokenizer",lambda root:object())
    monkeypatch.setattr(runner,"token_audit",lambda *args:dict(gate_status="PASS"))
    monkeypatch.setattr(runner,"configuration",lambda *args:dict(config,configuration_sha256="changed-code"))
    monkeypatch.setattr(runner,"post",lambda *args:pytest.fail("Unexpected request"))
    assert runner.main(["--project-root",str(tmp_path),"--diagnostic-run-id",runner.RUN_ID,"--status-only"])==2
    output=capsys.readouterr().out
    assert '"ready_for_execution":false' in output
    assert identity in output
    assert before=={p.name:p.read_bytes() for p in (tmp_path/runner.BASE).iterdir()}


def test_old_namespace_never_executed(tmp_path,monkeypatch,population):
    rows,payloads,registry,verified,_,config=population
    monkeypatch.setattr(runner,"BASE",runner.PARENT_BASE)
    with pytest.raises(ValueError,match="read-only"):
        runner.execute(tmp_path,rows,payloads,registry,verified,config)
    with pytest.raises(ValueError,match="read-only"):
        runner.main(["--project-root",str(tmp_path),"--execute"])
    assert list(tmp_path.iterdir())==[]


def test_unknown_run_rejected():
    with pytest.raises(SystemExit): runner.main(["--diagnostic-run-id","unknown","--status-only"])


def test_duplicate_citation_still_rejected(population):
    rows,payloads,registry,verified,values,_=population
    identity=rows[0]["query_id"]
    value=dict(values[identity],cited_chunk_ids=values[identity]["cited_chunk_ids"]*2)
    with pytest.raises(ValueError,match="Duplicate citation"):
        gate.validate(value,payloads[identity],registry,verified,method=runner.method)


def test_configuration_lineage_and_exact_prepared_order(tmp_path,monkeypatch,population):
    rows,_,_,_,_,_=population
    prompt=tmp_path/runner.INPUT
    prompt.parent.mkdir(parents=True)
    prompt.write_text("".join(runner.common.json_text(r)+"\n" for r in rows),encoding="utf-8")
    parent=tmp_path/runner.PARENT_BASE
    parent.mkdir(parents=True)
    old=dict(configuration_sha256="old-config",prompt_sha256={r["query_id"]:runner.digest(r["prompt_text"]) for r in rows},
             input_sha256={runner.INPUT:runner.method.previous.sha(prompt)})
    (parent/runner.CONFIG).write_text(json.dumps(old),encoding="utf-8")
    (parent/runner.JOURNAL).write_text(json.dumps(dict(query_id="CIC-VAL-012485",configuration_sha256="old-config"))+"\n",encoding="utf-8")
    for relative in (runner.COVERAGE,runner.method.BASE+"bundle.json","src/rag/run_v5_diagnostic.py",
                     "src/rag/positive_evidence_v5.py","src/rag/audit_qwen_prompt_lengths.py"):
        path=tmp_path/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text("synthetic",encoding="utf-8")
    monkeypatch.setattr(runner.qwen,"tokenizer_files",lambda root:[])
    monkeypatch.setattr(runner,"package_version",lambda name:"synthetic")
    before={p.name:p.read_bytes() for p in parent.iterdir()}
    config=runner.configuration(tmp_path,rows,{"gate_status":"PASS"})
    assert config["diagnostic_run_id"]==runner.RUN_ID and config["request_contract_version"]=="v2"
    assert config["parent_aborted_namespace"]==runner.PARENT_BASE
    assert config["parent_state"]=="aborted_before_durable_response"
    assert config["parent_completed_count"]==0 and config["parent_ambiguous_query_id"]=="CIC-VAL-012485"
    assert config["configuration_sha256"]!="old-config"
    assert config["selected_query_ids"]==[r["query_id"] for r in rows]
    assert len(config["request_schema_sha256"])==60
    with pytest.raises(ValueError,match="order changed"):
        runner.configuration(tmp_path,rows[::-1],{})
    changed=copy.deepcopy(rows); changed[0]["prompt_text"]+="changed"
    prompt.write_text("".join(runner.common.json_text(r)+"\n" for r in changed),encoding="utf-8")
    with pytest.raises(ValueError,match="differ"):
        runner.configuration(tmp_path,changed,{})
    assert before=={p.name:p.read_bytes() for p in parent.iterdir()}
