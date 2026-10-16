"""Synthetic contracts plus read-only historical identity; no model or network."""
import base64
import copy
import json
from pathlib import Path

import pytest

from test_rag_v5_diagnostic import population  # Reuse the existing evidence fixture.
from src.rag import association_response_v2 as contract
from src.rag import run_v5_contract_v2 as runner


def ustc(population,prediction="benign"):
    rows,payloads,registry,verified,_,_=population
    row=next(r for r in rows if r["source_dataset"]=="USTC-TFC2016" and r["classifier_prediction"]==prediction)
    return copy.deepcopy(payloads[row["query_id"]]),copy.deepcopy(registry),copy.deepcopy(verified)


def response(p,state):
    return dict(query_id=p["query_id"],reasoning_decision=state["reasoning_decision"],evidence_status=state["evidence_status"],
                supported_class=state["supported_class"],explanation="Supplied TRAIN associations are noncausal." if state["claim_ids"] else "No qualifying approved claim.",
                claim_ids=list(state["claim_ids"]),cited_chunk_ids=sorted({c for chunks in state["claim_chunks"].values() for c in chunks}))


@pytest.mark.parametrize("pattern,status",[("none","INSUFFICIENT"),("predicted","SUPPORTED"),("alternative","CONTRADICTED"),("both","CONFLICTING")])
def test_admissible_schema_matches_frozen_validator(population,pattern,status):
    p,registry,verified=ustc(population)
    if pattern=="none": p["trusted_evidence"]=[]
    if pattern=="alternative": p["classifier_prediction"]="malware"
    if pattern=="both": p["trusted_evidence"]+=ustc(population,"malware")[0]["trusted_evidence"]
    state=contract.admissible(p,registry,verified); schema=contract.schema(state,p["query_id"])
    assert state["evidence_status"]==status
    assert schema["properties"]["evidence_status"]["enum"]==[status]
    assert "uniqueItems" not in json.dumps(schema)
    if pattern=="none":
        assert schema["properties"]["claim_ids"]["maxItems"]==0
        assert schema["properties"]["cited_chunk_ids"]["maxItems"]==0
        assert schema["properties"]["supported_class"]["enum"]==[None]
    if pattern=="predicted": assert schema["properties"]["supported_class"]["enum"]==["benign"]
    value=response(p,state); before=copy.deepcopy(value); legacy=contract.to_legacy(value)
    contract.runtime.validator.validate(legacy,p,registry,verified,method=contract.runtime.method)
    assert value==before
    for key in ("query_id","reasoning_decision","evidence_status","supported_class","cited_chunk_ids"):
        assert legacy[key]==value[key]
    test_payload=copy.deepcopy(p)
    test_payload.update(query_id=p["query_id"].replace("-VAL-","-TEST-"),source_split="test")
    test_value=dict(legacy,query_id=test_payload["query_id"])
    assert contract.admissible(test_payload,registry,verified,source_split="test")==state
    contract.runtime.validator.validate(test_value,test_payload,registry,verified,
                                        method=contract.runtime.method,source_split="test")
    with pytest.raises(ValueError,match="split/ID mismatch"):
        contract.runtime.validator.validate(test_value,test_payload,registry,verified,method=contract.runtime.method)


def test_multiple_alternatives_conflict_and_unrelated_exclusion(population):
    rows,payloads,registry,verified,_,_=copy.deepcopy(population)
    p=next(p for p in payloads.values() if p["source_dataset"]=="CIC-Darknet2020" and p["classifier_prediction"]=="NonVPN")
    p["classifier_prediction"]="Non-Tor"  # NonVPN-vs-Tor is unrelated.
    assert contract.admissible(p,registry,verified)["claim_ids"]==[]
    p["trusted_evidence"]=[]
    for supported in ("Tor","VPN"):
        source=next(x for x in payloads.values() if x["source_dataset"]=="CIC-Darknet2020" and x["classifier_prediction"]==supported)
        chunk=copy.deepcopy(source["trusted_evidence"][0]);claim=registry[chunk["document_id"]]
        claim.update(contrasted_class="Non-Tor",reference_classes=["Non-Tor"])
        chunk["chunk_text"]=contract.runtime.method.render(claim)
        verified[chunk["chunk_id"]]["chunk_text"]=chunk["chunk_text"]
        p["trusted_evidence"].append(chunk)
    state=contract.admissible(p,registry,verified)
    assert state["pattern"]=="ALTERNATIVE_ONLY" and state["evidence_status"]=="CONFLICTING"
    contract.runtime.validator.validate(contract.to_legacy(response(p,state)),p,registry,verified,method=contract.runtime.method)


@pytest.mark.parametrize("mutation",["dataset","feature","state","unapproved","provenance"])
def test_invalid_supplied_claim_fails_closed(population,mutation):
    p,registry,verified=ustc(population)
    chunk=p["trusted_evidence"][0]; claim=registry[chunk["document_id"]]
    if mutation=="dataset": claim["source_dataset"]="CIC-Darknet2020"
    elif mutation=="feature": claim["canonical_feature"]="not a canonical feature"
    elif mutation=="state": claim.update(semantic_state="LOWER",direction="lower")
    elif mutation=="unapproved": claim["decision"]="REJECTED"
    else: verified[chunk["chunk_id"]]["provenance_status"]="REQUIRES_REVIEW"
    chunk["chunk_text"]=contract.runtime.method.render(claim)
    verified[chunk["chunk_id"]]["chunk_text"]=chunk["chunk_text"]
    with pytest.raises(ValueError): contract.admissible(p,registry,verified)


@pytest.mark.parametrize("ids",[["z","a"],["a","a"],["invented"]])
def test_adapter_never_repairs_claim_ids(population,ids):
    p,registry,verified=ustc(population);state=contract.admissible(p,registry,verified)
    value=response(p,state);value["claim_ids"]=ids
    legacy=contract.to_legacy(value)
    assert json.loads(legacy["explanation"][len("CLAIM_IDS: "):])==ids
    assert value["claim_ids"]==ids
    with pytest.raises(ValueError): contract.runtime.validator.validate(legacy,p,registry,verified,method=contract.runtime.method)


def test_no_claim_adapter_does_not_rewrite_explanation(population):
    p,registry,verified=ustc(population);p["trusted_evidence"]=[]
    value=response(p,contract.admissible(p,registry,verified));value["explanation"]="different model text"
    legacy=contract.to_legacy(value)
    assert legacy["explanation"]=="different model text"
    with pytest.raises(ValueError): contract.runtime.validator.validate(legacy,p,registry,verified,method=contract.runtime.method)


def test_pattern_selection_deterministic_without_model_results(population,monkeypatch):
    values=[]
    for pattern in range(4):
        for n in range(15):
            p,registry,verified=ustc(population);p["query_id"]=f"USTC-VAL-{pattern}-{n}"
            if pattern==0:p["trusted_evidence"]=[]
            if pattern==2:p["classifier_prediction"]="malware"
            if pattern==3:p["trusted_evidence"]+=ustc(population,"malware")[0]["trusted_evidence"]
            values.append(p)
    monkeypatch.setattr(runner.common,"QUERY_COUNTS",{"USTC-TFC2016":60})
    monkeypatch.setattr(runner.runtime,"jsonl",lambda *args:pytest.fail("Previous model results must not be read"))
    first=runner.select_diagnostic(iter(values),registry,verified)
    second=runner.select_diagnostic(iter(reversed(values)),registry,verified)
    assert first==second and len(first)==60
    assert {pattern:sum(s["pattern"]==pattern for p,s in first) for pattern in contract.PATTERNS}==dict.fromkeys(contract.PATTERNS,15)
    values[0]["source_split"]="test"
    with pytest.raises(ValueError):runner.select_diagnostic(values,registry,verified)


def test_labels_excluded_from_runtime(population):
    p,registry,verified=ustc(population);p["ground_truth"]="forbidden"
    with pytest.raises(ValueError): contract.admissible(p,registry,verified)


def test_raw_v2_and_derived_payload_separate_with_resume(tmp_path,population,monkeypatch):
    p,registry,verified=ustc(population);state=contract.admissible(p,registry,verified)
    row=dict(query_id=p["query_id"],source_dataset=p["source_dataset"],prompt_text=contract.prompt(p,registry,verified,state),response_schema=contract.schema(state,p["query_id"]))
    value=response(p,state)
    packet=dict(model=runner.runtime.MODEL,usage=dict(prompt_tokens=100,completion_tokens=10,total_tokens=110),choices=[dict(finish_reason="stop",message=dict(content=json.dumps(value)))])
    body=json.dumps(packet).encode();calls=[]
    def post(request):calls.append(request);return body
    monkeypatch.setattr(runner.runtime,"post",post)
    config=dict(configuration_sha256="synthetic",prompt_sha256={p["query_id"]:runner.runtime.digest(row["prompt_text"])},token_audit=dict(prompt_tokens={p["query_id"]:100}))
    original=runner.LEGACY_OUTCOMES
    def interrupt(*args):raise KeyboardInterrupt()
    monkeypatch.setattr(runner,"LEGACY_OUTCOMES",interrupt)
    with runner.runtime_scope(runner.DIAGNOSTIC,1):
        with pytest.raises(KeyboardInterrupt):runner.runtime.execute(tmp_path,[row],{p["query_id"]:p},registry,verified,config)
        raw=list(runner.runtime.jsonl(tmp_path,runner.runtime.BASE+runner.runtime.RAW))
        assert base64.b64decode(raw[0]["response_base64"])==body
        monkeypatch.setattr(runner,"LEGACY_OUTCOMES",original)
        result=runner.runtime.execute(tmp_path,[row],{p["query_id"]:p},registry,verified,config,resume=True)
        assert result["accepted"]==1 and len(calls)==1
        saved=list(runner.runtime.jsonl(tmp_path,runner.runtime.BASE+runner.runtime.VALIDATED))[0]
        assert saved["raw_v2_model_response"]==value
        assert saved["model_explanation"]==value["explanation"]
        assert saved["derived_validator_payload"]==contract.to_legacy(value)
        assert saved["raw_v2_response_sha256"]==runner.runtime.digest(runner.common.json_text(value))
        assert saved["derived_validator_payload_sha256"]==runner.runtime.digest(runner.common.json_text(contract.to_legacy(value)))


def test_contract_namespaces_are_new():
    assert len(set(runner.BASES.values()))==2
    assert runner.runtime.NEW_BASE not in runner.BASES.values()
    assert runner.comparison.BASE not in runner.BASES.values()
    assert all("/v4/" not in p and "/test/" not in p for p in runner.BASES.values())
    root=Path(__file__).resolve().parents[1]
    historical=json.loads((root/runner.BASES[runner.COMPARISON]/"configuration.json").read_text(encoding="utf-8"))
    assert contract.FROZEN_VALIDATOR_SHA256=="9b47110801b0bd9a6a689f4d419ff28d99a257429e8d0724653d722a6a78caed"
    assert historical["code_sha256"]["src/rag/positive_evidence_v5.py"]==contract.FROZEN_VALIDATOR_SHA256


@pytest.mark.parametrize("mode",["--status-only","--token-audit"])
def test_offline_modes_no_writes_or_requests(tmp_path,monkeypatch,mode):
    monkeypatch.setattr(runner,"prepare",lambda *args:([],{},{},{},{},{}))
    monkeypatch.setattr(runner,"input_hashes",lambda *args:{})
    monkeypatch.setattr(runner,"configuration",lambda *args:{})
    monkeypatch.setattr(runner.runtime.qwen,"tokenizer_files",lambda root:[])
    monkeypatch.setattr(runner.runtime.qwen,"load_tokenizer",lambda root:object())
    monkeypatch.setattr(runner.runtime,"token_audit",lambda *args:dict(gate_status="PASS"))
    monkeypatch.setattr(runner.runtime,"post",lambda *args:pytest.fail("Unexpected request"))
    assert runner.main(["--project-root",str(tmp_path),"--run-id",runner.DIAGNOSTIC,mode])==0
    assert list(tmp_path.iterdir())==[]


def test_future_cohort_reads_only_frozen_ids(tmp_path,monkeypatch):
    ids=[f"USTC-VAL-{n:06d}" for n in range(4205)]
    path=tmp_path/runner.comparison.historical.FREEZE
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(completed_validation_ids=ids)),encoding="utf-8")
    monkeypatch.setattr(runner.comparison,"FREEZE_SHA256",runner.runtime.method.previous.sha(path))
    monkeypatch.setattr(runner.runtime,"jsonl",lambda *args:pytest.fail("Model responses must not be read"))
    assert runner.frozen_ids(tmp_path)==ids


def test_contextual_evidence_retained_in_prompt(population):
    rows,payloads,registry,verified,_,_=population
    p=copy.deepcopy(next(p for p in payloads.values() if p["source_dataset"]=="CIC-Darknet2020" and p["classifier_prediction"]=="NonVPN"))
    p["classifier_prediction"]="Non-Tor"
    state=contract.admissible(p,registry,verified)
    text=contract.prompt(p,registry,verified,state)
    assert state["claim_ids"]==[]
    assert p["trusted_evidence"][0]["document_id"] in text.split("\nCONTEXTUAL_ONLY_CLAIM_IDS:\n")[1]
    assert json.loads(text.split("\nINPUT_JSON:\n")[1])==p
