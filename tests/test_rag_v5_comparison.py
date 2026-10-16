"""Synthetic fixed-cohort tests; no model, tokenizer, network or traffic data."""
import json
from pathlib import Path

import pytest

from src.rag import run_v5_comparison as comparison


def write(root,path,value,lines=False):
    target=root/path
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text("".join(json.dumps(r)+"\n" for r in value) if lines else json.dumps(value),encoding="utf-8")


@pytest.fixture
def cohort(tmp_path,monkeypatch):
    ids=[f"CIC-VAL-{i:06d}" for i in range(4205)]
    freeze=dict(completed_validation_ids=ids)
    write(tmp_path,comparison.historical.FREEZE,freeze)
    monkeypatch.setattr(comparison,"FREEZE_SHA256",comparison.runtime.method.previous.sha(tmp_path/comparison.historical.FREEZE))
    rows=[dict(query_id=q,source_split="validation",source_dataset="CIC-Darknet2020",classifier_prediction="Tor",response_valid=True) for q in ids]
    for source,part in zip(comparison.historical.SOURCES,(rows[:1200],rows[1200:1206],rows[1206:])):
        write(tmp_path,source,part,True)
    return ids,rows


def test_exact_historical_population_and_order(tmp_path,cohort):
    ids,rows=cohort
    ordered,population=comparison.historical_population(tmp_path)
    assert ordered==ids and len(population)==4205
    assert population[ids[0]]==("CIC-Darknet2020","Tor")


@pytest.mark.parametrize("mutation",["duplicate","missing","test"])
def test_invalid_historical_sources_fail_closed(tmp_path,cohort,mutation):
    ids,rows=cohort
    last=rows[1206:]
    if mutation=="duplicate": last.append(rows[0])
    elif mutation=="missing": last.pop()
    else: last[0]=dict(last[0],source_split="test")
    write(tmp_path,comparison.historical.SOURCES[-1],last,True)
    with pytest.raises(ValueError): comparison.historical_population(tmp_path)


def test_changed_freeze_fails(tmp_path,cohort):
    write(tmp_path,comparison.historical.FREEZE,dict(completed_validation_ids=cohort[0][::-1]))
    with pytest.raises(ValueError,match="freeze hash"):
        comparison.historical_population(tmp_path)


def test_context_linkage_includes_zero_claim_rows(tmp_path,cohort,monkeypatch):
    ids,_=cohort
    ordered,population=comparison.historical_population(tmp_path)
    rows=[dict(query_id=q,source_dataset="CIC-Darknet2020",classifier_prediction="Tor",source_split="validation",
               rag_partition="development_validation",traffic_profile="synthetic",evidence_records_json="[]",
               ground_truth="must not propagate",classifier_confidence="must not propagate") for q in reversed(ids)]
    monkeypatch.setattr(comparison.common,"completed",lambda *args:None)
    monkeypatch.setattr(comparison.common,"rows",lambda *args:iter(rows))
    selected=comparison.selected_contexts(tmp_path,ordered,population)
    assert set(selected)==set(ids)
    assert all(set(p)=={"query_id","source_dataset","classifier_prediction","source_split","traffic_profile","trusted_evidence"} for p in selected.values())
    rows[0]["classifier_prediction"]="VPN"
    with pytest.raises(ValueError,match="prediction/dataset"):
        comparison.selected_contexts(tmp_path,ordered,population)


def test_runtime_scope_restores_diagnostic_identity_on_failure():
    runtime=comparison.runtime
    original=(runtime.BASE,runtime.NEW_BASE,runtime.RUN_ID,runtime.summary)
    with pytest.raises(RuntimeError):
        with comparison.runtime_scope():
            assert runtime.BASE==comparison.BASE and runtime.RUN_ID==comparison.RUN_ID
            assert runtime.summary({})["remaining"]==4205
            raise RuntimeError("interrupted")
    assert (runtime.BASE,runtime.NEW_BASE,runtime.RUN_ID,runtime.summary)==original


@pytest.mark.parametrize("mode",["--token-audit","--status-only"])
def test_offline_modes_no_writes_or_requests(tmp_path,monkeypatch,mode):
    monkeypatch.setattr(comparison,"prepare",lambda root:([],{},{},{},{}))
    monkeypatch.setattr(comparison,"input_hashes",lambda root:{})
    monkeypatch.setattr(comparison,"configuration",lambda *args:{})
    monkeypatch.setattr(comparison.runtime.qwen,"tokenizer_files",lambda root:[])
    monkeypatch.setattr(comparison.runtime.qwen,"load_tokenizer",lambda root:object())
    monkeypatch.setattr(comparison.runtime,"token_audit",lambda *args:dict(gate_status="PASS"))
    monkeypatch.setattr(comparison.runtime,"post",lambda *args:pytest.fail("Unexpected request"))
    assert comparison.main(["--project-root",str(tmp_path),mode])==0
    assert list(tmp_path.iterdir())==[]


def test_separate_comparative_namespace():
    assert comparison.BASE!=comparison.runtime.NEW_BASE
    assert "stage08_comparison/v5f_historical4205_comparison/" in comparison.BASE
    assert comparison.summary({})["evaluation_scope"]=="fixed_historical_v4_validation_cohort_not_unseen_validation"


def test_comparison_reuses_raw_first_resume(tmp_path,monkeypatch):
    runtime=comparison.runtime
    monkeypatch.setattr(comparison,"COUNT",2)
    rows=[dict(query_id=f"CIC-VAL-{i:06d}",prompt_text="synthetic") for i in range(2)]
    payloads={r["query_id"]:{} for r in rows}
    config=dict(configuration_sha256="synthetic",prompt_sha256={r["query_id"]:runtime.digest(r["prompt_text"]) for r in rows},
                token_audit=dict(prompt_tokens={r["query_id"]:1 for r in rows}))
    calls=[]
    monkeypatch.setattr(runtime,"request",lambda row,payload:dict(query_id=row["query_id"]))
    def post(body):
        calls.append(body["query_id"])
        return b"durable synthetic bytes"
    monkeypatch.setattr(runtime,"post",post)
    def interrupted(*args): raise KeyboardInterrupt()
    monkeypatch.setattr(runtime,"outcomes",interrupted)
    with comparison.runtime_scope():
        with pytest.raises(KeyboardInterrupt): runtime.execute(tmp_path,rows,payloads,{},{},config)
        assert (tmp_path/comparison.BASE/runtime.RAW).exists()
        assert not (tmp_path/comparison.BASE/runtime.VALIDATED).exists()
        monkeypatch.setattr(runtime,"outcomes",lambda raw,*args:dict(query_id=raw["query_id"],validation_status="REJECTED",validated_response=None))
        result=runtime.execute(tmp_path,rows,payloads,{},{},config,resume=True)
        assert result["completed"]==2 and result["remaining"]==0
        runtime.execute(tmp_path,rows,payloads,{},{},config,resume=True)
    assert calls==[r["query_id"] for r in rows]
    assert not (tmp_path/runtime.NEW_BASE).exists()
