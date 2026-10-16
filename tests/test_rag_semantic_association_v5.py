"""Synthetic tests only; no real traffic, held-out labels, network or models."""
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.rag import build_semantic_association_v5 as method
from src.rag import positive_evidence_v5 as gate
from src.rag import run_provenance_aware_reranking as rerank
from src.rag.run_rag_retrieval import semantic_bin
from src.rag import audit_evidence_coverage_v5 as coverage


def test_exact_aggregate_bins():
    values=[-4,-3,-2,-1,-.5,-.499,0,.499,.5,1,2,3,4]
    states=method.semantic_states(values)
    expected={"extremely low":0,"very low":0,"low":0,"moderately low":0,"typical":1,
              "moderately high":2,"high":2,"very high":2,"extremely high":2}
    assert list(states)==[expected[semantic_bin(v)] for v in values]
    with pytest.raises(ValueError): method.semantic_states([float("nan")])


def test_registered_pairwise_population():
    pairs=list(itertools.combinations(method.common.CLASSES["CIC-Darknet2020"],2))
    assert pairs==[("Non-Tor","NonVPN"),("Non-Tor","Tor"),("Non-Tor","VPN"),("NonVPN","Tor"),("NonVPN","VPN"),("Tor","VPN")]
    assert list(itertools.combinations(method.common.CLASSES["USTC-TFC2016"],2))==[("benign","malware")]
    assert (6*61+48)*3==method.POLICY["planned_candidates"]==1242
    assert method.corpus.validate_development_id(method.DEVELOPMENT_ID)=="v5f_validation"


def test_group_weighting_not_row_pseudoreplication():
    spec=dict(label="label",group="group")
    frame=pd.DataFrame([dict(label="a",group="large",x=1)]*100+[dict(label="a",group="small",x=-1)])
    size,counts=method.group_state_counts(frame,spec,"x")
    assert sorted(size.tolist())==[1,100]
    prevalence=counts[:,2]/size.to_numpy()
    assert np.median(prevalence)==.5
    assert np.mean(frame.x>=.5)> .99
    assert (counts.sum(axis=1)==size.to_numpy()).all()


def test_bootstrap_and_leave_one_reproducible(monkeypatch):
    monkeypatch.setitem(method.POLICY,"bootstrap_replicates",200)
    a,b=[.8,.9,.8,.9,.85,.95],[.1,.2,.1,.2,.15,.05]
    first=method.assess(a,b,6,42)
    assert first==method.assess(a,b,6,42)
    assert first["decision"]=="APPROVED" and first["effect_size"]==1
    assert first["bootstrap_interval"][0]>0 and first["minimum_loo_difference"]>0
    for values in ([0,1,2,3,4],[0,0,1,1,2,2]):
        actual=method.leave_one_medians(values)
        expected=[np.median(np.delete(sorted(values),i)) for i in range(len(values))]
        assert np.array_equal(actual,expected)


def test_unstable_and_weak_rejected(monkeypatch):
    monkeypatch.setitem(method.POLICY,"bootstrap_replicates",200)
    weak=method.assess([.5]*6,[.5]*6,6,42)
    assert weak["decision"]=="REJECTED"
    assert "SMALL_PREVALENCE_DIFFERENCE" in weak["rejection_reasons"]
    unstable=method.assess([0,0,.2,.4,.4,.4],[.2]*6,6,42)
    assert "LEAVE_ONE_GROUP_OUT_SIGN_UNSTABLE" in unstable["rejection_reasons"]


@pytest.fixture
def association_case(monkeypatch):
    monkeypatch.setitem(method.POLICY,"bootstrap_replicates",200)
    row=method.assess([.9]*6,[.1]*6,6,42)
    row.update(claim_id="ASSOC-fixture",evidence_method_version=method.VERSION,source_dataset="CIC-Darknet2020",
               supported_class="VPN",contrasted_class="NonVPN",reference_classes=["NonVPN"],canonical_feature="flow duration",
               semantic_state="HIGHER",direction="higher",source_type=method.SOURCE_TYPE,
               source_artifacts={"synthetic/train.csv":"a"*64},configuration_sha256="b"*64,code_sha256="c"*64,
               statistics_link=method.BASE+"statistics.json#ASSOC-fixture")
    sources,docs,claims=method.derived_tables([row])
    item=dict(chunk_id="chunk-a",document_id=row["claim_id"],chunk_text=docs[0]["normalized_text"],source_type=method.SOURCE_TYPE,
              decision_claim_ids_json='["ASSOC-fixture"]')
    payload=dict(query_id="CIC-VAL-000001",source_dataset="CIC-Darknet2020",source_split="validation",classifier_prediction="VPN",
                 traffic_profile="Encrypted network traffic observation. Predicted traffic class: VPN. Observed traffic profile relative to the training distribution: flow duration is high. Retrieve trusted cybersecurity evidence relevant to the predicted class and these observed traffic characteristics.",trusted_evidence=[item])
    verified={"chunk-a":dict(item,provenance_status="VERIFIED",authorized_for_downstream="YES",quality_status="KEEP")}
    response=dict(query_id=payload["query_id"],reasoning_decision="SUPPORT_CLASS",evidence_status="SUPPORTED",supported_class="VPN",
                  explanation='CLAIM_IDS: ["ASSOC-fixture"]',cited_chunk_ids=["chunk-a"])
    return row,payload,verified,response,claims


def test_exact_matching_and_supported_only_audit(association_case):
    row,payload,_,_,claims=association_case
    query=dict(source_dataset=payload["source_dataset"],retrieval_query_text=payload["traffic_profile"])
    chunk=payload["trusted_evidence"][0]
    matched=rerank.matching_decision_claims(query,chunk,claims)
    assert matched==claims and {c["supported_class"] for c in matched}=={"VPN"}
    assert claims[0]["contrasted_class"]=="NonVPN"
    for feature in ("Flow Duration","flow duration is low"):
        changed=dict(query,retrieval_query_text=query["retrieval_query_text"].replace("flow duration is high",feature))
        assert rerank.matching_decision_claims(changed,chunk,claims)==[]
    edited=dict(query,confidence=1,true_class_label="ignored",classifier_correct="ignored")
    assert rerank.matching_decision_claims(edited,chunk,claims)==matched
    claims[0]["semantic_state"]="LOWER"
    with pytest.raises(ValueError): rerank.matching_decision_claims(query,chunk,claims)


def test_pairwise_validator_requires_named_contrast(association_case):
    row,payload,verified,response,_=association_case
    registry={row["claim_id"]:row}
    before=copy.deepcopy(response)
    gate.validate(response,payload,registry,verified,method=method)
    assert response==before
    payload["classifier_prediction"]="NonVPN"; response["evidence_status"]="CONTRADICTED"
    gate.validate(response,payload,registry,verified,method=method)
    payload["classifier_prediction"]="Tor"
    with pytest.raises(ValueError): gate.validate(response,payload,registry,verified,method=method)
    response.update(reasoning_decision="ABSTAIN",evidence_status="INSUFFICIENT",supported_class=None,
                    explanation="No qualifying approved claim.",cited_chunk_ids=[])
    gate.validate(response,payload,registry,verified,method=method)


@pytest.mark.parametrize("mutation",["context","quote","provenance","state","dataset","claim_id","test","ground_truth"])
def test_no_unverified_or_contextual_positive(association_case,mutation):
    row,payload,verified,response,_=association_case
    if mutation=="context": payload["trusted_evidence"][0]["decision_claim_ids_json"]="[]"
    if mutation=="quote": payload["trusted_evidence"][0]["chunk_text"]="unsupported prose"
    if mutation=="provenance": verified["chunk-a"]["provenance_status"]="UNVERIFIED"
    if mutation=="state": row["semantic_state"]="LOWER"
    if mutation=="dataset": row["source_dataset"]="USTC-TFC2016"
    if mutation=="claim_id": response["explanation"]='CLAIM_IDS: ["unknown"]'
    if mutation=="test": payload["source_split"]="test"
    if mutation=="ground_truth": payload["ground_truth"]="VPN"
    with pytest.raises(ValueError): gate.validate(response,payload,{row["claim_id"]:row},verified,method=method)


def test_conflict_cannot_be_cherry_picked(association_case):
    row,payload,verified,response,_=association_case
    other=dict(row,claim_id="ASSOC-other",supported_class="NonVPN",contrasted_class="VPN",reference_classes=["VPN"])
    item=dict(chunk_id="chunk-b",document_id=other["claim_id"],chunk_text=method.render(other),decision_claim_ids_json='["ASSOC-other"]')
    payload["trusted_evidence"].append(item)
    verified["chunk-b"]=dict(item,provenance_status="VERIFIED",authorized_for_downstream="YES",quality_status="KEEP")
    registry={r["claim_id"]:r for r in (row,other)}
    with pytest.raises(ValueError): gate.validate(response,payload,registry,verified,method=method)
    response.update(reasoning_decision="ABSTAIN",supported_class=None,evidence_status="CONFLICTING",cited_chunk_ids=["chunk-a","chunk-b"],
                    explanation='CLAIM_IDS: ["ASSOC-fixture","ASSOC-other"]')
    gate.validate(response,payload,registry,verified,method=method)


def test_frozen_input_allowlist_and_separate_namespace():
    assert method.BASE!=method.previous.BASE
    assert method.VERSION!=method.previous.VERSION
    assert all(spec["train"].endswith("/train.csv") for spec in method.previous.SPECS.values())
    assert method.POLICY["planned_candidates"]==1242
    assert method.previous.POLICY["planned_candidates"]==340


def test_bundle_hash_and_independent_audit(tmp_path,monkeypatch,association_case):
    row,_,_,_,_=association_case
    frozen=dict(inputs={},configuration_sha256="b"*64,code_sha256={"src/rag/build_semantic_association_v5.py":"c"*64})
    arrays={"synthetic_counts":np.array([[1,2,3],[3,2,1]])}
    monkeypatch.setattr(method,"freeze",lambda root:frozen)
    monkeypatch.setattr(method,"statistics",lambda root,f:([row],arrays))
    assert method.main(["--project-root",str(tmp_path),"--execute"])==0
    assert method.main(["--project-root",str(tmp_path),"--audit-only"])==0
    assert method.verify_bundle(tmp_path)==[row]
    with pytest.raises(ValueError,match="refusing overwrite"):
        method.main(["--project-root",str(tmp_path),"--execute"])
    (tmp_path/method.BASE/"statistics.json").write_text("[]",encoding="utf-8")
    with pytest.raises(ValueError,match="hash mismatch"): method.verify_bundle(tmp_path)


def test_source_identity_mismatch_fails_before_generation(tmp_path,monkeypatch):
    monkeypatch.setattr(method.previous,"freeze",lambda root:(_ for _ in ()).throw(ValueError("Frozen TRAIN/scaler/boundary hash mismatch")))
    with pytest.raises(ValueError,match="hash mismatch"): method.freeze(tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_generation_train_only_reproducible_and_retains_group_counts(tmp_path,monkeypatch):
    dataset="USTC-TFC2016"
    spec=dict(method.previous.SPECS[dataset],train="train.csv",scaler="scaler.csv",boundary="boundary.csv")
    rows=[]
    for label in ("benign","malware"):
        for family in range(6):
            for i in range(10):
                high=i<(8 if label=="malware" else 2)
                rows.append(dict(binary_label=label,semantic_family=f"{label}-{family}",bidirectional_duration_ms=1 if high else -1))
    frame=pd.DataFrame(rows)
    frame.to_csv(tmp_path/"train.csv",index=False)
    boundary=frame.groupby("semantic_family").size().to_dict()
    monkeypatch.setattr(method.previous,"SPECS",{dataset:spec})
    monkeypatch.setattr(method.previous,"feature_map",lambda *args:{"bidirectional duration ms":("bidirectional_duration_ms",0,1)})
    monkeypatch.setattr(method.previous,"train_boundary",lambda *args:boundary)
    monkeypatch.setitem(method.POLICY,"planned_candidates",3)
    monkeypatch.setitem(method.POLICY,"bootstrap_replicates",200)
    real=pd.read_csv
    opened=[]
    def read(path,**kwargs):
        assert path==tmp_path/"train.csv"
        assert set(kwargs["usecols"])=={"binary_label","semantic_family","bidirectional_duration_ms"}
        opened.append(path)
        return real(path,**kwargs)
    monkeypatch.setattr(pd,"read_csv",read)
    frozen=dict(inputs={p:"a"*64 for p in ("train.csv","scaler.csv","boundary.csv")},configuration_sha256="b"*64,
                code_sha256={"src/rag/build_semantic_association_v5.py":"c"*64})
    records,arrays=method.statistics(tmp_path,frozen)
    again,repeated=method.statistics(tmp_path,frozen)
    assert records==again and all(np.array_equal(a,repeated[k]) for k,a in arrays.items())
    assert len(opened)==2 and len(records)==3
    assert sum(r["decision"]=="APPROVED" for r in records)==2
    assert arrays["ustc_feature_000"].shape==(12,3)
    assert np.array_equal(arrays["ustc_feature_000"].sum(axis=1),arrays["ustc_row_counts"])
    boundary["heldout_group"]=1
    with pytest.raises(ValueError,match="membership/count"):
        method.statistics(tmp_path,frozen)


def test_stage2_adapter_uses_existing_quality_contract_and_create_only(tmp_path,monkeypatch,association_case):
    row,_,_,_,_=association_case
    source,document,_=method.derived_tables([row])
    docs=[]; manifest=[]; quality=[]
    for i in range(12):
        identity=f"CONTEXT-{i}"
        src=dict(source[0],document_id=identity,source_type="peer_reviewed_research")
        doc=dict(document[0],**src)
        doc.update(corpus_revision="V2",is_new_document="NO",**method.corpus.term_counts(doc["normalized_text"]),task_aligned="NO")
        docs.append(doc); manifest.append(src)
        quality.extend(method.corpus.quality_rows(identity,method.corpus.source_units(doc["normalized_text"]),"evidence_quality"))
    manifest.append(dict(source[0],document_id="TOR-SPEC-001",authorized_for_retrieval="NO"))
    review=[dict(document_id=f"CONTEXT-{i}",review_status="APPROVED",review_notes="synthetic") for i in range(3)]
    monkeypatch.setattr(method,"verify_bundle",lambda root:[row])
    monkeypatch.setattr(method.corpus,"build",lambda *args:copy.deepcopy((docs,manifest,review,quality,[])))
    assert method.main(["--project-root",str(tmp_path),"--stage2"])==0
    assert list(tmp_path.iterdir())==[]
    assert method.main(["--project-root",str(tmp_path),"--stage2","--execute"])==0
    claims=method.load_decision_claims(tmp_path,method.DEVELOPMENT_ID)
    assert claims==method.derived_tables([row])[2]
    with pytest.raises(ValueError,match="refusing overwrite"):
        method.main(["--project-root",str(tmp_path),"--stage2","--execute"])
    assert not (tmp_path/method.previous.BASE).exists()
