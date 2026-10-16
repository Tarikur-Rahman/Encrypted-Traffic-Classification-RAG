"""Synthetic offline tests: never read real TRAIN/validation/TEST populations."""
import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.rag import build_train_evidence_v5 as empirical
from src.rag import positive_evidence_v5 as gate
from src.rag import audit_evidence_coverage_v5 as coverage
from src.rag import run_provenance_aware_reranking as reranking
from src.rag import verify_retrieval_provenance_v3 as provenance


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


@pytest.fixture
def claim_case():
    row=dict(claim_id="EMP-fixture",source_dataset="USTC-TFC2016",supported_class="malware",
             canonical_feature="bidirectional duration ms",direction="higher",source_type=empirical.SOURCE_TYPE,
             reference_classes=["benign"],group_key="semantic_family",effect_size=1.,group_direction_fraction=1.,
             median_difference_ci=[2.,4.],source_artifacts={"synthetic/train.csv":"a"*64},
             configuration_sha256="b"*64,decision="APPROVED")
    text=empirical.render(row)
    item=dict(chunk_id="chunk-a",document_id=row["claim_id"],chunk_text=text,
              decision_claim_ids_json=json.dumps([row["claim_id"]]))
    payload=dict(query_id="USTC-VAL-000001",source_dataset="USTC-TFC2016",source_split="validation",
                 classifier_prediction="malware",traffic_profile="Encrypted network traffic observation. Predicted traffic class: malware. "
                 "Observed traffic profile relative to the training distribution: bidirectional duration ms is high. "
                 "Retrieve trusted cybersecurity evidence relevant to the predicted class and these observed traffic characteristics.",trusted_evidence=[item])
    verified={"chunk-a":dict(item,provenance_status="VERIFIED",authorized_for_downstream="YES",quality_status="KEEP")}
    response=dict(query_id=payload["query_id"],reasoning_decision="SUPPORT_CLASS",evidence_status="SUPPORTED",
                  supported_class="malware",explanation='CLAIM_IDS: ["EMP-fixture"]',cited_chunk_ids=["chunk-a"])
    return row,payload,verified,response


def test_effect_and_direction_gate_deterministic(monkeypatch):
    monkeypatch.setitem(empirical.POLICY,"bootstrap_replicates",200)
    first=empirical.assess([2,2,3,3,4,4],[-2,-2,-3,-3,-4,-4],600,600,6,42)
    assert first==empirical.assess([2,2,3,3,4,4],[-2,-2,-3,-3,-4,-4],600,600,6,42)
    assert first["decision"]=="APPROVED" and first["effect_size"]==1
    assert first["median_difference_ci"][0]>0
    assert first["minimum_signed_leave_one_group_out_delta"]==1


@pytest.mark.parametrize("a,b,n,reason",[
    ([0]*6,[0]*6,600,"SMALL_OR_OPPOSITE_EFFECT"),
    ([2]*6,[-2]*6,10,"INSUFFICIENT_SAMPLE_OR_GROUPS"),
    ([2]*4+[-2]*2,[-3]*6,600,"GROUP_DIRECTION_UNSTABLE"),
    ([0.1]*6,[-2]*6,600,"TRAIN_RELATIVE_DIRECTION_NOT_ESTABLISHED"),
    ([2]*6,[3]*6,600,"SMALL_OR_OPPOSITE_EFFECT"),
])
def test_reject_weak_unstable_or_insufficient(a,b,n,reason):
    result=empirical.assess(a,b,n,n,6,42)
    assert result["decision"]=="REJECTED" and reason in result["rejection_reasons"]


def test_delta_ties_and_direction():
    assert empirical.delta([1,1],[1,1])==0
    assert empirical.delta([0,1],[2,3])==-1
    assert empirical.delta([2,3],[0,1])==1


def test_boundary_reads_only_train_metadata(tmp_path):
    relative="metadata/boundary.csv"
    write_csv(tmp_path/relative,[dict(dataset="CIC-Darknet2020",split="TRAIN",flow_id="g",group_row_count="5"),
                                dict(dataset="CIC-Darknet2020",split="VALIDATION",flow_id="ignored",group_row_count="NOT_READ"),
                                dict(dataset="CIC-Darknet2020",split="TEST",flow_id="ignored2",group_row_count="NOT_READ")])
    spec=dict(empirical.SPECS["CIC-Darknet2020"],boundary=relative)
    assert empirical.train_boundary(tmp_path,spec)=={"g":5}


@pytest.mark.parametrize("changed", ["train.csv", "boundary.csv"])
def test_frozen_hash_rejects_modified_input(tmp_path,monkeypatch,changed):
    (tmp_path/"train.csv").write_bytes(b"synthetic train\n")
    (tmp_path/"boundary.csv").write_bytes(b"synthetic boundary\n")
    (tmp_path/"config.json").write_text(json.dumps(empirical.POLICY))
    monkeypatch.setattr(empirical,"INPUTS",{name:empirical.sha(tmp_path/name) for name in ("train.csv","boundary.csv")})
    monkeypatch.setattr(empirical,"CODE",())
    monkeypatch.setattr(empirical,"CONTEXT_FILES",())
    monkeypatch.setattr(empirical,"CONFIG","config.json")
    assert empirical.freeze(tmp_path)["inputs"]==empirical.INPUTS
    (tmp_path/changed).write_bytes(b"altered\n")
    with pytest.raises(ValueError,match="hash mismatch"):
        empirical.freeze(tmp_path)


def test_source_allowlist_contains_no_heldout_row_files():
    assert {s["train"] for s in empirical.SPECS.values()}=={
        "data/processed/USTC-TFC2016/train.csv","data/processed/CIC-Darknet2020/train.csv"}
    assert all(not p.endswith(("validation.csv","test.csv")) for p in empirical.INPUTS)
    assert empirical.corpus.validate_development_id(empirical.DEVELOPMENT_ID) is not False


def test_generation_reads_only_registered_train_columns_and_checks_membership(tmp_path,monkeypatch):
    import pandas as pd
    dataset="USTC-TFC2016"
    spec=dict(empirical.SPECS[dataset],train="data/train.csv",scaler="metadata/scaler.csv",boundary="metadata/split.csv")
    rows=[]
    boundary=[]
    for label,value in (("benign",-2),("malware",2)):
        for family in range(6):
            name=f"{label}-{family}"
            boundary.append(dict(dataset=dataset,split="train",semantic_family=name,flow_count="20"))
            rows.extend(dict(bidirectional_duration_ms=value,binary_label=label,semantic_family=name,
                             source_pcap="not consumed",target_binary="not consumed") for _ in range(20))
    write_csv(tmp_path/spec["train"],rows)
    write_csv(tmp_path/spec["boundary"],boundary)
    write_csv(tmp_path/spec["scaler"],[dict(feature_name="bidirectional_duration_ms",retained="True",train_mean="10",train_std="2")])
    monkeypatch.setattr(empirical,"SPECS",{dataset:spec})
    monkeypatch.setattr(empirical.corpus,"CANONICAL_DECISION_FEATURES",{dataset:frozenset({"bidirectional duration ms"})})
    monkeypatch.setitem(empirical.POLICY,"planned_candidates",2)
    monkeypatch.setitem(empirical.POLICY,"bootstrap_replicates",200)
    frozen=dict(inputs={v:"a"*64 for k,v in spec.items() if k in {"train","scaler","boundary"}},configuration_sha256="b"*64)
    real=pd.read_csv
    seen=[]
    def read(path,**kwargs):
        assert path==tmp_path/spec["train"]
        assert set(kwargs["usecols"])=={"bidirectional_duration_ms","binary_label","semantic_family"}
        seen.append(path)
        return real(path,**kwargs)
    monkeypatch.setattr(pd,"read_csv",read)
    result=empirical.statistics(tmp_path,frozen)
    assert len(result)==2 and all(r["decision"]=="APPROVED" for r in result)
    assert seen==[tmp_path/spec["train"]]
    assert {r["class_median"] for r in result}=={6,14}
    assert result==empirical.statistics(tmp_path,frozen)
    boundary[0]["flow_count"]="21"
    write_csv(tmp_path/spec["boundary"],boundary)
    with pytest.raises(ValueError,match="membership/count"):
        empirical.statistics(tmp_path,frozen)


@pytest.mark.parametrize("dataset", ["USTC-TFC2016", "CIC-Darknet2020"])
def test_real_producer_boundary_schemas(tmp_path,dataset):
    # Full real producer headers; held-out labels/counts are deliberately invalid
    # sentinels to prove only membership metadata is consumed for those splits.
    spec=dict(empirical.SPECS[dataset],boundary="boundary.csv")
    if dataset=="USTC-TFC2016":
        base=dict(dataset=dataset,binary_label="NOT_READ",semantic_family="family",split="train",
                  flow_count="7",capture_group_count="1",run_id="20260919T123108Z",timestamp_utc="frozen")
    else:
        base=dict(dataset=dataset,flow_id="flow",split="TRAIN",group_row_count="7",
                  non_tor_rows="NOT_READ",nonvpn_rows="NOT_READ",tor_rows="NOT_READ",vpn_rows="NOT_READ",
                  has_label_conflict="NOT_READ",seed="42",split_method="StratifiedGroupKFold_5_shuffle_seed42_best_3_1_1_of_20")
    rows=[base]
    for split in spec["boundary_splits"][1:]:
        rows.append(dict(base,**{spec["boundary_group"]:split, "split":split,spec["boundary_count"]:"NOT_READ"}))
    write_csv(tmp_path/spec["boundary"],rows)
    boundary=empirical.train_boundary(tmp_path,spec)
    assert boundary=={base[spec["boundary_group"]]:7}
    empirical.validate_train_membership(dict(boundary),boundary)
    for heldout in spec["boundary_splits"][1:]:
        with pytest.raises(ValueError,match="membership/count"):
            empirical.validate_train_membership({**boundary,heldout:1},boundary)
    for invalid in ({}, {"unknown":7}, {base[spec["boundary_group"]]:6}):
        with pytest.raises(ValueError,match="membership/count"):
            empirical.validate_train_membership(invalid,boundary)


@pytest.mark.parametrize("mutation", ["empty_train","duplicate","cross_split_duplicate","blank","unknown_split","wrong_case","dataset","missing_count","zero"])
def test_boundary_fail_closed(tmp_path,mutation):
    spec=dict(empirical.SPECS["CIC-Darknet2020"],boundary="boundary.csv")
    row=dict(dataset="CIC-Darknet2020",flow_id="g",split="TRAIN",group_row_count="5")
    rows=[row]
    if mutation=="empty_train": row["split"]="TEST"
    if mutation=="duplicate": rows.append(dict(row))
    if mutation=="cross_split_duplicate": rows.append(dict(row,split="VALIDATION"))
    if mutation=="blank": row["flow_id"]=" "
    if mutation=="unknown_split": row["split"]="HOLDOUT"
    if mutation=="wrong_case": row["split"]="train"
    if mutation=="dataset": row["dataset"]="USTC-TFC2016"
    if mutation=="missing_count": del row["group_row_count"]
    if mutation=="zero": row["group_row_count"]="0"
    write_csv(tmp_path/spec["boundary"],rows)
    with pytest.raises(ValueError):
        empirical.train_boundary(tmp_path,spec)


def test_no_filename_fallback_when_manifest_absent(tmp_path):
    spec=dict(empirical.SPECS["CIC-Darknet2020"],boundary="missing.csv")
    (tmp_path/"train.csv").write_text("Flow ID\ng\n",encoding="utf-8")
    with pytest.raises(ValueError,match="Missing local file"):
        empirical.train_boundary(tmp_path,spec)


def test_derived_claim_preserves_source_location_and_supported_only(claim_case):
    row,payload,_,_=claim_case
    sources,docs,claims=empirical.derived_tables([row])
    assert sources[0]["source_type"]==empirical.SOURCE_TYPE
    assert "not proof or a causal rule" in docs[0]["normalized_text"]
    assert claims[0]["source_location"]=="document" and claims[0]["contrasted_class"]==""
    query=dict(source_dataset=payload["source_dataset"],retrieval_query_text=payload["traffic_profile"])
    assert reranking.matching_decision_claims(query,payload["trusted_evidence"][0],claims)==claims
    query["retrieval_query_text"]=query["retrieval_query_text"].replace("is high","is low")
    assert reranking.matching_decision_claims(query,payload["trusted_evidence"][0],claims)==[]


def test_v5_positive_and_contradiction(claim_case):
    row,payload,verified,response=claim_case
    original=copy.deepcopy(response)
    gate.validate(response,payload,{row["claim_id"]:row},verified)
    assert response==original
    payload["classifier_prediction"]="benign"
    response["evidence_status"]="CONTRADICTED"
    gate.validate(response,payload,{row["claim_id"]:row},verified)
    row["reference_classes"]=[]
    # Refresh text to isolate reference semantics, rather than the verbatim gate.
    payload["trusted_evidence"][0]["chunk_text"]=empirical.render(row)
    verified["chunk-a"]["chunk_text"]=empirical.render(row)
    with pytest.raises(ValueError,match="contrast"):
        gate.validate(response,payload,{row["claim_id"]:row},verified)


@pytest.mark.parametrize("mutation",["unknown_claim","uncited","direction","canonical_case","unapproved",
                                    "unverified","altered_text","wrong_query","extra_field","gt","confidence"])
def test_v5_rejects_unsupported_inference(claim_case,mutation):
    row,payload,verified,response=claim_case
    if mutation=="unknown_claim": response["explanation"]='CLAIM_IDS: ["unknown"]'
    if mutation=="uncited": response["cited_chunk_ids"]=[]
    if mutation=="direction": payload["traffic_profile"]=payload["traffic_profile"].replace("is high","is low")
    if mutation=="canonical_case": payload["traffic_profile"]=payload["traffic_profile"].replace("bidirectional duration ms","Bidirectional duration ms")
    if mutation=="unapproved": row["decision"]="REJECTED"
    if mutation=="unverified": verified["chunk-a"]["provenance_status"]="UNKNOWN"
    if mutation=="altered_text": payload["trusted_evidence"][0]["chunk_text"]="fabricated"
    if mutation=="wrong_query": response["query_id"]="wrong"
    if mutation=="extra_field": response["extra"]="forbidden"
    if mutation=="gt": payload["true_class_label"]="malware"
    if mutation=="confidence": payload["classifier_confidence"]=0.99
    with pytest.raises(ValueError):
        gate.validate(response,payload,{row["claim_id"]:row},verified)


def test_contextual_text_cannot_create_positive(claim_case):
    row,payload,verified,response=claim_case
    payload["trusted_evidence"][0]["decision_claim_ids_json"]="[]"
    with pytest.raises(ValueError):
        gate.validate(response,payload,{row["claim_id"]:row},verified)
    response.update(reasoning_decision="ABSTAIN",evidence_status="INSUFFICIENT",supported_class=None,
                    explanation="No qualifying approved claim.",cited_chunk_ids=[])
    gate.validate(response,payload,{row["claim_id"]:row},verified)


def test_conflicting_requires_multiple_real_classes(claim_case):
    row,payload,verified,response=claim_case
    other=dict(row,claim_id="EMP-other",supported_class="benign",reference_classes=["malware"])
    item=dict(chunk_id="chunk-b",document_id=other["claim_id"],chunk_text=empirical.render(other),
              decision_claim_ids_json='["EMP-other"]')
    payload["trusted_evidence"].append(item)
    verified["chunk-b"]=dict(item,provenance_status="VERIFIED",authorized_for_downstream="YES",quality_status="KEEP")
    registry={r["claim_id"]:r for r in (row,other)}
    with pytest.raises(ValueError,match="Ambiguous"):
        gate.validate(response,payload,registry,verified)
    response.update(reasoning_decision="ABSTAIN",evidence_status="CONFLICTING",supported_class=None,
                    explanation='CLAIM_IDS: ["EMP-fixture","EMP-other"]',cited_chunk_ids=["chunk-a","chunk-b"])
    gate.validate(response,payload,registry,verified)


def test_coverage_stops_zero_and_negligible_and_missing_strata():
    assert not coverage.coverage_gate([])
    assert not coverage.coverage_gate([dict(queries=10000,contexts_with_claims=0)])
    assert not coverage.coverage_gate([dict(queries=10000,contexts_with_claims=99)])
    assert coverage.coverage_gate([dict(queries=10000,contexts_with_claims=100)])


def test_streaming_coverage_recomputes_matches_and_prepares_bounded_prompts(claim_case,tmp_path,monkeypatch):
    row,payload,verified,_=claim_case
    common=empirical.common
    monkeypatch.setattr(common,"CLASSES",{"USTC-TFC2016":("benign","malware")})
    monkeypatch.setattr(common,"QUERY_COUNTS",{"USTC-TFC2016":2})
    monkeypatch.setitem(empirical.POLICY,"diagnostic_per_stratum",1)
    paths,query_path,_,_,_,rank_paths=coverage.contexts.paths_for(empirical.DEVELOPMENT_ID)
    queries=[]; ranks=[]; context_rows=[]
    chunks={}
    for i in range(10):
        identity=f"chunk-{i}"
        chunks[identity]=dict(chunk_id=identity,document_id=row["claim_id"] if i==0 else "context",chunk_text=empirical.render(row) if i==0 else "Context only.")
    verified={k:dict(v,provenance_status="VERIFIED",authorized_for_downstream="YES",quality_status="KEEP") for k,v in chunks.items()}
    for n,prediction in enumerate(("malware","benign")):
        identity=f"USTC-VAL-{n}"
        query=dict(query_id=identity,source_dataset="USTC-TFC2016",source_split="validation",rag_partition="development_validation",
                   predicted_class_label=prediction,confidence="0.5",class_probabilities_json='{"benign":0.5,"malware":0.5}',retrieval_query_text=payload["traffic_profile"])
        queries.append(query)
        for i,chunk in enumerate(chunks.values()):
            claim_ids=[row["claim_id"]] if i==0 else []
            ranks.append(dict(query_id=identity,source_dataset=query["source_dataset"],chunk_id=chunk["chunk_id"],document_id=chunk["document_id"],
                              provenance_status="VERIFIED",authorized_for_retrieval="YES",final_rank=str(i+1),decision_claim_match_count=str(len(claim_ids)),
                              decision_claim_ids_json=json.dumps(claim_ids),decision_claim_classes_json=json.dumps(["malware"] if claim_ids else []),
                              decision_claim_features_json=json.dumps([row["canonical_feature"]] if claim_ids else [])))
        item=dict(chunks["chunk-0"],source_type=empirical.SOURCE_TYPE,rerank_rank=1,decision_claim_match_count="1",
                  decision_claim_ids_json=json.dumps([row["claim_id"]]),decision_claim_classes_json='["malware"]',
                  decision_claim_features_json=json.dumps([row["canonical_feature"]]))
        context_rows.append(dict(query_id=identity,source_dataset=query["source_dataset"],source_split="validation",rag_partition="development_validation",
                                 classifier_prediction=prediction,traffic_profile=query["retrieval_query_text"],evidence_records_json=json.dumps([item]),
                                 selected_chunk_ids='["chunk-0"]',context_text=coverage.contexts.context_text([item]),context_token_count="400"))
    tables={query_path:queries,rank_paths[0]:ranks,paths[0]:context_rows}
    monkeypatch.setattr(common,"rows",lambda root,name,*args:iter(tables[name]))
    def completed(root,config,*args):
        if config==rank_paths[2]:
            return dict(development_id=empirical.DEVELOPMENT_ID,profile="task_relevance",task_alignment_weight="0.05",relationship_weight="0.05")
        return dict(development_id=empirical.DEVELOPMENT_ID,profile="evidence_diverse",top_n="5",token_budget="2048",max_chunks_per_document="0",diversity_threshold="1.0")
    monkeypatch.setattr(common,"completed",completed)
    report,prompts=coverage.scan(tmp_path,{row["claim_id"]:row},chunks,verified)
    assert report["status"]=="PASS" and report["totals"]["queries"]==2
    assert report["totals"]["predicted_class_support"]==report["totals"]["alternative_class_support"]==1
    assert len(prompts)==2 and list(tmp_path.iterdir())==[]
    assert all("classifier_confidence" not in p["prompt_text"] and p["token_audit_status"]=="NOT_TOKEN_AUDITED" for p in prompts)
    assert (report,prompts)==coverage.scan(tmp_path,{row["claim_id"]:row},chunks,verified)
    ranks[0]["decision_claim_classes_json"]='["benign","malware"]'
    with pytest.raises(ValueError,match="Stage-6 claim audit"):
        coverage.scan(tmp_path,{row["claim_id"]:row},chunks,verified)


def test_v4_source_unchanged():
    from src.rag import positive_evidence_v4
    # Normalize checkout line endings only, preserving every code character.
    source=Path(positive_evidence_v4.__file__).read_bytes()
    expected="5c3970e91b255bad122a358392390b760b5c7ba3c09afa26ed9a8a075c092270"
    assert any(hashlib.sha256(value).hexdigest()==expected for value in (source,source.replace(b"\r\n",b"\n"),source.replace(b"\r\n",b"\n").replace(b"\n",b"\r\n")))


def test_empirical_source_not_globally_authorized(claim_case):
    row,payload,_,_=claim_case
    chunk=dict(payload["trusted_evidence"][0],source_type=empirical.SOURCE_TYPE,quality_status="KEEP",
               source_provenance_status="VERIFIED",corpus_partition="trusted",authorized_for_retrieval="YES")
    with pytest.raises(ValueError,match="verified derivation"):
        provenance.verify_chunks([chunk])
    docs={row["claim_id"]:{"normalized_text":empirical.render(row)}}
    assert provenance.verify_chunks([chunk],docs)[0]["provenance_status"]=="VERIFIED"
