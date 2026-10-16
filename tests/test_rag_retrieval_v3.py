import pytest

from src.rag import run_rag_retrieval_v3 as retrieval
from src.rag import phase2_common as common


def test_semantic_profile_reused_without_identifier_semantics():
    features = {f"packet_feature_{i}": i/10 for i in range(50)}
    representation = common.json_text(dict(predictor_count=50, features=features))
    a = retrieval.v2.semantic_profile("USTC-TFC2016", "benign", representation)
    assert a == retrieval.v2.semantic_profile("USTC-TFC2016", "benign", representation)
    assert a["semantic_profile_top_features"] == 12
    assert "packet feature" in a["retrieval_query_text"]
    assert "query_id" not in a["retrieval_query_text"]
    assert retrieval.v2.PREFIX == "Represent this sentence for searching relevant passages:"
    assert (retrieval.v2.TOP_K, retrieval.v2.DEPTH, retrieval.v2.RRF_K, retrieval.v2.DENSE_BATCH_SIZE) == (10, 50, 60, 256)


def test_balanced_retrieval_explicit_split(phase2_case, monkeypatch):
    from types import SimpleNamespace
    validation=dict(phase2_case.query)
    test=dict(validation,query_id='USTC-TEST-000001',source_split='test',rag_partition='final_test')
    calls=[]
    def retrieve(evidence,rows,model,counts):
        calls.extend(rows)
        for row in rows:
            yield row,[dict(query_id=row['query_id'],source_dataset=row['source_dataset'],
                           retrieval_method=method,rank=rank,chunk_id=f'C-{rank}',document_id=f'D-{rank}')
                       for method in retrieval.v2.METHODS for rank in range(1,11)]
    monkeypatch.setattr(retrieval.v2,'retrieve_batch',retrieve)
    policy=SimpleNamespace(count=lambda text:1)
    assert len(list(retrieval.balanced_batch(None,[validation],None,policy))[0][1])==30
    result=list(retrieval.balanced_batch(None,[test],None,policy,source_split='test'))
    assert result[0][0]==test and len(result[0][1])==30
    assert retrieval.hypotheses(test,source_split='test')==retrieval.hypotheses(validation)
    assert all(not {'ground_truth','true_class_label','classifier_correct'}.intersection(row) for row in calls)
    for batch,scope in (([test],'validation'),([validation],'test'),([test,validation],'test'),
                        ([dict(test,rag_partition='development_validation')],'test'),
                        ([dict(validation,query_id=test['query_id'])],'validation')):
        calls.clear()
        with pytest.raises(ValueError):list(retrieval.balanced_batch(None,batch,None,policy,source_split=scope))
        assert not calls
    for field in ('ground_truth','true_class_label','classifier_correct'):
        with pytest.raises(ValueError):
            list(retrieval.balanced_batch(None,[dict(test,**{field:'forbidden'})],None,policy,source_split='test'))


def test_hypothesis_intents_keep_dataset_vocabularies_isolated(phase2_case):
    query = dict(phase2_case.query)
    query.update(source_dataset="CIC-Darknet2020",predicted_class_label="Tor",
        class_probabilities_json=common.json_text({"Non-Tor":.1,"NonVPN":.1,"Tor":.7,"VPN":.1}))
    assert retrieval.hypotheses(query) == ["Tor","Non-Tor","NonVPN","VPN"]
    query.update(source_dataset="USTC-TFC2016",predicted_class_label="benign",
        class_probabilities_json=common.json_text({"benign":.8,"malware":.2}))
    assert retrieval.hypotheses(query) == ["benign","malware"]
    assert retrieval.hypotheses(query) == retrieval.hypotheses(dict(query))


def test_hypothesis_intents_ignore_ground_truth(phase2_case):
    query=dict(phase2_case.query)
    intents = retrieval.retrieval_intents(query)
    with_truth = retrieval.retrieval_intents(dict(query,true_class_label="malware",true_class_id=1,classifier_correct=False))
    assert intents == with_truth
    assert all(not {"true_class_label","true_class_id","classifier_correct"}.intersection(row) for _,row in with_truth)


def test_hypothesis_merge_is_deterministic_deduplicated_and_auditable(phase2_case):
    query = dict(phase2_case.query)
    labels = retrieval.hypotheses(query)
    discoveries = {label:[] for label in labels}
    for label in labels:
        for method in retrieval.v2.METHODS:
            for rank in range(1,11):
                discoveries[label].append(dict(query_id=query["query_id"],source_dataset=query["source_dataset"],
                    retrieval_method=method,rank=rank,chunk_id=f"C-{rank}",document_id=f"D-{rank}",
                    retrieval_score=1/rank,tfidf_score="",dense_score="",tfidf_rank="",dense_rank="",rrf_score=""))
    merged = retrieval.merge_intents(query,discoveries)
    assert len(merged) == 30
    hybrid = [row for row in merged if row["retrieval_method"] == "hybrid_rrf"]
    assert len({row["chunk_id"] for row in hybrid}) == 10
    assert all(common.parse_json(row["retrieval_intents_json"]) == labels for row in hybrid)
    reversed_merge = retrieval.merge_intents(query,dict(reversed(list(discoveries.items()))))
    assert hybrid == [row for row in reversed_merge if row["retrieval_method"] == "hybrid_rrf"]


def test_diagnostics_all_requested_counts(phase2_case):
    diagnostics = retrieval.Diagnostics()
    diagnostics.add(phase2_case.query, phase2_case.results)
    rows = list(diagnostics.rows())
    overall = next(r for r in rows if r["summary_level"] == "overall")
    assert overall["query_count"] == 1 and overall["retrieval_rows"] == 30
    assert overall["rank_min"] == 1 and overall["rank_max"] == 10
    assert overall["top1_unique_chunks"] == 1 and overall["top1_dominant_share"] == 1
    assert overall["unique_retrieval_query_texts"] == 1
    assert len(common.parse_json(overall["unique_chunks_by_rank_json"])) == 10
    assert all(set(row) == set(retrieval.DIAGNOSTICS_HEADER) for row in rows)
    with pytest.raises(ValueError, match="Duplicate"):
        diagnostics.add(phase2_case.query, phase2_case.results)


@pytest.mark.parametrize("change", ["rank", "query", "partition"])
def test_invalid_diagnostic_coverage_rejected(phase2_case, change):
    query, results = dict(phase2_case.query), [dict(r) for r in phase2_case.results]
    if change == "rank": results[0]["rank"] = "2"
    if change == "query": results[0]["query_id"] = "UNKNOWN"
    if change == "partition": query["rag_partition"] = "test"
    with pytest.raises(ValueError):
        retrieval.Diagnostics().add(query, results)


def test_dry_retrieval_never_executes(tmp_path, monkeypatch):
    monkeypatch.setattr(retrieval, "preflight", lambda root: ())
    monkeypatch.setattr(retrieval, "execute", lambda *args: pytest.fail("Executed retrieval"))
    assert retrieval.main(["--project-root", str(tmp_path)]) == 0
    assert not list(tmp_path.iterdir())
    assert all("v3" in path for path in retrieval.OUTPUTS)


def test_namespaced_retrieval_and_provenance_paths_preserve_legacy_constants():
    from src.rag import verify_retrieval_provenance_v3 as provenance
    assert retrieval.output_paths() == retrieval.OUTPUTS
    paths = retrieval.output_paths("v5_validation")
    assert paths[0] == "results/rag/v5_validation/stage04_retrieval/query_records.csv"
    assert all("/v5_validation/" in path and "/test/" not in path.lower() for path in paths)
    assert provenance.output_paths() == provenance.OUTPUTS
    assert provenance.retrieval_paths("v5_validation")[0] == "results/rag/v5_validation/stage04_retrieval/retrieval_results.csv"
    assert provenance.output_paths("v5_validation")[0].startswith("results/rag/v5_validation/stage05_provenance/")
    assert retrieval.EXPERIMENT == "phase2_dev_retrieval_v3"


def test_deterministic_score_ties(phase2_case):
    chunks = list(reversed(list(phase2_case.chunks.values())))
    ordered = retrieval.v2.ranked([1]*len(chunks), chunks)
    assert [chunks[i]["chunk_id"] for i, _, _ in ordered] == sorted(phase2_case.chunks)


def test_execute_confined_to_v3_with_fake_encoder(tmp_path, phase2_case, monkeypatch):
    from types import SimpleNamespace
    case = phase2_case
    query = dict.fromkeys(retrieval.v2.QUERY_FIELDS, "")
    query.update({key: value for key, value in case.query.items() if key in query})
    query.update(classifier_model="lightgbm", predicted_class_id="0", query_text="Saved original query", semantic_profile_top_features=12,
                 semantic_profile_feature_names_json="[]", semantic_profile_bins_json="[]")
    monkeypatch.setattr(retrieval.v2, "query_rows", lambda root: iter([query]))
    monkeypatch.setattr(retrieval.v2, "load_query_encoder", lambda root, policy: SimpleNamespace(device="cpu"))
    monkeypatch.setattr(retrieval.index.v1, "model_identity", lambda root: {})
    def fake_batch(evidence, batch, model, counts):
        for row in batch:
            yield row, [dict(item, retrieval_score=float(item["retrieval_score"]), configuration_name=retrieval.v2.CONFIGURATION_NAME,
                             tfidf_score="", dense_score="", tfidf_rank="", dense_rank="", rrf_score="") for item in case.results]
    monkeypatch.setattr(retrieval.v2, "retrieve_batch", fake_batch)
    evidence = SimpleNamespace(chunks=list(case.chunks.values()), configuration={})
    retrieval.execute(tmp_path, evidence, case.policy, {query["query_id"]: 12}, (), {"dense_device": "cpu"}, [], [])
    assert {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()} == set(retrieval.OUTPUTS)
    records = list(common.rows(tmp_path, retrieval.OUTPUTS[0]))
    assert records[0]["experiment_id"] == "phase2_dev_retrieval_v3"
