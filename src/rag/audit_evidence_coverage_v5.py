"""Offline label-free V5 coverage gate and bounded diagnostic prompt preparation.

No requests, model loading, token audit, or Stage 2--7 execution. SQLite stores
query profiles on temporary disk; at most one query's candidates/context is live.
Diagnostic prompts remain NOT_TOKEN_AUDITED and are never executed by this tool.
"""
import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from src.rag import phase2_common as common
from src.rag import build_train_evidence_v5 as empirical
from src.rag import positive_evidence_v5 as gate
from src.rag import run_provenance_aware_reranking as reranking
from src.rag import build_trusted_contexts as contexts
from src.rag import verify_retrieval_provenance_v3 as provenance
from src.rag import build_evidence_index_v2 as index

BASE = "results/rag/v5e_validation/pre_llm/"
FORBIDDEN = {"true_class_id","true_class_label","classifier_correct","ground_truth","target_id","target_binary"}
require = common.require


def coverage_gate(groups, *, method=empirical):
    # Operational viability, not an evidence threshold or a target status rate.
    return bool(groups) and all(row["contexts_with_claims"] / row["queries"] >= method.POLICY["min_coverage_fraction"]
                                and row["contexts_with_claims"] >= method.POLICY["diagnostic_per_stratum"] for row in groups)


def scan(root, registry, chunks, verified, *, method=empirical):
    development = method.DEVELOPMENT_ID
    context_paths, query_path, _, _, _, rank_paths = contexts.paths_for(development)
    rank_config = common.completed(root, rank_paths[2], rank_paths[3])
    require(rank_config.get("development_id")==development and rank_config.get("profile")=="task_relevance"
            and rank_config.get("task_alignment_weight")=="0.05" and rank_config.get("relationship_weight")=="0.05",
            "Registered reranking configuration differs")
    config = common.completed(root, context_paths[2], context_paths[3])
    require(config.get("development_id")==development, "Context development identity differs")
    require(all(config.get(k)==v for k,v in dict(profile="evidence_diverse",top_n="5",token_budget="2048",
                                               max_chunks_per_document="0",diversity_threshold="1.0").items()), "Registered context configuration differs")
    groups, diagnostics = {}, {}
    with TemporaryDirectory(prefix="v5-coverage-") as temporary:
        with sqlite3.connect(str(Path(temporary)/"queries.sqlite")) as db:
            db.execute("PRAGMA cache_size=-8192")
            db.execute("CREATE TABLE q (id TEXT PRIMARY KEY, payload TEXT, ranks INTEGER DEFAULT 0, matched INTEGER DEFAULT 0, predicted INTEGER DEFAULT 0, alternative INTEGER DEFAULT 0, context INTEGER DEFAULT 0)")
            counts = Counter()
            for query in common.rows(root, query_path):
                require(not FORBIDDEN.intersection(query), "Evaluation labels prohibited in coverage")
                common.validate_query(query)
                require("-VAL-" in query["query_id"] and "-TEST-" not in query["query_id"], "Non-validation query")
                # Keep only fields consumed by matching and the separate validator.
                value = {k:query[k] for k in ("query_id","source_dataset","predicted_class_label","retrieval_query_text")}
                db.execute("INSERT INTO q (id,payload) VALUES (?,?)", (query["query_id"],common.json_text(value)))
                counts[query["source_dataset"]] += 1
            require(dict(counts)==common.QUERY_COUNTS, "Incomplete full validation population")
            db.execute("CREATE TABLE candidate (query TEXT, chunk TEXT, rank INTEGER, PRIMARY KEY(query,chunk), UNIQUE(query,rank))")
            claims = method.derived_tables(list(registry.values()))[2]
            for row in common.rows(root, rank_paths[0], reranking.TASK_HEADER):
                found = db.execute("SELECT payload,ranks FROM q WHERE id=?",(row["query_id"],)).fetchone()
                require(found is not None, "Unknown reranking query")
                query, mask = common.parse_json(found[0]),found[1]
                chunk = chunks.get(row["chunk_id"])
                require(chunk is not None and row["source_dataset"]==query["source_dataset"]
                        and row["document_id"]==chunk["document_id"] and row["provenance_status"]=="VERIFIED"
                        and row["authorized_for_retrieval"]=="YES", "Unverified reranking row")
                matched = reranking.matching_decision_claims(query,chunk,claims)
                ids = sorted(r["claim_id"] for r in matched)
                classes = sorted({r["supported_class"] for r in matched})
                features = sorted({r["canonical_feature"] for r in matched})
                require(int(row["decision_claim_match_count"])==len(ids)
                        and common.parse_json(row["decision_claim_ids_json"])==ids
                        and common.parse_json(row["decision_claim_classes_json"])==classes
                        and common.parse_json(row["decision_claim_features_json"])==features, "Stage-6 claim audit differs")
                rank = int(row["final_rank"])
                require(1<=rank<=10 and not mask & (1 << (rank-1)), "Invalid/duplicate rerank rank")
                prediction = query["predicted_class_label"]
                db.execute("INSERT INTO candidate VALUES (?,?,?)",(row["query_id"],row["chunk_id"],rank))
                db.execute("UPDATE q SET ranks=?,matched=matched+?,predicted=MAX(predicted,?),alternative=MAX(alternative,?) WHERE id=?",
                           (mask | (1 << (rank-1)),len(ids),int(prediction in classes),int(bool(set(classes)-{prediction})),row["query_id"]))
            require(db.execute("SELECT COUNT(*) FROM q WHERE ranks!=1023").fetchone()[0]==0, "Incomplete Stage-6 population")
            for row in common.rows(root, context_paths[0], contexts.HEADER):
                require(not FORBIDDEN.intersection(row) and row["source_split"]=="validation"
                        and row["rag_partition"]=="development_validation", "Non-validation/labelled context")
                found = db.execute("SELECT payload,matched,predicted,alternative,context FROM q WHERE id=?",(row["query_id"],)).fetchone()
                require(found is not None and found[4]==0, "Unknown/duplicate context")
                query = common.parse_json(found[0])
                require(row["source_dataset"]==query["source_dataset"] and row["classifier_prediction"]==query["predicted_class_label"]
                        and row["traffic_profile"]==query["retrieval_query_text"], "Context/query mismatch")
                evidence = common.parse_json(row["evidence_records_json"])
                require(len(evidence)<=5 and common.parse_json(row["selected_chunk_ids"])==[e["chunk_id"] for e in evidence]
                        and row["context_text"]==contexts.context_text(evidence) and int(row["context_token_count"])<=2048, "Context integrity differs")
                for item in evidence:
                    candidate = db.execute("SELECT rank FROM candidate WHERE query=? AND chunk=?",(row["query_id"],item["chunk_id"])).fetchone()
                    require(candidate and candidate[0]==int(item["rerank_rank"]), "Context candidate/rank differs")
                    expected = reranking.matching_decision_claims(query,chunks[item["chunk_id"]],claims)
                    require(common.parse_json(item.get("decision_claim_ids_json","[]"))==sorted(c["claim_id"] for c in expected)
                            and int(item.get("decision_claim_match_count",0))==len(expected)
                            and common.parse_json(item.get("decision_claim_classes_json","[]"))==sorted({c["supported_class"] for c in expected})
                            and common.parse_json(item.get("decision_claim_features_json","[]"))==sorted({c["canonical_feature"] for c in expected}), "Context claim audit differs")
                payload = dict(query_id=row["query_id"],source_dataset=row["source_dataset"],source_split="validation",
                               classifier_prediction=row["classifier_prediction"],traffic_profile=row["traffic_profile"],trusted_evidence=evidence)
                matches = gate.qualifying(payload,registry,verified,method=method)
                key = (row["source_dataset"],row["classifier_prediction"])
                stats = groups.setdefault(key,Counter())
                stats.update(queries=1,queries_with_matched_claims=int(found[1]>0),predicted_class_support=found[2],
                             alternative_class_support=found[3],stage6_claim_matches=found[1],contexts_with_claims=int(bool(matches)),
                             contexts_with_any_matched_claim=int(any(int(e.get("decision_claim_match_count",0))>0 for e in evidence)))
                if matches:
                    # Hash-priority selection is seed-specific and independent of input order,
                    # labels, correctness and evidence status. Maximum 10 prompts per stratum.
                    priority = hashlib.sha256(f"42|{row['query_id']}".encode()).hexdigest()
                    sample = diagnostics.setdefault(key,[])
                    sample.append((priority,row["query_id"],payload))
                    sample.sort(key=lambda v:(v[0],v[1]))
                    del sample[method.POLICY["diagnostic_per_stratum"]:]
                db.execute("UPDATE q SET context=1 WHERE id=?",(row["query_id"],))
            require(db.execute("SELECT COUNT(*) FROM q WHERE context!=1").fetchone()[0]==0, "Incomplete Stage-7 population")
    expected = {(d,c) for d in common.CLASSES for c in common.CLASSES[d]}
    require(set(groups)==expected, "Missing validation prediction stratum")
    summaries = [dict(source_dataset=d,prediction=c,**groups[d,c]) for d,c in sorted(groups)]
    passed = coverage_gate(summaries,method=method)
    prompts = []
    if passed:
        for key in sorted(diagnostics):
            for _,identity,payload in diagnostics[key]:
                prompts.append(dict(query_id=identity,source_dataset=key[0],classifier_prediction=key[1],source_split="validation",
                                    prompt_text=gate.prompt(payload,registry,verified,method=method),token_audit_status="NOT_TOKEN_AUDITED",
                                    diagnostic_scope="claim-covered_stratified_validation_not_population_estimate"))
    fields = ("queries","queries_with_matched_claims","predicted_class_support","alternative_class_support","stage6_claim_matches","contexts_with_claims","contexts_with_any_matched_claim")
    totals = {k:sum(r[k] for r in summaries) for k in fields}
    by_dataset = {d:{k:sum(r[k] for r in summaries if r["source_dataset"]==d) for k in fields} for d in common.CLASSES}
    return dict(status="PASS" if passed else "STOP_NEGLIGIBLE_CLAIM_COVERAGE",strata=summaries,by_dataset=by_dataset,
                totals=totals,test_rows=0,ground_truth_used=0,diagnostic_prompts=len(prompts),
                next_step="Manual local Qwen token audit and dedicated V5 diagnostic runner review required; do not use V4 runtime"),prompts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path("."))
    parser.add_argument("--development-id",choices=("v5e_validation","v5f_validation"),default="v5e_validation")
    parser.add_argument("--execute",action="store_true",help="Publish audit and, only on PASS, bounded diagnostic prompts; never run an LLM")
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    method = empirical
    if args.development_id == "v5f_validation":
        from src.rag import build_semantic_association_v5 as method
    base = f"results/rag/{method.DEVELOPMENT_ID}/pre_llm/"
    targets = [base+"coverage_audit.json",base+"diagnostic_prompts.jsonl"]
    if args.execute:
        require(not any(empirical.path(root,p,False).exists() for p in targets), "Pre-LLM outputs already exist")
    records = method.verify_bundle(root)
    # Stage-2 rechecks KEEP/source-location rules and empirical source linkage.
    loader = method.load_decision_claims if args.development_id == "v5f_validation" else empirical.corpus.load_decision_claims
    saved_claims = loader(root,method.DEVELOPMENT_ID)
    require(saved_claims==method.derived_tables(records)[2], "Stage-2 empirical registry differs")
    registry = {r["claim_id"]:r for r in records if r["decision"]=="APPROVED"}
    chunks = common.unique(index.validated_chunks(root,inputs=index.input_paths(method.DEVELOPMENT_ID),outputs=index.output_paths(method.DEVELOPMENT_ID)),"chunk_id")
    verified = provenance.load_verified(root,method.DEVELOPMENT_ID)
    verified = {k:dict(v,chunk_text=chunks[k]["chunk_text"]) for k,v in verified.items()}
    before = common.snapshot(root, (*contexts.paths_for(method.DEVELOPMENT_ID)[0], *reranking.paths_for(method.DEVELOPMENT_ID)[0], contexts.paths_for(method.DEVELOPMENT_ID)[1]))
    report,prompts = scan(root,registry,chunks,verified,method=method)
    if args.development_id == "v5f_validation":
        report["claim_audit"] = method.audit(records,method.freeze(root))
        report["contexts_with_claims_definition"] = "Approved observed-state claims involving the evaluated hypothesis; other pairwise claims remain contextual"
    require(before==common.snapshot(root,[r["path"] for r in before]), "Coverage inputs changed")
    if args.execute:
        target = empirical.path(root,targets[0],False)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("x",encoding="utf-8") as stream:
            stream.write(common.json_text(report)+"\n")
        if report["status"]=="PASS":
            with empirical.path(root,targets[1],False).open("x",encoding="utf-8") as stream:
                for row in prompts:
                    stream.write(common.json_text(row)+"\n")
    print(common.json_text(report))
    return 0 if report["status"]=="PASS" else 2


if __name__=="__main__":
    raise SystemExit(main())
