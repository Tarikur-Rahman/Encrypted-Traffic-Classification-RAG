"""Transparent single-method reranking; neutral inverse-rank default."""
import re
from collections import defaultdict

from src.rag import phase2_common as common
from src.rag import verify_retrieval_provenance_v3 as provenance
from src.rag import build_evidence_index_v2 as index

BASE = "results/rag/reranking/"
OUTPUTS = tuple(BASE + name + ".csv" for name in ("reranked_evidence", "reranking_summary", "reranking_configuration", "reranking_validation_checklist"))
HEADER = tuple("query_id source_dataset chunk_id document_id source_type original_method original_rank original_score provenance_status authorized_for_retrieval retrieval_rank_component retrieval_weight source_preference_component source_preference_weight task_alignment_component task_alignment_weight final_rerank_score final_rank".split())
TASK_HEADER = (*HEADER, "relationship_richness_component", "relationship_weight", "retrieval_intents_json", "reranking_profile",
               "decision_claim_match_count", "decision_claim_ids_json", "decision_claim_classes_json", "decision_claim_features_json")
PROFILES = ("legacy", "task_relevance")
STOP_WORDS = frozenset("about above after again against all also analysis and any are around based been between can class classes classification classifier compared describes different during each evidence features from have into its may more most network observation observed other packet packets profile protocol relative relevant security that their these they this through traffic using were what when which with".split())
require = common.require


def paths_for(development_id=None):
    if development_id is None:
        return OUTPUTS, common.QUERY_PATH, provenance.RESULTS, provenance.output_paths(), index.INPUTS, index.OUTPUTS
    index.corpus_v2.validate_development_id(development_id)
    base = f"results/rag/{development_id}/stage06_reranking/"
    outputs = tuple(base+name+".csv" for name in ("reranked_evidence", "reranking_summary", "reranking_configuration", "reranking_validation_checklist"))
    retrieval = provenance.retrieval_paths(development_id)
    query = f"results/rag/{development_id}/stage04_retrieval/query_records.csv"
    return outputs, query, retrieval[0], provenance.output_paths(development_id), index.input_paths(development_id), index.output_paths(development_id)


def observable_terms(query):
    """Observed-profile terms only; discard every label in this dataset's vocabulary."""
    text = query.get("retrieval_query_text", "")
    marker = "Observed traffic profile relative to the training distribution: "
    if marker in text:
        text = text.split(marker, 1)[1].split(". Retrieve", 1)[0]
    for label in common.CLASSES[query["source_dataset"]]:
        text = re.sub(rf"\b{re.escape(label)}\b", " ", text, flags=re.I)
    return frozenset(term.casefold() for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text)
                     if term.casefold() not in STOP_WORDS)


def relevance_components(query, evidence_text):
    """Observed-feature overlap only; claim matching is evaluated separately."""
    terms = observable_terms(query)
    words = {term.casefold() for term in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", evidence_text)}
    overlap = len(terms & words) / max(1, len(terms))
    return overlap, 0.0


PROFILE_DIRECTIONS = {
    "moderately high": "higher", "high": "higher", "very high": "higher", "extremely high": "higher",
    "moderately low": "lower", "low": "lower", "very low": "lower", "extremely low": "lower",
    "typical": "typical",
}


def observed_profile_directions(query):
    text = query.get("retrieval_query_text", "")
    markers = ("Observed traffic profile relative to the training distribution: ", "Observed traffic profile: ")
    profile_found = False
    for marker in markers:
        if marker in text:
            text = text.split(marker, 1)[1].split(". Retrieve", 1)[0]
            profile_found = True
            break
    if not profile_found:
        return {}
    bins = "|".join(re.escape(value) for value in sorted(PROFILE_DIRECTIONS, key=len, reverse=True))
    found = {}
    pattern = re.compile(rf"([^,;\n.]+?)\s+is\s+({bins})(?=$|[,;\n.])", re.I)
    for match in pattern.finditer(text):
        feature = match.group(1).strip()
        direction = PROFILE_DIRECTIONS[match.group(2).casefold()]
        if feature:
            found[feature] = direction
    return found


def matching_decision_claims(query, chunk, claims, profile_directions=None):
    directions = profile_directions if profile_directions is not None else observed_profile_directions(query)
    matched = []
    for claim in claims:
        require(claim["review_status"] == "APPROVED" and claim["authorized_for_decision_evidence"] == "YES",
                "Unapproved decision claim supplied to reranking")
        if "evidence_method_version" in claim:
            from src.rag import build_semantic_association_v5 as association
            require(claim["evidence_method_version"] == association.VERSION
                    and claim["semantic_state"] in association.DIRECTIONS
                    and claim["direction"] == association.DIRECTIONS[claim["semantic_state"]]
                    and claim["source_type"] == association.SOURCE_TYPE
                    and claim["supported_class"] in common.CLASSES.get(claim["source_dataset"], ())
                    and claim["contrasted_class"] in common.CLASSES.get(claim["source_dataset"], ())
                    and claim["supported_class"] != claim["contrasted_class"], "Invalid association claim semantics")
        if (claim["source_dataset"] == query["source_dataset"] and
                claim["document_id"] == chunk["document_id"] and
                claim["evidence_quote"] in chunk["chunk_text"] and
                claim["canonical_feature"] in directions and
                directions[claim["canonical_feature"]] == claim["direction"]):
            matched.append(claim)
    by_id = {claim["claim_id"]: claim for claim in matched}
    return [by_id[identity] for identity in sorted(by_id)]


def options(args):
    preferences = common.parse_json(args.source_preferences_json)
    require(isinstance(preferences, dict) and set(preferences) <= set(common.corpus.foundation.SOURCE_CATEGORIES), "Unknown source preference")
    preferences = {key: common.finite(value) for key, value in preferences.items()}
    require(all(-1 <= value <= 1 for value in preferences.values()), "Source preference must be in [-1,1]")
    profile = getattr(args, "profile", "legacy")
    require(profile in PROFILES, "Unknown reranking profile")
    task_weight = getattr(args, "task_alignment_weight", None)
    relationship_weight = getattr(args, "relationship_weight", None)
    if profile == "task_relevance":
        require(task_weight is not None and relationship_weight is not None, "Explicit task-alignment and relationship weights required for task_relevance")
        require(not getattr(args, "task_alignments", None), "Task-relevance profile computes its own ranking signals")
    else:
        task_weight = 0 if task_weight is None else task_weight
        relationship_weight = 0 if relationship_weight is None else relationship_weight
    weights = (common.finite(args.retrieval_weight), common.finite(args.source_preference_weight),
               common.finite(task_weight), common.finite(relationship_weight))
    require(all(value >= 0 for value in weights) and 1 <= args.top_k <= 10, "Invalid reranking configuration")
    require(args.method in provenance.METHODS, "Select exactly one method")
    require(not weights[1] or preferences, "Explicit preferences required for nonzero weight")
    require(profile == "task_relevance" or not weights[2] or args.task_alignments,
            "Explicit task components required for nonzero weight")
    return dict(method=args.method, top_k=args.top_k, retrieval_weight=weights[0], source_preference_weight=weights[1],
                task_alignment_weight=weights[2], relationship_weight=weights[3], source_preferences=preferences, profile=profile)


def rerank(query, candidates, verified, config, alignments=None, evidence=None, decision_claims=None):
    alignments = alignments or {}
    require(candidates and len({r["chunk_id"] for r in candidates}) == len(candidates), "Missing/duplicate candidates")
    require(len({int(r["rank"]) for r in candidates}) == len(candidates), "Duplicate ranks")
    output = []
    directions = observed_profile_directions(query) if config.get("profile", "legacy") == "task_relevance" else {}
    if config.get("profile", "legacy") == "task_relevance":
        require(evidence is not None and decision_claims is not None, "Validated Stage-2 decision-claim artifact required")
    for row in candidates:
        require(row["query_id"] == query["query_id"] and row["source_dataset"] == query["source_dataset"] and row["retrieval_method"] == config["method"], "Query/method mismatch")
        record = verified.get(row["chunk_id"])
        require(record is not None and record["provenance_status"] == "VERIFIED" and record["authorized_for_downstream"] == record["authorized_for_retrieval"] == "YES" and record["quality_status"] == "KEEP", "Unauthorized/unknown chunk")
        require(record["document_id"] == row["document_id"], "Document mismatch")
        rank = int(row["rank"])
        require(1 <= rank <= 10, "Invalid original rank")
        original_score = common.finite(row["retrieval_score"])
        key = (query["query_id"], row["chunk_id"])
        if config.get("profile", "legacy") == "task_relevance":
            require(row["chunk_id"] in evidence, "Missing task-relevance evidence text")
            chunk = evidence[row["chunk_id"]]
            task, _ = relevance_components(query,chunk["chunk_text"])
            claims = matching_decision_claims(query, chunk, decision_claims, directions)
            relationship = float(bool(claims))
            intents = common.parse_json(row.get("retrieval_intents_json", "[]"))
            require(isinstance(intents,list) and intents and set(intents) <= set(common.CLASSES[query["source_dataset"]]), "Invalid hypothesis retrieval audit")
        else:
            require(not config["task_alignment_weight"] or key in alignments, "Missing explicit task component")
            task, relationship, intents = common.finite(alignments.get(key, 0)), 0.0, []
        require((0 <= task <= 1 if config.get("profile") == "task_relevance" else -1 <= task <= 1)
                and 0 <= relationship <= 1, "Task ranking signal outside configured range")
        source = config["source_preferences"].get(record["source_type"], 0)
        base = 1/rank
        score = common.finite(config["retrieval_weight"]*base + config["source_preference_weight"]*source + config["task_alignment_weight"]*task + config.get("relationship_weight",0)*relationship)
        output.append(dict(query_id=query["query_id"], source_dataset=query["source_dataset"], chunk_id=row["chunk_id"], document_id=row["document_id"], source_type=record["source_type"],
                           original_method=row["retrieval_method"], original_rank=rank, original_score=original_score, provenance_status="VERIFIED", authorized_for_retrieval="YES",
                           retrieval_rank_component=base, retrieval_weight=config["retrieval_weight"], source_preference_component=source, source_preference_weight=config["source_preference_weight"],
                           task_alignment_component=task, task_alignment_weight=config["task_alignment_weight"], final_rerank_score=score,
                           **({"relationship_richness_component":relationship,"relationship_weight":config["relationship_weight"],
                               "retrieval_intents_json":common.json_text(intents),"reranking_profile":"task_relevance",
                               "decision_claim_match_count":len(claims),
                               "decision_claim_ids_json":common.json_text(sorted({claim["claim_id"] for claim in claims})),
                               "decision_claim_classes_json":common.json_text(sorted({claim["supported_class"] for claim in claims})),
                               "decision_claim_features_json":common.json_text(sorted({claim["canonical_feature"] for claim in claims}))}
                              if config.get("profile")=="task_relevance" else {})))
    output.sort(key=lambda r: (-r["final_rerank_score"], r["original_rank"], r["chunk_id"]))
    return [dict(row, final_rank=i) for i, row in enumerate(output[:config["top_k"]], 1)]


def build(root, args):
    config = options(args)
    development_id = getattr(args, "development_id", None)
    require((config["profile"] == "legacy") == (development_id is None), "Improved profile requires --development-id; legacy profile must omit it")
    outputs, query_path, retrieval_path, provenance_outputs, index_inputs, index_outputs = paths_for(development_id)
    paths = [query_path, retrieval_path, *provenance_outputs]
    decision_claims = None
    if config["profile"] == "task_relevance":
        paths.extend((*index_inputs,index_outputs[0]))
        claims_path = index.corpus_v2.decision_claim_path(development_id)
        paths.append(claims_path)
        if development_id == "v5f_validation":
            from src.rag.build_semantic_association_v5 import load_decision_claims
            decision_claims = load_decision_claims(root, development_id)
        else:
            decision_claims = index.corpus_v2.load_decision_claims(root, development_id)
    if args.task_alignments:
        paths.append(args.task_alignments)
    before = common.snapshot(root, paths)
    queries, verified = common.queries(root, query_path), provenance.load_verified(root,development_id)
    chunks = common.unique(index.validated_chunks(root, inputs=index_inputs, outputs=index_outputs), "chunk_id") if config["profile"] == "task_relevance" else None
    groups = defaultdict(list)
    for row in provenance.validated_retrieval(root, queries, verified, retrieval_path):
        if row["retrieval_method"] == config["method"]:
            groups[row["query_id"]].append(row)
    alignments = {}
    if args.task_alignments:
        for row in common.rows(root, args.task_alignments, ("query_id", "chunk_id", "task_alignment_component")):
            key = (row["query_id"], row["chunk_id"])
            require(key not in alignments and row["query_id"] in groups and any(c["chunk_id"] == row["chunk_id"] for c in groups[row["query_id"]]), "Unknown/duplicate task component")
            alignments[key] = common.finite(row["task_alignment_component"])
    records = [row for identity, query in sorted(queries.items()) for row in
               rerank(query, groups[identity], verified, config, alignments, chunks, decision_claims)]
    require(before == common.snapshot(root, paths), "Inputs changed")
    saved = dict(config, source_preferences=common.json_text(config["source_preferences"]), input_metadata_json=common.json_text(before),
                 stage="PROVENANCE_AWARE_EVIDENCE_RERANKING", score_formula="retrieval_weight/rank + source_preference_weight*source_component + task_alignment_weight*task_component",
                 tie_policy="score descending, original rank ascending, chunk ID ascending", task_components="explicit curated values; not relevance judgments")
    if development_id: saved["development_id"] = development_id
    if config["profile"] == "task_relevance":
        saved.update(profile="task_relevance",score_formula="retrieval/rank plus configurable observed-feature overlap and relationship-richness ranking signals",
            task_components="class vocabulary excluded; ranking signals only; no support, contradiction, or evidence-status labels",
            relationship_signal="exact matches to manually approved Stage-2 decision claims only; no generic linguistic fallback")
    else:
        saved.pop("profile")
        saved.pop("relationship_weight")
    header = TASK_HEADER if config["profile"] == "task_relevance" else HEADER
    return [(outputs[0], header, records), (outputs[1], common.SUMMARY_HEADER, common.summary(records)),
            (outputs[2], common.CONFIG_HEADER, common.config_rows(saved)), (outputs[3], common.corpus.foundation.CHECKLIST_HEADER, common.checks("reranking", {"queries": len(queries), "rows": len(records), "single_method": config["method"], "verified_only": True}))]


def main(argv=None):
    parser = common.parser(__doc__)
    parser.add_argument("--method", choices=provenance.METHODS, default="hybrid_rrf")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--retrieval-weight", type=float, default=1)
    parser.add_argument("--source-preference-weight", type=float, default=0)
    parser.add_argument("--source-preferences-json", default="{}")
    parser.add_argument("--task-alignment-weight", type=float)
    parser.add_argument("--task-alignments", help="Project-relative CSV of explicit query/chunk component values")
    parser.add_argument("--profile", choices=PROFILES, default="legacy")
    parser.add_argument("--relationship-weight", type=float)
    parser.add_argument("--development-id")
    return common.run_tables(parser.parse_args(argv), "PROVENANCE_AWARE_RERANKING", build)


if __name__ == "__main__":
    raise SystemExit(main())
