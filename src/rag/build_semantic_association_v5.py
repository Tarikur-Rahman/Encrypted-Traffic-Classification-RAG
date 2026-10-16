"""TRAIN semantic-state associations, separately registered from median-v1.

CLI never accesses validation/TEST row files. --stage2 reuses existing corpus
construction without modifying the code hashes in the preserved v5e bundle.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from collections import Counter

from src.rag import build_train_evidence_v5 as previous
from src.rag import phase2_common as common

corpus = previous.corpus
require = common.require
path = previous.path
SOURCE_TYPE = previous.SOURCE_TYPE
VERSION = "train-semantic-association-v2"
DEVELOPMENT_ID = "v5f_validation"
BASE = "data/raw/rag/evidence_sources_v2/v5f_validation/"
CONFIG = "configs/rag_semantic_association_v5.json"
STATES = ("LOWER", "TYPICAL", "HIGHER")
DIRECTIONS = dict(LOWER="lower", TYPICAL="typical", HIGHER="higher")
POLICY = dict(version=VERSION, development_id=DEVELOPMENT_ID, seed=42,
              states=list(STATES), minimum_groups_ustc=6, minimum_groups_cic=20,
              minimum_prevalence_difference=0.1, minimum_delta=0.474,
              bootstrap_replicates=10000, confidence_level=0.95,
              interval_scope="marginal_exploratory_not_familywise", planned_candidates=1242,
              min_coverage_fraction=0.01, diagnostic_per_stratum=10)
EXTRA = ("evidence_method_version", "semantic_state", "target_group_count", "contrast_group_count",
         "target_prevalence", "contrast_prevalence", "effect_size", "bootstrap_interval_json",
         "minimum_loo_difference", "minimum_loo_delta", "source_artifacts_json", "configuration_sha256",
         "code_sha256", "statistics_link", "source_type")
CLAIM_HEADER = (*corpus.DECISION_CLAIM_HEADER, *EXTRA)
ARTIFACTS = (*previous.ARTIFACTS, "group_counts.npz")


def freeze(root):
    # Reuse the exact frozen sources/boundary readers and old configuration
    # integrity check; do not mutate their constants or negative experiment.
    result = previous.freeze(root)
    require(json.loads(previous.path(root,CONFIG).read_text(encoding="utf-8")) == POLICY, "Association policy differs")
    result = dict(result, version=VERSION, configuration_sha256=previous.sha(previous.path(root,CONFIG)),
                  parent_configuration_sha256=result["configuration_sha256"])
    result["code_sha256"] = dict(result["code_sha256"], **{
        "src/rag/build_semantic_association_v5.py":previous.sha(previous.path(root,"src/rag/build_semantic_association_v5.py"))})
    return result


def semantic_states(values):
    """Exact aggregate of frozen Stage-4 bins, including -0.5 and +0.5."""
    import numpy as np
    values = np.asarray(values,dtype=float)
    require(np.isfinite(values).all(), "Nonfinite standardized feature")
    return np.where(values <= -0.5, 0, np.where(values >= 0.5, 2, 1))


def bootstrap_medians(values, rng, replicates):
    """Ordinary equal-group bootstrap via exact empirical histogram sampling.

    Multinomial counts avoid repeatedly allocating tens of thousands of groups.
    This is distributionally identical to iid resampling group prevalences.
    """
    import numpy as np
    support, counts = np.unique(values,return_counts=True)
    n = len(values)
    output = np.empty(replicates)
    batch = max(1,min(128,1000000//len(support)))
    for start in range(0,replicates,batch):
        samples = rng.multinomial(n,counts/n,size=min(batch,replicates-start)).cumsum(axis=1)
        lo = (samples > (n-1)//2).argmax(axis=1)
        hi = (samples > n//2).argmax(axis=1)
        output[start:start+len(samples)] = (support[lo]+support[hi])/2
    return output


def leave_one_medians(values):
    import numpy as np
    a = np.sort(values)
    n = len(a)
    require(n>=2,"At least two groups required")
    deleted = np.arange(n)
    left, right = (n-2)//2, (n-1)//2
    return (a[left+(deleted<=left)] + a[right+(deleted<=right)])/2


def assess(a,b,minimum_groups,seed):
    import numpy as np
    a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    require(len(a)>=2 and len(b)>=2 and np.isfinite(a).all() and np.isfinite(b).all()
            and ((0<=a)&(a<=1)).all() and ((0<=b)&(b<=1)).all(), "Invalid group prevalences")
    target,contrast = float(np.median(a)),float(np.median(b))
    effect = previous.delta(a,b)
    loo_difference = min(float(leave_one_medians(a).min()-contrast),float(target-leave_one_medians(b).max()))
    sa,sb = np.sort(a),np.sort(b)
    total = effect*len(a)*len(b)
    ca = np.searchsorted(sb,a,"left")-(len(b)-np.searchsorted(sb,a,"right"))
    cb = (len(a)-np.searchsorted(sa,b,"right"))-np.searchsorted(sa,b,"left")
    loo_delta = min(float(((total-ca)/((len(a)-1)*len(b))).min()),float(((total-cb)/(len(a)*(len(b)-1))).min()))
    rng = np.random.default_rng(seed)
    differences = bootstrap_medians(a,rng,POLICY["bootstrap_replicates"])-bootstrap_medians(b,rng,POLICY["bootstrap_replicates"])
    tail = (1-POLICY["confidence_level"])/2
    ci = [float(x) for x in np.quantile(differences,[tail,1-tail])]
    reasons=[]
    if min(len(a),len(b)) < minimum_groups: reasons.append("INSUFFICIENT_GROUPS")
    if target-contrast < POLICY["minimum_prevalence_difference"]: reasons.append("SMALL_PREVALENCE_DIFFERENCE")
    if effect < POLICY["minimum_delta"]: reasons.append("SMALL_OR_OPPOSITE_GROUP_EFFECT")
    if loo_difference<=0 or loo_delta<=0: reasons.append("LEAVE_ONE_GROUP_OUT_SIGN_UNSTABLE")
    if ci[0]<=0: reasons.append("BOOTSTRAP_INTERVAL_INCLUDES_ZERO")
    return dict(target_group_count=len(a),contrast_group_count=len(b),target_prevalence=target,
                contrast_prevalence=contrast,prevalence_difference=target-contrast,effect_size=effect,
                bootstrap_interval=ci,minimum_loo_difference=loo_difference,minimum_loo_delta=loo_delta,
                direction_consistency=float(np.mean(a>contrast)),rejection_reasons=reasons,
                decision="REJECTED" if reasons else "APPROVED")


def group_state_counts(frame,spec,raw):
    """Numerators/denominators retained; rows are never independent replicates."""
    import numpy as np
    keys=[spec["label"],spec["group"]]
    grouped=frame.groupby(keys,sort=True)
    size=grouped.size()
    temporary=frame[keys].copy()
    states=semantic_states(frame[raw].to_numpy())
    values=[]
    for state in range(3):
        temporary["state_hit"]=(states==state).astype("int64")
        values.append(temporary.groupby(keys,sort=True)["state_hit"].sum().reindex(size.index).to_numpy())
    counts=np.stack(values,axis=1)
    require((counts.sum(axis=1)==size.to_numpy()).all(),"State counts do not cover TRAIN groups")
    return size,counts


def statistics(root,frozen):
    import numpy as np
    import pandas as pd
    records,arrays=[],{}
    for dataset,spec in previous.SPECS.items():
        features=previous.feature_map(root,dataset,spec)
        boundary=previous.train_boundary(root,spec)
        frame=pd.read_csv(previous.path(root,spec["train"]),usecols=[v[0] for v in features.values()]+[spec["label"],spec["group"]],
                          dtype={spec["label"]:str,spec["group"]:str})
        require(not frame.isna().any().any() and set(frame[spec["label"]])==set(common.CLASSES[dataset]),"Invalid TRAIN population")
        previous.validate_train_membership(frame.groupby(spec["group"]).size().to_dict(),boundary)
        purity=frame.groupby(spec["group"])[spec["label"]].nunique()
        excluded=set(purity[purity!=1].index)
        excluded_rows=int(frame[spec["group"]].isin(excluded).sum())
        frame=frame[~frame[spec["group"]].isin(excluded)]
        prefix="ustc" if dataset=="USTC-TFC2016" else "cic"
        group_totals=frame.groupby([spec["label"],spec["group"]],sort=True).size()
        labels=group_totals.index.get_level_values(0).to_numpy(dtype=str)
        if prefix=="ustc": require(not excluded and set(Counter(labels).values())=={6},"Frozen TRAIN family design differs")
        arrays[prefix+"_group_ids"]=np.asarray([hashlib.sha256(f"{dataset}|{g}".encode()).hexdigest() for _,g in group_totals.index])
        arrays[prefix+"_classes"]=labels
        arrays[prefix+"_row_counts"]=group_totals.to_numpy()
        for f,(feature,(raw,_,_)) in enumerate(sorted(features.items())):
            size,counts=group_state_counts(frame,spec,raw)
            key=f"{prefix}_feature_{f:03d}"
            arrays[key]=counts
            prevalence=counts/size.to_numpy()[:,None]
            for first,second in itertools.combinations(common.CLASSES[dataset],2):
                for state_number,state in enumerate(STATES):
                    a,b=prevalence[labels==first,state_number],prevalence[labels==second,state_number]
                    # One two-sided comparison per unordered pair/state; orientation
                    # follows measured median prevalence, never desired class counts.
                    supported,contrasted=first,second
                    if np.median(a)<np.median(b): a,b,supported,contrasted=b,a,second,first
                    identity="ASSOC-"+hashlib.sha256(f"{VERSION}|{dataset}|{first}|{second}|{feature}|{state}".encode()).hexdigest()[:20]
                    row=assess(a,b,POLICY["minimum_groups_ustc" if prefix=="ustc" else "minimum_groups_cic"],42+int(identity[6:14],16))
                    row.update(claim_id=identity,evidence_method_version=VERSION,source_dataset=dataset,
                               supported_class=supported,contrasted_class=contrasted,reference_classes=[contrasted],
                               canonical_feature=feature,semantic_state=state,direction=DIRECTIONS[state],source_type=SOURCE_TYPE,
                               group_key=spec["group"],group_counts_key=key,excluded_conflicting_groups=len(excluded),excluded_conflicting_rows=excluded_rows,
                               source_artifacts={p:frozen["inputs"][p] for p in (spec["train"],spec["scaler"],spec["boundary"])},
                               configuration_sha256=frozen["configuration_sha256"],code_sha256=frozen["code_sha256"]["src/rag/build_semantic_association_v5.py"],
                               statistics_link=BASE+"statistics.json#"+identity)
                    records.append(row)
        del frame
    require(len(records)==POLICY["planned_candidates"],"Association comparison population differs")
    return sorted(records,key=lambda r:r["claim_id"]),arrays


def render(row):
    # Compact full-line rendering; detailed hashes/counts reside in the linked,
    # verified statistics row and claim CSV, not repeated inside model context.
    return (f"Claim {row['claim_id']} ({VERSION}; {SOURCE_TYPE}): In the frozen {row['source_dataset']} TRAIN corpus, "
            f"{row['semantic_state']} {row['canonical_feature']} occurred more frequently among {row['supported_class']} groups "
            f"than {row['contrasted_class']} groups under the registered group-level association criterion. "
            f"Median prevalences {row['target_prevalence']:.8g} versus {row['contrast_prevalence']:.8g}; "
            f"Cliff delta {row['effect_size']:.8g}; bootstrap interval {common.json_text(row['bootstrap_interval'])}. "
            "This is an empirical association, not causation or proof of true class.")


def derived_tables(records):
    sources,docs,claims=[],[],[]
    for row in records:
        if row["decision"]!="APPROVED": continue
        identity,text=row["claim_id"],render(row)
        source=dict.fromkeys(corpus.v1.MANIFEST_HEADER,"")
        source.update(document_id=identity,title="Frozen TRAIN pairwise semantic association "+identity,source_type=SOURCE_TYPE,
                      publisher="Project TRAIN empirical derivation (not external)",local_path=BASE+"statistics.json",
                      traffic_topics=row["source_dataset"],security_topics="Noncausal class-conditional TRAIN association",
                      corpus_partition="trusted",source_provenance_status="VERIFIED",authorized_for_retrieval="YES",
                      curation_notes="Registered semantic-association gate; provenance="+row["statistics_link"])
        docs.append(dict(source,extraction_status="PASS",extraction_review_status="APPROVED",character_count=str(len(text)),
                         word_count=str(len(text.split())),normalized_text=text,extraction_notes="Deterministic derived association; no paper quotation"))
        sources.append(source)
        claim=dict(zip(corpus.DECISION_CLAIM_HEADER,(identity,identity,row["source_dataset"],"document",text,row["canonical_feature"],
                   row["direction"],row["supported_class"],row["contrasted_class"],"APPROVED","YES")))
        for field in EXTRA:
            value=row.get(field)
            if field=="bootstrap_interval_json": value=common.json_text(row["bootstrap_interval"])
            if field=="source_artifacts_json": value=common.json_text(row["source_artifacts"])
            claim[field]=str(value)
        claims.append(claim)
    quality=[r for doc in docs for r in corpus.quality_rows(doc["document_id"],corpus.source_units(doc["normalized_text"]),"evidence_quality")]
    corpus.validate_decision_claim_rows(docs,quality,[{k:c[k] for k in corpus.DECISION_CLAIM_HEADER} for c in claims])
    return sources,docs,claims


def audit(records,frozen):
    approved=[r for r in records if r["decision"]=="APPROVED"]
    return dict(evidence_method_version=VERSION,candidate_claims=len(records),approved_claims=len(approved),rejected_claims=len(records)-len(approved),
                rejection_reasons=dict(Counter(reason for r in records for reason in r["rejection_reasons"])),
                approved_by_dimension={k:dict(Counter(r[k] for r in approved)) for k in ("source_dataset","supported_class","contrasted_class","canonical_feature","semantic_state")},
                approved_by_dataset_class_contrast=dict(Counter("/".join(r[k] for k in ("source_dataset","supported_class","contrasted_class")) for r in approved)),
                duplicate_claims=len(records)-len({r["claim_id"] for r in records}),validation_rows_used=0,test_rows_used=0,
                validation_correctness_used=0,provenance_failures=0,interval_scope=POLICY["interval_scope"],provenance=frozen)


def verify_bundle(root):
    frozen=freeze(root)
    bundle=json.loads(previous.path(root,BASE+"bundle.json").read_text(encoding="utf-8"))
    require(bundle["provenance"]==frozen and set(bundle["files"])==set(ARTIFACTS)-{"bundle.json"},"Association bundle identity differs")
    require(all(previous.sha(previous.path(root,BASE+n))==h for n,h in bundle["files"].items()),"Association artifact hash mismatch")
    records=json.loads(previous.path(root,BASE+"statistics.json").read_text(encoding="utf-8"))
    for name,rows in zip(previous.ARTIFACTS[1:4],derived_tables(records)):
        require(list(previous.read_csv(root,BASE+name))==rows,"Association source/claim rendering differs")
    require(json.loads(previous.path(root,BASE+"claim_audit.json").read_text(encoding="utf-8"))==audit(records,frozen),"Association audit differs")
    return records


def verify_documents(root,development_id,documents):
    require(development_id==DEVELOPMENT_ID,"Wrong association development identity")
    expected={d["document_id"]:d for d in derived_tables(verify_bundle(root))[1]}
    actual={d["document_id"]:d for d in documents if d["source_type"]==SOURCE_TYPE}
    require(set(actual)==set(expected),"Association document population differs")
    require(all(all(str(actual[i][k])==str(v) for k,v in d.items()) for i,d in expected.items()),"Association document altered")


def load_decision_claims(root,development_id):
    require(development_id==DEVELOPMENT_ID,"Wrong association identity")
    outputs=corpus.output_paths(development_id)
    _,docs=corpus.v1.read_table(root,outputs[0],corpus.DOCUMENT_HEADER)
    _,quality=corpus.v1.read_table(root,outputs[3],corpus.IMPROVED_QUALITY_HEADER)
    verify_documents(root,development_id,docs)
    header,claims=corpus.v1.read_table(root,corpus.decision_claim_path(development_id),CLAIM_HEADER)
    require(tuple(header)==CLAIM_HEADER and claims==derived_tables(verify_bundle(root))[2],"Association claim registry altered")
    corpus.validate_decision_claim_rows(docs,quality,[{k:c[k] for k in corpus.DECISION_CLAIM_HEADER} for c in claims])
    return claims


def stage2(root,execute):
    records=verify_bundle(root)
    sources,added,claims=derived_tables(records)
    require(claims,"No approved associations; stop before Stage 2")
    paths=(*corpus.output_paths(DEVELOPMENT_ID),corpus.decision_claim_path(DEVELOPMENT_ID))
    require(not any(previous.path(root,p,False).exists() for p in paths),"Stage-2 outputs exist; refusing overwrite")
    docs,manifest,review,quality,before=corpus.build(root,"evidence_quality",*previous.CONTEXT_FILES,DEVELOPMENT_ID)
    for doc in added:
        counts=corpus.term_counts(doc["normalized_text"])
        docs.append(dict(doc,corpus_revision="V2",is_new_document="YES",**counts,task_aligned="YES" if any(counts.values()) else "NO"))
        quality.extend(corpus.quality_rows(doc["document_id"],corpus.source_units(doc["normalized_text"]),"evidence_quality"))
        review.append(dict(document_id=doc["document_id"],review_status="APPROVED",review_notes="Registered deterministic TRAIN association rendering"))
    manifest.extend(sources)
    corpus.validate_decision_claim_rows(docs,quality,[{k:c[k] for k in corpus.DECISION_CLAIM_HEADER} for c in claims])
    verify_documents(root,DEVELOPMENT_ID,docs)
    checks=corpus.checks(docs,manifest,review,quality,before,"evidence_quality",DEVELOPMENT_ID,claims)
    for row in checks:
        if row.get("check")=="decision_claim_policy": row["details"]="Registered TRAIN semantic-association gate; independently hash verified"
    if execute:
        headers=(corpus.DOCUMENT_HEADER,corpus.v1.MANIFEST_HEADER,corpus.v1.REVIEW_HEADER,corpus.IMPROVED_QUALITY_HEADER,corpus.SUMMARY_HEADER,corpus.CHECK_HEADER,CLAIM_HEADER)
        tables=(docs,sorted(manifest,key=lambda r:r["document_id"]),sorted(review,key=lambda r:r["document_id"]),
                sorted(quality,key=lambda r:(r["document_id"],int(r["unit_order"]))),corpus.summary_rows(docs,quality),checks,claims)
        require(before==corpus.file_metadata(root,[r["path"] for r in before]),"Stage-2 inputs changed")
        for name,header,rows in zip(paths,headers,tables):
            target=previous.path(root,name,False); target.parent.mkdir(parents=True,exist_ok=True)
            corpus.v1.write_table(target,header,rows,mode="x")
    print("EVIDENCE_CORPUS_ASSOCIATION="+("PASS" if execute else "READY_FOR_CONSTRUCTION"))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=Path("."))
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--audit-only",action="store_true")
    parser.add_argument("--stage2",action="store_true")
    args=parser.parse_args(argv); root=args.project_root.resolve(strict=True)
    corpus.validate_development_id(DEVELOPMENT_ID)
    require(not args.audit_only or not(args.execute or args.stage2),"Conflicting action flags")
    if args.stage2: stage2(root,args.execute); return 0
    if args.audit_only:
        import numpy as np
        records=verify_bundle(root)
        reproduced,arrays=statistics(root,freeze(root))
        require(records==reproduced,"Independent association recomputation differs")
        with np.load(previous.path(root,BASE+"group_counts.npz"),allow_pickle=False) as saved:
            require(set(saved.files)==set(arrays) and all(np.array_equal(saved[k],v) for k,v in arrays.items()),"Group counts differ")
        print(common.json_text(audit(records,freeze(root)))); return 0
    require(args.execute,"Choose --execute, --audit-only, or --stage2")
    require(not any(previous.path(root,BASE+n,False).exists() for n in ARTIFACTS),"Association outputs exist; refusing overwrite")
    frozen=freeze(root); records,arrays=statistics(root,frozen); tables=derived_tables(records)
    require(frozen==freeze(root),"Frozen sources changed")
    import numpy as np
    directory=(root/BASE)
    previous.path(root,BASE+"bundle.json",False)  # path/link guard before directory creation
    directory.mkdir(parents=True,exist_ok=True)
    for name,value in (("statistics.json",records),("claim_audit.json",audit(records,frozen))):
        with (directory/name).open("x",encoding="utf-8") as stream: stream.write(common.json_text(value)+"\n")
    with (directory/"group_counts.npz").open("xb") as stream: np.savez_compressed(stream,**arrays)
    for name,header,rows in zip(previous.ARTIFACTS[1:4],(corpus.v1.MANIFEST_HEADER,corpus.v1.DOCUMENT_HEADER,CLAIM_HEADER),tables):
        corpus.v1.write_table(directory/name,header,rows,mode="x")
    bundle=dict(provenance=frozen,files={n:previous.sha(directory/n) for n in ARTIFACTS if n!="bundle.json"})
    with (directory/"bundle.json").open("x",encoding="utf-8") as stream: stream.write(common.json_text(bundle)+"\n")
    print(common.json_text(audit(records,frozen))); return 0


if __name__=="__main__":
    raise SystemExit(main())
