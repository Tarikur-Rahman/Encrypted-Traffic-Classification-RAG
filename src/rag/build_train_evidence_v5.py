"""Create-only, TRAIN-only empirical evidence for v5e_validation (offline).

No model, validation population, TEST data, or classifier output is read here.
Approval means passing the registered empirical gate, NOT human literature review
and NOT proof of an individual flow's class. Split manifests are boundary metadata;
only their TRAIN entries are selected and no held-out label counts are consumed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from importlib.metadata import version as package_version
from collections import Counter
from pathlib import Path

from src.rag import build_evidence_corpus_v2 as corpus
from src.rag import phase2_common as common

require = common.require
DEVELOPMENT_ID = "v5e_validation"
SOURCE_TYPE = "FROZEN_TRAIN_EMPIRICAL_EVIDENCE"
BASE = "data/raw/rag/evidence_sources_v2/v5e_validation/"
CONFIG = "configs/rag_train_evidence_v5.json"
VERSION = "train-group-median-v1"
INPUTS = {
    "data/processed/USTC-TFC2016/train.csv": "e303924868f0a891a21e569e2f331e62836d83020763337cb01c7f9f506cd989",
    "data/processed/USTC-TFC2016/preprocessing_parameters.csv": "dde8371f743c8f3dccacdab36cab0227d7b42ec42a5720e5dffa7217b362845c",
    "data/processed/CIC-Darknet2020/train.csv": "40eb6ccd8125a39f64a12cc54f4d838d7f67f22cf610c8d2c9a3d08ba3a41007",
    "metadata/traffic/cic_preprocessing_parameters.csv": "fe9879dad98e09baf5ce5b10be8acf91f1dadf472f688a56647e414346227f39",
    "metadata/traffic/cic_split_manifest.csv": "6757dac8bd242cfc7abfab6094e9ef78af14da1059fc2f7e76813220ffe74c0c",
    "results/datasets/USTC-TFC2016/preprocessing/20260919T123108Z/split_family_assignment.csv": "b1254da4d6ca37e451d0536980bc656eb7cf360d7bd8037e38846d18bd68f2e0",
}
SPECS = {
    "USTC-TFC2016": dict(train=list(INPUTS)[0], scaler=list(INPUTS)[1], boundary=list(INPUTS)[5],
                         label="binary_label", group="semantic_family", boundary_group="semantic_family",
                         boundary_dataset="USTC-TFC2016", boundary_count="flow_count",
                         train_split="train", boundary_splits=("train", "validation", "test")),
    "CIC-Darknet2020": dict(train=list(INPUTS)[2], scaler=list(INPUTS)[3], boundary=list(INPUTS)[4],
                          label="Label", group="Flow ID", boundary_group="flow_id",
                          boundary_dataset="CIC-Darknet2020", boundary_count="group_row_count",
                          train_split="TRAIN", boundary_splits=("TRAIN", "VALIDATION", "TEST")),
}
# Fixed BEFORE validation coverage is inspected. Delta .5 means pairwise strict
# dominance minus reverse dominance >= 50 percentage points; not a significance
# claim. Bonferroni bootstrap tails cover the 340 planned candidates nominally.
POLICY = dict(version=VERSION, development_id=DEVELOPMENT_ID, seed=42,
              minimum_rows=100, minimum_groups_ustc=6, minimum_groups_cic=20,
              minimum_abs_delta=0.5, minimum_abs_standardized_median=0.5,
              minimum_group_direction_fraction=0.8, bootstrap_replicates=10000,
              familywise_alpha=0.05, planned_candidates=340,
              min_coverage_fraction=0.01, diagnostic_per_stratum=10)
ARTIFACTS = ("statistics.json", "additional_manifest.csv", "additional_documents.csv",
             "decision_evidence_claims.csv", "claim_audit.json", "bundle.json")
CODE = ("src/rag/build_train_evidence_v5.py", "src/rag/build_evidence_corpus_v2.py",
        "src/rag/run_rag_retrieval.py", "src/rag/phase2_common.py")
CONTEXT_FILES = ("data/raw/rag/evidence_sources_v2/additional_manifest.csv",
                 "data/raw/rag/evidence_sources_v2/additional_documents.csv")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path(root, relative, exists=True):
    target = corpus.v1.local_path(root, relative, must_exist=exists)
    if exists:
        require(target.is_file() and target.stat().st_nlink == 1, "Unaliased regular input required")
    return target


def read_csv(root, relative):
    corpus.v1.configure_csv_field_limit()
    with path(root, relative).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), "CSV header differs")
        for row in reader:
            require(None not in row and all(v is not None for v in row.values()), "Malformed CSV")
            yield row


def freeze(root):
    config = json.loads(path(root, CONFIG).read_text(encoding="utf-8"))
    require(config == POLICY, "Registered empirical policy differs")
    observed = {name: sha(path(root, name)) for name in INPUTS}
    require(observed == INPUTS, "Frozen TRAIN/scaler/boundary hash mismatch")
    return dict(inputs=observed, configuration_sha256=sha(path(root, CONFIG)),
                code_sha256={name: sha(path(root, name)) for name in CODE},
                contextual_source_sha256={name:sha(path(root,name)) for name in CONTEXT_FILES}, version=VERSION,
                environment=dict(python=platform.python_version(), numpy=package_version("numpy"), pandas=package_version("pandas")))


def train_boundary(root, spec):
    # Both producers assign whole groups, but their split vocabularies and
    # count columns differ. Accept their exact schemas, not guessed aliases.
    members, seen = {}, set()
    required = {"dataset", "split", spec["boundary_group"], spec["boundary_count"]}
    for row in read_csv(root, spec["boundary"]):
        require(required <= set(row), "Frozen boundary schema mismatch: " + spec["boundary"])
        require(row["dataset"] == spec["boundary_dataset"], "Frozen boundary dataset mismatch")
        require(row["split"] in spec["boundary_splits"], "Unrecognized frozen boundary split")
        group = row[spec["boundary_group"]]
        # Inspect held-out group IDs only to reject duplicate/cross-split
        # assignments; never consume held-out labels or population counts.
        require(group.strip() and group not in seen, "Blank/duplicate frozen boundary group")
        seen.add(group)
        if row["split"] != spec["train_split"]:
            continue
        count = int(row[spec["boundary_count"]])
        require(count > 0, "Nonpositive frozen TRAIN group count")
        members[group] = count
    require(members, "Empty frozen TRAIN boundary: " + spec["boundary_dataset"])
    return members


def validate_train_membership(group_counts, boundary):
    # Exact equality rejects missing, extra, held-out and miscounted groups.
    # freeze() additionally anchors the complete TRAIN file's bytes, preventing
    # different rows with the same group counts from replacing frozen rows.
    require(boundary and group_counts == boundary, "TRAIN membership/count differs from frozen split")


def feature_map(root, dataset, spec):
    from src.rag.run_rag_retrieval import readable_feature
    features = {}
    for row in read_csv(root, spec["scaler"]):
        retained = row.get("retained", row.get("retained_final")) in {"True", "YES"}
        name = row["feature_name"]
        canonical = readable_feature(name)
        if not retained or canonical in {"protocol", "ip version"}:
            continue
        require(canonical in corpus.CANONICAL_DECISION_FEATURES[dataset] and canonical not in features,
                "Incompatible/missing canonical feature")
        mean = common.finite(row.get("train_mean", row.get("train_scaler_mean")))
        scale = common.finite(row.get("train_std", row.get("train_scaler_scale")))
        require(scale > 0, "Invalid frozen scaler")
        features[canonical] = (name, mean, scale)
    require(set(features) == corpus.CANONICAL_DECISION_FEATURES[dataset], "Canonical feature population differs")
    return features


def delta(a, b):
    import numpy as np
    ordered = np.sort(b)
    return float((np.searchsorted(ordered, a, side="left").sum()
                  - (len(b) - np.searchsorted(ordered, a, side="right")).sum()) / (len(a) * len(b)))


def assess(a, b, n_class, n_reference, minimum_groups, seed):
    """Equal-weight real-group medians; no population/classifier information."""
    import numpy as np
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    require(len(a) and len(b) and np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite/empty group statistics")
    median, reference = float(np.median(a)), float(np.median(b))
    sign = 1 if median >= 0 else -1
    effect = delta(a, b)
    # A TRAIN-relative observed high/low must agree with the class-v-reference
    # contrast. The reference must lie on the other side of zero; typical values
    # are never turned into a positive directional claim.
    fraction = float(np.mean(sign * a >= POLICY["minimum_abs_standardized_median"]))
    reasons = []
    if min(n_class, n_reference) < POLICY["minimum_rows"] or min(len(a), len(b)) < minimum_groups:
        reasons.append("INSUFFICIENT_SAMPLE_OR_GROUPS")
    if sign * median < POLICY["minimum_abs_standardized_median"] or sign * reference > 0:
        reasons.append("TRAIN_RELATIVE_DIRECTION_NOT_ESTABLISHED")
    if sign * effect < POLICY["minimum_abs_delta"]:
        reasons.append("SMALL_OR_OPPOSITE_EFFECT")
    if fraction < POLICY["minimum_group_direction_fraction"]:
        reasons.append("GROUP_DIRECTION_UNSTABLE")
    # Exact leave-one-group-out effect minimum, without quadratic pair storage.
    sa, sb = np.sort(a), np.sort(b)
    contribution_a = np.searchsorted(sb, a, "left") - (len(b)-np.searchsorted(sb, a, "right"))
    contribution_b = (len(a)-np.searchsorted(sa, b, "right")) - np.searchsorted(sa, b, "left")
    total = effect * len(a) * len(b)
    loo = min(float(np.min(sign*(total-contribution_a)/((len(a)-1)*len(b)))) if len(a)>1 else -1,
              float(np.min(sign*(total-contribution_b)/(len(a)*(len(b)-1)))) if len(b)>1 else -1)
    if loo < POLICY["minimum_abs_delta"]:
        reasons.append("LEAVE_ONE_GROUP_OUT_EFFECT_UNSTABLE")
    ci = None
    if not reasons:
        rng = np.random.default_rng(seed)
        differences = np.empty(POLICY["bootstrap_replicates"])
        for i in range(len(differences)):
            differences[i] = np.median(rng.choice(a, len(a))) - np.median(rng.choice(b, len(b)))
        tail = POLICY["familywise_alpha"] / (2 * POLICY["planned_candidates"])
        ci = [float(v) for v in np.quantile(differences, [tail, 1-tail])]
        if (ci[0] <= 0 if sign > 0 else ci[1] >= 0):
            reasons.append("GROUP_BOOTSTRAP_DIRECTION_UNCERTAIN")
    return dict(class_standardized_median=median, reference_standardized_median=reference,
                direction="higher" if sign > 0 else "lower", effect_size=effect,
                effect_measure="Cliffs_delta_of_equal_weight_group_medians",
                group_direction_fraction=fraction, minimum_signed_leave_one_group_out_delta=loo,
                median_difference_ci=ci, ci_method="group_bootstrap_Bonferroni_340_nominal_not_population_inference",
                groups_class=len(a), groups_reference=len(b), n_class=n_class, n_reference=n_reference,
                rejection_reasons=reasons, decision="REJECTED" if reasons else "APPROVED")


def statistics(root, frozen):
    import numpy as np
    import pandas as pd
    records = []
    for dataset, spec in SPECS.items():
        features, boundary = feature_map(root, dataset, spec), train_boundary(root, spec)
        # One TRAIN table at a time, numeric columns and two grouping columns only.
        columns = [v[0] for v in features.values()] + [spec["label"], spec["group"]]
        frame = pd.read_csv(path(root, spec["train"]), usecols=columns,
                            dtype={spec["label"]: str, spec["group"]: str})
        require(not frame.isna().any().any(), "Missing TRAIN feature/label/group")
        require(np.isfinite(frame[[v[0] for v in features.values()]].to_numpy()).all(), "Nonfinite TRAIN values")
        require(set(frame[spec["label"]]) == set(common.CLASSES[dataset]), "TRAIN class vocabulary differs")
        validate_train_membership(frame.groupby(spec["group"]).size().to_dict(), boundary)
        # Label-conflicting Flow IDs cannot serve as independent class groups.
        purity = frame.groupby(spec["group"])[spec["label"]].nunique()
        excluded = set(purity[purity != 1].index)
        excluded_rows = int(frame[spec["group"]].isin(excluded).sum())
        frame = frame[~frame[spec["group"]].isin(excluded)]
        grouped = frame.groupby([spec["label"], spec["group"]], sort=True)
        medians = grouped[[v[0] for v in features.values()]].median()
        labels = medians.index.get_level_values(0)
        counts = frame[spec["label"]].value_counts()
        group_counts = Counter(labels)
        if dataset == "USTC-TFC2016":
            require(not excluded and set(group_counts.values()) == {6}, "Frozen six TRAIN families per class required")
        for label in common.CLASSES[dataset]:
            for feature, (raw, mean, scale) in sorted(features.items()):
                identity = "EMP-" + hashlib.sha256(f"{VERSION}|{dataset}|{label}|{feature}".encode()).hexdigest()[:20]
                a, b = medians.loc[labels == label, raw].to_numpy(), medians.loc[labels != label, raw].to_numpy()
                require(np.isfinite(a).all() and np.isfinite(b).all(), "Nonfinite TRAIN statistic")
                row = assess(a, b, int(counts[label]), int(len(frame)-counts[label]),
                             POLICY["minimum_groups_ustc" if dataset == "USTC-TFC2016" else "minimum_groups_cic"],
                             POLICY["seed"] + int(identity[4:12], 16))
                row.update(claim_id=identity, source_dataset=dataset, supported_class=label,
                           canonical_feature=feature, raw_feature=raw, source_type=SOURCE_TYPE,
                           reference_classes=sorted(set(common.CLASSES[dataset])-{label}),
                           reference_population="all other TRAIN classes, equal-weight label-pure groups",
                           group_key=spec["group"], excluded_conflicting_groups=len(excluded), excluded_conflicting_rows=excluded_rows,
                           class_median=mean+scale*row["class_standardized_median"],
                           reference_median=mean+scale*row["reference_standardized_median"],
                           median_definition="median of group medians, inverse transformed with frozen TRAIN scaler",
                           source_artifacts={p:frozen["inputs"][p] for p in (spec["train"],spec["scaler"],spec["boundary"])},
                           configuration_sha256=frozen["configuration_sha256"], version=VERSION)
                records.append(row)
        del frame, grouped, medians
    require(len(records) == POLICY["planned_candidates"], "Candidate count differs")
    return sorted(records, key=lambda r:r["claim_id"])


def render(row):
    """Standalone empirical statement, never a quotation attributed to a paper."""
    return (f"Claim {row['claim_id']}: In the frozen {row['source_dataset']} TRAIN corpus, "
            f"{row['canonical_feature']} was {row['direction']} for {row['supported_class']} than "
            f"the reference classes {', '.join(row['reference_classes'])} using equal-weight {row['group_key']} medians. "
            f"This is an association, not proof or a causal rule. source_type={SOURCE_TYPE}; "
            f"delta={row['effect_size']:.8g}; direction_fraction={row['group_direction_fraction']:.8g}; "
            f"median_difference_ci={common.json_text(row['median_difference_ci'])}; "
            f"sources={common.json_text(row['source_artifacts'])}; configuration_sha256={row['configuration_sha256']}; "
            f"statistics={BASE}statistics.json#{row['claim_id']}.")


def derived_tables(records):
    sources, docs, claims = [], [], []
    for row in records:
        if row["decision"] != "APPROVED":
            continue
        text, identity = render(row), row["claim_id"]
        source = dict.fromkeys(corpus.v1.MANIFEST_HEADER, "")
        source.update(document_id=identity, title="Frozen TRAIN empirical association " + identity,
                      source_type=SOURCE_TYPE, publisher="Project TRAIN empirical derivation (not external)",
                      local_path=BASE+"statistics.json", traffic_topics=row["source_dataset"],
                      security_topics="Class-conditional training association; no causal inference",
                      corpus_partition="trusted", source_provenance_status="VERIFIED", authorized_for_retrieval="YES",
                      curation_notes="Registered statistical gate approval; not a literature quote or manual review")
        doc = dict(source, extraction_status="PASS", extraction_review_status="APPROVED",
                   character_count=str(len(text)), word_count=str(len(text.split())), normalized_text=text,
                   extraction_notes="Deterministic rendering of hash-linked TRAIN statistics; no extraction/OCR")
        claim = dict(zip(corpus.DECISION_CLAIM_HEADER,
                     (identity,identity,row["source_dataset"],"document",text,row["canonical_feature"],row["direction"],
                      row["supported_class"],"","APPROVED","YES")))
        sources.append(source); docs.append(doc); claims.append(claim)
    quality = [q for doc in docs for q in corpus.quality_rows(doc["document_id"], corpus.source_units(doc["normalized_text"]), "evidence_quality")]
    corpus.validate_decision_claim_rows(docs, quality, claims)
    return sources, docs, claims


def audit(records, frozen):
    approved = [r for r in records if r["decision"] == "APPROVED"]
    return dict(candidate_claims=len(records), approved_claims=len(approved), rejected_claims=len(records)-len(approved),
                candidates_by_dataset=dict(Counter(r["source_dataset"] for r in records)),
                candidates_by_class=dict(Counter(r["source_dataset"]+"/"+r["supported_class"] for r in records)),
                rejected_by_dataset=dict(Counter(r["source_dataset"] for r in records if r["decision"]!="APPROVED")),
                rejected_by_class=dict(Counter(r["source_dataset"]+"/"+r["supported_class"] for r in records if r["decision"]!="APPROVED")),
                rejection_reasons=dict(Counter(reason for r in records for reason in r["rejection_reasons"])),
                approved_by_dataset=dict(Counter(r["source_dataset"] for r in approved)),
                approved_by_class=dict(Counter(r["source_dataset"]+"/"+r["supported_class"] for r in approved)),
                approved_by_feature=dict(Counter(r["source_dataset"]+"/"+r["canonical_feature"] for r in approved)),
                approved_by_source_type={SOURCE_TYPE:len(approved)}, external_claims=0, empirical_claims=len(approved),
                effect_sizes=[r["effect_size"] for r in records], duplicate_claims=len(records)-len({r["claim_id"] for r in records}),
                validation_rows_used=0, test_rows_used=0, validation_correctness_used=0,
                provenance_failures=0, provenance=frozen,
                scope="Only allowlisted TRAIN row files; split manifests used solely for TRAIN membership/counts",
                inference_limit="Associations within frozen training groups; no claim of held-out discrimination or causation")


def bundle_tables(root, records):
    sources, docs, claims = derived_tables(records)
    # Existing Tor/VPN papers stay contextual: no new external claim is inferred.
    external_sources = list(read_csv(root,CONTEXT_FILES[0]))
    external_docs = list(read_csv(root,CONTEXT_FILES[1]))
    require({r["document_id"] for r in external_sources}=={"RESEARCH-LASHKARI-2017-TOR","RESEARCH-DRAPER-GIL-2016-VPN"},
            "Reviewed contextual source population differs")
    lookup = common.unique(external_sources,"document_id")
    require(len(external_docs)==len(lookup), "Contextual source linkage differs")
    for doc in external_docs:
        require(corpus.v1.usable_document(doc) and doc["source_type"]=="peer_reviewed_research"
                and all(doc[k]==lookup[doc["document_id"]][k] for k in corpus.v1.MANIFEST_HEADER), "Contextual source metadata differs")
    return external_sources+sources,external_docs+docs,claims


def verify_bundle(root):
    """Verify every derived input and code/config identity before downstream use."""
    frozen = freeze(root)
    bundle = json.loads(path(root, BASE+"bundle.json").read_text(encoding="utf-8"))
    require(bundle["provenance"] == frozen, "Evidence derivation code/config/source changed")
    require(set(bundle["files"]) == set(ARTIFACTS)-{"bundle.json"}, "Incomplete empirical bundle")
    require(all(sha(path(root, BASE+name)) == digest for name,digest in bundle["files"].items()), "Empirical evidence hash mismatch")
    records = json.loads(path(root, BASE+"statistics.json").read_text(encoding="utf-8"))
    expected = bundle_tables(root, records)
    for name, rows in zip(ARTIFACTS[1:4], expected):
        require(list(read_csv(root, BASE+name)) == rows, "Derived text/claim differs from statistics")
    require(json.loads(path(root, BASE+"claim_audit.json").read_text(encoding="utf-8")) == audit(records,frozen), "Claim audit differs")
    return records


def verify_documents(root, development_id, documents):
    empirical = [r for r in documents if r["source_type"] == SOURCE_TYPE]
    if not empirical and development_id != DEVELOPMENT_ID:
        return
    require(development_id == DEVELOPMENT_ID, "Empirical evidence requires its registered identity")
    _, expected, _ = derived_tables(verify_bundle(root))
    require({r["document_id"] for r in empirical} == {r["document_id"] for r in expected}, "Empirical document population differs")
    lookup = {r["document_id"]:r for r in expected}
    for row in empirical:
        require(all(str(row[k]) == str(v) for k,v in lookup[row["document_id"]].items()), "Empirical document metadata/text altered")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--execute", action="store_true")
    action.add_argument("--audit-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.project_root.resolve(strict=True)
    corpus.validate_development_id(DEVELOPMENT_ID)
    if args.audit_only:
        records = verify_bundle(root)
        require(records==statistics(root,freeze(root)), "Independent TRAIN recomputation differs from recorded statistics")
        print(common.json_text(audit(records, freeze(root))))
        return 0
    require(not any(path(root, BASE+name, False).exists() for name in ARTIFACTS), "Empirical artifacts already exist; refusing overwrite")
    frozen = freeze(root)
    records = statistics(root, frozen)
    tables = bundle_tables(root, records)
    require(freeze(root) == frozen, "Inputs changed during derivation")
    # All computation and validation precede create-only publication; bundle last
    # is the completion marker. Partial output is never accepted automatically.
    directory = path(root, BASE, False)
    directory.mkdir(parents=True, exist_ok=True)
    objects = {"statistics.json":records, "claim_audit.json":audit(records,frozen)}
    for name, value in objects.items():
        with (directory/name).open("x",encoding="utf-8") as stream:
            stream.write(common.json_text(value)+"\n")
    for name,header,rows in zip(ARTIFACTS[1:4], (corpus.v1.MANIFEST_HEADER,corpus.v1.DOCUMENT_HEADER,corpus.DECISION_CLAIM_HEADER),tables):
        corpus.v1.write_table(directory/name,header,rows,mode="x")
    bundle = dict(provenance=frozen, files={name:sha(directory/name) for name in ARTIFACTS if name != "bundle.json"})
    with (directory/"bundle.json").open("x",encoding="utf-8") as stream:
        stream.write(common.json_text(bundle)+"\n")
    print(common.json_text(audit(records,frozen)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
