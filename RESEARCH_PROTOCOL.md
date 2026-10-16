# Research Protocol v0.1 — Frozen Before Data Processing

## Objective
Build and evaluate a provenance-aware RAG system that supplements encrypted-traffic classification with traceable cybersecurity evidence.

## Phase I
1. Acquire and inventory raw datasets.
2. Audit schemas and authoritative labels.
3. Dataset-specific processing.
4. Freeze predictor/label/group contracts.
5. Create leakage-safe train/validation/test splits.
6. Fit preprocessing on TRAIN only.
7. Train Random Forest, 1D-CNN, XGBoost, LightGBM and CNN-LSTM independently on each dataset.
8. Evaluate with a common protocol.
9. Freeze one independently trained LightGBM backbone per dataset.

## USTC primary task
Binary `benign` vs `malware`. The only authoritative binary label at acquisition is the release's top-level `Benign/` or `Malware/` category. Family names are provenance/group metadata, not predictors.

NFStream Layer-7 dissection is disabled (`n_dissections=0`). IPs, MACs, raw ports, absolute timestamps and engineering IDs are excluded from the primary predictor set.

## CIC primary task
CIC-Darknet2020 is treated as a separately processed secondary benchmark using the published flow CSV. No NFStream re-extraction is permitted for the baseline. Exact label columns and mappings are not guessed: the schema audit must run first and a label/feature contract must be frozen from the observed official CSV.

## Split policy
No packet/row-random split where rows from the same capture/session can cross partitions. USTC uses source-capture groups. CIC grouping is frozen only after its schema audit identifies authoritative grouping fields; otherwise a deterministic fallback must be documented and justified.

## Preprocessing
No learned transformation before split. Missing-value handling, scaling, constant-feature removal and learned feature selection are fit on TRAIN only and replayed unchanged to validation/test.

## Model selection
Validation data controls early stopping and any model choice. TEST is one-time final evaluation and never influences hyperparameters, preprocessing, thresholds or the LightGBM-backbone decision.

## Phase II
Structured security query -> cybersecurity evidence corpus -> indexing -> retrieval -> provenance verification -> provenance-aware reranking -> trusted context -> constrained LLM reasoning -> evidence-supported decision fusion.

The corpus begins with official MITRE ATT&CK STIX 2.1, CAPEC and CWE. Every evidence unit stores source URL, source authority, source version, retrieval time, content hash and lineage.

## Security evaluation
Clean retrieval, knowledge poisoning, indirect prompt injection, provenance spoofing, combined attacks and ablations. Report retrieval metrics, attack success rate, poisoned retrieval rate, decision manipulation rate, detection precision/recall/F1, false rejection, abstention, evidence support and latency.

## Result-storage requirement
Keep row-level predictions/probabilities, long-form overall metrics, per-class metrics, confusion matrix in long form, calibration bins, training history, retrieval/evidence records, attack records and artifact indexes. Stage-01 audit indexes contain only relative paths and byte sizes. Figures must never require retraining a model.

## V5 validation preparation decision, 2026-10-09

This addendum applies only to the unexecuted `v5_validation` development
experiment. It does not revise V4 or any completed experiment. Numeric
configuration is frozen in the existing `configs/rag.json` under
`v5_validation`. Scientific execution readiness remains BLOCKED: no reviewed
directional claim is eligible under the current training-relative matcher.
No validation/TEST observations, labels, correctness or outcome statistics
were used for these decisions. Tests are implementation examples, not policy.

### Authorized contextual source additions

Use the existing paired `--additional-manifest` / `--additional-documents`
mechanism with `data/raw/rag/evidence_sources_v2/additional_manifest.csv` and
`data/raw/rag/evidence_sources_v2/additional_documents.csv`. These are curated
inputs, not Stage-2 outputs. Existing manifests and normalized corpus outputs
remain untouched. Both local PDFs have nonempty text layers on every page;
full normalized text is retained without OCR. Source authorization is based on
the local publication identity corroborated by the original dataset owner's
citation, not filename, directory or generic publisher reputation.

| Document ID | Publication identity and primary authority | Dataset / relevant measurement | Decision |
|---|---|---|---|
| RESEARCH-LASHKARI-2017-TOR | Arash Habibi Lashkari, Gerard Draper Gil, Mohammad Saiful Islam Mamun, Ali A. Ghorbani; Characterization of Tor Traffic using Time based Features; ICISSP 2017, SCITEPRESS, pp. 253-262; https://doi.org/10.5220/0006105602530262; corroborated by https://www.unb.ca/cic/datasets/tor.html | ISCXTor2016; bidirectional/forward/backward inter-arrival-time statistics, duration and rates | Retrieval authorized. Feature definitions and selection/performance analyses do not provide a qualifying higher/lower Tor versus Non-Tor claim. |
| RESEARCH-DRAPER-GIL-2016-VPN | Gerard Draper-Gil, Arash Habibi Lashkari, Mohammad Saiful Islam Mamun, Ali A. Ghorbani; Characterization of Encrypted and VPN Traffic using Time-related Features; ICISSP 2016, SCITEPRESS, pp. 407-414; https://doi.org/10.5220/0005740704070414; corroborated by https://www.unb.ca/cic/datasets/vpn.html | ISCXVPN2016; flow duration, inter-arrival-time statistics and rates | Retrieval authorized. Shorter extraction timeouts improving classification are not a shorter-duration VPN/non-VPN evidence rule. |

Both publications predate this project and are independent of its outputs.
The additional-document ingestion path assigns source_location `document`
to its paragraph units; it does not use the page locations of NEW_SOURCES.
No claim location is inferred from the PDF filename or printed page number.

### Directional evidence review disposition

The existing `data/raw/rag/decision_evidence_claims.csv` remains header-only.
Approved counts are zero for USTC benign/malware and CIC Non-Tor/NonVPN/Tor/VPN.
No minimum count is imposed. Retrieval authorization does not authorize claims.

The local Lichy (2023-labelled source) and Shekhawat (2019) papers were also
reviewed. Feature importance is not direction. Certificate/key properties,
connection periodicity, and an illustrative executable-size/entropy decision
tree do not establish exact identities with the 48 USTC traffic features.
CIC-Darknet2020 documentation describes dataset composition; Tor specifications
describe cells/channels, not a packet-size directional class rule. RFC protocol
mechanics likewise supplied no qualifying target-class mapping. TOR-SPEC-001
remains unauthorized.

Targeted external review considered original dataset documentation and primary
research, without deriving new statistics from dataset labels:

| Source | Authorship/year/venue | Dataset and feature relation | Disposition |
|---|---|---|---|
| Tor-nonTor dataset documentation, https://www.unb.ca/cic/datasets/tor.html | UNB Canadian Institute for Cybersecurity; dataset 2016; official documentation | ISCXTor2016; capture design and flow extraction, no explicit canonical-feature direction | Authority corroboration only; no claim. |
| VPN-nonVPN dataset documentation, https://www.unb.ca/cic/datasets/vpn.html | UNB Canadian Institute for Cybersecurity; dataset 2016; official documentation | ISCXVPN2016; OpenVPN capture design, no directional canonical-feature rule | Authority corroboration only; no claim. |
| CIC-Darknet2020, https://www.unb.ca/cic/datasets/darknet2020.html | UNB Canadian Institute for Cybersecurity; dataset 2020; official documentation | Combined Tor/VPN datasets; no explicit measurable class-direction rule | Existing source retained; no claim. |
| USTC-TFC2016 repository, https://github.com/davidyslu/USTC-TFC2016 | David Y. S. Lu repository; dataset 2016; repository documentation | Dataset composition and provenance, no qualifying canonical-feature direction | Documentation review only; no new retrieval authorization. |
| An encrypted traffic classification method based on autoencoders and convolutional neural networks, https://doi.org/10.1371/journal.pone.0333276 | Shengwei Xu, Jijie Han, Jianbo Wang, Yijie Bai; 2025; PLOS One | USTC-TFC2016 and ISCXVPN2016; effective byte length after 784-byte trimming and excluding zero padding | Reject mapping to canonical packet size, flow bytes or duration: different measurement. Not added to corpus. Independent pre-project publication. |

None of these inspected sources supplied an admissible claim. This is a
bounded targeted literature review, not a proof that no qualifying evidence
exists anywhere. Search snippets, secondary summaries and classifier
performance were not used to authorize evidence.

### Frozen ranking and context parameters

Stage 6: `task_relevance`, `hybrid_rrf`, top-k 10, retrieval weight 1,
source-preference weight 0 and empty preferences, task-alignment weight 0.05,
relationship weight 0.05. The two auxiliary components each lie in [0,1].
Their total contribution is at most 0.1. This conservative budget is below
the inverse-rank gap 1/2 - 1/3 = 1/6, so original ranks 1 and 2 retain their
order and remain above ranks 3-10 in Stage 6. Lower ranks may reorder.
Equal division avoids declaring either uncalibrated signal more reliable.
This is a design choice, not an estimated optimal weight. The previous
top-two-preserving policy motivates the total budget, not transferring its
old class-metadata signal to the new lexical signal.

Stage 7: `evidence_diverse`, diversity threshold 1.0, top-n 5, token budget
2048, max chunks per document 0. Threshold 1.0 is the exact endpoint requiring
all nonempty observable query terms to appear in a fallback chunk, rather
than choosing an empirically tuned fractional cutoff. It is only exposure
priority, not evidence qualification; candidates can still enter through
ordinary rerank-order fill. Top-n 5 retains the existing final-k policy;
2048 preserves the current context budget; cap 0 preserves the uncapped
default without inventing a per-document restriction. A later LLM budget is
separate. Stage-6 top-two preservation does not override Stage-7 exposure
selection or token limits.

All numeric choices are fixed before V5 execution and independent of outcome.
No numeric parameter remains unresolved. Existing CLIs do not automatically
read configs/rag.json: after scientific readiness is resolved, its values must
be passed explicitly; no production defaults were changed.

### Remaining scientific blocker and minimal proposal (not implemented)

Current Stage 6 compares a source claim's direction with a query's direction
relative to the project's training distribution. A statement that class A has
larger values than class B does not establish that an individual observation
is above this training reference, nor that A is supported. The claim schema
does not encode the comparison reference or measurement calibration. No
external comparative claim is approved under this ambiguity.

Before enabling such claims, minimally represent the direction reference
(training-relative versus class-comparative), comparator and measurement
definition, and reject cross-reference matching by default. Class-comparative
statements may remain contextual evidence, but must not become claim-backed
support merely through a high/low bin. A quantitative bridge, if pursued,
requires a separately approved method based on compatible measurements and
TRAIN-only references, never validation/TEST labels. This is a proposal only;
no schema, matcher, validator or V4 behavior was changed.

Stage-2 paired inputs are technically prepared, but DO NOT EXECUTE YET for
the intended directional-evidence V5 experiment. Freeze the reference semantics
or explicitly approve a zero-claim contextual experiment before create-only
Stage-2 outputs are materialized. No experimental outputs were created here.

### Evidence-enriched V5: registered TRAIN-only derivation (v5e_validation)

This new experiment does not replace or modify the executed zero-claim
`v5_validation` experiment or frozen V4. It is not scientifically cleared for
LLM execution until its manual derivation, provenance, and coverage audits pass.
`v5e_validation` satisfies the existing safe-development-id grammar.

The authoritative inputs are `data/processed/USTC-TFC2016/train.csv`, its
`preprocessing_parameters.csv`, `data/processed/CIC-Darknet2020/train.csv`,
`metadata/traffic/cic_preprocessing_parameters.csv`,
`metadata/traffic/cic_split_manifest.csv`, and
`results/datasets/USTC-TFC2016/preprocessing/20260919T123108Z/split_family_assignment.csv`.
Their SHA-256 identities were inspected and anchored in the generator before
examining validation coverage. These are current frozen-artifact identities,
not a claim that historical cryptographic attestations already existed.
Split manifests contain multiple splits but only TRAIN membership/count fields
are consumed. No held-out row file or held-out label count is used. TRAIN labels
are used only offline. No IP address, timestamp, capture path, confidence,
classifier correctness, validation label, or TEST label is emitted as evidence.

`configs/rag_train_evidence_v5.json` registers the method. There are 340 planned
dataset/class/feature candidates: 2*48 USTC and 4*61 CIC. Features retain their
exact dataset-specific Stage-4 canonical names. The numeric inputs are already
standardized by the frozen TRAIN scaler; inverse-transforming group medians is
used only for descriptive raw-scale audit statistics.

USTC uses its actual six training semantic families per class, never validation
families. CIC uses exact Flow IDs, never invented endpoint/time groups.
Label-conflicting Flow IDs are excluded before every class/reference comparison;
excluded counts are recorded. Each label-pure group contributes one median, so
large captures and repeated flow groups cannot dominate the association.
The reference is every other TRAIN class's label-pure groups, equally weighted.

Approval requires >=100 class/reference rows, >=6 USTC groups or >=20 CIC groups
on each side, a class standardized group-median statistic >=0.5 in magnitude,
and a reference statistic on the opposite side of zero. This explicitly bridges
the curated direction to the existing TRAIN-relative high/low bins. At least
80% of class groups must agree at that same bin boundary. Signed Cliff's delta
of group medians must be >=0.5, including after removing any one group from
either side. Delta 0.5 is an explicitly registered conservative engineering
criterion (strict-pair dominance minus reverse dominance >=50 percentage points),
not a literature-derived cutoff or a validation-optimized setting.

For candidates surviving those gates, 10,000 seeded group bootstrap replicates
estimate the class-minus-reference median difference. The nominal percentile
interval uses alpha=0.05/340 with two tails and must exclude zero in the claimed
direction. Seed is 42 plus a deterministic claim-ID offset. The interval is a
sensitivity audit under group resampling, not proof of population significance:
six USTC families provide limited generalizability, groups may be dependent,
and tail quantiles are coarse. No p-value or causal claim is reported. Every
failure has an explicit reason; incomplete provenance blocks the whole run.
Numeric libraries and Python versions, source hashes, code hashes, configuration
hash, and all candidate statistics are recorded. `--audit-only` independently
recomputes the TRAIN statistics and checks byte-level artifact linkage.

Approved statements are deterministic derived documents labelled
`FROZEN_TRAIN_EMPIRICAL_EVIDENCE`. Approval is the registered empirical gate,
not human literature review. Statements identify their claim, dataset, class,
feature, reference, effect, stability/uncertainty, source hashes, configuration
hash, and statistics-row link. They explicitly disclaim causation and individual
classification proof. They are never attributed to a paper. Existing Tor/VPN
dataset papers remain contextual; the previous reviews established no admissible
directional external claim. No external decision claim is created in this pass.

Only this new identity accepts hash-verified derived documents at Stage 2 and
Stage 5. No global source-category whitelist is widened. Stages 3, 4, 6 and 7
retain their existing methodology, including exact matches, supported-class-only
audits, claim-first exposure, full validation population and bounded-memory
Stage 7. Use the same explicit .05/.05 Stage-6 weights and Stage-7 5/2048/0/1.0
parameters as the prior registered design. No output namespace is overwritten.

The separate `positive_evidence_v5.py` gate uses the existing six response
fields and SUPPORT_CLASS/ABSTAIN vocabulary. SUPPORTED means unique approved
association evidence supports the predicted class; CONTRADICTED means unique
alternative-class association evidence explicitly compares against that class.
Neither means confirmed correctness/error. Multiple claim-supported classes
require CONFLICTING; absent claims require INSUFFICIENT. Exact claim IDs,
canonical observed direction, standalone verbatim derived line, cited supplied
chunk, independent VERIFIED/authorized/KEEP provenance and matching dataset are
mandatory. Explanation has the narrow grammar `CLAIM_IDS: ["..."]`; contextual
fallback cannot qualify. V4 prompts/runtime/validator are untouched. Fusion stays
retain_classifier. The existing V4 runtime must NOT be used for these prompts.

Before Stage 8, `audit_evidence_coverage_v5` checks every validation query and
recomputes Stage-6 claim linkage against Stage-7 contexts using temporary SQLite
storage. It reports predicted/alternative coverage by dataset and predicted
class without labels. The operational stop rule, chosen before seeing coverage,
is <1% claim-backed contexts or <10 such queries in ANY prediction stratum.
This is a minimum-viability rule, not a target reasoning-status rate. STOP means
no diagnostic prompts and no LLM run; do not tune the evidence gates to pass it.
On PASS, at most ten claim-covered queries per stratum are selected by seeded
SHA-256 priority. This is a diagnostic sample, not a population estimate. Prompts
are explicitly NOT_TOKEN_AUDITED. A separate local Qwen token audit and review
of a dedicated V5 diagnostic execution adapter remain necessary before LLM use;
no full-run authorization or runner is introduced here.

Manual commands (WSL, from project root; none were executed during preparation):

```bash
PY=/home/dark_bird/miniforge3/envs/ustc-gpu/bin/python
"$PY" -m pytest tests/test_rag_train_evidence_v5.py tests/test_rag_corpus_v2.py tests/test_rag_index_v2.py tests/test_rag_reranking.py tests/test_rag_trusted_context.py -q
"$PY" -m src.rag.build_train_evidence_v5 --project-root . --execute
"$PY" -m src.rag.build_train_evidence_v5 --project-root . --audit-only
```

Inspect the audit BEFORE proceeding: stop if no claims survive, any provenance
check fails, or any leakage count is nonzero. The following commands are only
for the new identity and are listed for manual use after those checks pass.

```bash
D=v5e_validation
E=data/raw/rag/evidence_sources_v2/v5e_validation
S2=(--project-root . --profile evidence_quality --development-id "$D" --additional-manifest "$E/additional_manifest.csv" --additional-documents "$E/additional_documents.csv" --decision-claims "$E/decision_evidence_claims.csv")
"$PY" -m src.rag.build_evidence_corpus_v2 "${S2[@]}"
"$PY" -m src.rag.build_evidence_corpus_v2 "${S2[@]}" --execute
"$PY" -m src.rag.build_evidence_index_v2 --project-root . --development-id "$D"
"$PY" -m src.rag.build_evidence_index_v2 --project-root . --development-id "$D" --execute
"$PY" -m src.rag.run_rag_retrieval_v3 --project-root . --profile hypothesis_balanced --development-id "$D"
"$PY" -m src.rag.run_rag_retrieval_v3 --project-root . --profile hypothesis_balanced --development-id "$D" --execute
"$PY" -m src.rag.verify_retrieval_provenance_v3 --project-root . --development-id "$D"
"$PY" -m src.rag.verify_retrieval_provenance_v3 --project-root . --development-id "$D" --execute
S6=(--project-root . --profile task_relevance --development-id "$D" --method hybrid_rrf --top-k 10 --retrieval-weight 1 --source-preference-weight 0 --source-preferences-json '{}' --task-alignment-weight 0.05 --relationship-weight 0.05)
"$PY" -m src.rag.run_provenance_aware_reranking "${S6[@]}"
"$PY" -m src.rag.run_provenance_aware_reranking "${S6[@]}" --execute
S7=(--project-root . --profile evidence_diverse --development-id "$D" --diversity-threshold 1.0 --top-n 5 --token-budget 2048 --max-chunks-per-document 0)
"$PY" -m src.rag.build_trusted_contexts "${S7[@]}"
"$PY" -m src.rag.build_trusted_contexts "${S7[@]}" --execute
"$PY" -m src.rag.audit_evidence_coverage_v5 --project-root .
"$PY" -m src.rag.audit_evidence_coverage_v5 --project-root . --execute
```

Run one command at a time and require PASS/READY before the next. Artifact
publication is create-only. Approved/rejected counts and coverage remain unknown
until manual execution; no positive-status count is promised.

### TRAIN semantic association v2: v5f_validation (registered 2026-10-11)

The completed `v5e_validation` experiment is an immutable negative result: 340
candidates, zero approvals, independently reproduced. Its outputs, generator,
configuration and all code files listed in its provenance hash map are preserved.
This section supersedes no v1 criterion. A new estimand and a new identity are
introduced before observing any validation coverage or correctness.

Read-only diagnosis of the saved v1 statistics:

| Gate failure | USTC (96) | CIC (244) |
|---|---:|---:|
| Absolute TRAIN-relative direction (D) | 88 | 226 |
| Group agreement (G) | 92 | 223 |
| Small/opposite effect (E) | 57 | 223 |
| Leave-one-group-out effect (L) | 67 | 223 |
| Bootstrap direction uncertain (B) | 2 | 0 |
| Effect passes but absolute direction fails | 32 | 16 |

Exact overlapping rejection patterns: USTC D+G+E+L=54, D+G=23,
D+G+L=9, G=4, G+L=1, B=2, D+E+L=2, G+E+L=1.
CIC D+G+E+L=194, E+L=4, D+E+L=16, D+G=15, G=5,
G+E+L=9, D=1. Rejection counts overlap and must not be summed as candidates.
V1 computed bootstrap intervals only after all earlier gates passed. Only two
USTC candidates reached that step: benign bidirectional SYN packets, CI
[-2.4003064588593794,0], and benign source-to-destination SYN packets, CI
[-2.6252555118213223,0]. Neither excludes zero. CIC had zero bootstrap evaluations.
Consequently, bootstrap-pass/group-agreement-fail is NOT MEASURED for the skipped
candidates; zero recorded bootstrap passes is not evidence they would all fail.

The recorded higher/lower direction counts are USTC 16/80 and CIC 31/213. Of the
class-standardized median summaries, 86/96 USTC and 198/244 CIC lie inside the
typical bin. Their ranges are [-0.779094,1.245445] and [-1.123193,1.840664].
These are distributions across candidate summaries, NOT distributions of the
individual group medians. V1 did not save individual group medians, so their
full distributions cannot be reconstructed from the saved bundle alone.

Strongest absolute effects include USTC benign/malware bidirectional duration ms
and bidirectional max packet inter-arrival time ms (delta -/+0.944444), yet
duration class medians are both typical (-0.217106 and -0.010742). CIC strongest
by class: Non-Tor forward header length +0.610229 (both medians near -0.064),
NonVPN backward header length -0.488750 (both near -0.0446), Tor average packet
size +0.445585, VPN backward packet length min +0.687708. These explain why a
relative ordering cannot establish the former absolute high/low claim.

USTC has six TRAIN families per class. CIC label-pure group counts are Non-Tor
27,570, NonVPN 8,754, Tor 106 and VPN 11,058; five label-conflicting Flow IDs were
excluded by the existing rule. Pooled-rest references mix these very unequal
class populations. Pooling can conceal pair-specific relationships, but v1
saved no pairwise effects, so actual cancellation is not established by these
artifacts. V2 will measure the contrasts explicitly; no claim of improvement is
made before its independent audit.

New method: `train-semantic-association-v2`; development ID `v5f_validation`.
For each exact canonical feature and TRAIN group, count rows in LOWER (z<=-0.5),
TYPICAL (-0.5<z<0.5), HIGHER (z>=0.5). These are exact unions of frozen Stage-4
bins. Their boundaries are not estimated or tuned. For each group, divide counts
by its own row count; groups then have equal weight. No flow-level replication
is used for uncertainty or effect sizes. USTC uses semantic families and CIC
exact Flow IDs, excluding label-conflicting groups before comparisons exactly
as in v1. Original file hashes and full TRAIN membership/count checks remain.

Compare USTC benign/malware and all six unordered CIC class pairs. There are
(48 + 6*61)*3 = 1,242 planned feature/pair/state comparisons. Each comparison is
two-sided and appears once. The class with the larger median group prevalence
becomes supported_class; the other is the explicit contrasted_class. Ties use
canonical class order and fail the practical-effect gate. This is an empirical
orientation, not a desired class/status target.

`configs/rag_semantic_association_v5.json` fixes: >=6 USTC groups or >=20 CIC
groups per side; median prevalence difference >=0.10; Cliff's delta >=0.474;
strictly positive median difference AND delta after removing any group from
either side; and a two-sided 95% equal-group bootstrap median-difference interval
excluding zero. Bootstrap uses 10,000 replicates and seed42 plus deterministic
claim-ID offset. Histogram multinomial resampling is equivalent to sampling
groups with replacement, with bounded temporary memory. Every candidate gets
an interval, even if other gates fail, avoiding v1's missing diagnostic issue.

Delta >=0.474 follows the conventional large-effect boundary documented by the
effsize package author, citing Romano et al. (2006):
https://softeng.polito.it/software/effsize/cliff.delta.html . The 10-percentage-point
prevalence floor and group-count floors are explicitly registered design choices,
not universal statistical constants. Six USTC families limit extrapolation.
The 95% intervals are MARGINAL EXPLORATORY intervals, NOT familywise confidence
or confirmatory discoveries across 1,242 comparisons. Correlated features and
possible dependence between Flow IDs further limit inferential interpretation.
No familywise significance, causal relationship, or held-out accuracy is claimed.
This uncertainty scope is emitted in the audit and V5 prompt policy. Thresholds
must not be adjusted based on surviving counts or validation outcomes.

The seven create-only evidence artifacts are statistics.json, group_counts.npz,
additional_manifest.csv, additional_documents.csv, decision_evidence_claims.csv,
claim_audit.json and bundle.json under
`data/raw/rag/evidence_sources_v2/v5f_validation/`. The compressed archive retains
exact state numerators, group row denominators, TRAIN classes, and hashed group
identities for each canonical-feature key; source group names/IPs are not exposed
to retrieval. Independent audit rederives all statistics and archive arrays.

The claim CSV retains all eleven Stage-2 claim columns and adds:
evidence_method_version, semantic_state, target_group_count, contrast_group_count,
target_prevalence, contrast_prevalence, effect_size, bootstrap_interval_json,
minimum_loo_difference, minimum_loo_delta, source_artifacts_json,
configuration_sha256, code_sha256, statistics_link, source_type.
Statistics also retain direction consistency, exact group-count archive key,
excluded-group counts, class pair and rejection reasons. Every approved line
identifies its exact claim and measured association; the complete provenance
resides in the hash-verified claim/statistics linkage, not fabricated paper prose.

To preserve the hashes of the completed v1 experiment, Stage 2 uses the new
module's `--stage2` adapter. It composes the existing corpus builder, source-unit
quality review, exact canonical-feature and source_location checks, summary and
create-only writers, then adds verified v2 derived documents. It does not change
the old corpus builder. Stage 5 verifies v5f derivation; Stage 6 loads its exact
extended registry and matches canonical feature/state without confidence/GT.
Only supported_class contributes to class-priority audit. Stage 7 is unchanged.

The separate V5 validator accepts an explicit method parameter; v1 remains the
default and V4 is untouched. For v2, exact named contrasts involving the evaluated
class qualify. An association supporting the evaluated class against a named
alternative can be SUPPORTED; one supporting a competitor against the evaluated
class can be CONTRADICTED. An unrelated pair cannot contradict the evaluated class.
Genuinely competing qualifying classes require CONFLICTING. No qualifying
association gives INSUFFICIENT. Verbatim complete cited lines, registered exact
claim IDs, approved provenance, canonical observation and state remain mandatory.
These are TRAIN associations, not proof of true class or classifier error.

Pre-LLM audit accepts `--development-id v5f_validation` and recomputes claim
matches. It reports any claim-backed context separately from contexts containing
qualifying claims involving the hypothesis. The latter must cover >=1% and >=10
queries in EACH prediction stratum before a bounded diagnostic is prepared.
No TEST rows or evaluation labels are accepted. Zero/negligible coverage means
STOP, not threshold tuning. Diagnostic prompts remain NOT_TOKEN_AUDITED and
cannot be passed to the V4 executor. No LLM execution is included here.

Manual commands; run separately, requiring PASS/READY before the next command:

```bash
PY=/home/dark_bird/miniforge3/envs/ustc-gpu/bin/python
"$PY" -m pytest tests/test_rag_semantic_association_v5.py tests/test_rag_train_evidence_v5.py tests/test_rag_corpus_v2.py tests/test_rag_reranking.py tests/test_rag_trusted_context.py -q
"$PY" -m src.rag.build_semantic_association_v5 --project-root . --execute
"$PY" -m src.rag.build_semantic_association_v5 --project-root . --audit-only
```

Stop if the independent audit fails or no association survives. Otherwise:

```bash
D=v5f_validation
"$PY" -m src.rag.build_semantic_association_v5 --project-root . --stage2
"$PY" -m src.rag.build_semantic_association_v5 --project-root . --stage2 --execute
"$PY" -m src.rag.build_evidence_index_v2 --project-root . --development-id "$D"
"$PY" -m src.rag.build_evidence_index_v2 --project-root . --development-id "$D" --execute
S4=(--project-root . --profile hypothesis_balanced --development-id "$D")
"$PY" -m src.rag.run_rag_retrieval_v3 "${S4[@]}"
"$PY" -m src.rag.run_rag_retrieval_v3 "${S4[@]}" --execute
"$PY" -m src.rag.verify_retrieval_provenance_v3 --project-root . --development-id "$D"
"$PY" -m src.rag.verify_retrieval_provenance_v3 --project-root . --development-id "$D" --execute
S6=(--project-root . --profile task_relevance --development-id "$D" --method hybrid_rrf --top-k 10 --retrieval-weight 1 --source-preference-weight 0 --source-preferences-json '{}' --task-alignment-weight 0.05 --relationship-weight 0.05)
"$PY" -m src.rag.run_provenance_aware_reranking "${S6[@]}"
"$PY" -m src.rag.run_provenance_aware_reranking "${S6[@]}" --execute
S7=(--project-root . --profile evidence_diverse --development-id "$D" --diversity-threshold 1.0 --top-n 5 --token-budget 2048 --max-chunks-per-document 0)
"$PY" -m src.rag.build_trusted_contexts "${S7[@]}"
"$PY" -m src.rag.build_trusted_contexts "${S7[@]}" --execute
"$PY" -m src.rag.audit_evidence_coverage_v5 --project-root . --development-id "$D"
"$PY" -m src.rag.audit_evidence_coverage_v5 --project-root . --development-id "$D" --execute
```
