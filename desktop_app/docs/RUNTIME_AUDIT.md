# Read-only runtime audit

Completed before implementation using PowerShell reads, listings, Git status/diff,
and source searches. No project Python scripts, imports, installers, or builds ran
during the audit. The original protection baseline is retained. Figure files and
`final_paper_figures/` were already present at this task's start and absent from the
older baseline; a separate task baseline preserves them without deleting user work.

1. USTC extraction: `src.data.ustc_prepare_dataset.load_streamer()` returns the
   version-checked NFStreamer; use it with that module's `NFSTREAM_OPTIONS`.
   `flow_records` additionally attaches release labels and is inappropriate for
   unlabeled external captures. `prepare_dataset` writes protected outputs and
   must never be called.
2. Frozen NFStream 6.6.0 options: decode_tunnels=True, snapshot_length=1536,
   idle_timeout=120, active_timeout=1800, accounting_mode=0, n_dissections=0,
   statistical_analysis=True, splt_analysis=0, n_meters=0, max_nflows=0,
   performance_report=0, system_visibility_mode=0. They match configs/nfstream.json.
3. Raw USTC columns: all 60 feature_name entries in its preprocessing CSV. Native
   endpoint/DPI columns may be present but are not model inputs.
4. USTC ordered 50 predictors: retained=True rows sorted by final_feature_order
   (1–50); use existing `materialize_phase1_interface.predictor_order`, including
   its existing integrity-inventory checks. No independent feature list.
5. Preprocessing: no serialized scaler/encoder is necessary. Both saved contracts
   contain train medians and StandardScaler means/scales. USTC's integrity audit
   reconstructs `(finite-or-median - train_mean)/train_std`; CIC's
   `verify_materialization` explicitly reconstructs the equivalent transform from
   CSV parameters. Rehydrating StandardScaler and invoking transform is equivalent;
   no fit, feature engineering, or changed thresholds is permitted. USTC accepts
   real numeric/missing values; CIC uses its existing numeric_candidates coercion.
6. USTC model: models/USTC-TFC2016/20260919T183945Z/lightgbm.txt, native Booster,
   binary objective, classes benign/malware, threshold >=0.5 for malware.
7. New USTC CSV: raw native flow features, or explicitly declared already-frozen
   model features. Schema alone cannot distinguish raw from standardized values.
8. CIC ordered 62 predictors: retained_final=YES, final_predictor_order 0–61 in
   metadata/traffic/cic_preprocessing_parameters.csv, validated by predictor_order.
9. CIC CSV: raw flow-derived candidates (77) or explicitly declared frozen 62
   standardized model features. Class order Non-Tor, NonVPN, Tor, VPN. Model:
   models/CIC-Darknet2020/20260920T145831Z/lightgbm.txt, multiclass argmax.
10. No validated PCAP-to-CIC extractor exists. It remains unsupported.
11. `materialize_phase1_interface.infer` supplies frozen probabilities;
    `observation` supplies prediction/class mapping, representation and query
    semantics. Its schema template accepts validation/test only. GUI external
    records must replace template identity/provenance explicitly, never retain a
    fabricated benchmark split, label, or ground truth. semantic_profile builds
    the same training-relative retrieval text from the authorized features.
12. `run_rag_retrieval_v3.load_index`, `run_rag_retrieval.load_query_encoder` and
    `retrieve_batch` can load the existing index and retrieve for a one-row batch.
13. `verify_retrieval_provenance_v3.verify_chunks` verifies validated index chunks;
    it does not require a batch experiment or write output.
14. `run_provenance_aware_reranking.rerank` accepts one query. Frozen configuration
    comes from reranking_configuration.csv; task components use the existing
    final_test.upstream.alignment and metadata/rag/task_alignment_v3_policy.csv.
15. `build_trusted_contexts.construct` supports one query, but validates scope as
    validation/test only. External input must stop here rather than forge a split.
16. `materialize_constrained_llm_experiment.prompt_record` and
    `llm_v3_runtime.generate` support single prompts, subject to that same scope
    and the frozen Qwen token budget, model identity and response validators.
17. `run_evidence_supported_fusion.fuse` is the unchanged fusion kernel; it also
    validates validation/test scope and must not receive forged external records.
18. Batch execute/build/main routines in extraction, preprocessing, corpus/index,
    retrieval, verification, reranking, contexts, prompt materialization, subset
    preparation, diagnostic execution, and final_test stages use canonical output
    paths. None is invoked. Their pure/load kernels have been inspected separately.
19. GUI orchestration writes only app runtime sessions, caches, logs and exports.
    Imports disable bytecode before protected modules load. Python write guard and
    resolved output-path checks enforce the boundary. No corpus/index regeneration.
20. LLM default: loopback http://127.0.0.1:8000/v1/chat/completions, model
    provenance-qwen3-4b; local Qwen3-4B-Instruct-2507-FP8 tokenizer. Existing transport
    requires no API credential, forbids redirects/non-loopback endpoints. A running
    matching model server is required; model files alone do not establish readiness.

Module-level audit: inspected modules define constants/functions/classes; the USTC
module only registers an existing Npcap DLL directory on import. Heavy retrieval
imports are lazy. No protected batch CLI will run. Future runtime availability and
regression outcomes are reported separately, not inferred from these source reads.
