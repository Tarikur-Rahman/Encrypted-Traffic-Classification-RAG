# External Phase-II runtime validation — 2026-10-04

Full external Phase-II acceptance passed for USTC CSV, real USTC PCAP and CIC
CSV. No rejected response is used for fusion. Frozen experimental scopes,
validators, prompts, evidence and model assets are unchanged.

## Verified status

| Capability | Actual result |
| --- | --- |
| External trusted context | WORKING; exact frozen-field parity for both contracts |
| Local constrained LLM | WORKING; unchanged V4 validation required |
| External fusion | WORKING; only newly generated validated responses enter fusion |
| USTC raw CSV full E2E | WORKING: validated INSUFFICIENT response, final class benign |
| USTC PCAP full E2E | WORKING: real Facetime capture, validated INSUFFICIENT response, final class benign |
| CIC raw CSV full E2E | WORKING: regression response validated, final class Non-Tor |
| CIC PCAP | NOT SUPPORTED BY FROZEN CONTRACT |

The original USTC failure was reproduced before the fix. Structure, class
vocabulary, supported-class relationship and citation membership passed. The
exact failure was `V4 positive response requires nonempty EVIDENCE_QUOTE and
EVIDENCE_LINK`: the model claimed SUPPORTED with a broad multiline passage that
did not match the frozen one-line feature-to-class grammar. A saved frozen USTC
prompt passed against the same live server, isolating the defect from the model
installation, transport and frozen validator.

The desktop now applies the frozen policy's explicit “rejected for retry” rule:
one bounded retry repeats the exact validator predicate and mandatory V4 rules
32–33. The first raw response is neither edited nor accepted. The retry generates
a new raw response, which must independently pass the unchanged structure,
response and positive-evidence validators. Any second failure remains blocked.
The accepted USTC CSV and PCAP responses were ABSTAIN/INSUFFICIENT with null
supported class and no citations; retain-classifier fusion produced benign.
The CIC regression also returned ABSTAIN/INSUFFICIENT and fused to Non-Tor.

Live frozen-prompt structural checks used the first three saved held-out prompts:
CIC-TEST-000146 and CIC-TEST-000226 passed original schema, identity, citations,
semantics and token checks without fallback. CIC-TEST-000092 failed the original
positive-evidence gate. Both successes and failures are retained in the report.
The focused acceptance script passed CSV first, then real PCAP, then CIC.

Evidence: [focused acceptance](../runtime/diagnostics/ustc_contract_fix_acceptance.json),
[USTC diagnostic](../runtime/diagnostics/ustc_llm_validation/validator_error.txt),
[CIC diagnostic](../runtime/diagnostics/cic_llm_validation/validator_error.txt), and
[typed comparison](../runtime/diagnostics/llm_validation_comparison.json).
No model, policy, validator or evidence was weakened, and no invalid output was
normalized into an accepted response.

Earlier multi-flow diagnostics remain available as fail-closed evidence; they
predate the bounded semantic retry and are not the final acceptance result.

## Architecture and immutable reuse

All changes and outputs are inside `desktop_app`. The following frozen sources
were inspected read-only before implementation: trusted-context construction,
V4 prompt materialization, `llm_v3_runtime`, constrained LLM runner/importer,
fusion, common validators, positive-evidence V4, and final-test downstream/LLM.
Frozen configuration, policy and saved outputs were also inspected.

- `backend/external_phase2/trusted_context_runtime.py` is an attributed exact
  compatibility wrapper for `build_trusted_contexts.construct` (lines 19–57).
  It reuses the original context text/token functions and reads actual frozen
  limits: top 5, 2048 context tokens, maximum 2 chunks per document. Whole chunks
  are selected deterministically; verification, authorization, quality, linkage
  and score reconstruction checks remain mandatory.
- `prompt_runtime.py` streams one immutable V4 prompt template and validates its
  policy; it does not index or regenerate the multi-gigabyte prompt population.
  The input envelope preserves actual evidence, allowlisted citations and class
  semantics while identifying external scope explicitly.
- `llm_runtime.py` reuses original single-prompt generation and response validation
  function code objects. A private per-contract class lookup adds `external`
  only in desktop memory. Original module globals, evaluation validators, files
  and batch entry points are untouched. Regression checks verify this boundary.
- `fusion_runtime.py` reproduces only the original small `fuse` function
  (lines 14–44), with attribution and exact saved-field regression. Default
  `retain_classifier` remains fixed in the GUI. Fusion revalidates identity,
  response, citations, canonical authorization and verified provenance.
- External artifacts retain `source_dataset=external`, their analysis contract,
  `source_split=external_inference`, and `evaluation_scope=external_runtime`.
  They have neither ground truth nor fabricated validation/test membership.

## Local model and launcher

The model is `provenance-qwen3-4b` served only at
`http://127.0.0.1:8000/v1/chat/completions`, using existing local
`models/rag/llm/Qwen3-4B-Instruct-2507-FP8` assets. Preflight checks the expected
model and the chat route's required-field validation without generating text.
Remote endpoints and redirects are prohibited.

`scripts/Start Local LLM.ps1` wraps the user-supplied validated command in WSL
Ubuntu-24.04 and the existing `rag-llm` environment. The server uses CUDA GPU 0,
0.83 GPU memory utilization, one sequence, eager execution, generation config
vllm and seed 42, with the supplied deep-GEMM/FlashInfer sampler settings.
The older supplied 4096 context limit is replaced by the actual immutable
successful execution configuration's 5120 limit, with 1024 completion tokens,
temperature 0, retry limit 3 (the original semantic runtime caps total attempts
at 3) and timeout 120 seconds. No packages or model assets were installed.

Persistent caches, logs and temporary outputs go to `runtime/local-llm`.
Linux abstract IPC sockets avoid unsupported socket files on the Windows mount.
The live run used the already-installed vLLM 0.30.0. Token counts matched the
local frozen tokenizer. Generation is real; acceptance of an individual model
claim still depends on the unchanged V4 validator.

## Outputs, statuses and failures

Per-flow outputs resolve beneath:
`runtime/analyses/<analysis_id>/phase2/<stage>/<flow_number>/<attempt>/`.
Stages are `trusted_context` (context.json), `llm` (prompt.json, runtime.json,
and response.json only on acceptance), and `fusion` (decision.json only on
successful fusion). Separate attempt IDs prevent stale responses being reused.
The existing aggregate per-flow result and session remain app-local.

Every stage write uses `assert_desktop_write_path`, which resolves paths and
rejects traversal or links outside desktop_app. Existing backend write guards
remain installed; Python bytecode writing is disabled before frozen imports.
Input files are read-only and tested hashes are preserved.

Context failure blocks LLM/fusion. Offline LLM preserves completed context and
reports OFFLINE. After the one bounded semantic retry, invalid JSON, rejected
semantics/grounding, transport fallback, truncation or unauthorized citations
block fusion. No fallback-generated
abstention is presented as a successful external model decision. Valid genuine
ABSTAIN responses remain eligible for the original retain-classifier policy.
Only structured final response fields are stored/displayed; no hidden reasoning.

The progress page reports actual stages. Context/reasoning pages display
External Runtime scope, selected evidence/documents/chunks, token budget/count,
provenance, context text, endpoint status, model, usage, latency, retries and
validated response fields. The decision view includes final class, evidence
status, classifier prediction/confidence, supported class, fusion rule and citations.

## Regression and desktop validation

- 40 backend tests pass, with one intentionally skipped duplicate CIC parameter
  in the USTC-only vocabulary regression. The focused external Phase-II tests
  also pass; [JUnit](../runtime/diagnostics/external-phase2-tests.xml).
- Both contracts have exact selected chunk/document/order/context text/token
  parity, immutable prompt-prefix and semantic payload parity, and exact fusion
  final class/status/reason/citation/provenance/explanation parity.
- Negative tests cover identity, contract, citation, authorization, score,
  write containment, offline preservation and fallback rejection.
- 12 frontend tests passed; TypeScript, production build and Windows portable
  packaging passed. The rebuilt executable is under `release/`.
- Packaged historical smoke passed 15 viewport/page combinations, 345 text
  contrast checks, native selection, JSON/CSV export and reload persistence,
  with zero page errors. [Report](../runtime/packaged-smoke-report.json).
- Packaged external UI smoke passed all four inputs (USTC raw CSV, CIC raw CSV,
  CIC model features and Facetime PCAP), incompatible-input and outside-app export
  rejection, with zero page errors. CIC flows showed real final decisions and
  Earlier USTC flows showed the pre-fix blocked state. Results are recorded in
  [runtime smoke report](../runtime/runtime-smoke-report.json). The focused final
  live acceptance is the newer `ustc_contract_fix_acceptance.json` report above.

Commands (from desktop_app):

```powershell
.venv\Scripts\python.exe -B -m pytest tests -q
npm test
npm run package
node scripts/smoke.mjs --packaged
node scripts/runtime-smoke.mjs --packaged
.venv\Scripts\python.exe -B scripts/test_external_phase2_live.py
.venv\Scripts\python.exe -B scripts/verify_main_project_unchanged.py --baseline runtime/runtime-task-baseline.json --report runtime/external-phase2-protection-after.json
```

## Protection

Before and after verification passed: **27,272 / 27,272 unchanged**; zero added,
removed or changed protected files. Git status/diff match the authoritative task
baseline. The original 27,228-file baseline remains retained; the later baseline
also covers 44 figure artifacts that predated runtime work. No protected cache
or Python bytecode was created. Main-project files modified: **ZERO**.

[Before](../runtime/external-phase2-protection-before.json) ·
[After](../runtime/external-phase2-protection-after.json).

This task adds `backend/external_phase2/`, the local server launch wrappers, live
acceptance script and external Phase-II tests. It updates desktop adapters,
backend/API and Electron route access, context/decision/progress UI and runtime
smoke tests. Documentation changes are limited to this file and `README.md`.
The earlier input-runtime audit and all research source remain unchanged.
