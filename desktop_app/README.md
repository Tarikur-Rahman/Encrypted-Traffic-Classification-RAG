# ProvenanceGuard ETI desktop

A local Electron + React research desktop around the existing, read-only project.
All application source, dependencies, sessions, logs, indexes, and build output are
inside `desktop_app/`. No training or experiment regeneration is performed.

Current live acceptance is complete for USTC CSV, real USTC PCAP, and CIC CSV.
Rejected raw responses remain fail-closed; the desktop performs one bounded
validator-guided retry and fuses only a newly generated response that passes the
unchanged frozen V4 validator. See [validation results](docs/RUNTIME_VALIDATION.md).

## Launch this checkout

Double-click **Start ProvenanceGuard.cmd**, or the portable executable in
`release/ProvenanceGuard-ETI-1.0.0.exe`. The executable is a companion to this
research checkout; keep it in `desktop_app/release/`. Python, models, and research
artifacts are intentionally not copied into the executable.

The isolated Node runtime and Python environment were installed for this checkout.
If using a terminal without Node on PATH:

```powershell
cd desktop_app
$env:PATH = "$PWD\tools\node-v22.22.0-win-x64;$env:PATH"
npm run dev
```

Electron starts FastAPI automatically on an OS-assigned loopback port and stops
its owned process when the application exits. No browser or separate backend
terminal is required. `npm start` opens the production frontend after a build.

## Fresh installation

Use Python 3.11+ and Node 22+. Run these commands from `desktop_app`:

```powershell
py -3.11 -m venv .venv
$env:TEMP = "$PWD\runtime\cache\tmp"
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force $env:TEMP | Out-Null
.venv\Scripts\python.exe -B -m pip install --cache-dir runtime/cache/pip --index-url https://download.pytorch.org/whl/cpu torch==2.6.0
.venv\Scripts\python.exe -B -m pip install --cache-dir runtime/cache/pip -r backend/requirements-gui.txt
$env:npm_config_cache = "$PWD\runtime\npm-cache"
$env:ELECTRON_CACHE = "$PWD\runtime\electron-cache"
npm ci
npm run build
npm start
```

Do not install GUI packages into the root research environment. This checkout's
GUI environment was created with `--system-site-packages` for read-only access to
existing scientific dependencies; GUI dependencies were installed locally.
For a completely fresh machine, install the project's necessary runtime
dependencies into **desktop_app/.venv**, preserving the research requirements file.

## Actual compatibility and research boundaries

The file picker inspects CSV/PCAP/PCAPNG before offering compatible model contracts.
For external CSVs, explicitly declare **raw flow measurements** or **model features
already standardized using this frozen transform**. Names alone cannot establish scaling.
Missing predictors are listed; they are never filled with invented zeros.

| New input | Behavior |
| --- | --- |
| USTC CSV | Frozen 60-column raw-flow contract to 50 predictors, or declared 50 model features |
| USTC PCAP / PCAPNG | Original NFStreamer 6.6.0 with exact frozen options, then frozen preprocessing and model |
| CIC CSV | Frozen 77-column raw-flow contract to 62 predictors, or declared 62 model features |
| CIC PCAP | Unsupported: no validated packet-to-CIC extractor exists |
| External Phase II | Fresh query, retrieval, provenance, reranking, trusted context, constrained local LLM and fusion |
| Validation Phase II | Also fresh trusted context; constrained V4 LLM/fusion requires the configured local server |
| Historical Phase II | Separate, explicitly labeled saved-output review |

The original integrity-audit implementations explicitly reconstruct the fitted
transform from saved TRAIN statistics. The desktop rehydrates those parameters
without fitting and verifies their parity against real raw and saved validation
rows. Existing `infer` and `observation` functions perform model prediction and
representation construction. External identifiers and provenance replace the
observation template's benchmark identity before storage or downstream calls.
External traffic always has `source_dataset=external`, an explicit
`analysis_contract`, and no invented ground truth.

The original evaluation validators remain unchanged. Desktop-only external
adapters use the frozen context and fusion algorithms with explicit
`source_split=external_inference` and `evaluation_scope=external_runtime`.
External flows are never relabeled as validation/test. Exact context and fusion
regression tests cover both model contracts. This is an external inference
runtime using frozen Phase-II algorithms and immutable research artifacts.

Fresh LLM execution uses only the original loopback endpoint
`http://127.0.0.1:8000/v1/chat/completions`, model `provenance-qwen3-4b`, original
V4 semantic fail-closed configuration, 5120-token envelope and 1024 completion
limit. No API key or external service is used. An absent server leaves reasoning
and fusion unavailable while preserving trusted context. Rejected responses,
invalid citations and unverified evidence block external fusion.

Start the existing WSL `rag-llm` environment in a separate PowerShell terminal:

```powershell
powershell -ExecutionPolicy Bypass -File "scripts/Start Local LLM.ps1"
```

This wrapper uses the user-supplied vLLM command and reads the frozen successful
execution configuration: 5120 context tokens, 1024 completion tokens, temperature
0, seed 42, three attempts and a 120-second request timeout. The supplied older
4096-token server limit is increased to match that frozen 5120-token envelope.
It uses existing local model assets, WSL Ubuntu-24.04, the existing `rag-llm`
Conda environment and CUDA GPU 0; it installs nothing. All persistent server
caches and logs are redirected into `runtime/local-llm`. Wait for
`Application startup complete` in `runtime/local-llm/server.log`, then run fresh
Phase II. The reasoning page can check model/endpoint availability without
generating an analysis response. Close the server terminal to stop it.

All new outputs, including native exports, must resolve inside `desktop_app`.
Caches, extracted flows, interface records, per-flow Phase-II records and audit
logs live under `runtime`. Selected source files remain read-only.
External stage artifacts live in
`runtime/analyses/<analysis_id>/phase2/{trusted_context,llm,fusion}/<flow_number>/<attempt>/`.
Each retry has a separate attempt directory. Only validated structured final
response fields are persisted; hidden chain-of-thought is not displayed.

The frozen held-out final outputs contain **1,200** LLM-reviewed queries: **800 CIC**
and **400 USTC**. New Analysis defaults to this completed subset so the entire
evidence trace is available. Uncheck the scope option to inspect the full saved
Phase-I population. Missing LLM/fusion outputs remain explicitly unavailable.
The application does not extrapolate subset decisions to other flows.

## Recommended workflow

1. Open **New analysis**, select a file, then choose a compatible frozen contract.
2. Choose **Completed held-out research results** or use the native file selector
   for a compatible CSV or USTC capture. Declare the CSV stage before starting.
3. **Analysis progress** reports running, complete, reused, unavailable, or
   not-applicable stages. Historical preprocessing is labeled reused.
4. Inspect class/confidence distributions on **Classification**.
5. Search, sort, filter, and paginate **Flow Explorer**. Open one flow.
6. Review its class probabilities, safe traffic representation, and provenance.
7. Choose **Review with Provenance-Aware RAG**. This reads the selected flow's
   authoritative stage outputs, rather than invoking RAG over the entire batch.
8. Explore retrieval, verification, rank movement, trusted context, constrained
   final explanation, and final classification.
9. Export JSON or CSV using a native save dialog. Choose a **new** filename.

The scientific stages remain: structured query → corpus → index → retrieval →
provenance verification → provenance-aware reranking → trusted context →
constrained reasoning → fusion → final classification. No additional screening
stages were added. Scores, probabilities, evidence, and explanations come from
the actual project. Numerical provenance scores absent from the research output
are shown as “Not reported.” A model's confidence is not evidence certainty.

## Architecture

- `frontend/src/pages/`: all eleven screens; React Router and restrained slate UI.
- `frontend/src/stores/analysis.ts`: Zustand analysis identity persistence; results
  are restored from backend session files after reload.
- `frontend/src/components/`: Recharts distributions and reusable presentation.
- `electron/`: sandboxed renderer, restricted preload IPC, native dialogs, service lifecycle.
- `backend/adapters/`: input contracts, existing Phase-I calls, saved Phase-II reads,
  CSV/JSON export serialization.
- `backend/external_phase2/`: external identity gates, exact attributed context
  and fusion compatibility wrappers, V4 prompt adaptation, and reuse of the
  original single-prompt generation/validation code through private class lookup
  bindings. Frozen module globals and evaluation validators remain unchanged.
- `backend/session/`: atomic JSON session persistence and real stage events.
- `backend/main.py`: authenticated FastAPI routes, Pydantic requests, worker queue.
- `backend/utils/write_guard.py`: accidental Python write protection outside the app.
- `runtime/`: generated CSV seek indexes, sessions, logs, test exports, screenshots.
- `scripts/`: development launch, packaging, protection verification, desktop tests.
- `tests/`: real-artifact backend integration and React route/state checks.

CSV seek indexes preserve quoted multiline records and are invalidated by source
path, size, and modification time. Index construction runs on the worker thread;
the first Phase-II review can take a minute for multi-gigabyte research artifacts.
Subsequent reviews reuse the local indexes. Heavy inference is only triggered by
an explicit Phase-I operation, never by navigation. Flow tables use server-side
pagination, class/search/confidence filtering, and sorting through TanStack Table.

The API lives behind an unpredictable per-launch bearer token retained in the
Electron main process. The renderer receives a narrow API request bridge, never
the token or unrestricted Node access. The renderer has `contextIsolation: true`,
`nodeIntegration: false`, and `sandbox: true`. Navigation, window creation, webviews,
and permission requests are blocked. Source links accept HTTPS only. See the
[Electron security guidance](https://www.electronjs.org/docs/latest/tutorial/security).

Exports are created exclusively (`wx`), preventing overwrites. The native layer
refuses export destinations in research folders, including resolved links. JSON
contains the complete session record; CSV provides one row per flow with any
available final decision. Raw LLM responses are excluded from the API and exports.

## Build and packaging

```powershell
cd desktop_app
npm run build
npm run package
```

The portable build appears under `release/`. Its runtime is supplied by this
checkout. If you relocate the executable, set `PG_PROJECT_ROOT` to the authoritative
research root. `PG_PYTHON` optionally selects the isolated Python interpreter.
Neither option changes any research configuration.

The production Content Security Policy permits only local scripts. Development
mode permits Vite's inline React refresh preamble on the local development origin.

## Validation

```powershell
.venv\Scripts\python.exe -B -m pytest -c pytest.ini tests/test_backend.py -q --basetemp=runtime/pytest-temp
npm test
npm run test:desktop
```

The desktop test launches real Electron and real FastAPI, reads canonical CIC
results, selects a flow, visits evidence and decision pages at three resolutions,
exports both formats, tests the native picker bridge with automated dialog return
values, and checks state persistence. It uses no fabricated production results.
See `runtime/desktop-smoke-report.json` and `runtime/screenshots/`.

## Proof that the main project is unchanged

Before implementation, all **27,228** existing files outside `desktop_app/` were
inventoried and SHA-256 hashed, including ignored/untracked datasets, models, and
the root virtual environment. `.git` internals are excluded from content hashing;
Git status, unstaged diff, and staged diff are separately recorded. The original
Git state already had many untracked research files and is preserved as-is.

```powershell
.venv\Scripts\python.exe -B scripts/verify_main_project_unchanged.py
```

The verifier compares contents and file membership, then checks the original Git
state outside the application. It writes `verification-report.json` and exits
nonzero on added, removed, or changed protected files. The original baseline is
`protected-baseline.json`; the script refuses to overwrite it. Do not regenerate
the baseline to hide a difference. Python runs with bytecode writing disabled.

## Troubleshooting

- **Python service does not start:** create the GUI environment and install its
  requirements. Inspect `runtime/python-service.log` and `runtime/backend.log`.
- **Node is not on PATH:** use the included `.cmd` launcher or add the isolated
  `tools/node-v22.22.0-win-x64` directory to your shell PATH.
- **PCAP or external CSV unavailable:** see the contract table above. This is not
  repaired by renaming columns, choosing another classifier, or retraining.
- **First evidence review takes time:** the worker is indexing large saved CSVs.
  Progress remains available. No research artifact is rewritten.
- **A flow has no final decision:** the local LLM is offline, a stage failed
  validation, or no historical result exists. Read the displayed reason. External
  context remains available when the LLM is offline; retry Phase II after startup.
- **Reasoning reports INSUFFICIENT:** this is a real returned outcome, including
  fail-closed constrained-response validation. It is not a GUI error.
- **Interrupted analysis:** sessions survive restart. Interrupted operations show
  a clear message; create a new analysis or retry Phase II. There is no silent rerun.
- **Export rejected:** choose an unused filename inside `desktop_app/runtime/`.

Sessions and exports can contain research paths and evidence text; store them with
the same care as the local research outputs. The application makes no cloud calls.

## Runtime regression and protection checks

```powershell
.venv\Scripts\python.exe -B -m pytest -q --junitxml=runtime/backend-test-results.xml
npm test
npm run build
npm run test:desktop
node scripts/runtime-smoke.mjs --packaged --require-live
.venv\Scripts\python.exe -B scripts/test_external_phase2_live.py
.venv\Scripts\python.exe -B scripts/verify_main_project_unchanged.py --baseline runtime/runtime-task-baseline.json --report runtime/runtime-task-verification.json
```

The original 27,228-file baseline is retained. This task's baseline contains
27,272 files, including figure artifacts that already existed when this task
started. Verification compares all current protected files, and the completion
report separately checks every original baseline file. No root research package
is installed or upgraded. The desktop reads existing scientific packages and
uses its own CPU PyTorch installation because Windows Application Control blocks
the research environment's CUDA DLL.
