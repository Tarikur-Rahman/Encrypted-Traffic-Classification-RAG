# Validation record

The application uses real canonical research artifacts in backend and Electron
integration tests. Test-only native dialog responses are automated; production
does not contain mock predictions, evidence, scores, or final decisions.

## Verified

- TypeScript type checking and Vite production compilation.
- Electron packaging into a Windows portable executable and unpacked app.
- Eight Python checks: authenticated health, CSV contracts, PCAP/PCAPNG refusal,
  multiline CSV indexes and invalidation, protection comparison, real saved
  Phase-I/Phase-II integration, both frozen LightGBM models, CSV export escaping.
- Twelve frontend checks: all eleven routes and Zustand session hydration.
- Desktop launch starts Python automatically and establishes authenticated health.
- A real CIC session loads 800 saved completed-subset flows and reads the complete
  evidence trace for `CIC-TEST-000092`.
- The first example's INSUFFICIENT result is intentionally preserved; the GUI
  does not turn missing positive evidence into a supported classification.
- Native file-picker IPC returns local paths for CSV, PCAP, and PCAPNG.
- JSON and CSV export write user-selected new destinations.
- Layout tests check 1280×720, 1440×900, and 1920×1080.
- Computed text contrast is checked for visible research-page text; reports
  record checked counts and any failures.

Generated reports and screenshots live under `runtime/`:

- `desktop-smoke-report.json`
- `packaged-smoke-report.json`
- `screenshots/`

`verification-report.json` at the app root records the full protected-file check.
During testing, pytest initially resolved a relative cache path against the root
research project and added five cache files under a new root `runtime/` directory.
The verifier detected this. Those exact generated files were removed, and a full
rehash confirmed all 27,228 original files and original Git state were unchanged.
The app now has its own `pytest.ini`, with the cache provider disabled, to prevent
recurrence. No existing research file needed restoration.

## Deliberate limitations

- Arbitrary raw CSV and PCAP inference are unavailable because there is no
  reusable frozen external-input preprocessing entrypoint in the supplied code.
- Fresh Phase-II retrieval and LLM execution are not performed; actual saved
  stage outputs are reviewed with explicit “reused” statuses.
- Only saved reasoning-subset flows have final explanations and decisions.
- The portable executable depends on the adjacent research checkout and isolated
  Python environment; it does not bundle or copy the research assets.
- Native dialog appearance is supplied by Windows. Automated testing substitutes
  dialog return values while exercising the real preload and main-process code.

These boundaries are visible in the UI and README. They are not presented as
completed capabilities.
