from __future__ import annotations
import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from src.common.io import write_json_atomic
from src.common.hashing import sha256_file

EXCLUDE_EXACT = {
    "id", "expiration_id", "src_ip", "dst_ip", "src_mac", "dst_mac", "src_oui", "dst_oui",
    "src_port", "dst_port",
    "bidirectional_first_seen_ms", "bidirectional_last_seen_ms",
    "src2dst_first_seen_ms", "src2dst_last_seen_ms",
    "dst2src_first_seen_ms", "dst2src_last_seen_ms",
    "application_name", "application_category_name", "application_is_guessed",
    "application_confidence", "requested_server_name", "client_fingerprint",
    "server_fingerprint", "user_agent", "content_type",
}
EXCLUDE_SUBSTRINGS = ("ip_address", "filename", "source_file", "row_id")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    root = args.project_root.resolve()
    native_root = root / "data/interim/USTC-TFC2016/nfstream_native"
    files = sorted(native_root.rglob("*.csv"))
    if not files:
        raise SystemExit("No native NFStream CSVs. Run extraction first.")

    # Schema must be identical across every capture.
    headers = [list(pd.read_csv(p, nrows=0).columns) for p in files]
    canonical = headers[0]
    if any(h != canonical for h in headers[1:]):
        raise RuntimeError("NFStream schema differs across USTC capture outputs.")
    if any(c in canonical for c in (
        "application_name", "requested_server_name", "user_agent"
    )):
        raise RuntimeError("Unexpected L7/DPI columns found despite n_dissections=0.")

    predictor_candidates = [
        c for c in canonical
        if c not in EXCLUDE_EXACT
        and not any(x in c.lower() for x in EXCLUDE_SUBSTRINGS)
    ]
    audit = {
        "native_columns": canonical,
        "excluded_columns": [c for c in canonical if c not in predictor_candidates],
        "candidate_predictors": predictor_candidates,
        "candidate_predictor_count": len(predictor_candidates),
        "status": "CANDIDATE_FEATURE_CONTRACT_REQUIRES_VALIDATION",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    contract_path = root / "metadata/contracts" / f"ustc_feature_contract_candidate_{stamp}.json"
    if not args.execute:
        print(json.dumps(audit, indent=2))
        print("PREFLIGHT_ONLY=True")
        return 0

    output = root / "data/processed/USTC-TFC2016/flows_candidate.csv"
    if output.exists():
        raise RuntimeError(f"Refusing overwrite: {output}")
    frames = []
    for p in files:
        rel = p.relative_to(native_root)
        category = rel.parts[0].lower()
        family = rel.parts[1] if len(rel.parts) > 1 else p.stem
        if category not in {"benign", "malware"}:
            raise RuntimeError(f"Unexpected category: {category}")
        df = pd.read_csv(p)
        frame = df[predictor_candidates].copy()
        frame.insert(0, "capture_group", str(rel.parent).replace("\\", "/"))
        frame.insert(1, "family", family)
        frame.insert(2, "target_binary", 0 if category == "benign" else 1)
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    if merged[predictor_candidates].isna().any().any():
        # Do not impute before split; preserve and report.
        audit["missing_predictor_cells"] = int(merged[predictor_candidates].isna().sum().sum())
    else:
        audit["missing_predictor_cells"] = 0
    merged.to_csv(output, index=False)
    audit.update({
        "processed_candidate": str(output.relative_to(root)).replace("\\", "/"),
        "processed_candidate_sha256": sha256_file(output),
        "rows": int(len(merged)),
        "status": "MATERIALIZED_CANDIDATE_NOT_SPLIT_NOT_PREPROCESSED",
    })
    write_json_atomic(contract_path, audit)
    print(json.dumps(audit, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
