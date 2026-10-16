from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.common.io import write_json_atomic
from src.common.hashing import sha256_file

SEED = 20260919

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    root = args.project_root.resolve()
    inp = args.input if args.input.is_absolute() else root / args.input
    df = pd.read_csv(inp)
    for c in (args.target, args.group):
        if c not in df.columns:
            raise KeyError(c)

    groups = df[args.group].astype(str).to_numpy()
    y = df[args.target].to_numpy()
    idx = np.arange(len(df))

    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.60, random_state=SEED)
    train_idx, temp_idx = next(gss1.split(idx, y, groups))
    temp_groups = groups[temp_idx]
    temp_y = y[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=SEED + 1)
    val_rel, test_rel = next(gss2.split(temp_idx, temp_y, temp_groups))
    val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]

    parts = {"train": train_idx, "validation": val_idx, "test": test_idx}
    group_sets = {k: set(groups[v].tolist()) for k, v in parts.items()}
    overlaps = {
        "train_validation": len(group_sets["train"] & group_sets["validation"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "validation_test": len(group_sets["validation"] & group_sets["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Group leakage detected: {overlaps}")

    summary = {
        "seed": SEED,
        "target": args.target,
        "group": args.group,
        "input_sha256": sha256_file(inp),
        "rows": {k: int(len(v)) for k, v in parts.items()},
        "groups": {k: int(len(group_sets[k])) for k in parts},
        "overlaps": overlaps,
        "class_counts": {
            k: {str(cls): int((y[v] == cls).sum()) for cls in np.unique(y)}
            for k, v in parts.items()
        },
    }
    print(json.dumps(summary, indent=2))
    if not args.execute:
        print("PREFLIGHT_ONLY=True")
        return 0

    out_dir = inp.parent / "splits_v1"
    if out_dir.exists():
        raise RuntimeError(f"Refusing overwrite: {out_dir}")
    out_dir.mkdir(parents=True)
    for name, indices in parts.items():
        df.iloc[indices].to_csv(out_dir / f"{name}.csv", index=False)
    summary["output_sha256"] = {
        name: sha256_file(out_dir / f"{name}.csv") for name in parts
    }
    write_json_atomic(out_dir / "split_manifest.json", summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
