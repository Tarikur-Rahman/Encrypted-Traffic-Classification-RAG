from __future__ import annotations
import argparse, subprocess
from pathlib import Path

REPO = "https://github.com/davidyslu/USTC-TFC2016.git"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    dst = args.project_root.resolve() / "data/raw/USTC-TFC2016"
    if any(dst.iterdir()):
        raise SystemExit(f"Destination is not empty; refusing to merge raw data: {dst}")
    cmd = ["git", "clone", "--depth", "1", REPO, str(dst)]
    print("COMMAND:", " ".join(cmd))
    if not args.execute:
        print("PREFLIGHT_ONLY=True")
        return 0
    subprocess.run(cmd, check=True)
    for required in ("Benign", "Malware"):
        if not (dst / required).is_dir():
            raise RuntimeError(f"Expected release directory missing: {required}")
    print("USTC_ACQUISITION=PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
