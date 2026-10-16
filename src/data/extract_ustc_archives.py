from __future__ import annotations
import argparse
from pathlib import Path, PurePosixPath
import py7zr

def safe_member(name: str) -> bool:
    p = PurePosixPath(name.replace("\\", "/"))
    return not p.is_absolute() and ".." not in p.parts

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    root = args.project_root.resolve()
    src = root / "data/raw/USTC-TFC2016"
    out = root / "data/interim/USTC-TFC2016/pcaps"
    archives = sorted(src.rglob("*.7z"))
    pcaps = sorted(src.rglob("*.pcap")) + sorted(src.rglob("*.pcapng"))
    print(f"RAW_PCAPS={len(pcaps)}")
    print(f"ARCHIVES={len(archives)}")
    if not args.execute:
        print("PREFLIGHT_ONLY=True")
        return 0
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Output already populated: {out}")
    out.mkdir(parents=True, exist_ok=True)

    # Copy direct PCAPs while retaining authoritative Benign/Malware parent category.
    import shutil
    for p in pcaps:
        category = "Benign" if "Benign" in p.parts else "Malware" if "Malware" in p.parts else None
        if category is None:
            raise RuntimeError(f"Cannot establish USTC category for {p}")
        dst_dir = out / category / p.stem
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst_dir / p.name)

    for archive in archives:
        category = "Benign" if "Benign" in archive.parts else "Malware" if "Malware" in archive.parts else None
        if category is None:
            raise RuntimeError(f"Cannot establish USTC category for {archive}")
        dst_dir = out / category / archive.stem
        dst_dir.mkdir(parents=True, exist_ok=True)
        with py7zr.SevenZipFile(archive, mode="r") as z:
            names = z.getnames()
            bad = [n for n in names if not safe_member(n)]
            if bad:
                raise RuntimeError(f"Unsafe 7z member(s) in {archive}: {bad[:3]}")
            z.extractall(path=dst_dir)
    print("USTC_ARCHIVE_EXTRACTION=PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
