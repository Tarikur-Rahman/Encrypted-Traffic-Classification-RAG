"""Create a read-only inventory of the current main project for compatibility audits.

The snapshot is deliberately stored inside desktop_app so an audit never mutates
the authoritative research tree. Cache/build directories are excluded because
they are generated runtime noise rather than project contracts.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("PG_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
OUT = ROOT / "desktop_app" / "runtime" / "compatibility_audit" / "pre_task_main_project_snapshot.json"
EXCLUDED = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True
        ).stdout
    except OSError:
        return ""


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in EXCLUDED for part in relative.parts):
            continue
        # The output itself is not part of the source snapshot.
        if relative == OUT.relative_to(ROOT):
            continue
        try:
            files.append({
                "path": relative.as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            })
        except OSError as exc:
            files.append({"path": relative.as_posix(), "error": str(exc)})
    payload = {
        "snapshot_kind": "pre_task_current_main_project",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(ROOT),
        "excluded_directories": sorted(EXCLUDED),
        "git_status_short": git("status", "--short").splitlines(),
        "git_diff_name_status": git("diff", "--name-status").splitlines(),
        "file_count": len(files),
        "files": files,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(OUT)
    print(f"files={len(files)}")


if __name__ == "__main__":
    main()
