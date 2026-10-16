"""Read-only content inventory of the current research project."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'desktop_app/runtime/redesign_audit/pre_task_snapshot.json'
EXCLUDE = {'desktop_app', '.git', '.venv', '__pycache__', '.pytest_cache', 'node_modules'}

def main():
    if OUT.exists():
        raise SystemExit('Snapshot already exists; refusing to replace the baseline.')
    files = {}
    for directory, names, filenames in os.walk(ROOT):
        names[:] = [name for name in names if name not in EXCLUDE]
        for name in filenames:
            path = Path(directory) / name
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            files[path.relative_to(ROOT).as_posix()] = {'size': path.stat().st_size, 'sha256': digest}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump({'root': str(ROOT), 'excluded': sorted(EXCLUDE), 'files': files}, stream, indent=2)
    print(f'Snapshotted {len(files)} research files: {OUT}', flush=True)

if __name__ == '__main__':
    main()
