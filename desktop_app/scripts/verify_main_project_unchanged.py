"""Full content baseline: includes ignored/untracked files; excludes Git internals.

Git internals are recorded separately (status/diff) because read-only Git commands
may update index housekeeping. No research file, virtualenv, or dataset is excluded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'desktop_app' / 'protected-baseline.json'

def inventory(root):
    result = {}
    for directory, dirs, files in os.walk(root, followlinks=False):
        if Path(directory) == root:
            dirs[:] = [d for d in dirs if d not in {'desktop_app', '.git'}]
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                result[relative] = {'link': os.readlink(path)}
            else:
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                result[relative] = {'size': path.stat().st_size, 'sha256': digest}
    return result

def git_state(root):
    return {name: subprocess.run(['git', *args], cwd=root, capture_output=True,
            text=True, encoding='utf-8', errors='replace').stdout
            for name, args in [('status', ['status', '--short', '--', '.', ':!desktop_app']),
                               ('diff', ['diff', '--', '.', ':!desktop_app']),
                               ('staged', ['diff', '--cached', '--', '.', ':!desktop_app'])]}

def compare(before, after):
    return {'added': sorted(after.keys() - before.keys()),
            'removed': sorted(before.keys() - after.keys()),
            'changed': sorted(k for k in before.keys() & after.keys() if before[k] != after[k])}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--baseline', type=Path, default=BASE)
    parser.add_argument('--report', type=Path, default=BASE.parent / 'verification-report.json')
    args = parser.parse_args()
    baseline = args.baseline.resolve()
    report = args.report.resolve()
    if not baseline.is_relative_to(BASE.parent) or not report.is_relative_to(BASE.parent):
        raise SystemExit('Protection records must stay inside desktop_app.')
    if args.record and baseline.exists():
        raise SystemExit('Refusing to replace the original baseline.')
    files = inventory(ROOT)
    state = git_state(ROOT)
    if args.record:
        with baseline.open('x', encoding='utf-8') as out:
            json.dump({'files': files, 'git': state}, out, indent=2)
        print(f'Baseline recorded: {len(files)} protected files.')
        return
    before = json.loads(baseline.read_text(encoding='utf-8'))
    delta = compare(before['files'], files)
    result = dict(delta, git_unchanged=before['git'] == state, protected_files=len(files))
    result['passed'] = not any(delta.values()) and result['git_unchanged']
    report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)

if __name__ == '__main__':
    main()
