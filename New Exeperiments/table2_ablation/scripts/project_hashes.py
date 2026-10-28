"""Hash every project file, excluding only the Experiment 2 directory."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'supervisor_requested_experiments/table2_ablation'

def snapshot(label):
    target = OUT / 'audit' / f'main_project_file_hashes_{label}.csv'
    count = total = 0
    with target.open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path','size_bytes','sha256'])
        for directory, dirs, files in os.walk(ROOT):
            dirs[:] = sorted(d for d in dirs if Path(directory) / d != OUT)
            for name in sorted(files):
                p = Path(directory) / name
                with p.open('rb') as stream:
                    sha = hashlib.file_digest(stream, 'sha256').hexdigest()
                size = p.stat().st_size
                writer.writerow([p.relative_to(ROOT).as_posix(), size, sha])
                count += 1
                total += size
                if count % 25000 == 0:
                    print(f'{label}: {count} files, {total/1e9:.2f} GB hashed', flush=True)
    print(f'{label}: COMPLETE {count} files, {total/1e9:.2f} GB', flush=True)
    if label == 'after':
        def read(path):
            with path.open(encoding='utf-8',newline='') as f:
                return {r['path']:r for r in csv.DictReader(f)}
        before=read(OUT/'audit/main_project_file_hashes_before.csv')
        after=read(target)
        changed=[p for p in before if before[p] != after.get(p)]
        added=sorted(set(after)-set(before))
        text=f'# Main-project change report\n\nExisting main-project files modified/deleted: {len(changed)}\n\nFiles added outside Experiment 2: {len(added)}\n\nSHA-256 compared for {len(before)} files, including .git, environments and Table 1. Only table2_ablation excluded.\n\nChanged: {changed}\n\nAdded: {added}\n'
        (OUT/'audit/main_project_change_report.md').write_text(text,encoding='utf-8')
        assert not changed and not added, text
        report_path=OUT/'logs/final_terminal_report.txt'
        report=report_path.read_text(encoding='utf-8').replace('Existing main-project files modified: see mandatory SHA-256 after-comparison',
            f'Existing main-project files modified: ZERO (all {len(before)} files SHA-256 verified; only table2_ablation excluded)')
        report_path.write_text(report,encoding='utf-8')
        manifest_path=OUT/'run_manifest.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['existing_main_project_files_modified']=0
        manifest['existing_files_hash_verified']=len(before)
        manifest['output_hashes']={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in OUT.rglob('*') if p.is_file() and p!=manifest_path}
        manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
        print(report)

if __name__ == '__main__':
    snapshot(sys.argv[1])
