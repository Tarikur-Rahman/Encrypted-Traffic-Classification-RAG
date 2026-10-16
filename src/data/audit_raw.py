"""Read-only inventories. Provenance fields are not ML predictors."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
from src.common.audit import AuditRun

TAIL = ['run_id', 'timestamp_utc']
HEADERS = {
    'raw_file_manifest.csv': ['dataset', 'relative_path', 'extension', 'size_bytes', 'authoritative_binary_label'] + TAIL,
    'dataset_summary.csv': ['dataset', 'file_count', 'total_bytes', 'pcap_count', 'csv_count', 'benign_pcap_count', 'malware_pcap_count'] + TAIL,
    'extension_summary.csv': ['dataset', 'extension', 'file_count', 'total_bytes'] + TAIL,
    'pcap_inventory.csv': ['dataset', 'binary_label', 'semantic_family', 'capture_group', 'relative_path', 'size_bytes'] + TAIL,
    'family_inventory.csv': ['dataset', 'binary_label', 'semantic_family', 'pcap_count', 'total_bytes'] + TAIL,
    'ustc_audit_checks.csv': ['check_name', 'observed_value', 'expected_value', 'status', 'details'] + TAIL,
}

def classify(path: Path, base: Path) -> str:
    return {'Benign': 'benign', 'Malware': 'malware'}.get(path.relative_to(base).parts[0], '')

def provenance(path: Path, base: Path):
    rel = path.relative_to(base)
    label = classify(path, base)
    family = (Path(rel.parts[1]).stem if len(rel.parts) == 2 else rel.parts[1]) if label else ''
    # The full relative path uniquely identifies one physical source, even when
    # different nested directories contain identically named PCAPs.
    return label, family, rel.as_posix()

def audit_raw(root: Path, dataset: str) -> Path:
    if dataset not in {'USTC-TFC2016', 'CIC-Darknet2020'}:
        raise ValueError(dataset)
    base = root / 'data' / 'raw' / dataset
    if not base.is_dir():
        raise FileNotFoundError(base)
    run = AuditRun(root, dataset, 'raw_audit')
    rows, pcaps = [], []
    for path in sorted(p for p in base.rglob('*') if p.is_file()):
        label, family, group = provenance(path, base) if dataset == 'USTC-TFC2016' else ('', '', '')
        row = dict(dataset=dataset, relative_path=path.relative_to(base).as_posix(), extension=path.suffix.lower(),
                   size_bytes=path.stat().st_size, authoritative_binary_label=label)
        rows.append(row)
        if row['extension'] == '.pcap':
            pcaps.append({k: row[k] for k in ['dataset', 'relative_path', 'size_bytes']} |
                         dict(binary_label=label, semantic_family=family, capture_group=group))
    summary = dict(dataset=dataset, file_count=len(rows), total_bytes=sum(r['size_bytes'] for r in rows),
                   pcap_count=len(pcaps), csv_count=sum(r['extension'] == '.csv' for r in rows),
                   benign_pcap_count=sum(r['binary_label'] == 'benign' for r in pcaps),
                   malware_pcap_count=sum(r['binary_label'] == 'malware' for r in pcaps))
    tables = {'raw_file_manifest.csv': rows, 'dataset_summary.csv': [summary],
              'extension_summary.csv': [dict(dataset=dataset, extension=ext,
                  file_count=sum(r['extension'] == ext for r in rows),
                  total_bytes=sum(r['size_bytes'] for r in rows if r['extension'] == ext))
                  for ext in sorted({r['extension'] for r in rows})]}
    if dataset == 'USTC-TFC2016':
        families = []
        for label, family in sorted({(r['binary_label'], r['semantic_family']) for r in pcaps if r['binary_label']}):
            members = [r for r in pcaps if (r['binary_label'], r['semantic_family']) == (label, family)]
            families.append(dict(dataset=dataset, binary_label=label, semantic_family=family,
                                 pcap_count=len(members), total_bytes=sum(r['size_bytes'] for r in members)))
        checks = [
            ('raw_file_count', len(rows), 26), ('pcap_count', len(pcaps), 24),
            ('benign_pcap_count', summary['benign_pcap_count'], 14),
            ('malware_pcap_count', summary['malware_pcap_count'], 10),
            ('semantic_family_count', len(families), 20),
            ('benign_family_count', sum(r['binary_label'] == 'benign' for r in families), 10),
            ('malware_family_count', sum(r['binary_label'] == 'malware' for r in families), 10),
            ('duplicate_capture_group_count', sum(n-1 for n in Counter(r['capture_group'] for r in pcaps).values()), 0),
            ('missing_binary_label_pcap_count', sum(not r['binary_label'] for r in pcaps), 0)]
        tables.update({'pcap_inventory.csv': pcaps, 'family_inventory.csv': families,
                       'ustc_audit_checks.csv': [dict(check_name=k, observed_value=v, expected_value=e,
                            status='PASS' if v == e else 'FAIL',
                            details='Duplicates count excess occurrences; families are binary-label/family pairs.')
                            for k, v, e in checks]})
    for name, table in tables.items():
        run.write(name, HEADERS[name], table)
    return run.finish()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--dataset', choices=['USTC-TFC2016', 'CIC-Darknet2020'], required=True)
    args = parser.parse_args()
    out = audit_raw(args.project_root.resolve(), args.dataset)
    print(out)
    if args.dataset == 'USTC-TFC2016':
        import csv
        with (out / 'ustc_audit_checks.csv').open(encoding='utf-8', newline='') as stream:
            failures = [r['check_name'] for r in csv.DictReader(stream) if r['status'] == 'FAIL']
        if failures:
            print('FAIL: ' + ', '.join(failures))
            return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
