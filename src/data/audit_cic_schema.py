"""Descriptive audit only: no label normalization or target mapping."""
from __future__ import annotations
import argparse
import csv
from collections import Counter
from pathlib import Path
import pandas as pd
from src.common.audit import AuditRun

LABEL_CANDIDATES = {'label', 'label.1', 'category', 'traffic', 'class', 'application', 'type', 'network', 'privacy', 'target'}
PREFIX = ['dataset', 'source_file']
TAIL = ['run_id', 'timestamp_utc']
HEADERS = {
    'cic_dataset_summary.csv': PREFIX + ['rows', 'columns', 'size_bytes', 'candidate_label_column_count'] + TAIL,
    'schema_columns.csv': PREFIX + ['column_index', 'column_name', 'dtype', 'missing_count', 'missing_fraction', 'unique_count', 'is_candidate_label'] + TAIL,
    'label_value_counts.csv': PREFIX + ['label_column', 'label_value', 'count', 'fraction'] + TAIL,
    'missing_values.csv': PREFIX + ['column_name', 'missing_count', 'missing_fraction'] + TAIL,
    'duplicate_column_names.csv': PREFIX + ['collision_type', 'normalized_name', 'column_name', 'column_index'] + TAIL,
    'cic_candidate_label_summary.csv': PREFIX + ['label_column', 'unique_value_count', 'missing_count', 'missing_fraction'] + TAIL,
}

def audit_cic_schema(root: Path) -> Path:
    dataset = 'CIC-Darknet2020'
    base = root / 'data' / 'raw' / dataset
    files = sorted(p for p in base.rglob('*') if p.is_file() and p.suffix.lower() == '.csv')
    if not files:
        raise FileNotFoundError(f'No CSV files under {base}')
    run = AuditRun(root, dataset, 'schema_audit')
    tables = {name: [] for name in HEADERS}
    for path in files:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream, strict=True)
            raw_names = next(reader)
            for record, values in enumerate(reader, 2):
                if len(values) != len(raw_names):
                    raise ValueError(f'{path}: record {record} has {len(values)} fields; expected {len(raw_names)}')
        # Let pandas distinguish duplicate headers, including any existing
        # suffixed names. Use those exact identities in every output table.
        names = list(pd.read_csv(path, nrows=0, encoding='utf-8-sig').columns)
        candidates = [n.strip().lower() in LABEL_CANDIDATES or 'label' in n.strip().lower() for n in names]
        # Only empty fields are
        # missing: literal NA, NULL, <NA>, whitespace and numeric labels survive.
        frame = pd.read_csv(path, header=0,
                            dtype={name: 'string' for name, yes in zip(names, candidates) if yes},
                            keep_default_na=False, na_values=[''], low_memory=False, encoding='utf-8-sig')
        common = dict(dataset=dataset, source_file=path.relative_to(base).as_posix())
        def add(name, **values):
            tables[name].append(common | values)
        add('cic_dataset_summary.csv', rows=len(frame), columns=len(names), size_bytes=path.stat().st_size,
            candidate_label_column_count=sum(candidates))
        exact = Counter(raw_names)
        folded = Counter(n.casefold() for n in raw_names)
        for i, name in enumerate(frame.columns):
            series = frame.iloc[:, i]
            missing = int(series.isna().sum())
            fraction = missing / len(frame) if len(frame) else 0.0
            unique = int(series.nunique(dropna=True))
            add('schema_columns.csv', column_index=i, column_name=name, dtype=str(series.dtype),
                missing_count=missing, missing_fraction=fraction, unique_count=unique, is_candidate_label=candidates[i])
            add('missing_values.csv', column_name=name, missing_count=missing, missing_fraction=fraction)
            raw_name = raw_names[i]
            for collision, normalized, count in [('exact', raw_name, exact[raw_name]), ('case_insensitive', raw_name.casefold(), folded[raw_name.casefold()])]:
                if count > 1:
                    add('duplicate_column_names.csv', collision_type=collision, normalized_name=normalized, column_name=name, column_index=i)
            if candidates[i]:
                add('cic_candidate_label_summary.csv', label_column=name, unique_value_count=unique,
                    missing_count=missing, missing_fraction=fraction)
                for value, count in series.value_counts(dropna=False, sort=False).items():
                    add('label_value_counts.csv', label_column=name, label_value='' if pd.isna(value) else value,
                        count=int(count), fraction=int(count) / len(frame))
    for name, rows in tables.items():
        run.write(name, HEADERS[name], rows)
    return run.finish()

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    print(audit_cic_schema(args.project_root.resolve()))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
