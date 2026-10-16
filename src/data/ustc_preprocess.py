"""Family-disjoint USTC splitting and train-only preprocessing; preflight by default.

For each class, exhaustively evaluate all 1,260 assignments of 10 families to
6/2/2 train/validation/test slots. Minimize squared deviation of class flow
fractions from 60/20/20. Integer arithmetic avoids floating-point tie ambiguity;
a seeded permutation of sorted family names breaks ties. Only labels, family
identities and flow counts inform this allocation, never predictor values or
model performance. Row proportions need not be exactly 60/20/20.

Fit medians, constant-column removal and StandardScaler on TRAIN only. An
entirely missing training predictor is an error: no training median exists.
train_std records StandardScaler.scale_ (the actual inference denominator,
normally population standard deviation), including its numerical safeguards.
Dropped constants have train_std=0 and an empty final_feature_order.
missing_before counts original missing cells; infinite_before is separate.
train_missing_before counts training missing cells after infinity replacement.

All outputs are CSV, with no dataframe index. Completed files are published
exclusively and cannot overwrite existing artifacts. Publication is atomic per
file, not across all eight files; interruption can leave a partial run. Existing
outputs are never deleted or automatically reused. Analysis is in memory.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from itertools import combinations
import os
from pathlib import Path
import random
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.api.types import is_complex_dtype, is_numeric_dtype
from sklearn.preprocessing import StandardScaler

from src.common.io import write_csv_atomic
from src.data.ustc_prepare_dataset import DATASET, METADATA, feature_role

SEED = 20260919
EXPECTED_ROWS = 562549
EXPECTED_PREDICTORS = 60
SPLITS = ('train', 'validation', 'test')
FAMILY_COUNTS = dict(train=6, validation=2, test=2)
TARGET = {'benign': 0, 'malware': 1}
OUTPUT_METADATA = ['target_binary'] + METADATA
PARAMETER_COLUMNS = ['feature_name', 'retained', 'drop_reason', 'train_median',
                     'train_mean', 'train_std', 'final_feature_order']
TAIL = ['run_id', 'timestamp_utc']
RESULT_HEADERS = {
    'split_distribution.csv': ['dataset', 'split', 'class', 'flow_count', 'percentage_within_split', 'percentage_of_class'] + TAIL,
    'split_family_assignment.csv': ['dataset', 'binary_label', 'semantic_family', 'split', 'flow_count', 'capture_group_count'] + TAIL,
    'feature_processing.csv': ['dataset', 'feature_name', 'original_dtype', 'missing_before', 'infinite_before', 'train_missing_before', 'retained', 'drop_reason', 'final_feature_order'] + TAIL,
    'processing_summary.csv': ['dataset', 'input_rows', 'train_rows', 'validation_rows', 'test_rows', 'input_predictor_count', 'final_predictor_count', 'zero_variance_removed_count', 'train_family_count', 'validation_family_count', 'test_family_count'] + TAIL,
}


def inspect_input(frame: pd.DataFrame, *, expected_rows: int, expected_predictors: int) -> list[str]:
    if len(frame) != expected_rows:
        raise ValueError(f'Expected {expected_rows} input rows; found {len(frame)}.')
    if not frame.columns.is_unique or not frame.index.is_unique:
        raise ValueError('Input column names and row indexes must be unique.')
    if 'target_binary' in frame or not set(METADATA).issubset(frame.columns):
        raise ValueError('Expected native flows with four metadata columns and no existing target_binary.')
    if frame[METADATA].isna().any().any() or frame[METADATA].eq('').any().any():
        raise ValueError('Every input row must have complete provenance metadata.')
    if set(frame.binary_label) != set(TARGET):
        raise ValueError('binary_label must contain exactly benign and malware.')
    if frame.groupby('semantic_family').binary_label.nunique().ne(1).any():
        raise ValueError('Each semantic_family must belong to exactly one binary class.')
    family_counts = frame.groupby('binary_label').semantic_family.nunique()
    if any(family_counts.get(label, 0) != 10 for label in TARGET):
        raise ValueError('Expected 10 benign and 10 malware semantic families.')
    for key, fields in [('capture_group', ['binary_label', 'semantic_family', 'source_pcap']),
                        ('source_pcap', ['binary_label', 'semantic_family', 'capture_group'])]:
        if frame.groupby(key)[fields].nunique().ne(1).any().any():
            raise ValueError(f'Inconsistent provenance for {key}.')
    predictors = [c for c in frame.columns if feature_role(c) == 'predictor_candidate']
    if len(predictors) != expected_predictors:
        raise ValueError(f'Expected {expected_predictors} predictor candidates; found {len(predictors)}.')
    nonnumeric = [c for c in predictors if not is_numeric_dtype(frame[c]) or is_complex_dtype(frame[c])]
    if nonnumeric:
        raise ValueError('Predictors must be real numeric columns; no encoding is allowed: ' + ', '.join(nonnumeric))
    return predictors


def assign_families(frame: pd.DataFrame) -> pd.DataFrame:
    """Find the globally best squared-error flow allocation per binary class."""
    inventory = frame.groupby(['binary_label', 'semantic_family'], sort=True).agg(
        flow_count=('capture_group', 'size'), capture_group_count=('capture_group', 'nunique')).reset_index()
    assignments = {}
    rng = random.Random(SEED)
    for label in TARGET:
        counts = inventory.loc[inventory.binary_label == label].set_index('semantic_family').flow_count
        families = sorted(counts.index)
        if len(families) != 10:
            raise ValueError(f'Expected 10 families for {label}.')
        rng.shuffle(families)
        total = int(counts.sum())
        best_score, best = None, None
        for train in combinations(families, 6):
            rest = [family for family in families if family not in train]
            train_count = sum(int(counts[family]) for family in train)
            for validation in combinations(rest, 2):
                test = tuple(family for family in rest if family not in validation)
                validation_count = sum(int(counts[family]) for family in validation)
                test_count = total - train_count - validation_count
                score = ((5 * train_count - 3 * total) ** 2
                         + (5 * validation_count - total) ** 2
                         + (5 * test_count - total) ** 2)
                if best_score is None or score < best_score:
                    best_score, best = score, (train, validation, test)
        for split, families_in_split in zip(SPLITS, best):
            for family in families_in_split:
                if family in assignments:
                    raise ValueError('Semantic family names must be unique across binary classes.')
                assignments[family] = split
    inventory['split'] = inventory.semantic_family.map(assignments)
    return inventory


def split_rows(frame: pd.DataFrame, assignment: pd.DataFrame) -> dict[str, np.ndarray]:
    if not assignment.semantic_family.is_unique:
        raise ValueError('A semantic family has more than one assignment.')
    mapping = assignment.set_index('semantic_family')['split']
    row_split = frame.semantic_family.map(mapping)
    parts = {split: np.flatnonzero(row_split.eq(split).to_numpy()) for split in SPLITS}
    validate_split(frame, parts)
    return parts


def validate_split(frame: pd.DataFrame, parts: dict[str, np.ndarray]) -> None:
    if set(parts) != set(SPLITS):
        raise ValueError('Expected train, validation and test partitions.')
    positions = np.concatenate([parts[split] for split in SPLITS])
    if not np.array_equal(np.sort(positions), np.arange(len(frame))):
        raise ValueError('Split row sets must be disjoint and cover every input row exactly once.')
    for left, right in combinations(SPLITS, 2):
        for key in ['semantic_family', 'capture_group', 'source_pcap']:
            if set(frame.iloc[parts[left]][key]) & set(frame.iloc[parts[right]][key]):
                raise ValueError(f'{key} overlap between {left} and {right}.')
    for split, per_class in FAMILY_COUNTS.items():
        subset = frame.iloc[parts[split]]
        counts = subset.groupby('binary_label').semantic_family.nunique()
        if (set(counts.index) != set(TARGET) or any(counts[label] != per_class for label in TARGET)
                or subset.semantic_family.nunique() != 2 * per_class):
            raise ValueError(f'{split} must have {per_class} families per binary class.')


def fit_transform(frame: pd.DataFrame, predictors: list[str], parts: dict[str, np.ndarray]):
    """All learned parameters depend exclusively on parts['train']."""
    validate_split(frame, parts)
    if not predictors or any(c in OUTPUT_METADATA or feature_role(c) != 'predictor_candidate' for c in predictors):
        raise ValueError('Only Stage-2 predictor candidates may be fitted.')
    for c in predictors:
        if not is_numeric_dtype(frame[c]) or is_complex_dtype(frame[c]):
            raise ValueError(f'Predictor must be real numeric: {c}')
    original = frame[predictors]
    clean = original.replace([np.inf, -np.inf], np.nan)
    train = clean.iloc[parts['train']]
    medians = train.median()
    if not np.isfinite(medians.to_numpy(dtype=float)).all():
        bad = medians.index[~np.isfinite(medians.to_numpy(dtype=float))]
        raise ValueError('Cannot estimate a finite TRAIN median for: ' + ', '.join(bad))
    train_imputed = train.fillna(medians)
    retained = train_imputed.nunique(dropna=False).gt(1)
    final = [c for c in predictors if retained[c]]
    if not final:
        raise ValueError('No predictors remain after zero-variance TRAIN filtering.')
    scaler = StandardScaler().fit(train_imputed[final])
    if not np.isfinite(scaler.mean_).all() or not np.isfinite(scaler.scale_).all():
        raise ValueError('Non-finite TRAIN scaling parameters.')
    parameters, feature_rows = [], []
    for c in predictors:
        keep = bool(retained[c])
        order = final.index(c) + 1 if keep else None
        decision = dict(feature_name=c, retained=keep, drop_reason='' if keep else 'zero_variance_train',
                        final_feature_order=order)
        parameters.append(dict(decision, train_median=float(medians[c]),
            train_mean=float(scaler.mean_[order - 1]) if keep else float(train_imputed[c].iloc[0]),
            train_std=float(scaler.scale_[order - 1]) if keep else 0.0))
        feature_rows.append(dict(decision, original_dtype=str(original[c].dtype),
            missing_before=int(original[c].isna().sum()),
            infinite_before=int(np.isinf(original[c].to_numpy(dtype=float)).sum()),
            train_missing_before=int(train[c].isna().sum())))
    processed = {}
    for split in SPLITS:
        positions = parts[split]
        source = frame.iloc[positions]
        values = scaler.transform(clean.iloc[positions][final].fillna(medians[final]))
        if not np.isfinite(values).all():
            raise ValueError(f'Processed predictors contain NaN or infinity in {split}.')
        output = pd.DataFrame(values, columns=final, index=source.index)
        output['target_binary'] = source.binary_label.map(TARGET)
        output[METADATA] = source[METADATA]
        if list(output.columns) != final + OUTPUT_METADATA or set(output.target_binary) != {0, 1}:
            raise ValueError(f'Invalid predictor order or target values in {split}.')
        processed[split] = output
    return processed, parameters, feature_rows


def result_tables(frame, parts, assignment, parameters, feature_rows):
    distribution = []
    for split in SPLITS:
        subset = frame.iloc[parts[split]]
        for label in TARGET:
            count = int(subset.binary_label.eq(label).sum())
            distribution.append({'split': split, 'class': label, 'flow_count': count,
                'percentage_within_split': 100.0 * count / len(subset),
                'percentage_of_class': 100.0 * count / int(frame.binary_label.eq(label).sum())})
    final_count = sum(r['retained'] for r in parameters)
    return {
        'split_distribution.csv': distribution,
        'split_family_assignment.csv': assignment.to_dict('records'),
        'feature_processing.csv': feature_rows,
        'processing_summary.csv': [dict(input_rows=len(frame),
            train_rows=len(parts['train']), validation_rows=len(parts['validation']), test_rows=len(parts['test']),
            input_predictor_count=len(parameters), final_predictor_count=final_count,
            zero_variance_removed_count=len(parameters) - final_count,
            train_family_count=frame.iloc[parts['train']].semantic_family.nunique(),
            validation_family_count=frame.iloc[parts['validation']].semantic_family.nunique(),
            test_family_count=frame.iloc[parts['test']].semantic_family.nunique())],
    }


def preprocess(root: Path, execute: bool = False) -> tuple[Path, Path]:
    root = root.resolve()
    source = root / 'data' / 'interim' / DATASET / 'flows.csv'
    output_dir = root / 'data' / 'processed' / DATASET
    now = datetime.now(timezone.utc)
    run_id = now.strftime('%Y%m%dT%H%M%SZ')
    timestamp = now.isoformat().replace('+00:00', 'Z')
    result_dir = root / 'results' / 'datasets' / DATASET / 'preprocessing' / run_id
    output_names = [f'{split}.csv' for split in SPLITS] + ['preprocessing_parameters.csv']
    if execute:
        for name in output_names:
            if (output_dir / name).exists():
                raise FileExistsError(f'Refusing to overwrite: {output_dir / name}')
        if result_dir.exists():
            raise FileExistsError(f'Refusing to reuse result directory: {result_dir}')
    with source.open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream))
    if len(header) != len(set(header)):
        raise ValueError('Duplicate input column names are not allowed.')
    frame = pd.read_csv(source, low_memory=False, encoding='utf-8-sig',
                        dtype={c: 'string' for c in METADATA})
    predictors = inspect_input(frame, expected_rows=EXPECTED_ROWS, expected_predictors=EXPECTED_PREDICTORS)
    print(f'input_rows={len(frame)}\ncandidate_predictor_count={len(predictors)}')
    print(f'semantic_family_count={frame.semantic_family.nunique()}')
    print(f'Method: family-disjoint 6/2/2 per class; exhaustive flow-balance optimization; seed={SEED}')
    for name in output_names:
        print(f'planned_output={output_dir / name}')
    print(f'planned_result_directory={result_dir}\nPREFLIGHT_ONLY={not execute}')
    if not execute:
        return output_dir, result_dir
    assignment = assign_families(frame)
    parts = split_rows(frame, assignment)
    processed, parameters, feature_rows = fit_transform(frame, predictors, parts)
    tables = result_tables(frame, parts, assignment, parameters, feature_rows)
    # Stage all files before publication. The shared CSV writer and hard links
    # fail if a concurrent writer creates any destination in the meantime.
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.ustc-preprocess-', dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        for split, output in processed.items():
            columns = list(output.columns)
            write_csv_atomic(stage / f'{split}.csv', columns,
                (dict(zip(columns, row)) for row in output.itertuples(index=False, name=None)))
        write_csv_atomic(stage / 'preprocessing_parameters.csv', PARAMETER_COLUMNS, parameters)
        for name, rows in tables.items():
            write_csv_atomic(stage / name, RESULT_HEADERS[name], (
                dict(row, dataset=DATASET, run_id=run_id, timestamp_utc=timestamp) for row in rows))
        result_dir.mkdir(parents=True, exist_ok=False)
        for name in output_names:
            os.link(stage / name, output_dir / name)
        for name in RESULT_HEADERS:
            os.link(stage / name, result_dir / name)
    print('VALIDATION=PASS')
    return output_dir, result_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    preprocess(args.project_root, execute=args.execute)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
