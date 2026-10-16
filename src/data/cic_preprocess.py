"""CIC-2: materialize the frozen split with TRAIN-only preprocessing.

Reads only the CIC raw CSV and four CIC-1 contracts. No split construction or
model execution occurs. Candidate source_order is zero-based within the frozen
77-candidate list. All output provenance fields are forbidden model inputs.

No extra identity file is needed: a complete existing output set must reproduce
the exact raw provenance/targets in each current frozen partition, the ordered
candidate contract, and the independently verified TRAIN parameters. Partial or
inconsistent output sets cannot be silently overwritten. Only the four named
outputs are replaced. Publication errors restore previous outputs; publication
is atomic per file, not crash-atomic across the four files. Use one publisher.

Verification re-reads saved CSVs and recomputes TRAIN statistics with NumPy;
it never calls fit, changes split assignments, or rewrites output files.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import cic_define_contract as contract

PROCESSED = 'data/processed/CIC-Darknet2020'
PARAMETERS = 'metadata/traffic/cic_preprocessing_parameters.csv'
OUTPUTS = {split: f'{PROCESSED}/{split.lower()}.csv' for split in contract.SPLITS}
PARAMETER_COLUMNS = ['source_order', 'feature_name', 'train_median', 'train_zero_variance',
                     'retained_final', 'final_predictor_order', 'train_scaler_mean', 'train_scaler_scale']
METADATA_COLUMNS = ['target_id', 'Label', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Src Port', 'Dst Port', 'Label.1']
TARGET = dict(zip(contract.LABELS, range(4)))
RTOL, ATOL = 1e-10, 1e-12
require = contract.require


def read_table(path, columns=None, *, textual=False):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    require(header and len(header) == len(set(header)), f'{path}: missing/duplicate header.')
    if columns is not None:
        require(header == columns, f'{path}: unexpected column order/schema.')
    return pd.read_csv(path, keep_default_na=False, encoding='utf-8-sig', float_precision='round_trip',
                       dtype=str if textual else {name: 'string' for name in METADATA_COLUMNS})


def load_inputs(root):
    directory = root / contract.OUTPUT_DIRECTORY
    main = contract.read_metadata(directory / contract.MAIN_FILE)
    expected = dict(**contract.IDENTITY, contract_status='FROZEN_FOR_MATERIALIZATION',
                    next_stage='CIC_PREPROCESSING_MATERIALIZATION', target_column='Label',
                    auxiliary_label_column='Label.1', class_count='4', split_group='Flow ID',
                    split_seed='42', split_method=contract.SPLIT_METHOD, **contract.POLICIES)
    require(len(main) == 1 and all(main[0].get(k) == v for k, v in expected.items()),
            'A matching frozen CIC-1 contract and preprocessing policy are required.')
    main = main[0]
    for name, expected_value in [('train_fraction', 0.6), ('validation_fraction', 0.2), ('test_fraction', 0.2)]:
        require(float(main[name]) == expected_value, 'Unexpected frozen split proportions.')
    mapping = read_table(directory / contract.MAPPING_FILE, contract.SCHEMAS[contract.MAPPING_FILE], textual=True)
    require(mapping.to_dict('records') == [dict(class_id=str(i), raw_label=label, canonical_label=label,
            probability_column=contract.PROBABILITY_COLUMNS[i]) for i, label in enumerate(contract.LABELS)],
            'Frozen label mapping/probability order differs.')
    raw = contract.read_source(root)
    observed_candidates, _ = contract.inspect_source(raw)
    features = read_table(directory / contract.FEATURE_FILE, contract.SCHEMAS[contract.FEATURE_FILE], textual=True)
    expected_features = contract.feature_contract(raw, observed_candidates).astype(str)
    semantic_fields = [name for name in contract.SCHEMAS[contract.FEATURE_FILE] if name != 'reason']
    require(features[semantic_fields].equals(expected_features[semantic_fields]), 'Frozen feature contract differs from source/policy.')
    candidates = features.loc[features.predictor_candidate == 'YES', 'column_name'].tolist()
    require(len(candidates) == 77 and not set(candidates) & set(contract.EXCLUDED), 'Expected exactly 77 allowed candidates.')
    manifest = read_table(directory / contract.MANIFEST_FILE, contract.SCHEMAS[contract.MANIFEST_FILE], textual=True)
    for name in ['group_row_count', *contract.CLASS_COUNT_COLUMNS, 'seed']:
        values = pd.to_numeric(manifest[name], errors='raise')
        require(np.isfinite(values).all() and values.ge(0).all() and values.eq(np.floor(values)).all(),
                'Manifest must contain nonnegative integer counts and seed.')
        manifest[name] = values.astype(np.int64)
    require(manifest.dataset.eq(contract.DATASET).all() and manifest.seed.eq(42).all()
            and manifest.split_method.eq(main['split_method']).all(), 'Manifest identity differs from frozen contract.')
    contract.validate_manifest(raw, manifest)
    for name, value in [('raw_rows', len(raw)), ('raw_columns', len(raw.columns)),
                        ('initial_predictor_candidates', len(candidates)), ('distinct_flow_id_groups', len(manifest)),
                        ('conflicting_flow_id_groups', int(manifest.has_label_conflict.eq('YES').sum()))]:
        require(int(main[name]) == value, f'Frozen contract {name} disagrees with source/manifest.')
    for split in contract.SPLITS:
        part = manifest.loc[manifest['split'] == split]
        for name in ['group_row_count', *contract.CLASS_COUNT_COLUMNS]:
            field = f'{split.lower()}_rows' if name == 'group_row_count' else f'{split.lower()}_{name}'
            require(int(main[field]) == int(part[name].sum()), f'Frozen split total differs: {field}.')
    row_splits = raw['Flow ID'].map(manifest.set_index('flow_id')['split'])
    require(row_splits.isin(contract.SPLITS).all(), 'Every raw row must map to one frozen split.')
    parts = {split: np.flatnonzero(row_splits.eq(split).to_numpy()) for split in contract.SPLITS}
    return raw, candidates, parts


def numeric_candidates(raw, candidates):
    return raw[candidates].apply(pd.to_numeric, errors='coerce').astype(np.float64).replace([np.inf, -np.inf], np.nan)


def provenance(raw, indices):
    result = raw.iloc[indices][METADATA_COLUMNS[1:]].reset_index(drop=True).copy()
    result.insert(0, 'target_id', result['Label'].map(TARGET))
    require(result.target_id.notna().all() and set(result.target_id) == set(range(4)), 'Each split requires all four targets.')
    return result


def fit_preprocessing(raw, candidates, parts):
    clean = numeric_candidates(raw, candidates)
    train = clean.iloc[parts['TRAIN']]
    medians = train.median()
    require(np.isfinite(medians.to_numpy()).all(), 'Every candidate needs a finite TRAIN median; all-missing TRAIN feature found.')
    imputed_train = train.fillna(medians)
    zero_variance = imputed_train.nunique(dropna=False).eq(1)
    retained = [name for name in candidates if not zero_variance[name]]
    require(retained, 'No predictors remain after TRAIN zero-variance removal.')
    scaler = StandardScaler().fit(imputed_train[retained])
    require(np.isfinite(scaler.mean_).all() and np.isfinite(scaler.scale_).all() and (scaler.scale_ > 0).all(),
            'Invalid TRAIN scaler parameters.')
    parameters = []
    for order, name in enumerate(candidates):
        keep = name in retained
        final_order = retained.index(name) if keep else None
        parameters.append(dict(source_order=order, feature_name=name, train_median=float(medians[name]),
            train_zero_variance='NO' if keep else 'YES', retained_final='YES' if keep else 'NO',
            final_predictor_order=final_order if keep else '',
            train_scaler_mean=float(scaler.mean_[final_order]) if keep else '',
            train_scaler_scale=float(scaler.scale_[final_order]) if keep else ''))
    outputs = {}
    for split, indices in parts.items():
        values = scaler.transform(clean.iloc[indices][retained].fillna(medians[retained]))
        require(np.isfinite(values).all(), f'Nonfinite processed predictors in {split}.')
        if split == 'TRAIN':
            require((np.ptp(values, axis=0) > 0).all(), 'Retained TRAIN predictor has zero variance.')
        outputs[split] = pd.concat([pd.DataFrame(values, columns=retained), provenance(raw, indices)], axis=1)
    return outputs, pd.DataFrame(parameters, columns=PARAMETER_COLUMNS)


def verify_materialization(root, raw, candidates, parts):
    """Read-only verification, independently recomputing raw TRAIN statistics."""
    parameters = read_table(root / PARAMETERS, PARAMETER_COLUMNS, textual=True)
    require(parameters.feature_name.tolist() == candidates
            and parameters.source_order.tolist() == list(map(str, range(len(candidates)))), 'Parameter candidate identity/order differs.')
    # Reconstruct directly from raw TRAIN, independently of fitting intermediates.
    train_raw = raw.iloc[parts['TRAIN']]
    matrix = np.column_stack([pd.to_numeric(train_raw[name], errors='coerce').to_numpy(dtype=float) for name in candidates])
    matrix[~np.isfinite(matrix)] = np.nan
    require(np.isfinite(matrix).any(axis=0).all(), 'No finite raw TRAIN values for a candidate.')
    medians = np.nanmedian(matrix, axis=0)
    imputed = np.where(np.isnan(matrix), medians, matrix)
    zero = np.ptp(imputed, axis=0) == 0
    keep = ~zero
    final = [name for name, retained in zip(candidates, keep) if retained]
    require(final and not set(final) & set(METADATA_COLUMNS), 'No valid retained predictors.')
    means = np.mean(imputed[:, keep], axis=0)
    variances = np.var(imputed[:, keep], axis=0, ddof=0)
    scales = np.sqrt(variances)
    # StandardScaler's documented numerical constant safeguard (float64 error
    # bound); no estimator is fitted during independent verification.
    n, eps = len(imputed), np.finfo(np.float64).eps
    scales[variances <= n * eps * variances + (n * means * eps) ** 2] = 1.0
    require(np.isfinite(means).all() and np.isfinite(scales).all() and (scales > 0).all(), 'Invalid independently recomputed TRAIN statistics.')
    require(parameters.train_zero_variance.tolist() == np.where(zero, 'YES', 'NO').tolist()
            and parameters.retained_final.tolist() == np.where(keep, 'YES', 'NO').tolist(), 'TRAIN zero-variance decisions differ.')
    require(parameters.loc[keep, 'final_predictor_order'].tolist() == list(map(str, range(len(final)))), 'Final predictor order differs.')
    require(parameters.loc[zero, ['final_predictor_order', 'train_scaler_mean', 'train_scaler_scale']].eq('').all().all(),
            'Removed features must have blank scaler/order fields.')
    for stored, expected, field in [(parameters.train_median, medians, 'median'),
            (parameters.loc[keep, 'train_scaler_mean'], means, 'mean'),
            (parameters.loc[keep, 'train_scaler_scale'], scales, 'scale')]:
        values = pd.to_numeric(stored, errors='raise').to_numpy(dtype=float)
        require(np.isfinite(values).all() and np.allclose(values, expected, rtol=RTOL, atol=ATOL),
                f'Independent TRAIN {field} verification failed.')
    saved_means = pd.to_numeric(parameters.loc[keep, 'train_scaler_mean']).to_numpy(dtype=float)
    saved_scales = pd.to_numeric(parameters.loc[keep, 'train_scaler_scale']).to_numpy(dtype=float)
    seen_groups = set()
    rows = {}
    for split, indices in parts.items():
        table = read_table(root / OUTPUTS[split], final + METADATA_COLUMNS)
        expected_provenance = provenance(raw, indices)
        require(table[METADATA_COLUMNS].astype(str).equals(expected_provenance.astype(str)),
                f'{split}: raw row/provenance/target identity or frozen split differs; refusing reuse.')
        groups = set(table['Flow ID'])
        require(not seen_groups & groups, 'Flow ID cross-split overlap.')
        seen_groups.update(groups)
        values = table[final].to_numpy(dtype=float)
        require(np.isfinite(values).all(), f'{split}: nonfinite saved predictors.')
        raw_values = np.column_stack([pd.to_numeric(raw.iloc[indices][name], errors='coerce').to_numpy(dtype=float) for name in final])
        expected = (np.where(np.isfinite(raw_values), raw_values, medians[keep]) - saved_means) / saved_scales
        require(np.allclose(values, expected, rtol=RTOL, atol=ATOL), f'{split}: saved predictors differ from frozen TRAIN transformation.')
        if split == 'TRAIN':
            require((np.ptp(values, axis=0) > 0).all(), 'Saved TRAIN has a constant retained predictor.')
        rows[f'{split.lower()}_rows'] = len(table)
    require(sum(rows.values()) == len(raw) == contract.EXPECTED_ROWS, 'Saved splits do not cover every raw row.')
    return dict(**rows, initial_predictors=len(candidates), removed_train_zero_variance=int(zero.sum()),
                final_predictors=len(final), removed_features=[name for name, removed in zip(candidates, zero) if removed],
                flow_id_overlap=0, preprocessing_leakage='PASS')


def check_existing(root, raw, candidates, parts):
    paths = [root / relative for relative in (*OUTPUTS.values(), PARAMETERS)]
    for path in paths:
        require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
                and (not path.exists() or path.is_file()), f'Unsafe canonical output: {path}')
    present = [path.exists() for path in paths]
    if any(present):
        require(all(present), 'Partial existing CIC preprocessing output set; identity requires review.')
        # Exact current split membership and source provenance establish rerun
        # identity without an extra metadata artifact. This also rejects stale
        # or corrupted parameters instead of silently overwriting them.
        verify_materialization(root, raw, candidates, parts)


def publish(root, outputs, parameters, raw, candidates, parts):
    check_existing(root, raw, candidates, parts)
    with TemporaryDirectory(prefix='.cic-preprocess-', dir=root) as temporary:
        staging = Path(temporary)
        tables = {OUTPUTS[split]: frame for split, frame in outputs.items()} | {PARAMETERS: parameters}
        require(set(tables) == {*OUTPUTS.values(), PARAMETERS}, 'Only four canonical preprocessing outputs are allowed.')
        previous = staging / 'previous'
        for relative, table in tables.items():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path, index=False, encoding='utf-8')
            if (root / relative).exists():
                backup = previous / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / relative, backup)
        replaced = []
        try:
            for relative in tables:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging / relative, destination)
                replaced.append(relative)
            return verify_materialization(root, raw, candidates, parts)
        except Exception:
            for relative in reversed(replaced):
                backup = previous / relative
                if backup.exists():
                    os.replace(backup, root / relative)
                else:
                    (root / relative).unlink()
            raise


def preprocess(root, *, execute=False):
    root = Path(root).resolve()
    raw, candidates, parts = load_inputs(root)
    check_existing(root, raw, candidates, parts)
    if not execute:
        print('PREFLIGHT_ONLY=True; frozen inputs validated; no parameters fitted or files written.')
        return None
    outputs, parameters = fit_preprocessing(raw, candidates, parts)
    summary = publish(root, outputs, parameters, raw, candidates, parts)
    print('CIC_DATA_PREP=PASS')
    for name, value in summary.items():
        print(f'{name}={value}')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    preprocess(args.project_root, execute=args.execute)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
