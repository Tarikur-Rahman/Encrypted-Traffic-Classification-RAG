"""CIC-1: freeze metadata and a grouped split, never fit preprocessing or models.

Five StratifiedGroupKFold folds (shuffle=True, seed=42) are constructed from
raw Label and exact Flow ID only. All 20 ordered validation/test fold choices
are scored. For split s with target t_s, minimize:

  sum_s (rows_s / rows - t_s)^2
  + (1/4) * sum_s sum_c (rows_sc / rows_c - t_s)^2.

This weights overall row balance and mean per-class balance equally, without
letting the majority class dominate the latter. Fraction arithmetic makes ties
exact. Ties use (sorted TRAIN fold tuple, VALIDATION fold, TEST fold), ascending.
Endpoint overlap is diagnostic only and cannot affect the chosen assignment.

Default invocation checks metadata prerequisites only. --execute validates the
source, constructs the split, and replaces exactly four canonical CSVs. Files
are staged before publication; ordinary publication errors restore the prior
files. Publication is atomic per file, not crash-atomic across all four files.
The main contract is published last. Run only one publisher at a time.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
from itertools import combinations, permutations
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

DATASET = 'CIC-Darknet2020'
TASK = 'five_model_4class_benchmark'
RAW_SOURCE = 'data/raw/CIC-Darknet2020/Darknet.CSV'
OUTPUT_DIRECTORY = 'metadata/traffic'
CLOSURE = 'metadata/traffic/ustc_experiment_closure.csv'
SEED = 42
EXPECTED_ROWS = 158616
EXPECTED_COLUMNS = 85
EXPECTED_SIZE = 73325208
EXPECTED_GROUPS = 79160
EXPECTED_CONFLICTS = 10
LABELS = ('Non-Tor', 'NonVPN', 'Tor', 'VPN')
EXPECTED_COUNTS = dict(zip(LABELS, (110442, 23863, 1392, 22919)))
PROBABILITY_COLUMNS = ('class_0_non_tor', 'class_1_nonvpn', 'class_2_tor', 'class_3_vpn')
PROVENANCE = ('Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Src Port', 'Dst Port')
EXCLUDED = (*PROVENANCE, 'Label', 'Label.1')
SPLITS = ('TRAIN', 'VALIDATION', 'TEST')
FRACTIONS = (Fraction(3, 5), Fraction(1, 5), Fraction(1, 5))
SPLIT_METHOD = 'StratifiedGroupKFold_5_shuffle_seed42_best_3_1_1_of_20'
OBJECTIVE = 'sum_split_squared_row_fraction_error+mean_class_sum_split_squared_class_fraction_error'
POLICIES = dict(
    port_policy='EXCLUDED_PROVENANCE_ONLY',
    nonfinite_policy='COERCE_NUMERIC_REPLACE_INF_WITH_MISSING',
    imputation_policy='TRAIN_MEDIAN_ONLY',
    constant_feature_policy='REMOVE_TRAIN_ZERO_VARIANCE_ONLY',
    scaling_policy='TRAIN_STANDARD_SCALER_ONLY',
    resampling_policy='NONE', pca_policy='NONE', feature_selection_policy='NONE',
)
MAIN_FILE = 'cic_phase1_contract.csv'
MAPPING_FILE = 'cic_label_mapping.csv'
FEATURE_FILE = 'cic_feature_contract.csv'
MANIFEST_FILE = 'cic_split_manifest.csv'
SCHEMAS = {
    MAPPING_FILE: ['class_id', 'raw_label', 'canonical_label', 'probability_column'],
    FEATURE_FILE: ['source_index', 'column_name', 'role', 'predictor_candidate',
                   'model_allowed', 'preprocessing_policy', 'reason'],
    MANIFEST_FILE: ['dataset', 'flow_id', 'split', 'group_row_count', 'non_tor_rows',
                    'nonvpn_rows', 'tor_rows', 'vpn_rows', 'has_label_conflict', 'seed', 'split_method'],
}
CLASS_COUNT_COLUMNS = SCHEMAS[MANIFEST_FILE][4:8]
IDENTITY = dict(dataset=DATASET, task=TASK, raw_source=RAW_SOURCE)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_metadata(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        require(fields and len(fields) == len(set(fields)), f'{path}: invalid header.')
        rows = list(reader)
    require(rows and all(None not in row and None not in row.values() for row in rows),
            f'{path}: empty or malformed metadata.')
    return rows


def protect_ustc(root):
    """The closure is the only USTC file this module reads."""
    expected = dict(dataset='USTC-TFC2016', stage_status='CLOSED_FROZEN',
                    frozen_for_retraining='YES', frozen_for_test_reuse='YES')
    rows = read_metadata(root / CLOSURE)
    matches = [row for row in rows if row.get('dataset') == expected['dataset']]
    require(len(matches) == 1 and all(matches[0].get(k) == v for k, v in expected.items()),
            'USTC protection requires one CLOSED_FROZEN record with both freeze flags YES.')


def check_output_identity(root):
    directory = root / OUTPUT_DIRECTORY
    require(directory.resolve().is_relative_to(root.resolve()), 'Output directory must stay inside project root.')
    for name in (*SCHEMAS, MAIN_FILE):
        path = directory / name
        require(not path.is_symlink() and (not path.exists() or path.is_file()),
                f'Refusing non-regular canonical output: {path}')
    main = directory / MAIN_FILE
    if main.exists():
        rows = read_metadata(main)
        require(len(rows) == 1 and all(rows[0].get(k) == v for k, v in IDENTITY.items()),
                'Existing CIC contract identity differs; refusing overwrite.')
    else:
        require(not any((directory / name).exists() for name in SCHEMAS),
                'Canonical CIC outputs exist without a main contract; ownership requires review.')


def read_source(root):
    path = root / RAW_SOURCE
    require(path.stat().st_size == EXPECTED_SIZE, f'Expected source size_bytes={EXPECTED_SIZE}.')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, [])
        require(len(header) == EXPECTED_COLUMNS, f'Expected {EXPECTED_COLUMNS} raw columns.')
        for number, row in enumerate(reader, 2):
            require(len(row) == len(header), f'Malformed source CSV at record {number}.')
    # Preserve exact source identifiers and labels, including suffixes assigned
    # by pandas to the duplicate raw Label headers. Never strip raw values.
    return pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False,
                       dtype={name: 'string' for name in EXCLUDED}, low_memory=False)


def inspect_source(frame):
    require(len(frame) == EXPECTED_ROWS, f'Expected {EXPECTED_ROWS} source rows; found {len(frame)}.')
    require(len(frame.columns) == EXPECTED_COLUMNS and frame.columns.is_unique,
            f'Expected {EXPECTED_COLUMNS} distinct pandas source columns.')
    require(set(EXCLUDED).issubset(frame.columns), 'Source is missing required label/provenance columns.')
    require(not frame[list(PROVENANCE)].isna().any().any()
            and not frame[list(PROVENANCE)].eq('').any().any(), 'Missing provenance/group value.')
    require(frame['Flow ID'].map(lambda value: isinstance(value, str)).all(),
            'Flow ID must retain its exact source string identity.')
    require(not frame['Label'].isna().any() and not frame['Label'].eq('').any(), 'Missing Label values.')
    require(set(frame['Label']) == set(LABELS), 'Label must contain exactly the four authorized raw classes.')
    require(frame['Label'].value_counts().to_dict() == EXPECTED_COUNTS, 'Unexpected source class counts.')
    candidates = [name for name in frame.columns if name not in EXCLUDED]
    require(len(candidates) == EXPECTED_COLUMNS - len(EXCLUDED), 'Unexpected predictor candidate count.')
    # This is representability validation only; nothing is imputed, scaled,
    # removed or assigned back into the source dataframe.
    missing_tokens = {'', 'nan', '+nan', '-nan', 'na', 'n/a', 'null', 'none', '<na>', '#na', '#n/a'}
    diagnostics = {}
    for name in candidates:
        source = frame[name]
        numeric = pd.to_numeric(source, errors='coerce')
        coerced = source.notna() & numeric.isna()
        tokens = source.loc[coerced].astype(str).str.strip().str.casefold()
        require(tokens.isin(missing_tokens).all(), f'Candidate {name!r} contains nonnumeric, non-missing text.')
        values = numeric.to_numpy(dtype=float, na_value=np.nan)
        diagnostics[name] = dict(missing=int(np.isnan(values).sum()),
                                 positive_inf=int(np.isposinf(values).sum()),
                                 negative_inf=int(np.isneginf(values).sum()))
    counts = group_counts(frame)
    require(len(counts) == EXPECTED_GROUPS, f'Expected {EXPECTED_GROUPS} distinct Flow ID groups; found {len(counts)}.')
    conflicts = int(counts.gt(0).sum(axis=1).gt(1).sum())
    require(conflicts == EXPECTED_CONFLICTS, f'Expected {EXPECTED_CONFLICTS} mixed-label Flow ID groups; found {conflicts}.')
    return candidates, diagnostics


def group_counts(frame):
    return (frame.groupby(['Flow ID', 'Label'], sort=True).size().unstack(fill_value=0)
            .reindex(columns=LABELS, fill_value=0))


def choose_folds(fold_counts):
    """Return the exact minimum over all 20 assignments, with lexical ties."""
    require(fold_counts.shape == (5, 4) and (fold_counts >= 0).all(), 'Expected five four-class fold counts.')
    totals = [int(value) for value in fold_counts.sum(axis=0)]
    require(all(totals), 'All four classes are required for balancing.')
    total = sum(totals)
    options = []
    for validation, test in permutations(range(5), 2):
        train = tuple(i for i in range(5) if i not in (validation, test))
        allocations = (fold_counts[list(train)].sum(axis=0), fold_counts[validation], fold_counts[test])
        score = Fraction(0)
        for counts, target in zip(allocations, FRACTIONS):
            score += (Fraction(int(sum(counts)), total) - target) ** 2
            score += sum((Fraction(int(counts[c]), totals[c]) - target) ** 2 for c in range(4)) / 4
        options.append((score, train, validation, test))
    return min(options)


def split_manifest(frame):
    y = frame['Label'].map(dict(zip(LABELS, range(4)))).to_numpy(dtype=int)
    groups = frame['Flow ID'].to_numpy(dtype=str)
    fold_ids = np.full(len(frame), -1, dtype=np.int8)
    coverage = np.zeros(len(frame), dtype=np.int8)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (_, indices) in enumerate(splitter.split(np.zeros((len(frame), 1)), y, groups)):
        require(fold < 5 and len(indices) and (indices >= 0).all() and (indices < len(frame)).all(),
                'Invalid fold assignment.')
        fold_ids[indices] = fold
        np.add.at(coverage, indices, 1)
    require((coverage == 1).all() and set(fold_ids) == set(range(5)), 'Every source row must occur in exactly one fold.')
    assignments = pd.DataFrame({'flow_id': groups, 'fold': fold_ids})
    require(assignments.groupby('flow_id')['fold'].nunique().eq(1).all(), 'Flow ID crosses folds.')
    counts = np.array([np.bincount(y[fold_ids == fold], minlength=4) for fold in range(5)])
    score, train, validation, test = choose_folds(counts)
    fold_split = {fold: 'TRAIN' for fold in train} | {validation: 'VALIDATION', test: 'TEST'}
    inventory = group_counts(frame)
    manifest = inventory.rename(columns=dict(zip(LABELS, CLASS_COUNT_COLUMNS))).rename_axis('flow_id').reset_index()
    manifest['dataset'] = DATASET
    manifest['split'] = manifest.flow_id.map(assignments.groupby('flow_id')['fold'].first().map(fold_split))
    manifest['group_row_count'] = manifest[CLASS_COUNT_COLUMNS].sum(axis=1)
    manifest['has_label_conflict'] = np.where(manifest[CLASS_COUNT_COLUMNS].gt(0).sum(axis=1) > 1, 'YES', 'NO')
    manifest['seed'] = SEED
    manifest['split_method'] = SPLIT_METHOD
    manifest = manifest[SCHEMAS[MANIFEST_FILE]]
    validate_manifest(frame, manifest)
    selection = dict(train_fold_ids=';'.join(map(str, train)), validation_fold_id=validation, test_fold_id=test,
                     balance_objective=OBJECTIVE, balance_score_exact=str(score),
                     fold_tie_break='lexicographic(sorted_train_folds,validation_fold,test_fold)',
                     sklearn_version=sklearn.__version__)
    return manifest, selection


def validate_manifest(frame, manifest):
    require(manifest.flow_id.is_unique and not manifest.flow_id.isna().any(), 'Duplicate/missing manifest Flow ID.')
    require(set(manifest.flow_id) == set(frame['Flow ID']), 'Manifest must cover every Flow ID exactly once.')
    require(manifest['split'].isin(SPLITS).all(), 'Missing or unknown split assignment.')
    expected = group_counts(frame)
    actual = manifest.set_index('flow_id').reindex(expected.index)
    require(np.array_equal(actual[CLASS_COUNT_COLUMNS].to_numpy(), expected.to_numpy()), 'Manifest class counts differ from raw groups.')
    require(np.array_equal(actual.group_row_count.to_numpy(), expected.sum(axis=1).to_numpy()), 'Manifest group row counts differ.')
    require(int(actual.group_row_count.sum()) == len(frame) == EXPECTED_ROWS, 'Not all raw rows are accounted for.')
    conflicts = np.where(expected.gt(0).sum(axis=1) > 1, 'YES', 'NO')
    require(np.array_equal(actual.has_label_conflict.to_numpy(), conflicts), 'Mixed-label group integrity failed.')
    for split in SPLITS:
        require(manifest.loc[manifest['split'] == split, CLASS_COUNT_COLUMNS].sum().gt(0).all(),
                f'All four classes must appear in {split}.')


def endpoint_overlap(frame, manifest):
    """Count shared distinct unordered IP pairs; no influence on splitting."""
    row_splits = frame['Flow ID'].map(manifest.set_index('flow_id')['split'])
    pairs = {split: set() for split in SPLITS}
    for src, dst, split in zip(frame['Src IP'], frame['Dst IP'], row_splits):
        pairs[split].add(tuple(sorted((src, dst))))
    return {f'endpoint_pair_overlap_{a.lower()}_{b.lower()}': len(pairs[a] & pairs[b])
            for a, b in combinations(SPLITS, 2)}


def feature_contract(frame, candidates):
    rows = []
    for index, name in enumerate(frame.columns):
        allowed = name in candidates
        if allowed:
            role, policy = 'predictor_candidate', ';'.join(POLICIES[key] for key in
                ('nonfinite_policy', 'imputation_policy', 'constant_feature_policy', 'scaling_policy'))
            reason = 'Retain in source order; fit preprocessing on TRAIN only in the next stage.'
        else:
            role = 'target' if name == 'Label' else 'auxiliary_label' if name == 'Label.1' else 'provenance_only'
            policy = 'EXACT_LABEL_MAPPING' if name == 'Label' else 'EXCLUDED_FROM_MODEL'
            reason = ('Exact Flow ID is split_group and provenance_only; never a predictor.' if name == 'Flow ID'
                      else 'Avoid direct service/endpoint shortcuts; preserve traffic-behavior feature boundary.' if name in ('Src Port', 'Dst Port')
                      else 'Only Label is the authorized modeling target; forbidden_predictor.' if name == 'Label.1'
                      else 'Exact four-class target; never a predictor.' if name == 'Label'
                      else 'Source identifier retained only as provenance; never a predictor.')
        rows.append(dict(source_index=index, column_name=name, role=role,
                         predictor_candidate='YES' if allowed else 'NO', model_allowed='YES' if allowed else 'NO',
                         preprocessing_policy=policy, reason=reason))
    return pd.DataFrame(rows, columns=SCHEMAS[FEATURE_FILE])


def build_contract(frame):
    candidates, numeric_diagnostics = inspect_source(frame)
    manifest, selection = split_manifest(frame)
    main = dict(**IDENTITY, raw_rows=len(frame), raw_columns=len(frame.columns),
                target_column='Label', auxiliary_label_column='Label.1', class_count=4,
                split_group='Flow ID', split_seed=SEED, train_fraction=0.60,
                validation_fraction=0.20, test_fraction=0.20, split_method=SPLIT_METHOD,
                initial_predictor_candidates=len(candidates), **POLICIES,
                contract_status='FROZEN_FOR_MATERIALIZATION', next_stage='CIC_PREPROCESSING_MATERIALIZATION',
                **selection, distinct_flow_id_groups=len(manifest),
                conflicting_flow_id_groups=int(manifest.has_label_conflict.eq('YES').sum()),
                **endpoint_overlap(frame, manifest))
    for split in SPLITS:
        part = manifest.loc[manifest['split'] == split]
        main[f'{split.lower()}_rows'] = int(part.group_row_count.sum())
        for column in CLASS_COUNT_COLUMNS:
            main[f'{split.lower()}_{column}'] = int(part[column].sum())
    mapping = pd.DataFrame([dict(class_id=i, raw_label=label, canonical_label=label,
                                probability_column=PROBABILITY_COLUMNS[i]) for i, label in enumerate(LABELS)],
                           columns=SCHEMAS[MAPPING_FILE])
    return {MAPPING_FILE: mapping, FEATURE_FILE: feature_contract(frame, candidates),
            MANIFEST_FILE: manifest, MAIN_FILE: pd.DataFrame([main])}, numeric_diagnostics


def publish(root, tables):
    require(set(tables) == {*SCHEMAS, MAIN_FILE}, 'Only the four canonical outputs may be published.')
    check_output_identity(root)
    directory = root / OUTPUT_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.cic-contract-', dir=directory) as temporary:
        staging = Path(temporary)
        previous = staging / 'previous'
        previous.mkdir()
        for name in (*SCHEMAS, MAIN_FILE):
            tables[name].to_csv(staging / name, index=False, encoding='utf-8')
            path = directory / name
            if path.exists():
                (previous / name).write_bytes(path.read_bytes())
        replaced = []
        try:
            for name in (*SCHEMAS, MAIN_FILE):
                os.replace(staging / name, directory / name)
                replaced.append(name)
        except Exception:
            for name in reversed(replaced):
                if (previous / name).exists():
                    os.replace(previous / name, directory / name)
                else:
                    (directory / name).unlink()
            raise


def define_contract(root, *, execute=False):
    root = Path(root).resolve()
    protect_ustc(root)
    check_output_identity(root)
    if not execute:
        print('PREFLIGHT_ONLY=True; source not read; no folds constructed; no files written.')
        print(f'raw_source={RAW_SOURCE}; seed={SEED}; output_directory={OUTPUT_DIRECTORY}')
        return None
    tables, diagnostics = build_contract(read_source(root))
    publish(root, tables)
    main = tables[MAIN_FILE].iloc[0]
    print(f'contract_status={main.contract_status}; predictor_candidates={main.initial_predictor_candidates}')
    for split in SPLITS:
        print(f'{split}: rows={main[f"{split.lower()}_rows"]}; class_counts='
              + str({label: main[f'{split.lower()}_{column}'] for label, column in zip(LABELS, CLASS_COUNT_COLUMNS)}))
    print(f'Flow ID groups={main.distinct_flow_id_groups}; intact mixed-label groups={main.conflicting_flow_id_groups}')
    print(f'folds: TRAIN={main.train_fold_ids}; VALIDATION={main.validation_fold_id}; TEST={main.test_fold_id}')
    for name, value in main.items():
        if name.startswith('endpoint_pair_overlap_'):
            print(f'{name}={value} (diagnostic only)')
    print('numeric_anomalies=' + str({name: values for name, values in diagnostics.items() if any(values.values())}))
    print('No preprocessing parameters fitted; no model execution or TEST evaluation.')
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Publish the four canonical CIC-1 metadata CSVs.')
    args = parser.parse_args(argv)
    define_contract(args.project_root, execute=args.execute)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
