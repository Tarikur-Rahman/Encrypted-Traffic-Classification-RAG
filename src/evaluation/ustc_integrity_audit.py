"""Read-only evidence audit of the frozen USTC experiment (no model loading).

Only --execute reads the datasets and publishes the eleven canonical audit files.
Independent descriptive TRAIN statistics verify saved preprocessing parameters;
no estimator or transformer is fitted. Numeric row matches use native pandas
equality/grouping, never digest-based identities. Model files are stat'ed only.

PASS concerns the available source and persisted evidence, not an attestation of
historical execution or the contents of opaque checkpoints. Source contracts fail
closed when the reviewed data-access/learning operations change. ROC's initial
+infinity sentinel, PR's terminal blank threshold and empty calibration bins are
documented format exceptions, not corrupt probabilities.
"""
from __future__ import annotations

import argparse
import ast
import csv
from itertools import combinations
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from pandas.api.types import is_complex_dtype, is_numeric_dtype

DATASET = 'USTC-TFC2016'
TASK = 'integrity_audit'
RUN = '20260919T183945Z'
SPLITS = ('train', 'validation', 'test')
ROWS = dict(train=367293, validation=97001, test=98255)
SEED, THRESHOLD, PREDICTOR_COUNT = 42, 0.5, 50
PROVENANCE = ['binary_label', 'semantic_family', 'capture_group', 'source_pcap']
METADATA = ['target_binary'] + PROVENANCE
FAMILIES = {
    'train': {'benign': {'BitTorrent', 'Facetime', 'FTP', 'MySQL', 'Outlook', 'Skype'},
              'malware': {'Cridex', 'Geodo', 'Htbot', 'Miuref', 'Nsis-ay', 'Tinba'}},
    'validation': {'benign': {'Weibo', 'WorldOfWarcraft'}, 'malware': {'Virut', 'Zeus'}},
    'test': {'benign': {'Gmail', 'SMB'}, 'malware': {'Neris', 'Shifu'}},
}
DIRECTIONS = ('bidirectional', 'src2dst', 'dst2src')
CANDIDATES = ['protocol', 'ip_version', 'tunnel_id'] + [
    f'{direction}_{suffix}' for direction in DIRECTIONS for suffix in (
        'duration_ms', 'packets', 'bytes', 'min_ps', 'mean_ps', 'stddev_ps', 'max_ps',
        'min_piat_ms', 'mean_piat_ms', 'stddev_piat_ms', 'max_piat_ms', 'syn_packets',
        'cwr_packets', 'ece_packets', 'urg_packets', 'ack_packets', 'psh_packets',
        'rst_packets', 'fin_packets')]
EXCLUDED = {'id', 'expiration_id', 'src_ip', 'src_mac', 'src_oui', 'src_port',
            'dst_ip', 'dst_mac', 'dst_oui', 'dst_port', 'vlan_id'} | {
    f'{direction}_{time}_seen_ms' for direction in DIRECTIONS for time in ('first', 'last')}
REMOVED = {'tunnel_id'} | {f'{d}_{flag}_packets' for d in DIRECTIONS for flag in ('cwr', 'ece', 'urg')}
ARTIFACTS = {'random_forest': 'random_forest.pkl', '1d_cnn': '1d_cnn.pt',
             'xgboost': 'xgboost.json', 'lightgbm': 'lightgbm.txt', 'cnn_lstm': 'cnn_lstm.pt'}
DEVICES = dict(zip(ARTIFACTS, ('cuml_cuda', 'cuda', 'cuda', 'lightgbm_cuda', 'cuda')))
HISTORY_REFERENCES = {'1d_cnn': (30, 3, 1.3902553872), 'cnn_lstm': (30, 19, 0.7580079576),
                      'xgboost': (300, 299, 0.0009274106), 'lightgbm': (246, 216, 0.0024597499)}
METRICS = 'accuracy precision recall f1 weighted_f1 mcc roc_auc auprc ece brier_score'.split()
PREFIX, TAIL = ['dataset', 'model', 'split'], ['run_id', 'timestamp_utc']
RESULT_SCHEMAS = {
    'metrics.csv': PREFIX + ['device'] + METRICS + ['threshold', 'seed'] + TAIL,
    'confusion_matrix.csv': PREFIX + ['actual_class', 'predicted_class', 'count'] + TAIL,
    'roc_curve.csv': PREFIX + ['fpr', 'tpr', 'threshold'] + TAIL,
    'pr_curve.csv': PREFIX + ['precision', 'recall', 'threshold'] + TAIL,
    'calibration_bins.csv': PREFIX + ['bin_index', 'lower_bound', 'upper_bound', 'count',
        'mean_probability', 'observed_positive_fraction', 'absolute_gap'] + TAIL,
    'training_history.csv': ['dataset', 'model', 'step', 'train_loss', 'validation_loss'] + TAIL,
}
PARAMETER_COLUMNS = ['feature_name', 'retained', 'drop_reason', 'train_median',
                     'train_mean', 'train_std', 'final_feature_order']
CHECK_COLUMNS = ['category', 'check_name', 'status', 'details']
OUTPUT_SCHEMAS = {
    'ustc_integrity_audit_checklist.csv': CHECK_COLUMNS,
    'ustc_split_family_assignment.csv': ['semantic_family', 'binary_label', 'split', 'rows'],
    'ustc_capture_group_split_assignment.csv': ['capture_group', 'source_pcap', 'semantic_family', 'binary_label', 'split', 'rows'],
    'ustc_cross_split_duplicate_check.csv': ['comparison', 'left_split', 'right_split',
        'shared_distinct_rows', 'left_matching_rows', 'right_matching_rows', 'matching_row_pairs',
        'numeric_duplicate', 'provenance_overlap', 'leakage_interpretation', 'status', 'examples'],
    'ustc_final_predictor_inventory.csv': ['predictor', 'status', 'reason'],
    'ustc_preprocessing_leakage_check.csv': CHECK_COLUMNS,
    'ustc_test_isolation_check.csv': CHECK_COLUMNS,
    'ustc_model_history_consistency.csv': ['model', 'recorded_steps', 'final_step', 'best_validation_step',
        'best_validation_loss', 'artifact', 'size_bytes', 'status', 'details'],
    'ustc_result_consistency_check.csv': CHECK_COLUMNS,
    'ustc_integrity_audit_metadata.csv': ['dataset', 'task', 'authoritative_run', 'audit_scope',
        'predictor_count', 'train_rows', 'validation_rows', 'test_rows', 'seed', 'threshold',
        'overall_status', 'finalization_authorized'],
}
FINDINGS = 'ustc_integrity_audit_findings.txt'
OUTPUT_NAMES = tuple(OUTPUT_SCHEMAS) + (FINDINGS,)
LIMITATION = ('Current source and persisted CSV consistency only; no historical code/execution '
              'attestation exists in these inputs. Checkpoint contents/restoration cannot be independently '
              'verified by file metadata. No models are deserialized or executed.')
PROTOCOL_EVIDENCE = ('Current source and persisted artifacts are consistent with the required '
                     'TRAIN → VALIDATION → freeze → TEST protocol; historical runtime execution '
                     'cannot be independently attested from these files alone.')


class Audit:
    def __init__(self):
        self.tables = {name: [] for name in OUTPUT_SCHEMAS}

    def check(self, category, name, ok, details='', *, warning=False):
        row = dict(category=category, check_name=name,
                   status='PASS' if bool(ok) else ('WARN' if warning else 'FAIL'), details=str(details))
        self.tables['ustc_integrity_audit_checklist.csv'].append(row)
        target = {'F': 'ustc_preprocessing_leakage_check.csv', 'G': 'ustc_test_isolation_check.csv'}
        if category in ('H', 'I', 'J', 'K'):
            target[category] = 'ustc_result_consistency_check.csv'
        if category in target:
            self.tables[target[category]].append(row.copy())
        return bool(ok)

    def section(self, category, action):
        try:
            action()
        except (ValueError, KeyError, TypeError, IndexError, OSError, ArithmeticError) as exc:
            self.check(category, 'section_completed', False, f'{type(exc).__name__}: {exc}')

    @property
    def passed(self):
        return bool(self.tables['ustc_integrity_audit_checklist.csv']) and not any(
            r['status'] == 'FAIL' for r in self.tables['ustc_integrity_audit_checklist.csv'])


def read_csv(path, *, usecols=None):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        columns = next(csv.reader(stream))
    if len(columns) != len(set(columns)):
        raise ValueError(f'Duplicate column names in {path}')
    return pd.read_csv(path, usecols=usecols, low_memory=False, float_precision='round_trip')


def finite_range(values, low=0.0, high=1.0):
    a = np.asarray(values, dtype=float)
    return bool(np.isfinite(a).all() and ((a >= low) & (a <= high)).all())


def close(actual, expected, *, rtol=1e-9, atol=1e-10):
    return bool(np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False))


def predictor_inventory(features):
    return [dict(predictor=c, status='SAFE' if c in CANDIDATES and c not in REMOVED else 'UNSAFE',
        reason='Retained native statistical/protocol predictor; provenance excluded.' if c in CANDIDATES and c not in REMOVED
        else 'Outside frozen native predictor allowlist or a TRAIN-constant removed feature.') for c in features]


def audit_splits(audit, frames, features):
    for split in SPLITS:
        frame = frames[split]
        audit.check('A', f'{split}_rows', len(frame) == ROWS[split], f'actual={len(frame)} expected={ROWS[split]}')
        audit.check('A', f'{split}_nonempty', not frame.empty)
        audit.check('A', f'{split}_unique_columns', frame.columns.is_unique)
        actual = [c for c in frame.columns if c not in METADATA]
        audit.check('A', f'{split}_predictor_schema', actual == features and len(actual) == PREDICTOR_COUNT
                    and set(frame.columns) == set(features + METADATA), f'predictors={actual}')
        audit.check('A', f'{split}_no_missing_values', not frame.isna().any().any())
        numeric = all(is_numeric_dtype(frame[c]) and not is_complex_dtype(frame[c]) for c in features)
        audit.check('A', f'{split}_numeric_predictors', numeric)
        if numeric:
            a = frame[features].to_numpy(dtype=float)
            audit.check('A', f'{split}_no_positive_infinity', not np.isposinf(a).any())
            audit.check('A', f'{split}_no_negative_infinity', not np.isneginf(a).any())
        constants = frame[features].nunique(dropna=False).loc[lambda x: x <= 1].index.tolist()
        if split == 'train':
            audit.check('A', 'train_final_predictors_nonconstant', not constants,
                        f'constant_predictors={constants}; final feature removal must use TRAIN only.')
        else:
            audit.check('A', f'{split}_constant_predictors_observed', not constants,
                        f'constant_predictors={constants}; split-local constancy is not leakage; '
                        'retain the frozen TRAIN-selected predictors.', warning=True)
        audit.check('A', f'{split}_binary_target', set(frame.target_binary) == {0, 1},
                    f'distribution={frame.target_binary.value_counts(dropna=False).to_dict()}')
        audit.check('A', f'{split}_target_encoding', np.array_equal(
            frame.binary_label.map({'benign': 0, 'malware': 1}), frame.target_binary))
        audit.check('A', f'{split}_complete_provenance', not frame[PROVENANCE].isna().any().any()
                    and not frame[PROVENANCE].eq('').any().any())
    audit.check('A', 'total_rows', sum(len(f) for f in frames.values()) == sum(ROWS.values()))
    inventory = predictor_inventory(features)
    audit.tables['ustc_final_predictor_inventory.csv'] = inventory
    audit.check('E', 'native_predictor_allowlist', len(inventory) == PREDICTOR_COUNT and
                all(r['status'] == 'SAFE' for r in inventory), 'Names checked independently of extraction/training imports.')
    # A perfect two-value label proxy is suspicious, but can also be genuine separability.
    proxies = [c for c in features if all(frames[s].groupby('target_binary')[c].nunique().eq(1).all()
               and frames[s][c].nunique() == 2 for s in SPLITS)]
    audit.check('E', 'two_value_target_proxy_investigation', not proxies,
                f'Potential direct target proxies={proxies}; manual investigation required if present.', warning=True)


def audit_provenance(audit, frames):
    for split, frame in frames.items():
        families = frame.groupby(['semantic_family', 'binary_label'], dropna=False).size().reset_index(name='rows')
        audit.tables['ustc_split_family_assignment.csv'].extend(dict(r, split=split) for r in families.to_dict('records'))
        actual = {label: set(frame.loc[frame.binary_label == label, 'semantic_family']) for label in ('benign', 'malware')}
        audit.check('B', f'{split}_frozen_family_assignment', actual == FAMILIES[split], str(actual))
        audit.check('B', f'{split}_family_class_identity', frame.groupby('semantic_family').binary_label.nunique().eq(1).all())
        cols = ['capture_group', 'source_pcap', 'semantic_family', 'binary_label']
        groups = frame.groupby(cols, dropna=False).size().reset_index(name='rows')
        audit.tables['ustc_capture_group_split_assignment.csv'].extend(dict(r, split=split) for r in groups.to_dict('records'))
        for key in ('capture_group', 'source_pcap'):
            others = [c for c in cols if c != key]
            audit.check('C', f'{split}_{key}_identity', frame.groupby(key)[others].nunique().eq(1).all().all(),
                        f'count={frame[key].nunique()}; names={sorted(frame[key].unique())}')
    for left, right in combinations(SPLITS, 2):
        for key, category in [('semantic_family', 'B'), ('capture_group', 'C'), ('source_pcap', 'C')]:
            shared = set(frames[left][key]) & set(frames[right][key])
            audit.check(category, f'{left}_{right}_{key}_overlap', not shared, f'count={len(shared)}; names={sorted(shared)}')


def duplicate_summary(left, right, columns, *, examples=3):
    """Grouped native equality avoids a Cartesian explosion for repeated rows."""
    a = left.groupby(columns, dropna=False, sort=False).size().rename('left_count').reset_index()
    b = right.groupby(columns, dropna=False, sort=False).size().rename('right_count').reset_index()
    shared = a.merge(b, on=columns, how='inner', validate='one_to_one')
    sample = []
    for _, row in shared.head(examples).iterrows():
        evidence = {c: row[c] for c in columns}
        for label, frame in [('left', left), ('right', right)]:
            mask = frame[columns].eq(row[columns]).all(axis=1)
            evidence[label + '_provenance'] = frame.loc[mask, PROVENANCE].head(2).to_dict('records')
        sample.append(evidence)
    return dict(shared_distinct_rows=len(shared), left_matching_rows=int(shared.left_count.sum()),
                right_matching_rows=int(shared.right_count.sum()),
                matching_row_pairs=sum(int(a) * int(b) for a, b in zip(shared.left_count, shared.right_count)),
                examples=str(sample))


def audit_duplicates(audit, frames, features):
    for label, cols in [('predictors', features), ('predictors_and_target', features + ['target_binary'])]:
        for split, frame in frames.items():
            repeats = int(frame.duplicated(cols, keep='first').sum())
            audit.check('D', f'{split}_{label}_within_split_duplicates', True,
                        f'excess_rows={repeats}; within-split repetition is not cross-split leakage.')
        for left, right in combinations(SPLITS, 2):
            row = duplicate_summary(frames[left], frames[right], cols)
            provenance_overlap = any(set(frames[left][c]) & set(frames[right][c])
                                     for c in ('capture_group', 'source_pcap'))
            count = row['shared_distinct_rows']
            status = 'FAIL' if provenance_overlap else ('WARN' if count else 'PASS')
            classification = dict(numeric_duplicate='YES' if count else 'NO',
                provenance_overlap='YES' if provenance_overlap else 'NO',
                leakage_interpretation='CRITICAL_FAIL' if provenance_overlap else ('WARN_ONLY' if count else 'NO_DUPLICATE'))
            audit.tables['ustc_cross_split_duplicate_check.csv'].append(dict(row, comparison=label,
                left_split=left, right_split=right, status=status, **classification))
            audit.check('D', f'{left}_{right}_{label}_duplicates', not (count or provenance_overlap),
                        f'{row}; {classification}; source/capture overlap is checked across the split pair; '
                        'numeric equality alone does not establish an identical provenance/capture observation.',
                        warning=not provenance_overlap)


def parameter_features(parameters):
    if list(parameters.columns) != PARAMETER_COLUMNS or not parameters.feature_name.is_unique:
        raise ValueError('Invalid preprocessing parameter schema or duplicate features.')
    flags = parameters.retained.astype(str).str.lower()
    if not flags.isin(['true', 'false']).all():
        raise ValueError('Invalid retained flags.')
    kept = parameters.loc[flags.eq('true')].sort_values('final_feature_order')
    if kept.final_feature_order.tolist() != list(range(1, PREDICTOR_COUNT + 1)):
        raise ValueError('Retained feature order/count violates frozen 50-predictor contract.')
    return kept.feature_name.tolist()


def audit_preprocessing(audit, parameters, flows, frames, features, native_columns):
    audit.check('F', 'native_extraction_contract', set(native_columns) == set(CANDIDATES) | EXCLUDED | set(PROVENANCE),
                f'native={len(set(native_columns) - set(PROVENANCE))}; candidates={len(CANDIDATES)}; excluded={len(EXCLUDED)}')
    audit.check('F', 'original_flow_count', len(flows) == sum(ROWS.values()), f'rows={len(flows)}')
    audit.check('F', 'parameter_candidates', set(parameters.feature_name) == set(CANDIDATES))
    flags = parameters.retained.astype(str).str.lower().eq('true')
    removed = set(parameters.loc[~flags, 'feature_name'])
    audit.check('F', 'removed_train_constants', removed == REMOVED, f'actual={sorted(removed)}')
    audit.check('F', 'drop_reasons', parameters.loc[~flags, 'drop_reason'].eq('zero_variance_train').all()
                and parameters.loc[flags, 'drop_reason'].fillna('').eq('').all()
                and parameters.loc[~flags, 'final_feature_order'].isna().all())
    p = parameters.set_index('feature_name')
    assignments = {family: split for split, frame in frames.items() for family in frame.semantic_family.unique()}
    audit.check('F', 'all_original_families_assigned', flows.semantic_family.isin(assignments).all())
    raw_train = flows.loc[flows.semantic_family.isin(frames['train'].semantic_family.unique())]
    audit.check('F', 'original_train_membership', len(raw_train) == len(frames['train']))
    # Independently calculate descriptive statistics, never call fit/fit_transform.
    for column in parameters.feature_name:
        # pandas may expose a read-only view; mutate only our owned writable copy.
        raw = pd.to_numeric(raw_train[column], errors='raise').to_numpy(dtype=float, copy=True)
        raw[~np.isfinite(raw)] = np.nan
        median = float(np.nanmedian(raw))
        imputed = np.where(np.isnan(raw), median, raw)
        mean, variance = float(np.mean(imputed)), float(np.var(imputed, ddof=0))
        keep = np.unique(imputed).size > 1
        # StandardScaler's numerical constant-feature safeguard, reproduced algebraically.
        n, eps = len(imputed), np.finfo(np.float64).eps
        scale = (1.0 if variance <= n * eps * variance + (n * mean * eps) ** 2 else math.sqrt(variance)) if keep else 0.0
        saved = p.loc[column]
        audit.check('F', f'{column}_train_median', close(saved.train_median, median), f'saved={saved.train_median}; TRAIN={median}')
        audit.check('F', f'{column}_train_mean', close(saved.train_mean, mean), f'saved={saved.train_mean}; TRAIN={mean}')
        audit.check('F', f'{column}_train_scale_variance', close(saved.train_std, scale),
                    f'saved_scale={saved.train_std}; TRAIN_scale={scale}; TRAIN_population_variance={variance}')
        audit.check('F', f'{column}_train_only_retention', (str(saved.retained).lower() == 'true') == keep,
                    f'TRAIN_unique_values={np.unique(imputed).size}')
    for split, processed in frames.items():
        source = flows.loc[flows.semantic_family.isin(processed.semantic_family.unique())].reset_index(drop=True)
        processed = processed.reset_index(drop=True)
        aligned = len(source) == len(processed) and source[PROVENANCE].equals(processed[PROVENANCE])
        audit.check('F', f'{split}_original_flow_alignment', aligned,
                    'Check original extraction row order and complete provenance; no reassignment or split changes.')
        if not aligned:
            continue
        mismatches = []
        for column in features:
            raw = pd.to_numeric(source[column], errors='raise').to_numpy(dtype=float)
            saved = p.loc[column]
            clean = np.where(np.isfinite(raw), raw, saved.train_median)
            expected = (clean - saved.train_mean) / saved.train_std
            if not close(processed[column], expected):
                mismatches.append(column)
        audit.check('F', f'{split}_persisted_transform_matches', not mismatches, f'mismatched_columns={mismatches}')


def syntax(source):
    return ast.dump(ast.parse(source), include_attributes=False)


def expression(node):
    return ast.unparse(node)


def source_contracts(preprocessing_source, training_source):
    """Conservative structural checks, not comment searches or code execution.

    These contracts intentionally reject changed learning/data-access operations
    for manual review. They are not a general Python program verifier.
    """
    pre, train = ast.parse(preprocessing_source), ast.parse(training_source)
    pf = {n.name: n for n in pre.body if isinstance(n, ast.FunctionDef)}
    tf = {n.name: n for n in train.body if isinstance(n, ast.FunctionDef)}

    def assignments(function, target):
        return [expression(n.value) for n in ast.walk(function) if isinstance(n, ast.Assign)
                and any(expression(t) == target for t in n.targets)]

    def assigned(function, target, expected):
        return [syntax(x) for x in assignments(function, target)] == [syntax(x) for x in expected]

    def calls(function, name):
        return [expression(n) for n in ast.walk(function) if isinstance(n, ast.Call)
                and expression(n.func) == name]

    def exact_calls(function, name, expected):
        return sorted(syntax(x) for x in calls(function, name)) == sorted(syntax(x) for x in expected)

    fit = pf['fit_transform']
    pre_ok = all(assigned(fit, target, [value]) for target, value in {
        'original': 'frame[predictors]', 'clean': 'original.replace([np.inf, -np.inf], np.nan)',
        'train': "clean.iloc[parts['train']]", 'medians': 'train.median()',
        'train_imputed': 'train.fillna(medians)', 'retained': 'train_imputed.nunique(dropna=False).gt(1)',
        'final': '[c for c in predictors if retained[c]]',
        'scaler': 'StandardScaler().fit(train_imputed[final])',
    }.items())
    fits = [expression(n) for n in ast.walk(pre) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr in ('fit', 'fit_transform', 'partial_fit', 'fit_resample')]
    pre_ok &= fits == ['StandardScaler().fit(train_imputed[final])']
    pre_ok &= assigned(pf['preprocess'], 'assignment', ['assign_families(frame)'])
    pre_ok &= assigned(pf['preprocess'], 'parts', ['split_rows(frame, assignment)'])
    pre_ok &= exact_calls(pf['preprocess'], 'fit_transform', ['fit_transform(frame, predictors, parts)'])
    # Subscript/attribute mutations of training statistics are outside this reviewed contract.
    protected = {'train', 'medians', 'train_imputed', 'retained', 'final', 'scaler', 'parts', 'predictors'}
    for n in ast.walk(fit):
        if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, (ast.Subscript, ast.Attribute)) and
                   any(isinstance(x, ast.Name) and x.id in protected for x in ast.walk(t)) for t in targets):
                pre_ok = False
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            # Statistics must not be updated after their approved TRAIN assignments.
            root = n.func.value
            while isinstance(root, (ast.Subscript, ast.Attribute)):
                root = root.value
            if isinstance(root, ast.Name) and root.id in protected and n.func.attr in {
                    'update', 'append', 'extend', 'insert', 'clear', 'pop', '__setitem__'}:
                pre_ok = False

    forbidden = {'smote', 'pca', 'resample', 'fit_resample', 'gridsearchcv', 'randomizedsearchcv',
                 'selectkbest', 'rfe', 'rfecv', 'calibratedclassifiercv', 'isotonicregression', 'train_test_split'}
    used = {n.id.lower() for tree in (pre, train) for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used |= {n.attr.lower() for tree in (pre, train) for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    no_extra_learning = not (used & forbidden)
    load = tf['load_data']
    no_early_test = any(isinstance(n, ast.For) and expression(n.target) == 'split'
        and syntax(expression(n.iter)) == syntax("('train', 'validation')") for n in load.body)
    no_early_test &= not any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                            and 'test' in n.value.lower() for n in ast.walk(load))
    test_reads = [n for n in ast.walk(train) if isinstance(n, ast.Constant) and n.value == 'test.csv']
    no_early_test &= len(test_reads) == 1 and test_reads[0] in list(ast.walk(tf['load_test']))
    predictor_inputs = assigned(load, 'features', ['kept.feature_name.tolist()'])
    data_constructors = [n for n in ast.walk(load) if isinstance(n, ast.Call) and expression(n.func) == 'TrainingData']
    expected_inputs = ["features", "frames['train'][features]", "frames['train'].target_binary.to_numpy(dtype=int)",
                       "frames['validation'][features]", "frames['validation'].target_binary.to_numpy(dtype=int)"]
    predictor_inputs &= len(data_constructors) == 1 and len(data_constructors[0].args) == 6 and all(
        syntax(expression(actual)) == syntax(expected) for actual, expected in zip(data_constructors[0].args, expected_inputs))
    predictor_inputs &= assigned(tf['gpu_features'], 'values', ['frame.to_numpy(dtype=np.float32, copy=True)'])
    tm = tf['train_model']
    allowed_fits = [
        'model.fit(gpu_features(cp, data.x_train), cp.asarray(data.y_train, dtype=cp.int32))',
        'model.fit(x_train, y_train, eval_set=eval_set, verbose=False)',
        "model.fit(data.x_train, data.y_train, eval_set=eval_set, eval_names=['train', 'validation'], "
        "eval_metric='binary_logloss', callbacks=[backend.early_stopping(30, first_metric_only=True, verbose=False), "
        'backend.record_evaluation(losses)])',
    ]
    fitting = exact_calls(tm, 'model.fit', allowed_fits)
    fitting &= assigned(tm, 'x_train', ['gpu_features(cp, data.x_train)'])
    fitting &= assigned(tm, 'y_train', ['cp.asarray(data.y_train, dtype=cp.int32)'])
    fitting &= assigned(tm, 'eval_set', [
        '[(data.x_train, data.y_train), (data.x_validation, data.y_validation)]',
        '[(x_train, y_train), (gpu_features(cp, data.x_validation), cp.asarray(data.y_validation, dtype=cp.int32))]'])
    neural = tf['train_neural']
    fitting &= assigned(neural, 'train_loader', ['loader(data.x_train, data.y_train, True)'])
    fitting &= assigned(neural, 'validation_loader', ['loader(data.x_validation, data.y_validation, False)'])
    # Exact gradient control flow: validation cannot update optimizer/gradients.
    epoch = next(n for n in neural.body if isinstance(n, ast.FunctionDef) and n.name == 'epoch_pass')
    gradient_contract = '''
def epoch_pass(training):
    model.train(training)
    total_loss, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for x, y in train_loader if training else validation_loader:
            x, y = x.to('cuda', non_blocking=True), y.to('cuda', non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x).squeeze(-1), y)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item()) * len(y)
            count += len(y)
    return total_loss / count
'''
    gradients = syntax(expression(epoch)) == syntax(gradient_contract)
    gradient_calls = [n for n in ast.walk(train) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute) and n.func.attr in ('backward', 'step', 'zero_grad')]
    gradients &= len(gradient_calls) == 3 and all(n in list(ast.walk(epoch)) for n in gradient_calls)
    epochs = tf['fit_neural_epochs']
    checkpoints = assigned(epochs, 'validation_loss', ['float(epoch_pass(False))'])
    checkpoints &= exact_calls(epochs, 'model.load_state_dict', ['model.load_state_dict(best_state)'])
    checkpoints &= any(isinstance(n, ast.If) and expression(n.test) == 'validation_loss < best_loss'
                       and assigned(n, '(best_loss, best_epoch)', ['(validation_loss, epoch)'])
                       and assigned(n, 'best_state', ['copy.deepcopy(model.state_dict())']) for n in ast.walk(epochs))
    checkpoints &= not any(isinstance(n, ast.Break) for n in ast.walk(epochs))
    checkpoints &= exact_calls(neural, 'fit_neural_epochs', ['fit_neural_epochs(model, epoch_pass)'])
    checkpoints &= assigned(tm, 'best_iteration', ['int(model.best_iteration) + 1', 'int(model.best_iteration_)'])
    checkpoints &= assigned(tm, 'predict', [
        'lambda frame: xgboost_probability(cp, model, frame, best_iteration)',
        'lambda frame: positive_probability(model, frame, num_iteration=best_iteration)'])

    # The entire finalization/TEST control flow must match, including the all-model barrier.
    run = tf['run']
    barrier = next(n for n in run.body if isinstance(n, ast.Try))
    boundary_contract = '''
for name in selected:
    print(f'Training {name} on {DEVICES[name]}')
    try:
        predict, history, save = train_model(name, data)
        record(name, 'validation', data.y_validation, predict(data.x_validation))
        publish_model(model_dir / ARTIFACTS[name], save)
        tables['training_history.csv'].extend(dict(row, dataset=DATASET, model=name,
            run_id=run_id, timestamp_utc=timestamp) for row in history)
        finalized[name] = predict
    except Exception as exc:
        raise RuntimeError(f'{name} failed on required backend {DEVICES[name]}: {exc}') from exc
x_test, y_test = load_test(root, data)
for name, predict in finalized.items():
    try:
        record(name, 'test', y_test, predict(x_test))
    except Exception as exc:
        raise RuntimeError(f'{name} TEST evaluation failed; no retry or retraining: {exc}') from exc
'''
    boundary = syntax('\n'.join(expression(n) for n in barrier.body)) == syntax(boundary_contract)
    boundary &= exact_calls(run, 'load_test', ['load_test(root, data)'])
    boundary &= exact_calls(run, 'train_model', ['train_model(name, data)'])
    boundary &= assigned(run, 'data', ['load_data(root)']) and assigned(run, 'finalized', ['{}'])
    # Reject disguised additional TEST accesses in helpers or data mutation in learning functions.
    for func in (tm, neural, epochs):
        for n in ast.walk(func):
            if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                if any(isinstance(t, (ast.Attribute, ast.Subscript)) and any(
                    isinstance(v, ast.Name) and v.id in {'data', 'eval_set', 'x_train', 'y_train', 'train_loader', 'validation_loader'}
                    for v in ast.walk(t)) for t in targets):
                    fitting = False
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                root = n.func.value
                while isinstance(root, (ast.Subscript, ast.Attribute)):
                    root = root.value
                if isinstance(root, ast.Name) and root.id in {'data', 'eval_set', 'x_train', 'y_train'} and n.func.attr in {
                        'update', 'append', 'extend', 'insert', 'clear', 'pop', '__setitem__'}:
                    fitting = False
            if isinstance(n, ast.Name) and 'test' in n.id.lower():
                fitting = False
            if isinstance(n, ast.Attribute) and 'test' in n.attr.lower():
                fitting = False
            if isinstance(n, ast.Call) and any(token in expression(n.func).lower()
                    for token in ('read_csv', 'load_test', 'load_data', 'open', 'exec', 'eval')):
                if expression(n.func) not in ('model.evals_result', 'model.eval', 'backend.record_evaluation', 'path.open'):
                    fitting = False
    for name, func in tf.items():
        if name not in ('train_model', 'gpu_smoke') and any(isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr in ('fit', 'fit_transform', 'partial_fit', 'fit_resample')
                for n in ast.walk(func)):
            no_extra_learning = False
    constants = {expression(t): n.value for n in train.body if isinstance(n, ast.Assign) for t in n.targets}
    fixed = all(isinstance(constants.get(k), ast.Constant) and constants[k].value == v
                for k, v in {'SEED': 42, 'THRESHOLD': 0.5, 'FEATURE_COUNT': 50, 'MAX_EPOCHS': 30, 'BATCH_SIZE': 1024}.items())
    threshold = assigned(tf['evaluate_predictions'], 'predicted', ['(p >= THRESHOLD).astype(int)'])
    return dict(preprocessing_train_only=bool(pre_ok), no_extra_learning=bool(no_extra_learning),
                initial_load_excludes_test=bool(no_early_test), fitting_inputs=bool(fitting),
                gradient_isolation=bool(gradients), validation_checkpoint_selection=bool(checkpoints),
                finalized_before_test=bool(boundary), fixed_seed_threshold_contract=bool(fixed and threshold),
                predictor_input_contract=bool(predictor_inputs))


def audit_sources(audit, preprocessing_source, training_source):
    evidence = source_contracts(preprocessing_source, training_source)
    audit.check('E', 'training_predictor_input_contract', evidence['predictor_input_contract'],
                'Only persisted retained feature names select TRAIN/validation predictors; targets/provenance remain separate.')
    for name in ('median_fit_train_only', 'variance_filter_train_only', 'scaler_fit_train_only',
                 'validation_not_used_for_fit', 'test_not_used_for_fit', 'feature_removal_train_only'):
        audit.check('F', name, evidence['preprocessing_train_only'], 'Executable AST operations plus independent persisted-statistic checks.')
    for name in ('no_pca', 'no_smote', 'no_balancing_resampling', 'no_global_feature_selection', 'no_target_informed_selection'):
        audit.check('F', name, evidence['no_extra_learning'] and evidence['preprocessing_train_only'])
    checks = {
        'test_not_used_for_training': evidence['fitting_inputs'] and evidence['initial_load_excludes_test'],
        'test_not_used_for_gradients': evidence['gradient_isolation'] and evidence['fitting_inputs'],
        'test_not_used_for_preprocessing': evidence['preprocessing_train_only'],
        'test_not_used_for_early_stopping': evidence['fitting_inputs'],
        'test_not_used_for_checkpoint_selection': evidence['validation_checkpoint_selection'] and evidence['fitting_inputs'],
        'test_not_used_for_feature_selection': evidence['preprocessing_train_only'] and evidence['no_extra_learning'],
        'test_not_used_for_threshold_tuning': evidence['fixed_seed_threshold_contract'] and evidence['finalized_before_test'],
        'test_not_used_for_hyperparameter_tuning': evidence['no_extra_learning'] and evidence['finalized_before_test'],
        'test_not_used_for_model_selection': evidence['finalized_before_test'],
        'test_not_used_for_calibration_fit': evidence['no_extra_learning'] and evidence['finalized_before_test'],
        'all_models_finalized_before_single_test_pass': evidence['finalized_before_test'] and evidence['initial_load_excludes_test'],
    }
    for name, ok in checks.items():
        audit.check('G', name, ok, 'Current executable source contract; persisted history checked separately. Unknown source changes require review.')
    audit.check('G', 'historical_execution_evidence_limit', False, LIMITATION, warning=True)


def confusion_metrics(counts):
    tn, fp, fn, tp = map(int, counts)
    n = tn + fp + fn + tp
    if min(tn, fp, fn, tp) < 0 or n == 0:
        raise ValueError('Invalid confusion counts.')
    def div(a, b):
        return a / b if b else 0.0
    f1 = div(2 * tp, 2 * tp + fp + fn)
    benign_f1 = div(2 * tn, 2 * tn + fp + fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(accuracy=(tp + tn) / n, precision=div(tp, tp + fp), recall=div(tp, tp + fn), f1=f1,
                weighted_f1=((tn + fp) * benign_f1 + (tp + fn) * f1) / n,
                mcc=div(tp * tn - fp * fn, denominator))


def audit_histories(audit, history, model_dir):
    audit.check('H', 'history_models', set(history.model) == set(HISTORY_REFERENCES),
                'Random Forest has no iterative history; no TEST history columns are permitted.')
    for model, filename in ARTIFACTS.items():
        artifact = model_dir / filename
        exists = artifact.is_file() and artifact.stat().st_size > 0
        audit.check('H', f'{model}_artifact_exists_nonempty', exists, f'{filename}; metadata only; not deserialized.')
        subset = history.loc[history.model == model]
        row = dict(model=model, recorded_steps=len(subset), final_step='', best_validation_step='',
                   best_validation_loss='', artifact=filename, size_bytes=artifact.stat().st_size if exists else 0)
        ok = exists
        if model == 'random_forest':
            ok &= subset.empty
        elif not subset.empty:
            steps = subset.step.to_numpy(dtype=float)
            losses = subset[['train_loss', 'validation_loss']].to_numpy(dtype=float)
            expected_count, expected_best, reference = HISTORY_REFERENCES[model]
            valid = finite_range(losses, 0, float('inf')) and np.array_equal(steps, np.arange(1, len(subset) + 1))
            audit.check('H', f'{model}_finite_contiguous_history', valid)
            best = subset.iloc[int(np.argmin(losses[:, 1]))]
            row.update(final_step=steps[-1], best_validation_step=best.step, best_validation_loss=best.validation_loss)
            length_ok = len(subset) == expected_count and steps[-1] == expected_count
            reference_ok = best.step == expected_best and close(best.validation_loss, reference, rtol=0, atol=5e-9)
            audit.check('H', f'{model}_recorded_steps', length_ok, f'actual={len(subset)} expected={expected_count}')
            audit.check('H', f'{model}_best_validation_reference', reference_ok,
                        f'CSV best_step={best.step}; CSV best_loss={best.validation_loss}; approximate reference={reference}')
            if model in ('xgboost', 'lightgbm'):
                stop_ok = len(subset) == 300 or len(subset) - int(best.step) == 30
                audit.check('H', f'{model}_early_stopping_consistency', stop_ok)
                ok &= stop_ok
            ok &= valid and length_ok and reference_ok
        else:
            ok = False
        audit.check('H', f'{model}_history_and_artifact', ok)
        audit.tables['ustc_model_history_consistency.csv'].append(dict(row, status='PASS' if ok else 'FAIL',
            details='CSV-derived history; references approximate. Artifact contents and restored weights are not inspected.'))


def audit_curves(audit, model, split, metric, counts, roc, pr, calibration):
    label = f'{model}_{split}'
    tn, fp, fn, tp = counts
    negatives, positives = tn + fp, fn + tp
    total = negatives + positives
    r = roc[['fpr', 'tpr', 'threshold']].to_numpy(dtype=float)
    p = pr[['precision', 'recall', 'threshold']].to_numpy(dtype=float)
    roc_ok = (len(r) >= 2 and finite_range(r[:, :2]) and np.isposinf(r[0, 2])
        and finite_range(r[1:, 2]) and (np.diff(r[:, 0]) >= 0).all() and (np.diff(r[:, 1]) >= 0).all()
        and (np.diff(r[:, 2]) < 0).all() and close(r[0, :2], [0, 0]) and close(r[-1, :2], [1, 1]))
    audit.check('K', label + '_roc_structure', roc_ok, 'Only the first ROC threshold may be +Inf (no-predicted-positive sentinel).')
    pr_ok = (len(p) >= 2 and finite_range(p[:, :2]) and np.isnan(p[-1, 2]) and finite_range(p[:-1, 2])
        and (np.diff(p[:-1, 2]) > 0).all() and (np.diff(p[:, 1]) <= 0).all()
        and close(p[0, 1], 1) and close(p[-1, :2], [1, 0]))
    audit.check('K', label + '_pr_structure', pr_ok, 'Only the terminal PR threshold may be blank.')
    if roc_ok:
        area = float(np.sum(np.diff(r[:, 0]) * (r[1:, 1] + r[:-1, 1]) / 2))
        audit.check('K', label + '_roc_auc_recomputed', close(metric.roc_auc, area), f'curve={area}; metrics={metric.roc_auc}')
        operating = r[np.flatnonzero(r[:, 2] >= THRESHOLD)[-1]]
        audit.check('K', label + '_roc_threshold_confusion', close(operating[:2], [fp / negatives, tp / positives]))
    if pr_ok:
        ap = float(-np.sum(np.diff(p[:, 1]) * p[:-1, 0]))
        audit.check('K', label + '_auprc_recomputed', close(metric.auprc, ap), f'average_precision={ap}; metrics={metric.auprc}')
    bins = calibration.sort_values('bin_index')
    edges = np.linspace(0, 1, 16)
    valid_bins = (len(bins) == 15 and np.array_equal(bins.bin_index, np.arange(15))
        and close(bins.lower_bound, edges[:-1]) and close(bins.upper_bound, edges[1:])
        and finite_range(bins['count'], 0, total) and np.equal(bins['count'], np.floor(bins['count'])).all()
        and bins['count'].sum() == total)
    audit.check('K', label + '_calibration_structure', valid_bins)
    occupied = bins.loc[bins['count'] > 0]
    empty = bins.loc[bins['count'] == 0]
    stats = ['mean_probability', 'observed_positive_fraction', 'absolute_gap']
    valid_stats = finite_range(occupied[stats]) and empty[stats].isna().all().all()
    if valid_stats:
        valid_stats &= close(occupied.absolute_gap, abs(occupied.mean_probability - occupied.observed_positive_fraction))
        valid_stats &= ((occupied.mean_probability >= occupied.lower_bound - 1e-12)
                        & (occupied.mean_probability <= occupied.upper_bound + 1e-12)).all()
    audit.check('K', label + '_calibration_statistics', valid_stats, 'Empty bins require blank statistics; occupied bins must be finite.')
    if valid_bins and valid_stats:
        ece = float((occupied['count'] * occupied.absolute_gap).sum() / total)
        audit.check('K', label + '_ece_recomputed', close(metric.ece, ece), f'bins={ece}; metrics={metric.ece}')
        audit.check('K', label + '_calibration_class_support', close(
            (occupied['count'] * occupied.observed_positive_fraction).sum(), positives))
    if not (roc_ok and pr_ok):
        return
    threshold_match = len(r) == len(p) and close(r[1:, 2][::-1], p[:-1, 2])
    audit.check('K', label + '_complete_roc_pr_thresholds', threshold_match)
    # Complete ROC points encode counts at each distinct score. These are existing
    # aggregate evidence, not new predictions; they also permit exact Brier checks.
    cumulative = r[:, :2] * np.array([negatives, positives])
    integer_counts = close(cumulative, np.rint(cumulative), rtol=0, atol=1e-6)
    audit.check('K', label + '_roc_integer_observation_counts', integer_counts)
    if not integer_counts:
        return
    increments = np.diff(np.rint(cumulative), axis=0)
    audit.check('K', label + '_distinct_score_counts', (increments.sum(axis=1) > 0).all())
    probabilities = r[1:, 2]
    brier = float(np.sum(increments[:, 0] * probabilities ** 2 + increments[:, 1] * (1 - probabilities) ** 2) / total)
    audit.check('K', label + '_brier_recomputed', close(metric.brier_score, brier), f'ROC-derived={brier}; metrics={metric.brier_score}')
    if threshold_match:
        positive = cumulative[1:, 1]
        precision = positive / cumulative[1:].sum(axis=1)
        audit.check('K', label + '_roc_pr_counts_consistent', close(precision[::-1], p[:-1, 0])
                    and close(r[1:, 1][::-1], p[:-1, 1]))
    if valid_bins and valid_stats:
        consistent = True
        for i, row in enumerate(bins.itertuples(index=False)):
            mask = (probabilities >= edges[i]) & ((probabilities <= edges[i + 1]) if i == 14 else (probabilities < edges[i + 1]))
            count = increments[mask].sum()
            consistent &= row.count == count
            if count:
                mean = np.sum(probabilities[mask] * increments[mask].sum(axis=1)) / count
                observed = increments[mask, 1].sum() / count
                consistent &= close([row.mean_probability, row.observed_positive_fraction], [mean, observed])
        audit.check('K', label + '_calibration_matches_score_counts', consistent)


def audit_results(audit, results, frames, run_id=RUN):
    expected_pairs = {(m, s) for m in ARTIFACTS for s in ('validation', 'test')}
    stamps = set()
    for name, frame in results.items():
        audit.check('I', name + '_schema', list(frame.columns) == RESULT_SCHEMAS[name])
        audit.check('I', name + '_nonempty', not frame.empty)
        audit.check('I', name + '_identity', frame[['dataset', 'run_id']].notna().all().all()
                    and frame.dataset.eq(DATASET).all() and frame.run_id.eq(run_id).all())
        parsed = pd.to_datetime(frame.timestamp_utc, errors='coerce', utc=True)
        audit.check('I', name + '_timestamps', parsed.notna().all() and parsed.dt.strftime('%Y%m%dT%H%M%SZ').eq(run_id).all())
        stamps.update(frame.timestamp_utc.dropna().unique())
        if name != 'training_history.csv':
            audit.check('I', name + '_model_split_coverage', set(zip(frame.model, frame.split)) == expected_pairs)
    audit.check('I', 'one_run_timestamp', len(stamps) == 1)
    metrics = results['metrics.csv']
    audit.check('I', 'exactly_ten_unique_metric_rows', len(metrics) == 10 and not metrics.duplicated(['model', 'split']).any()
                and set(zip(metrics.model, metrics.split)) == expected_pairs)
    audit.check('I', 'fixed_seed_threshold_gpu_devices', metrics.seed.eq(SEED).all() and metrics.threshold.eq(THRESHOLD).all()
                and metrics.device.eq(metrics.model.map(DEVICES)).all())
    for name in METRICS:
        audit.check('I', name + '_finite_range', finite_range(metrics[name], -1 if name == 'mcc' else 0, 1))
    for model, split in sorted(expected_pairs):
        def pair():
            label = model + '_' + split
            group = metrics.loc[(metrics.model == model) & (metrics.split == split)]
            if len(group) != 1:
                raise ValueError(f'{label}: expected exactly one metric row.')
            metric = group.iloc[0]
            cm = results['confusion_matrix.csv']
            cm = cm.loc[(cm.model == model) & (cm.split == split)]
            pairs = [('benign', 'benign'), ('benign', 'malware'), ('malware', 'benign'), ('malware', 'malware')]
            valid = len(cm) == 4 and set(zip(cm.actual_class, cm.predicted_class)) == set(pairs)
            valid &= finite_range(cm['count'], 0, ROWS[split]) and np.equal(cm['count'], np.floor(cm['count'])).all()
            audit.check('J', label + '_four_confusion_cells', valid)
            if not valid:
                return
            indexed = cm.set_index(['actual_class', 'predicted_class'])['count']
            counts = [int(indexed.loc[p]) for p in pairs]
            tn, fp, fn, tp = counts
            audit.check('J', label + '_confusion_total', sum(counts) == ROWS[split] == len(frames[split]),
                        f'confusion={sum(counts)} expected={ROWS[split]} actual_split={len(frames[split])}')
            y = frames[split].target_binary
            audit.check('J', label + '_class_support', tn + fp == int(y.eq(0).sum()) and fn + tp == int(y.eq(1).sum()))
            for name, value in confusion_metrics(counts).items():
                audit.check('J', label + '_' + name, close(metric[name], value), f'confusion={value}; metrics={metric[name]}')
            tables = []
            for name in ('roc_curve.csv', 'pr_curve.csv', 'calibration_bins.csv'):
                frame = results[name]
                tables.append(frame.loc[(frame.model == model) & (frame.split == split)])
            audit_curves(audit, model, split, metric, counts, *tables)
        audit.section('K', pair)
    audit.check('G', 'persisted_history_excludes_test', list(results['training_history.csv'].columns) == RESULT_SCHEMAS['training_history.csv'],
                'History contains TRAIN/validation losses only; schema cannot attest undocumented historical accesses.')


def complete_report(audit, frames, features, run_id=RUN):
    checks = audit.tables['ustc_integrity_audit_checklist.csv']
    # Warnings do not veto finalization, but a missing/failed critical section does.
    critical_pass = all(any(r['category'] == cat and r['status'] == 'PASS' for r in checks)
                        and not any(r['category'] == cat and r['status'] == 'FAIL' for r in checks)
                        for cat in 'ABCDEFGHIJK')
    conclusion = 'NO DIRECT LEAKAGE EVIDENCE FOUND' if critical_pass else 'LEAKAGE CHECKS FAILED OR INCOMPLETE'
    independent_duplicates = any(r['numeric_duplicate'] == 'YES' and r['provenance_overlap'] == 'NO'
                                for r in audit.tables['ustc_cross_split_duplicate_check.csv'])
    duplicate_caution = ('Numeric duplicates exist across independently sourced captures; numeric equality alone '
                         'does not establish leakage.' if independent_duplicates else
                         'Identical numeric vectors alone do not establish leakage; consult bounded provenance examples.')
    audit.check('L', 'near_perfect_tree_investigation', critical_pass,
                conclusion + '; near-perfect performance does not prove absence of leakage. '
                + duplicate_caution + ' High performance may reflect strong separability of this dataset/family split; '
                'external-dataset validation is required.')
    status = 'PASS' if audit.passed else 'FAIL'
    audit.tables['ustc_integrity_audit_metadata.csv'] = [dict(dataset=DATASET, task=TASK, authoritative_run=run_id,
        audit_scope=LIMITATION, predictor_count=len(features), train_rows=len(frames.get('train', [])),
        validation_rows=len(frames.get('validation', [])), test_rows=len(frames.get('test', [])),
        seed=SEED, threshold=THRESHOLD, overall_status=status, finalization_authorized='YES' if audit.passed else 'NO')]
    lines = [f'{DATASET} integrity audit of {run_id}', f'overall_status={status}',
             f'finalization_authorized={"YES" if audit.passed else "NO"} (within the documented evidence scope)', LIMITATION]
    categories = {'A': 'Dataset sanity', 'B': 'Family leakage', 'C': 'Capture leakage', 'D': 'Cross-split duplicates',
                  'E': 'Predictor leakage', 'F': 'Preprocessing leakage', 'G': 'TEST isolation',
                  'H': 'Model/history consistency', 'I': 'Result sanity', 'J': 'Confusion consistency', 'K': 'Curve/calibration consistency'}
    for category, label in categories.items():
        rows = [r for r in audit.tables['ustc_integrity_audit_checklist.csv'] if r['category'] == category]
        state = 'FAIL' if not rows or any(r['status'] == 'FAIL' for r in rows) else ('WARN' if any(r['status'] == 'WARN' for r in rows) else 'PASS')
        lines.append(f'{label}: {state}')
    lines += [PROTOCOL_EVIDENCE if critical_pass else
              'Current source/persisted artifact checks failed or are incomplete; historical runtime execution '
              'cannot be independently attested from these files alone.', conclusion, duplicate_caution,
              'Near-perfect performance is not proof of no leakage. Strong separability of this dataset/family split '
              'requires external-dataset validation.', 'No training, inference, preprocessing fitting, split changes, or TEST-driven selection performed.']
    for row in audit.tables['ustc_integrity_audit_checklist.csv']:
        if row['status'] != 'PASS':
            lines.append(f"{row['status']} {row['category']}/{row['check_name']}: {row['details'][:1200]}")
    return '\n'.join(lines) + '\n'


def check_destination(directory, identity):
    directory = Path(directory)
    # Reject redirects of the canonical audit directory or individual owned files.
    if directory.resolve() != directory.absolute():
        raise ValueError('Audit output directory must not redirect through a symlink/junction.')
    for name in OUTPUT_NAMES:
        target = directory / name
        if target.is_symlink() or (target.exists() and (not target.is_file() or target.stat().st_nlink > 1)):
            raise ValueError(f'Unsafe audit output target: {target}')
    metadata = directory / 'ustc_integrity_audit_metadata.csv'
    if metadata.exists():
        frame = read_csv(metadata)
        if len(frame) != 1 or any(k not in frame or str(frame.iloc[0][k]) != str(v) for k, v in identity.items()):
            raise FileExistsError('Existing audit identity differs: dataset/task/authoritative_run must all match.')
    elif any((directory / name).exists() for name in OUTPUT_NAMES):
        raise FileExistsError('Audit-owned files exist without identity metadata; refusing unverified overwrite.')


def publish_report(directory, audit, findings):
    """Overwrite only a matching audit identity; stage locally, metadata commits last.

    Files are atomically replaced individually, not as a filesystem transaction.
    A FAIL metadata marker prevents a mid-publication interruption retaining PASS.
    This command is intended for a single writer.
    """
    directory = Path(directory)
    metadata_name = 'ustc_integrity_audit_metadata.csv'
    metadata = audit.tables[metadata_name][0]
    identity = {k: metadata[k] for k in ('dataset', 'task', 'authoritative_run')}
    check_destination(directory, identity)
    directory.mkdir(parents=True, exist_ok=True)
    staged = {}
    def stage(name, rows=None):
        descriptor, path = tempfile.mkstemp(prefix='.' + name + '.', suffix='.tmp', dir=directory)
        staged[name] = Path(path)
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as stream:
            if name == FINDINGS:
                stream.write(findings)
            else:
                columns = OUTPUT_SCHEMAS[name]
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
    marker = None
    try:
        for name in OUTPUT_SCHEMAS:
            stage(name, audit.tables[name])
        stage(FINDINGS)
        check_destination(directory, identity)
        completed_metadata = staged.pop(metadata_name)
        try:
            stage(metadata_name, [dict(metadata, overall_status='FAIL', finalization_authorized='NO',
                                      audit_scope='Publication in progress or interrupted; rerun the same audit.')])
            marker = staged.pop(metadata_name)
            os.replace(marker, directory / metadata_name)
            for name, path in staged.items():
                os.replace(path, directory / name)
            os.replace(completed_metadata, directory / metadata_name)
        finally:
            completed_metadata.unlink(missing_ok=True)
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)
        if marker is not None:
            marker.unlink(missing_ok=True)


def run_audit(root):
    root = Path(root).resolve()
    output = root / 'results/audits' / DATASET / TASK
    check_destination(output, dict(dataset=DATASET, task=TASK, authoritative_run=RUN))
    audit, frames, features, results = Audit(), {}, [], {}
    parameters = flows = None
    base = root / 'data/processed' / DATASET
    try:
        parameters = read_csv(base / 'preprocessing_parameters.csv')
        features = parameter_features(parameters)
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        audit.check('F', 'parameters_readable_contract', False, str(exc))
    for split in SPLITS:
        try:
            frames[split] = read_csv(base / f'{split}.csv')
            audit.check('A', f'{split}_readable_unique_header', True)
        except (OSError, ValueError, StopIteration) as exc:
            audit.check('A', f'{split}_readable_unique_header', False, str(exc))
    if len(frames) == 3 and features:
        audit.section('A', lambda: audit_splits(audit, frames, features))
        audit.section('B', lambda: audit_provenance(audit, frames))
        audit.section('D', lambda: audit_duplicates(audit, frames, features))
        try:
            flow_path = root / 'data/interim' / DATASET / 'flows.csv'
            with flow_path.open(encoding='utf-8-sig', newline='') as stream:
                native = next(csv.reader(stream))
            flows = read_csv(flow_path, usecols=CANDIDATES + PROVENANCE)
            audit.section('F', lambda: audit_preprocessing(audit, parameters, flows, frames, features, native))
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            audit.check('F', 'original_flows_readable', False, str(exc))
    else:
        for category in ('A', 'B', 'C', 'D', 'E', 'F'):
            audit.check(category, 'required_dataset_evidence_available', False)
    try:
        audit_sources(audit, (root / 'src/data/ustc_preprocess.py').read_text(encoding='utf-8'),
                      (root / 'src/models/ustc_train_models.py').read_text(encoding='utf-8'))
    except (OSError, SyntaxError, ValueError, KeyError, StopIteration) as exc:
        audit.check('F', 'preprocessing_source_contract_readable', False, str(exc))
        audit.check('G', 'training_source_contract_readable', False, str(exc))
    for name, schema in RESULT_SCHEMAS.items():
        try:
            frame = read_csv(root / 'results/models' / DATASET / RUN / name)
            if list(frame.columns) != schema:
                raise ValueError(f'Invalid schema: {name}; actual={list(frame.columns)}')
            results[name] = frame
            audit.check('I', name + '_readable_exact_header', True)
        except (OSError, ValueError, StopIteration) as exc:
            audit.check('I', name + '_readable_exact_header', False, str(exc))
    if len(results) == 6 and len(frames) == 3:
        audit.section('I', lambda: audit_results(audit, results, frames))
    else:
        audit.check('I', 'required_result_evidence_available', False)
    history = results.get('training_history.csv', pd.DataFrame(columns=RESULT_SCHEMAS['training_history.csv']))
    audit.section('H', lambda: audit_histories(audit, history, root / 'models' / DATASET / RUN))
    findings = complete_report(audit, frames, features)
    publish_report(output, audit, findings)
    print(findings)
    print(f'audit_directory={output}')
    return 0 if audit.passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Read integrity evidence and write only canonical audit outputs.')
    args = parser.parse_args(argv)
    if not args.execute:
        print(f'AUDIT_NOT_EXECUTED=True\nauthoritative_run={RUN}\nUse --execute to run the read-only integrity checks.')
        return 0
    try:
        return run_audit(args.project_root)
    except (OSError, ValueError) as exc:
        print(f'AUDIT_ERROR={exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
