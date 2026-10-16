"""Audit, finalize and close one CIC run using saved evidence only.

No training/backend modules are imported and model files are only stat'ed.
Processed CSVs are streamed for headers, targets and Flow ID counts only.
Preprocessing checks validate the frozen TRAIN-only policy/parameter contract;
they rely on the completed CIC-2 leakage verification, rather than claiming to
reconstruct fitting provenance without raw TRAIN evidence. Saved probabilities
are unavailable: log loss/Brier are range-checked, not independently reproduced.
The requested probability-guard note relies on the completed run's evaluation
contract. Authoritative metric/loss strings are copied without normalization.

ROC areas use trapezoids; PR average precision uses sum((r[i]-r[i+1])*p[i]),
including the terminal (precision=1, recall=0) point. No trapezoidal PR area.
Confusion comparisons use rtol=1e-9, atol=1e-10; curve/calibration comparisons
use rtol=1e-8, atol=1e-9. These are saved-evidence consistency checks, not a
repeat evaluation of a model or reconstruction of row-level predictions.

Dry invocation writes nothing. --execute publishes the four final CSVs plus
the checklist after all checks pass. A failed execute writes only its checklist.
--close --execute requires existing finals/checklist to match a fresh successful
audit, then writes closure and checklist only. Same-run derived outputs may be
replaced; other-run identities are rejected. Ordinary publication errors roll
back replacements; the file set is not crash-atomic. Use a single publisher.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

DATASET = 'CIC-Darknet2020'
RUN = '20260920T145831Z'
TASK = 'privacy_network_4class'
MODELS = ('random_forest', '1d_cnn', 'xgboost', 'lightgbm', 'cnn_lstm')
SPLITS = ('validation', 'test')
LABELS = ('Non-Tor', 'NonVPN', 'Tor', 'VPN')
PROBABILITY_COLUMNS = ('class_0_non_tor', 'class_1_nonvpn', 'class_2_tor', 'class_3_vpn')
ROWS = dict(train=95170, validation=31723, test=31723)
GROUP_COUNT = 79160
CONFLICT_COUNT = 10
ARTIFACTS = dict(zip(MODELS, ('random_forest.pkl', '1d_cnn.pt', 'xgboost.json', 'lightgbm.txt', 'cnn_lstm.pt')))
DEVICES = dict(zip(MODELS, ('cuml_cuda', 'cuda', 'cuda', 'lightgbm_cuda', 'cuda')))
PROVENANCE = ['Label', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Src Port', 'Dst Port', 'Label.1']
METRICS = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1', 'mcc',
           'macro_ovr_roc_auc', 'macro_ovr_auprc', 'multiclass_log_loss', 'multiclass_brier_score', 'ece']
POLICIES = dict(port_policy='EXCLUDED_PROVENANCE_ONLY', nonfinite_policy='COERCE_NUMERIC_REPLACE_INF_WITH_MISSING',
    imputation_policy='TRAIN_MEDIAN_ONLY', constant_feature_policy='REMOVE_TRAIN_ZERO_VARIANCE_ONLY',
    scaling_policy='TRAIN_STANDARD_SCALER_ONLY', resampling_policy='NONE', pca_policy='NONE', feature_selection_policy='NONE')
MODEL_DIR = f'models/{DATASET}/{RUN}'
RESULT_DIR = f'results/models/{DATASET}/{RUN}'
FINAL_DIR = f'results/final/{DATASET}'
AUDIT_DIR = f'results/audits/{DATASET}/integrity_audit'
CHECKLIST = f'{AUDIT_DIR}/cic_integrity_audit_checklist.csv'
CLOSURE = 'metadata/traffic/cic_experiment_closure.csv'
IDENTITY_NOTE = f'dataset={DATASET};authoritative_run={RUN}'
PROBABILITY_NOTE = ('Run-time evaluation enforced probability bounds and row-sum agreement '
    'within absolute tolerance 1e-6; sklearn warnings are therefore treated as '
    'minor numerical-normalization warnings, not evidence of malformed probabilities.')
PREFIX = ['dataset', 'run_id', 'model', 'split']
INPUT_HEADERS = {
    'metrics.csv': PREFIX + ['device'] + METRICS + ['row_count'],
    'confusion_matrix.csv': PREFIX + ['actual_class_id', 'actual_class_label', 'predicted_class_id', 'predicted_class_label', 'count'],
    'training_history.csv': ['dataset', 'run_id', 'model', 'step', 'train_loss', 'validation_loss', 'is_best_validation_step'],
    'roc_curve.csv': PREFIX + ['class_id', 'class_label', 'fpr', 'tpr', 'threshold'],
    'pr_curve.csv': PREFIX + ['class_id', 'class_label', 'precision', 'recall', 'threshold'],
    'calibration_bins.csv': PREFIX + ['bin_index', 'lower_bound', 'upper_bound', 'count', 'mean_confidence', 'accuracy', 'absolute_gap'],
    'run_configuration.csv': ['dataset', 'run_id', 'model', 'key', 'value'],
}
TRAIN_COLUMNS = ['dataset', 'authoritative_run', 'task', 'model', 'device', 'step', 'train_loss',
    'validation_loss', 'is_best_validation_step', 'best_validation_step', 'best_validation_loss', 'predictor_count', 'class_count', 'seed']
EVALUATION_COLUMNS = ['dataset', 'authoritative_run', 'task', 'model', 'device'] + METRICS + [
    'row_count', 'predictor_count', 'class_count', 'seed', 'integrity_audit_status']
PER_CLASS_COLUMNS = ['dataset', 'authoritative_run', 'model', 'split', 'class_id', 'class_label',
                     'support', 'precision', 'recall', 'f1', 'roc_auc_ovr', 'average_precision_ovr']
FINAL_HEADERS = {
    'cic_train_results.csv': TRAIN_COLUMNS,
    'cic_validation_results.csv': EVALUATION_COLUMNS,
    'cic_test_results.csv': EVALUATION_COLUMNS,
    'cic_per_class_results.csv': PER_CLASS_COLUMNS,
}
CHECKLIST_COLUMNS = ['section', 'check', 'status', 'details']
CLOSURE_COLUMNS = ['dataset', 'authoritative_run', 'stage_status', 'integrity_audit_status', 'finalization_status',
    'train_rows', 'validation_rows', 'test_rows', 'initial_predictor_count', 'removed_train_zero_variance',
    'predictor_count', 'class_count', 'model_count', 'seed', 'train_results_path', 'validation_results_path',
    'test_results_path', 'per_class_results_path', 'model_artifact_directory', 'authoritative_result_directory',
    'integrity_audit_directory', 'frozen_for_retraining', 'frozen_for_test_reuse', 'phase_i_status', 'next_phase']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name, *, low=0, high=math.inf, integer=False):
    parsed = float(value)
    require(math.isfinite(parsed) and low <= parsed <= high, f'{name}: nonfinite/out-of-range value {value!r}.')
    require(not integer or parsed.is_integer(), f'{name}: expected an integer.')
    return int(parsed) if integer else parsed


def close_number(actual, expected, name, *, curve=False):
    require(math.isclose(actual, expected, rel_tol=1e-8 if curve else 1e-9,
                        abs_tol=1e-9 if curve else 1e-10), f'{name}: saved={actual}, derived={expected}.')


def csv_rows(path, *, schema=None, required=()):
    """Stream text unchanged, including blank terminal PR thresholds."""
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream, strict=True)
        header = reader.fieldnames
        require(header and len(header) == len(set(header)), f'{path}: missing/duplicate header.')
        require(schema is None or header == schema, f'{path}: unexpected CSV schema.')
        require(set(required).issubset(header), f'{path}: required columns missing.')
        for row in reader:
            require(None not in row and None not in row.values(), f'{path}: malformed CSV row.')
            yield row


def read_rows(path, **kwargs):
    rows = list(csv_rows(path, **kwargs))
    require(rows, f'{path}: empty CSV.')
    return rows


def evidence(root, filename):
    for row in csv_rows(root / RESULT_DIR / filename, schema=INPUT_HEADERS[filename]):
        require(row['dataset'] == DATASET and row['run_id'] == RUN, f'{filename}: authoritative dataset/run identity mismatch.')
        yield row


def pair_key(row):
    key = row['model'], row['split']
    require(key[0] in MODELS and key[1] in SPLITS, 'Unexpected model/split identity.')
    return key


def class_key(row):
    c = number(row['class_id'], 'class_id', high=3, integer=True)
    require(row['class_label'] == LABELS[c], 'Class label/order mismatch.')
    return (*pair_key(row), c)


@dataclass
class Audit:
    checklist: list = field(default_factory=list)
    finals: dict = field(default_factory=dict)
    passed: bool = False

    def check(self, section, name, operation, details):
        try:
            result = operation()
        except (ValueError, OSError, KeyError, TypeError, OverflowError, csv.Error) as exc:
            self.checklist.append(dict(section=section, check=name, status='FAIL', details=str(exc)))
            raise
        self.checklist.append(dict(section=section, check=name, status='PASS', details=details))
        return result

    def note(self, section, name, details):
        self.checklist.append(dict(section=section, check=name, status='PASS', details=details))


def frozen_contract(root):
    base = root / 'metadata/traffic'
    main = read_rows(base / 'cic_phase1_contract.csv')
    expected = dict(dataset=DATASET, task='five_model_4class_benchmark', raw_source='data/raw/CIC-Darknet2020/Darknet.CSV',
        raw_rows=str(sum(ROWS.values())), raw_columns='85', target_column='Label', auxiliary_label_column='Label.1',
        class_count='4', split_group='Flow ID', split_seed='42', initial_predictor_candidates='77',
        contract_status='FROZEN_FOR_MATERIALIZATION', **POLICIES)
    require(len(main) == 1 and all(main[0].get(k) == v for k, v in expected.items()), 'Frozen CIC contract differs.')
    for split, count in ROWS.items():
        require(number(main[0][f'{split}_rows'], 'split rows', integer=True) == count, 'Frozen row counts differ.')
    mapping = read_rows(base / 'cic_label_mapping.csv', schema=['class_id', 'raw_label', 'canonical_label', 'probability_column'])
    require(mapping == [dict(class_id=str(i), raw_label=label, canonical_label=label, probability_column=PROBABILITY_COLUMNS[i])
                        for i, label in enumerate(LABELS)], 'Frozen label mapping differs.')
    features = read_rows(base / 'cic_feature_contract.csv', required=['source_index', 'column_name', 'predictor_candidate', 'model_allowed'])
    require(len(features) == 85 and [r['source_index'] for r in features] == list(map(str, range(85)))
            and len({r['column_name'] for r in features}) == 85, 'Invalid source feature ordering.')
    require(all(r['predictor_candidate'] in ('YES', 'NO') and r['model_allowed'] == r['predictor_candidate'] for r in features),
            'Invalid model permission flags.')
    candidates = [r['column_name'] for r in features if r['predictor_candidate'] == 'YES']
    require(len(candidates) == 77 and {r['column_name'] for r in features if r['model_allowed'] == 'NO'} == set(PROVENANCE),
            'Identifier/auxiliary-label exclusion differs.')
    parameters = read_rows(base / 'cic_preprocessing_parameters.csv', schema=['source_order', 'feature_name', 'train_median',
        'train_zero_variance', 'retained_final', 'final_predictor_order', 'train_scaler_mean', 'train_scaler_scale'])
    require([r['feature_name'] for r in parameters] == candidates
            and [r['source_order'] for r in parameters] == list(map(str, range(77))), 'Preprocessing candidate order differs.')
    retained = []
    for row in parameters:
        number(row['train_median'], 'TRAIN median', low=-math.inf)
        require(row['retained_final'] in ('YES', 'NO'), 'Invalid retained flag.')
        keep = row['retained_final'] == 'YES'
        require(row['train_zero_variance'] == ('NO' if keep else 'YES'), 'TRAIN removal policy differs.')
        if keep:
            require(row['final_predictor_order'] == str(len(retained)), 'Final predictor ordering differs.')
            number(row['train_scaler_mean'], 'TRAIN mean', low=-math.inf)
            require(number(row['train_scaler_scale'], 'TRAIN scale') > 0, 'Nonpositive scaler scale.')
            retained.append(row['feature_name'])
        else:
            require(all(row[k] == '' for k in ['final_predictor_order', 'train_scaler_mean', 'train_scaler_scale']),
                    'Removed features must have blank scaler/order fields.')
    require(len(retained) == 62 and not set(retained) & set(PROVENANCE), 'Expected 62 allowed retained predictors.')
    manifest = read_rows(base / 'cic_split_manifest.csv', required=['dataset', 'flow_id', 'split', 'group_row_count',
        'non_tor_rows', 'nonvpn_rows', 'tor_rows', 'vpn_rows', 'has_label_conflict', 'seed', 'split_method'])
    require(len(manifest) == GROUP_COUNT and len({r['flow_id'] for r in manifest}) == GROUP_COUNT, 'Flow ID group coverage differs.')
    groups, totals, conflicts = {}, {s: [0] * 4 for s in ROWS}, 0
    for row in manifest:
        split = row['split'].lower()
        require(split in ROWS and row['split'] == split.upper() and row['flow_id'] != ''
                and row['dataset'] == DATASET and row['seed'] == '42' and row['split_method'] == main[0]['split_method'],
                'Manifest identity/grouping differs.')
        counts = [number(row[c], c, integer=True) for c in ('non_tor_rows', 'nonvpn_rows', 'tor_rows', 'vpn_rows')]
        require(sum(counts) > 0 and number(row['group_row_count'], 'group total', integer=True) == sum(counts), 'Group counts differ.')
        mixed = sum(n > 0 for n in counts) > 1
        require(row['has_label_conflict'] == ('YES' if mixed else 'NO'), 'Mixed-label group flag differs.')
        conflicts += mixed
        groups[row['flow_id']] = (split, tuple(counts))
        totals[split] = [a + b for a, b in zip(totals[split], counts)]
    require(conflicts == CONFLICT_COUNT, 'Mixed-label Flow ID group count differs.')
    for split, total in totals.items():
        require(sum(total) == ROWS[split] and all(n > 0 for n in total), 'Split/class totals differ.')
    return dict(features=retained, groups=groups, totals=totals)


def processed_structure(root, contract):
    """Stream only structural metadata; predictor values are never evaluated."""
    for split, expected_count in ROWS.items():
        count, observed = 0, defaultdict(lambda: [0] * 4)
        for row in csv_rows(root / f'data/processed/{DATASET}/{split}.csv', schema=contract['features'] + ['target_id'] + PROVENANCE):
            c = number(row['target_id'], 'target_id', high=3, integer=True)
            require(row['Label'] == LABELS[c] and row['Flow ID'] in contract['groups'], 'Processed target/provenance identity differs.')
            require(contract['groups'][row['Flow ID']][0] == split, 'Flow ID cross-split overlap or wrong assignment.')
            observed[row['Flow ID']][c] += 1
            count += 1
        expected = {g: counts for g, (s, counts) in contract['groups'].items() if s == split}
        require(count == expected_count and {g: tuple(v) for g, v in observed.items()} == expected,
                f'{split}: processed row/group/class coverage differs.')


def configuration(root, contract):
    rows = list(evidence(root, 'run_configuration.csv'))
    lookup = defaultdict(list)
    for i, row in enumerate(rows):
        require(row['model'] in (*MODELS, 'all') and row['key'], 'Unknown configuration model/key.')
        lookup[row['model'], row['key']].append((i, row['value']))
    require(all(len(values) == 1 or key == ('all', 'run_status') for key, values in lookup.items()), 'Duplicate configuration key.')
    def get(model, key):
        require(len(lookup[model, key]) == 1, f'Missing/duplicate configuration {model}/{key}.')
        return lookup[model, key][0][1]
    expected = dict(seed='42', class_count='4', predictor_count='62', preprocessing='ALREADY_FROZEN',
        test_selection='FORBIDDEN', test_evaluations_per_model='1', class_weights='NONE', resampling='NONE',
        additional_preprocessing='NONE', ALL_MODELS_FINALIZED='True')
    for split, count in ROWS.items():
        expected[f'{split}_rows'] = str(count)
    for key, value in expected.items():
        require(get('all', key) == value, f'Configuration {key} differs.')
    for key, value in [('class_mapping', dict(zip(map(str, range(4)), LABELS))),
                       ('probability_order', list(PROBABILITY_COLUMNS)), ('model_roster', list(MODELS)),
                       ('feature_names', contract['features'])]:
        require(json.loads(get('all', key)) == value, f'Configuration {key} differs.')
    states = lookup['all', 'run_status']
    require([value for _, value in states] == ['STARTED', 'COMPLETE'], 'Run must complete exactly once without failure.')
    gate_position = lookup['all', 'ALL_MODELS_FINALIZED'][0][0]
    require(states[0][0] < gate_position < states[-1][0], 'Invalid finalization-gate event order.')
    for model in MODELS:
        require(get(model, 'test_inference_attempt') == '1', f'{model}: expected one TEST attempt.')
        require(gate_position < lookup[model, 'test_inference_attempt'][0][0] < states[-1][0], 'TEST attempt precedes finalization gate.')
    for model in ('1d_cnn', 'cnn_lstm'):
        for key, value in dict(epochs='30', batch_size='1024', early_termination='False', loss='CrossEntropyLoss', output_logits='4').items():
            require(get(model, key) == value, f'{model}: neural configuration {key} differs.')
    for model in ('xgboost', 'lightgbm'):
        require(get(model, 'n_estimators') == '300' and get(model, 'num_class') == '4'
                and get(model, 'random_state') == '42', f'{model}: multiclass iteration/seed contract differs.')
    require(get('xgboost', 'objective') == 'multi:softprob' and get('xgboost', 'device') == 'cuda'
            and get('xgboost', 'early_stopping_rounds') == '30', 'XGBoost GPU/early-stopping contract differs.')
    require(get('lightgbm', 'objective') == 'multiclass' and get('lightgbm', 'device_type') == 'cuda'
            and get('all', 'lightgbm_early_stopping_rounds') == '30', 'LightGBM GPU/early-stopping contract differs.')
    require(get('random_forest', 'output_type') == 'cupy' and get('random_forest', 'random_state') == '42', 'RF CUDA/seed contract differs.')
    return dict(predictor_count=get('all', 'predictor_count'), class_count=get('all', 'class_count'), seed=get('all', 'seed'))


def artifacts(root):
    for name in ARTIFACTS.values():
        path = root / MODEL_DIR / name
        require(path.is_file() and path.stat().st_size > 0, f'Missing/empty model artifact: {name}.')


def metrics(root):
    rows = list(evidence(root, 'metrics.csv'))
    keys = [pair_key(row) for row in rows]
    require(len(rows) == 10 and len(set(keys)) == 10 and set(keys) == {(m, s) for m in MODELS for s in SPLITS}, 'Expected ten unique model/split metric rows.')
    for row in rows:
        require(row['device'] == DEVICES[row['model']], 'Metric device differs.')
        require(number(row['row_count'], 'metric row count', integer=True) == ROWS[row['split']], 'Metric row count differs.')
        for name in METRICS:
            number(row[name], name, low=-1 if name == 'mcc' else 0,
                   high=math.inf if name in ('multiclass_log_loss', 'multiclass_brier_score') else 1)
    return dict(zip(keys, rows))


def confusion(root, metric_rows, contract):
    cells = {}
    for row in evidence(root, 'confusion_matrix.csv'):
        a = number(row['actual_class_id'], 'actual class', high=3, integer=True)
        b = number(row['predicted_class_id'], 'predicted class', high=3, integer=True)
        key = (*pair_key(row), a, b)
        require(key not in cells and row['actual_class_label'] == LABELS[a] and row['predicted_class_label'] == LABELS[b],
                'Duplicate/mislabeled confusion cell.')
        cells[key] = number(row['count'], 'confusion count', integer=True)
    require(set(cells) == {(m, s, a, b) for m in MODELS for s in SPLITS for a in range(4) for b in range(4)}, 'Expected 16 confusion cells per model/split.')
    per_class = {}
    for (model, split), saved in metric_rows.items():
        matrix = [[cells[model, split, a, b] for b in range(4)] for a in range(4)]
        support = [sum(row) for row in matrix]
        predicted = [sum(matrix[a][b] for a in range(4)) for b in range(4)]
        n, correct = sum(support), sum(matrix[c][c] for c in range(4))
        require(n == ROWS[split] and support == contract['totals'][split], 'Confusion totals/class supports differ.')
        precision = [matrix[c][c] / predicted[c] if predicted[c] else 0.0 for c in range(4)]
        recall = [matrix[c][c] / support[c] if support[c] else 0.0 for c in range(4)]
        f1 = [2 * p * r / (p + r) if p + r else 0.0 for p, r in zip(precision, recall)]
        denominator = math.sqrt((n*n - sum(v*v for v in predicted)) * (n*n - sum(v*v for v in support)))
        derived = dict(accuracy=correct/n, macro_precision=sum(precision)/4, macro_recall=sum(recall)/4,
            macro_f1=sum(f1)/4, weighted_f1=sum(f*s for f, s in zip(f1, support))/n,
            mcc=(n*correct - sum(a*b for a, b in zip(support, predicted)))/denominator if denominator else 0.0)
        for name, value in derived.items():
            close_number(float(saved[name]), value, f'{model}/{split}/{name}')
        for c in range(4):
            per_class[model, split, c] = dict(support=support[c], precision=precision[c], recall=recall[c], f1=f1[c])
    return per_class


def curves(root, kind, metric_rows, per_class):
    """Stream full curves with O(number of model/split/classes) memory."""
    states = {}
    filename = f'{kind}_curve.csv'
    for row in evidence(root, filename):
        key = class_key(row)
        state = states.setdefault(key, dict(count=0, area=0.0, last=None, terminal=False))
        require(not state['terminal'], f'{key}: point after terminal PR endpoint.')
        if kind == 'roc':
            x, y = number(row['fpr'], 'FPR', high=1), number(row['tpr'], 'TPR', high=1)
            threshold = float(row['threshold'])
            if not state['count']:
                require((x, y) == (0.0, 0.0) and threshold == math.inf, 'ROC must start at (0,0), threshold=+inf.')
            else:
                number(threshold, 'ROC threshold', high=1)
                px, py, pt = state['last']
                require(x >= px and y >= py and threshold < pt, 'ROC ordering/thresholds invalid.')
                state['area'] += (x - px) * (y + py) / 2
        else:
            # x=recall descends, y=precision; AP uses the LEFT precision at
            # each recall drop, not the average of endpoint precisions.
            x, y = number(row['recall'], 'recall', high=1), number(row['precision'], 'precision', high=1)
            terminal = row['threshold'] == ''
            threshold = None if terminal else number(row['threshold'], 'PR threshold', high=1)
            if not state['count']:
                require(x == 1.0 and not terminal, 'PR must start at recall=1 with a finite threshold.')
                close_number(y, per_class[key]['support'] / ROWS[key[1]], 'PR starting prevalence', curve=True)
            else:
                px, py, pt = state['last']
                require(x <= px and (terminal or threshold > pt), 'PR recall/threshold ordering invalid.')
                state['area'] += (px - x) * py
            if terminal:
                require((x, y) == (0.0, 1.0), 'Invalid terminal PR endpoint.')
                state['terminal'] = True
        state['last'] = (x, y, threshold)
        state['count'] += 1
    expected = {(m, s, c) for m in MODELS for s in SPLITS for c in range(4)}
    require(set(states) == expected, f'Missing/extra {kind.upper()} class curves.')
    for key, state in states.items():
        require(state['count'] >= 2, f'{key}: insufficient curve points.')
        if kind == 'roc':
            require(state['last'][:2] == (1.0, 1.0), f'{key}: ROC must end at (1,1).')
        else:
            require(state['terminal'], f'{key}: missing terminal PR endpoint.')
        number(state['area'], f'{key}: reconstructed curve area', high=1 + 1e-10)
    metric = 'macro_ovr_roc_auc' if kind == 'roc' else 'macro_ovr_auprc'
    for model, split in metric_rows:
        macro = math.fsum(states[model, split, c]['area'] for c in range(4)) / 4
        close_number(float(metric_rows[model, split][metric]), macro, f'{model}/{split}/{metric}', curve=True)
    return {key: state['area'] for key, state in states.items()}


def calibration(root, metric_rows):
    groups = defaultdict(dict)
    for row in evidence(root, 'calibration_bins.csv'):
        key = pair_key(row)
        index = number(row['bin_index'], 'bin index', high=14, integer=True)
        require(index not in groups[key], 'Duplicate calibration bin.')
        groups[key][index] = row
    require(set(groups) == set(metric_rows), 'Calibration model/split coverage differs.')
    for key, bins in groups.items():
        require(set(bins) == set(range(15)), 'Expected all 15 calibration bins.')
        total, weighted_gap, weighted_accuracy = 0, 0.0, 0.0
        for i in range(15):
            row = bins[i]
            lo, hi = number(row['lower_bound'], 'bin lower', high=1), number(row['upper_bound'], 'bin upper', high=1)
            close_number(lo, i/15, 'bin lower')
            close_number(hi, (i+1)/15, 'bin upper')
            count = number(row['count'], 'bin count', integer=True)
            total += count
            if count:
                confidence = number(row['mean_confidence'], 'bin confidence', high=1)
                accuracy = number(row['accuracy'], 'bin accuracy', high=1)
                gap = number(row['absolute_gap'], 'bin gap', high=1)
                require(lo - 1e-12 <= confidence <= hi + 1e-12, 'Mean confidence outside its bin.')
                close_number(gap, abs(confidence-accuracy), 'bin absolute gap', curve=True)
                weighted_gap += count * gap
                weighted_accuracy += count * accuracy
            else:
                require(all(row[k] == '' for k in ('mean_confidence', 'accuracy', 'absolute_gap')), 'Empty bin statistics must be blank.')
        require(total == ROWS[key[1]], 'Calibration count totals differ.')
        close_number(weighted_gap / total, float(metric_rows[key]['ece']), f'{key}: ECE', curve=True)
        close_number(weighted_accuracy / total, float(metric_rows[key]['accuracy']), f'{key}: calibration accuracy', curve=True)


def training_history(root):
    groups = defaultdict(list)
    for row in evidence(root, 'training_history.csv'):
        require(row['model'] in MODELS[1:], 'RF or unknown model has fabricated training history.')
        groups[row['model']].append(row)
    require(set(groups) == set(MODELS[1:]), 'Missing iterative training model.')
    best = {}
    for model, rows in groups.items():
        count = len(rows)
        require(count == 30 if model in ('1d_cnn', 'cnn_lstm') else 1 <= count <= 300, f'{model}: invalid history length.')
        require([number(r['step'], 'history step', low=1, integer=True) for r in rows] == list(range(1, count+1)), 'History steps must be contiguous from 1.')
        losses = [number(r['validation_loss'], 'validation loss') for r in rows]
        for row in rows:
            number(row['train_loss'], 'train loss')
        flags = [r['is_best_validation_step'] for r in rows]
        require(all(flag in ('YES', 'NO') for flag in flags), 'Invalid best-step flag.')
        # Explicit boolean equality count; never sum the strings themselves.
        require(sum(flag == 'YES' for flag in flags) == 1, 'Expected exactly one best validation step.')
        index = min(range(count), key=losses.__getitem__)
        require(flags[index] == 'YES', f'{model}: best step must be earliest minimum validation loss.')
        best[model] = rows[index]
    return groups, best


def derive_finals(metric_rows, history, config, per_class, roc, pr):
    groups, best = history
    common = dict(dataset=DATASET, authoritative_run=RUN, task=TASK, **config)
    finals = {name: [] for name in FINAL_HEADERS}
    for model in MODELS:
        for row in groups.get(model, []):
            finals['cic_train_results.csv'].append(dict(common, model=model, device=DEVICES[model],
                **{key: row[key] for key in ('step', 'train_loss', 'validation_loss', 'is_best_validation_step')},
                best_validation_step=best[model]['step'], best_validation_loss=best[model]['validation_loss']))
        for split in SPLITS:
            row = metric_rows[model, split]
            finals[f'cic_{split}_results.csv'].append(dict(common, model=model, device=row['device'],
                **{key: row[key] for key in METRICS + ['row_count']}, integrity_audit_status='PASS'))
            for c, label in enumerate(LABELS):
                key = model, split, c
                finals['cic_per_class_results.csv'].append(dict(dataset=DATASET, authoritative_run=RUN,
                    model=model, split=split, class_id=c, class_label=label, **per_class[key],
                    roc_auc_ovr=roc[key], average_precision_ovr=pr[key]))
    return finals


def audit(root, run_id=RUN):
    report = Audit()
    try:
        report.check('identity', 'authoritative_run', lambda: require(run_id == RUN, 'Only the authoritative candidate run is permitted.'), IDENTITY_NOTE)
        contract = report.check('contracts', 'frozen_contract', lambda: frozen_contract(root), 'Frozen CIC-only policies and all five metadata inputs agree.')
        report.note('contracts', 'label_mapping', '0 Non-Tor;1 NonVPN;2 Tor;3 VPN;fixed probability order.')
        report.note('contracts', '62_predictors', '77 candidates;15 TRAIN-zero-variance removals;62 retained;all provenance and Label.1 excluded.')
        report.note('contracts', 'flow_id_grouping_contract', f'{GROUP_COUNT} unique groups;{CONFLICT_COUNT} mixed-label groups remain whole.')
        report.note('contracts', 'preprocessing_leakage_status', 'PASS for frozen TRAIN-only policy/parameter provenance; relies on completed CIC-2 leakage verification. No raw-data refitting or independent fitting-history claim.')
        report.check('structure', 'split_row_counts', lambda: processed_structure(root, contract), 'Processed metadata matches every frozen group/class count;zero Flow ID cross-split overlap.')
        config = report.check('configuration', 'test_isolation_configuration', lambda: configuration(root, contract), 'Completed run;finalization gate precedes exactly one recorded TEST attempt per model.')
        report.note('configuration', 'no_model_selection_from_test', 'test_selection=FORBIDDEN;fixed model order;no ranking or model choice.')
        report.check('artifacts', 'five_model_artifacts', lambda: artifacts(root), 'All five artifacts exist and are non-empty;stat only, no model contents read.')
        metric_rows = report.check('metrics', 'metric_coverage', lambda: metrics(root), 'Exactly 10 unique model/split rows with expected devices and counts.')
        report.note('metrics', 'metric_finiteness_ranges', 'All eleven metrics finite and within required ranges;log loss/Brier preserved, not recomputed without probabilities.')
        per_class = report.check('confusion', 'confusion_coverage_totals', lambda: confusion(root, metric_rows, contract), '16 unique cells per model/split;integer counts;totals and class supports agree.')
        report.note('confusion', 'accuracy_consistency', 'Accuracy, macro precision/recall/F1, weighted F1 and MCC independently agree with confusion evidence.')
        roc = report.check('curves', 'roc_coverage', lambda: curves(root, 'roc', metric_rows, per_class), 'All 40 ROC curves have valid thresholds, endpoints and monotone FPR/TPR.')
        report.note('curves', 'roc_auc_consistency', 'Macro trapezoidal area of stored full ROC points agrees with saved metrics.')
        pr = report.check('curves', 'pr_coverage', lambda: curves(root, 'pr', metric_rows, per_class), 'All 40 PR curves have valid endpoints, prevalence and threshold/recall ordering.')
        report.note('curves', 'average_precision_consistency', 'Macro non-interpolated AP from full stored PR recall drops agrees;no trapezoidal PR substitution.')
        report.check('calibration', 'calibration_coverage', lambda: calibration(root, metric_rows), '15 bins per model/split;[0,1] bounds;counts and bin statistics valid.')
        report.note('calibration', 'ece_consistency', 'Weighted absolute gaps reproduce saved ECE;weighted bin accuracies agree with saved accuracy.')
        history = report.check('history', 'training_histories', lambda: training_history(root), 'Four iterative models;finite losses;contiguous steps;one explicit YES;earliest validation minimum;no RF history.')
        report.note('history', 'neural_30_epoch_compliance', 'Both neural histories contain epochs 1..30;boosters may validly stop before 300.')
        report.note('probabilities', 'probability_contract_guard', PROBABILITY_NOTE)
        for name in ('no_model_loading', 'no_inference', 'no_retraining', 'no_hashes'):
            report.note('boundaries', name, 'Saved CSV evidence and artifact existence/size only.')
        report.finals = report.check('finalization', 'derive_final_results',
            lambda: derive_finals(metric_rows, history, config, per_class, roc, pr),
            'Final rows derive solely from validated saved evidence;authoritative metric/loss text preserved.')
        report.passed = True
    except (ValueError, OSError, KeyError, TypeError, OverflowError, csv.Error):
        # A failed checked operation already recorded the blocking evidence.
        report.passed = False
    return report


def serialized(rows):
    return [{key: '' if value is None else str(value) for key, value in row.items()} for row in rows]


def closure_row():
    return dict(dataset=DATASET, authoritative_run=RUN, stage_status='CLOSED_FROZEN',
        integrity_audit_status='PASS', finalization_status='PASS', train_rows=ROWS['train'], validation_rows=ROWS['validation'],
        test_rows=ROWS['test'], initial_predictor_count=77, removed_train_zero_variance=15, predictor_count=62,
        class_count=4, model_count=5, seed=42, train_results_path=f'{FINAL_DIR}/cic_train_results.csv',
        validation_results_path=f'{FINAL_DIR}/cic_validation_results.csv', test_results_path=f'{FINAL_DIR}/cic_test_results.csv',
        per_class_results_path=f'{FINAL_DIR}/cic_per_class_results.csv', model_artifact_directory=MODEL_DIR,
        authoritative_result_directory=RESULT_DIR, integrity_audit_directory=AUDIT_DIR,
        frozen_for_retraining='YES', frozen_for_test_reuse='YES', phase_i_status='COMPLETE', next_phase='PROVENANCE_AWARE_RAG')


def check_owner(root, relative):
    path = root / relative
    require(path.resolve().is_relative_to(root.resolve()) and not path.is_symlink(), 'Canonical output escapes project root.')
    if not path.exists():
        return
    require(path.is_file(), f'{path}: not a regular output file.')
    rows = read_rows(path)
    if relative == CHECKLIST:
        identities = [r for r in rows if r.get('section') == 'identity' and r.get('check') == 'authoritative_run']
        require(len(identities) == 1 and identities[0].get('details') == IDENTITY_NOTE, 'Existing audit belongs to another/unknown run.')
    else:
        require(all(r.get('dataset') == DATASET and r.get('authoritative_run') == RUN for r in rows),
                'Existing derived output belongs to another/unknown run; refusing overwrite.')


def publish(root, tables):
    allowed = {f'{FINAL_DIR}/{name}': header for name, header in FINAL_HEADERS.items()}
    allowed.update({CHECKLIST: CHECKLIST_COLUMNS, CLOSURE: CLOSURE_COLUMNS})
    require(set(tables).issubset(allowed), 'Only canonical derived outputs may be written.')
    # Check all canonical identities, even outputs not replaced in this mode.
    for relative in allowed:
        check_owner(root, relative)
    with TemporaryDirectory(prefix='.cic-finalize-', dir=root) as temporary:
        stage = Path(temporary)
        for i, (relative, rows) in enumerate(tables.items()):
            require(rows and all(set(row) == set(allowed[relative]) for row in rows), 'Invalid derived output schema.')
            with (stage / str(i)).open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=allowed[relative])
                writer.writeheader()
                writer.writerows(rows)
            if (root / relative).exists():
                shutil.copyfile(root / relative, stage / f'previous-{i}')
        replaced = []
        try:
            for i, relative in enumerate(tables):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage / str(i), destination)
                replaced.append((i, destination))
        except Exception:
            for i, destination in reversed(replaced):
                backup = stage / f'previous-{i}'
                if backup.exists():
                    os.replace(backup, destination)
                else:
                    destination.unlink()
            raise


def validate_final_outputs(root, report):
    require(report.passed, 'Closure requires a successful fresh audit.')
    for name, expected in report.finals.items():
        actual = read_rows(root / FINAL_DIR / name, schema=FINAL_HEADERS[name])
        require(actual == serialized(expected), f'{name}: saved final results differ from authoritative derived evidence.')
    checklist = read_rows(root / CHECKLIST, schema=CHECKLIST_COLUMNS)
    require(all(row['status'] == 'PASS' for row in checklist), 'Existing checklist contains a non-PASS status.')
    required = {(row['section'], row['check']) for row in report.checklist}
    keys = [(row['section'], row['check']) for row in checklist]
    require(len(keys) == len(set(keys)) and required.issubset(keys), 'Existing checklist is incomplete/duplicated.')
    check_owner(root, CHECKLIST)


def finalize(root, *, run_id=RUN, execute=False, close=False):
    root = Path(root).resolve()
    report = audit(root, run_id)
    if report.passed and close:
        try:
            report.check('closure', 'finalization_gate', lambda: validate_final_outputs(root, report),
                         'All four existing final files exactly match a fresh audit;existing checklist has only PASS.')
        except (ValueError, OSError, KeyError, TypeError, OverflowError, csv.Error):
            report.passed = False
    if execute and run_id == RUN:
        tables = {CHECKLIST: report.checklist}
        if report.passed:
            if close:
                tables[CLOSURE] = [closure_row()]
            else:
                tables.update({f'{FINAL_DIR}/{name}': rows for name, rows in report.finals.items()})
        publish(root, tables)
    if not report.passed:
        print('CIC_FINALIZATION=BLOCKED')
        for row in report.checklist:
            if row['status'] == 'FAIL':
                print(f"{row['section']}/{row['check']}: {row['details']}")
    else:
        print('CIC_CLOSURE=PASS' if close else 'CIC_FINALIZATION=PASS')
        print(f'DRY_RUN={not execute}')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--close', action='store_true')
    args = parser.parse_args(argv)
    try:
        return 0 if finalize(args.project_root, run_id=args.run_id, execute=args.execute, close=args.close).passed else 1
    except (ValueError, OSError, KeyError, TypeError, OverflowError, csv.Error) as exc:
        print(f'CIC_FINALIZATION=BLOCKED\n{exc}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
