"""Finalize frozen USTC history, validation and TEST into three gated CSVs.

Default invocation does not finalize; --execute reads two audit CSVs and three
authoritative result CSVs. Model files are checked for existence only. No dataset,
model contents, curves, or preprocessing inputs are read. Metric strings and the
best recorded validation loss are preserved without rounding. Rows use a fixed
model order, never a performance ranking. The training-history task column
records identity for the three-file set, alongside dataset/run in every file.
TRAIN classification metrics were not saved and are never fabricated here.

--close --execute performs metadata-only closure after verifying existing final
CSVs against the same authoritative evidence. It never invokes finalization or
rewrites experiment outputs. Without --execute, neither operation writes files.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import tempfile

DATASET = 'USTC-TFC2016'
RUN = '20260919T183945Z'
TASK = 'final_results'
TRAIN_FILE = 'ustc_train_results.csv'
EVALUATION_FILES = {'validation': 'ustc_validation_results.csv', 'test': 'ustc_test_results.csv'}
MODELS = ('random_forest', '1d_cnn', 'xgboost', 'lightgbm', 'cnn_lstm')
SPLITS = ('validation', 'test')
SPLIT_ROWS = dict(validation=97001, test=98255)
ARTIFACTS = dict(zip(MODELS, ('random_forest.pkl', '1d_cnn.pt', 'xgboost.json', 'lightgbm.txt', 'cnn_lstm.pt')))
DEVICES = dict(zip(MODELS, ('cuml_cuda', 'cuda', 'cuda', 'lightgbm_cuda', 'cuda')))
HISTORY_EXPECTED = {'1d_cnn': (30, 3), 'cnn_lstm': (30, 19), 'xgboost': (300, 299), 'lightgbm': (246, 216)}
METRICS = 'accuracy precision recall f1 weighted_f1 mcc roc_auc auprc ece brier_score'.split()
CONFUSION_FIELDS = ['true_negative', 'false_positive', 'false_negative', 'true_positive']
CLASS_PAIRS = (('benign', 'benign'), ('benign', 'malware'), ('malware', 'benign'), ('malware', 'malware'))
TRAIN_COLUMNS = ['dataset', 'authoritative_run', 'task', 'model', 'device', 'step', 'train_loss', 'validation_loss',
                 'is_best_validation_step', 'best_validation_step', 'best_validation_loss', 'predictor_count', 'seed', 'threshold']
EVALUATION_COLUMNS = (['dataset', 'authoritative_run', 'model', 'device'] + METRICS
                     + CONFUSION_FIELDS + ['total_rows', 'total_errors', 'predictor_count', 'seed', 'threshold', 'integrity_audit_status'])
OUTPUT_SCHEMAS = {TRAIN_FILE: TRAIN_COLUMNS, **{name: EVALUATION_COLUMNS for name in EVALUATION_FILES.values()}}
PREFIX, TAIL = ['dataset', 'model', 'split'], ['run_id', 'timestamp_utc']
INPUT_SCHEMAS = {
    'metrics.csv': PREFIX + ['device'] + METRICS + ['threshold', 'seed'] + TAIL,
    'confusion_matrix.csv': PREFIX + ['actual_class', 'predicted_class', 'count'] + TAIL,
    'training_history.csv': ['dataset', 'model', 'step', 'train_loss', 'validation_loss'] + TAIL,
}
CLOSURE_PATH = 'metadata/traffic/ustc_experiment_closure.csv'
CLOSURE_COLUMNS = [
    'dataset', 'authoritative_run', 'stage_status', 'integrity_audit_status', 'finalization_status',
    'train_rows', 'validation_rows', 'test_rows', 'predictor_count', 'model_count', 'seed', 'threshold',
    'train_results_path', 'validation_results_path', 'test_results_path', 'model_artifact_directory',
    'authoritative_result_directory', 'integrity_audit_directory',
    'frozen_for_retraining', 'frozen_for_test_reuse', 'next_dataset',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path, *, schema=None, required=()):
    """Keep numeric source text intact while rejecting malformed CSV structure."""
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames
        require(header and len(header) == len(set(header)), f'{path}: missing/duplicate CSV header.')
        require(set(required).issubset(header), f'{path}: missing required columns.')
        if schema is not None:
            require(header == schema, f'{path}: unexpected CSV schema.')
        rows = list(reader)
    require(rows, f'{path}: empty CSV.')
    require(all(None not in row and all(v is not None for v in row.values()) for row in rows),
            f'{path}: malformed CSV row width.')
    return rows


def number(value, field, *, low=0, high=math.inf, integer=False):
    try:
        parsed = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{field}: invalid numeric value {value!r}.') from exc
    require(math.isfinite(parsed) and low <= parsed <= high, f'{field}: nonfinite/out-of-range value {value!r}.')
    require(not integer or parsed.is_integer(), f'{field}: expected an integer, found {value!r}.')
    return int(parsed) if integer else parsed


def validate_gate(root):
    directory = root / 'results/audits' / DATASET / 'integrity_audit'
    expected = dict(dataset=DATASET, task='integrity_audit', authoritative_run=RUN,
                    overall_status='PASS', finalization_authorized='YES')
    rows = read_csv(directory / 'ustc_integrity_audit_metadata.csv', required=expected)
    require(len(rows) == 1 and all(rows[0][k] == v for k, v in expected.items()),
            'Integrity gate rejected: audit identity must match and status/authorization must be PASS/YES.')
    for field, value in [('predictor_count', 50), ('seed', 42), ('threshold', 0.5)]:
        if field in rows[0]:
            require(number(rows[0][field], field) == value, f'Integrity gate: inconsistent {field}.')
    checklist = read_csv(directory / 'ustc_integrity_audit_checklist.csv',
                         schema=['category', 'check_name', 'status', 'details'])
    require(all(row['status'] in ('PASS', 'WARN') for row in checklist),
            'Integrity gate rejected: checklist contains FAIL or an invalid status.')
    require(all(row['category'] and row['check_name'] for row in checklist), 'Integrity gate: incomplete checklist rows.')


def authoritative_rows(root, filename):
    rows = read_csv(root / 'results/models' / DATASET / RUN / filename, schema=INPUT_SCHEMAS[filename])
    require(all(r['dataset'] == DATASET and r['run_id'] == RUN for r in rows),
            f'{filename}: authoritative dataset/run mismatch.')
    return rows


def history_summary(rows):
    require({r['model'] for r in rows} == set(HISTORY_EXPECTED),
            'History must contain exactly the four iterative models, with no Random Forest history.')
    summary = {}
    for model, (count, expected_best) in HISTORY_EXPECTED.items():
        group = [r for r in rows if r['model'] == model]
        steps = [number(r['step'], f'{model} step', integer=True, low=1) for r in group]
        require(len(group) == count and steps == list(range(1, count + 1)),
                f'{model}: expected exactly {count} contiguous history steps ending at {count}.')
        losses = [number(r['validation_loss'], f'{model} validation_loss') for r in group]
        for row in group:
            number(row['train_loss'], f'{model} train_loss')
        best = min(range(len(group)), key=losses.__getitem__)  # Validation only; earliest minimum wins ties.
        require(steps[best] == expected_best, f'{model}: best validation step must be {expected_best}, found {steps[best]}.')
        summary[model] = dict(best_validation_step=steps[best], best_validation_loss=group[best]['validation_loss'])
    return summary


def confusion_summary(rows):
    expected = {(m, s, a, p) for m in MODELS for s in SPLITS for a, p in CLASS_PAIRS}
    keys = [(r['model'], r['split'], r['actual_class'], r['predicted_class']) for r in rows]
    require(len(keys) == len(expected) and set(keys) == expected, 'Confusion matrix must contain exactly four unique cells per model/split.')
    lookup = dict(zip(keys, rows))
    summary = {}
    for model in MODELS:
        for split in SPLITS:
            counts = [number(lookup[model, split, a, p]['count'], f'{model}/{split} confusion count',
                             integer=True, high=SPLIT_ROWS[split]) for a, p in CLASS_PAIRS]
            require(sum(counts) == SPLIT_ROWS[split], f'{model}/{split}: confusion total must be {SPLIT_ROWS[split]}.')
            summary[model, split] = dict(zip(CONFUSION_FIELDS, counts)) | dict(total_rows=sum(counts), total_errors=counts[1] + counts[2])
    return summary


def validate_confusion_metrics(row, counts):
    """Validate consistency without replacing any authoritative metric value."""
    tn, fp, fn, tp = (counts[c] for c in CONFUSION_FIELDS)
    def divide(a, b):
        return a / b if b else 0.0
    f1 = divide(2 * tp, 2 * tp + fp + fn)
    expected = dict(accuracy=(tn + tp) / counts['total_rows'], precision=divide(tp, tp + fp),
        recall=divide(tp, tp + fn), f1=f1,
        weighted_f1=((tn + fp) * divide(2 * tn, 2 * tn + fp + fn) + (tp + fn) * f1) / counts['total_rows'],
        mcc=divide(tp * tn - fp * fn, math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
    for metric, value in expected.items():
        require(math.isclose(float(row[metric]), value, rel_tol=1e-9, abs_tol=1e-10),
                f"{row['model']}/{row['split']}: {metric} disagrees with authoritative confusion counts.")


def build_results(root):
    validate_gate(root)  # Gate failure must stop before reading model results or writing anything.
    metrics = authoritative_rows(root, 'metrics.csv')
    expected = {(m, s) for m in MODELS for s in SPLITS}
    keys = [(r['model'], r['split']) for r in metrics]
    require(len(metrics) == 10 and set(keys) == expected, 'Metrics must contain exactly ten unique model/split rows.')
    lookup = dict(zip(keys, metrics))
    for row in metrics:
        for metric in METRICS:
            number(row[metric], metric, low=-1 if metric == 'mcc' else 0, high=1)
        require(number(row['seed'], 'seed') == 42 and number(row['threshold'], 'threshold') == 0.5,
                'Metrics must use seed=42 and threshold=0.5.')
        require(row['device'] == DEVICES[row['model']], f"{row['model']}: unexpected GPU device label.")
    confusion = confusion_summary(authoritative_rows(root, 'confusion_matrix.csv'))
    history_rows = authoritative_rows(root, 'training_history.csv')
    history = history_summary(history_rows)
    for model, filename in ARTIFACTS.items():
        require((root / 'models' / DATASET / RUN / filename).is_file(), f'{model}: missing model artifact {filename}.')
    finalized = {name: [] for name in OUTPUT_SCHEMAS}
    for model in MODELS:
        # Preserve every recorded step and source loss; RF has no rows here.
        for source in (r for r in history_rows if r['model'] == model):
            best = history[model]
            finalized[TRAIN_FILE].append(dict(dataset=DATASET, authoritative_run=RUN, task=TASK,
                model=model, device=lookup[model, 'validation']['device'], step=source['step'],
                train_loss=source['train_loss'], validation_loss=source['validation_loss'],
                is_best_validation_step='YES' if number(source['step'], 'step', integer=True) == best['best_validation_step'] else 'NO',
                best_validation_step=best['best_validation_step'], best_validation_loss=best['best_validation_loss'],
                predictor_count=50, seed=42, threshold=0.5))
        for split in SPLITS:
            source = lookup[model, split]
            counts = confusion[model, split]
            validate_confusion_metrics(source, counts)
            finalized[EVALUATION_FILES[split]].append(dict(dataset=DATASET, authoritative_run=RUN, model=model,
                device=source['device'], **{name: source[name] for name in METRICS}, **counts,
                predictor_count=50, seed=42, threshold=0.5, integrity_audit_status='PASS'))
    return finalized


def check_destination(directory):
    require(directory.resolve() == directory.absolute(), 'Canonical directory must not redirect through a symlink/junction.')
    existing = {}
    for filename, schema in OUTPUT_SCHEMAS.items():
        path = directory / filename
        require(not path.is_symlink(), 'Canonical output must not be a symlink.')
        if path.exists():
            require(path.is_file() and path.stat().st_nlink == 1, 'Unsafe canonical output target.')
            existing[filename] = read_csv(path, schema=schema)
    if existing:
        # TRAIN is the task-identity anchor. Unowned/partial evaluation files cannot
        # be silently adopted; schema plus dataset/run bind them to this file set.
        require(TRAIN_FILE in existing, 'Existing evaluation files have no training-history task identity; refusing overwrite.')
        for filename, rows in existing.items():
            identity = dict(dataset=DATASET, authoritative_run=RUN)
            if filename == TRAIN_FILE:
                identity['task'] = TASK
            if not all(all(row[key] == value for key, value in identity.items()) for row in rows):
                raise FileExistsError('Existing final result dataset/task/authoritative_run differs; refusing overwrite.')


def finalize(root):
    root = Path(root).resolve()
    tables = build_results(root)  # All validation completes before even creating the output directory.
    directory = root / 'results/final' / DATASET
    check_destination(directory)
    directory.mkdir(parents=True, exist_ok=True)
    staged = {}
    try:
        for filename, columns in OUTPUT_SCHEMAS.items():
            descriptor, temporary = tempfile.mkstemp(prefix='.' + filename + '.', suffix='.tmp', dir=directory)
            staged[filename] = Path(temporary)
            with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(tables[filename])
                stream.flush()
                os.fsync(stream.fileno())
        # Recheck authorization and existing output identity immediately before publication.
        validate_gate(root)
        check_destination(directory)
        # Atomic per file, not a three-file transaction. Publish the task identity
        # first so a partial first publication can be safely completed on rerun.
        for filename, temporary in staged.items():
            os.replace(temporary, directory / filename)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    return directory


def closure_record(root):
    """Validate stored evidence only; do not recreate final CSVs or open models."""
    expected = build_results(root)  # Read-only audit gate, authoritative identities, history and artifact checks.
    directory = root / 'results/final' / DATASET
    for filename, schema in OUTPUT_SCHEMAS.items():
        actual = read_csv(directory / filename, schema=schema)
        count = 606 if filename == TRAIN_FILE else 5
        require(len(actual) == count, f'Closure: {filename} must contain {count} rows, found {len(actual)}.')
        require(all(row['dataset'] == DATASET and row['authoritative_run'] == RUN for row in actual),
                f'Closure: {filename} dataset/authoritative run mismatch.')
        keys = ('model', 'step') if filename == TRAIN_FILE else ('model',)
        actual_rows = {tuple(row[key] for key in keys): row for row in actual}
        expected_rows = {tuple(str(row[key]) for key in keys): {key: str(row[key]) for key in schema}
                         for row in expected[filename]}
        require(len(actual_rows) == count and actual_rows == expected_rows,
                f'Closure: {filename} does not match validated authoritative results/history.')
    # Paths remain project-relative and portable. Closure declares the freeze; it
    # does not change permissions or alter the experiment's training code.
    return dict(dataset=DATASET, authoritative_run=RUN, stage_status='CLOSED_FROZEN',
        integrity_audit_status='PASS', finalization_status='PASS', train_rows=367293,
        validation_rows=97001, test_rows=98255, predictor_count=50, model_count=5, seed=42, threshold=0.5,
        train_results_path=f'results/final/{DATASET}/{TRAIN_FILE}',
        validation_results_path=f'results/final/{DATASET}/{EVALUATION_FILES["validation"]}',
        test_results_path=f'results/final/{DATASET}/{EVALUATION_FILES["test"]}',
        model_artifact_directory=f'models/{DATASET}/{RUN}',
        authoritative_result_directory=f'results/models/{DATASET}/{RUN}',
        integrity_audit_directory=f'results/audits/{DATASET}/integrity_audit',
        frozen_for_retraining='YES', frozen_for_test_reuse='YES', next_dataset='CIC-Darknet2020')


def check_closure_destination(path):
    require(not path.is_symlink() and path.resolve() == path.absolute(),
            'Closure output must not redirect through a symlink/junction.')
    if path.exists():
        require(path.is_file() and path.stat().st_nlink == 1, 'Unsafe closure output target.')
        identity = dict(dataset=DATASET, authoritative_run=RUN, stage_status='CLOSED_FROZEN')
        existing = read_csv(path, required=identity)
        if len(existing) != 1 or any(existing[0][key] != value for key, value in identity.items()):
            raise FileExistsError('Existing closure dataset/authoritative_run/stage_status differs; refusing overwrite.')


def close_experiment(root):
    root = Path(root).resolve()
    record = closure_record(root)  # All evidence must pass before any closure directory/file is created.
    destination = root / CLOSURE_PATH
    check_closure_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + destination.name + '.', suffix='.tmp', dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=CLOSURE_COLUMNS)
            writer.writeheader()
            writer.writerow(record)
            stream.flush()
            os.fsync(stream.fileno())
        # Revalidate authorization and stored-result consistency before publication.
        closure_record(root)
        check_closure_destination(destination)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--close', action='store_true', help='Verify completed results and write only the canonical closure CSV.')
    parser.add_argument('--execute', action='store_true', help='Execute the selected finalization or closure operation.')
    args = parser.parse_args(argv)
    action = 'CLOSURE' if args.close else 'FINALIZATION'
    if not args.execute:
        print(f'{action}_NOT_EXECUTED=True\nauthoritative_run={RUN}\nUse --execute to perform this operation.')
        return 0
    try:
        path = close_experiment(args.project_root) if args.close else finalize(args.project_root)
    except (OSError, ValueError, csv.Error) as exc:
        print(f'{action}_ERROR={exc}')
        return 1
    print(f'{action}=PASS\noutput={path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
