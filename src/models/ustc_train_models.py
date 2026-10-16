"""USTC Stage 4: five GPU baselines, finalized before one TEST pass on Linux/WSL.

Default preflight validates data and runs tiny synthetic GPU probes, never
real-data fitting. Optional backends are loaded only for selected models.
Seed 42 is reset before each independent model. GPU floating-point reductions
can still differ across hardware/library builds; no bitwise guarantee is made.

Neural inputs are float32 tensors of the ordered processed predictors, with no
new preprocessing. Best checkpoints track every strictly lower validation loss;
both neural models train for exactly 30 epochs before restoring the best checkpoint.
Calibration uses p(malware), with left-closed bins and p=1 in the last bin.
Empty-bin statistics and the terminal PR threshold are empty CSV cells.

TEST is opened only after every selected model is finalized and validated.
Model and result directories are exclusive. Artifacts are published atomically
without overwrite.
On model failure, earlier successful models/results are retained and the run
fails; no CPU fallback, automatic retries, or reuse of partial run directories.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import pickle
import platform
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_complex_dtype, is_numeric_dtype
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve)

from src.common.io import write_csv_atomic
from src.data.ustc_prepare_dataset import feature_role

DATASET = 'USTC-TFC2016'
SEED = 42
THRESHOLD = 0.5
FEATURE_COUNT = 50
BATCH_SIZE = 1024
MAX_EPOCHS = 30
MODELS = ('random_forest', '1d_cnn', 'xgboost', 'lightgbm', 'cnn_lstm')
PINS = {'pandas': '3.0.3', 'xgboost': '3.2.0', 'lightgbm': '4.7.0', 'torch': '2.14.0', 'cuml-cu13': '26.8.0'}
METADATA = ['target_binary', 'binary_label', 'semantic_family', 'capture_group', 'source_pcap']
ARTIFACTS = dict(zip(MODELS, ['random_forest.pkl', '1d_cnn.pt', 'xgboost.json', 'lightgbm.txt', 'cnn_lstm.pt']))
DEVICES = dict(zip(MODELS, ['cuml_cuda', 'cuda', 'cuda', 'lightgbm_cuda', 'cuda']))
PREFIX = ['dataset', 'model', 'split']
TAIL = ['run_id', 'timestamp_utc']
CSV_HEADERS = {
    'metrics.csv': PREFIX + ['device', 'accuracy', 'precision', 'recall', 'f1', 'weighted_f1', 'mcc', 'roc_auc', 'auprc', 'ece', 'brier_score', 'threshold', 'seed'] + TAIL,
    'confusion_matrix.csv': PREFIX + ['actual_class', 'predicted_class', 'count'] + TAIL,
    'roc_curve.csv': PREFIX + ['fpr', 'tpr', 'threshold'] + TAIL,
    'pr_curve.csv': PREFIX + ['precision', 'recall', 'threshold'] + TAIL,
    'calibration_bins.csv': PREFIX + ['bin_index', 'lower_bound', 'upper_bound', 'count', 'mean_probability', 'observed_positive_fraction', 'absolute_gap'] + TAIL,
    'training_history.csv': ['dataset', 'model'] + ['step', 'train_loss', 'validation_loss'] + TAIL,
}


@dataclass
class TrainingData:
    features: list[str]
    x_train: pd.DataFrame
    y_train: np.ndarray
    x_validation: pd.DataFrame
    y_validation: np.ndarray
    provenance: dict[str, set]


def read_csv_unique(path: Path) -> pd.DataFrame:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream))
    if len(header) != len(set(header)):
        raise ValueError(f'Duplicate CSV columns: {path}')
    return pd.read_csv(path, low_memory=False)


def load_data(root: Path) -> TrainingData:
    base = root / 'data' / 'processed' / DATASET
    parameters = read_csv_unique(base / 'preprocessing_parameters.csv')
    required = {'feature_name', 'retained', 'final_feature_order'}
    if not required.issubset(parameters.columns) or parameters.feature_name.isna().any() or not parameters.feature_name.is_unique:
        raise ValueError('Invalid preprocessing parameter schema or duplicate feature names.')
    flags = parameters.retained.astype(str).str.lower()
    if not flags.isin(['true', 'false']).all():
        raise ValueError('retained must contain only true/false.')
    kept = parameters.loc[flags == 'true'].copy()
    kept['final_feature_order'] = pd.to_numeric(kept.final_feature_order, errors='raise')
    kept = kept.sort_values('final_feature_order')
    if len(kept) != FEATURE_COUNT or kept.final_feature_order.tolist() != list(range(1, FEATURE_COUNT + 1)):
        raise ValueError(f'Expected exactly {FEATURE_COUNT} retained predictors with consecutive orders 1..{FEATURE_COUNT}.')
    features = kept.feature_name.tolist()
    forbidden = [c for c in features if c in METADATA or feature_role(c) != 'predictor_candidate']
    if forbidden:
        raise ValueError('Forbidden predictor columns: ' + ', '.join(forbidden))
    frames = {}
    for split in ('train', 'validation'):
        frame = read_csv_unique(base / f'{split}.csv')
        validate_frame(frame, features, split)
        frames[split] = frame
    for c in ('semantic_family', 'capture_group', 'source_pcap'):
        if set(frames['train'][c]) & set(frames['validation'][c]):
            raise ValueError(f'TRAIN/validation overlap in {c}.')
    return TrainingData(features, frames['train'][features], frames['train'].target_binary.to_numpy(dtype=int),
                        frames['validation'][features], frames['validation'].target_binary.to_numpy(dtype=int),
                        {c: set(frames['train'][c]) | set(frames['validation'][c])
                         for c in ('semantic_family', 'capture_group', 'source_pcap')})


def validate_frame(frame, features, split):
    if set(frame.columns) != set(features + METADATA):
        raise ValueError(f'{split}: expected exactly the retained predictors and five metadata columns.')
    if frame.empty:
        raise ValueError(f'{split}: dataset is empty.')
    nonnumeric = [c for c in features if not is_numeric_dtype(frame[c]) or is_complex_dtype(frame[c])]
    if nonnumeric or not np.isfinite(frame[features].to_numpy(dtype=float)).all():
        raise ValueError(f'{split}: predictors must be numeric and finite; invalid columns: {nonnumeric}')
    if not is_numeric_dtype(frame.target_binary) or set(frame.target_binary) != {0, 1}:
        raise ValueError(f'{split}: target_binary must contain both 0 and 1 only.')
    if frame[METADATA].isna().any().any() or frame[METADATA].eq('').any().any():
        raise ValueError(f'{split}: incomplete metadata.')
    mapped = frame.binary_label.map({'benign': 0, 'malware': 1})
    if mapped.isna().any() or not np.array_equal(mapped, frame.target_binary):
        raise ValueError(f'{split}: target mapping must be benign=0, malware=1.')


def load_test(root: Path, data: TrainingData):
    """Called once, only after all selected models and validation results are fixed."""
    frame = read_csv_unique(root / 'data' / 'processed' / DATASET / 'test.csv')
    validate_frame(frame, data.features, 'test')
    for column, seen in data.provenance.items():
        if seen & set(frame[column]):
            raise ValueError(f'TRAIN/validation/TEST overlap in {column}.')
    return frame[data.features], frame.target_binary.to_numpy(dtype=int)


def require_package(name: str, module_name: str | None = None):
    try:
        installed = version(name)
    except PackageNotFoundError as exc:
        raise RuntimeError(f'Missing dependency: {name}' + (f'=={PINS[name]}' if name in PINS else '')) from exc
    print(f'{name} version={installed}')
    if name in PINS and installed.split('+')[0] != PINS[name]:
        raise RuntimeError(f'Incompatible {name}: requires {PINS[name]}, found {installed}.')
    try:
        return importlib.import_module(module_name or name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(f'Cannot initialize {name}: {exc}') from exc


def seed_everything(torch=None):
    random.seed(SEED)
    np.random.seed(SEED)
    if torch is not None:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def model_parameters(name: str) -> dict:
    if name == 'random_forest':
        return dict(n_estimators=300, random_state=SEED, max_depth=None,
                    max_features='sqrt', bootstrap=True, n_streams=1, output_type='cupy')
    if name == 'xgboost':
        return dict(objective='binary:logistic', n_estimators=300, max_depth=6,
                    learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
                    tree_method='hist', device='cuda', random_state=SEED,
                    eval_metric='logloss', early_stopping_rounds=30)
    if name == 'lightgbm':
        return dict(objective='binary', n_estimators=300, num_leaves=31,
                    learning_rate=0.05, random_state=SEED, device_type='cuda',
                    data_random_seed=SEED, feature_fraction_seed=SEED, bagging_seed=SEED)
    raise ValueError(name)


def verify_xgboost_cuda(model):
    # Inspect actual learner state: XGBoost can warn and silently choose CPU.
    config = json.loads(model.get_booster().save_config())
    device = config['learner']['generic_param']['device']
    if not str(device).startswith('cuda'):
        raise RuntimeError(f'XGBoost GPU required; actual learner device is {device!r}. CPU fallback rejected.')


def load_cuml_rf():
    require_package('cuml-cu13', 'cuml')
    from cuml.ensemble import RandomForestClassifier
    return RandomForestClassifier


def load_cupy():
    try:
        cp = importlib.import_module('cupy')
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError('No CUDA device found.')
        device = cp.cuda.Device()
        properties = cp.cuda.runtime.getDeviceProperties(device.id)
        print(f'CuPy version={cp.__version__}, CUDA device={device.id}, GPU={properties["name"]}, '
              f'CUDA runtime={cp.cuda.runtime.runtimeGetVersion()}')
        cp.random.seed(SEED)
        return cp
    except Exception as exc:
        raise RuntimeError(f'CuPy CUDA unavailable; no CPU fallback: {exc}') from exc


def gpu_features(cp, frame):
    values = frame.to_numpy(dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError('Processed predictors overflow float32 GPU input.')
    return cp.asarray(values, dtype=cp.float32)


def cuml_probability(cp, model, x):
    classes = cp.asnumpy(cp.asarray(model.classes_)).tolist()
    if len(classes) != 2 or set(classes) != {0, 1}:
        raise ValueError('Model must expose classes 0 and 1.')
    probabilities = model.predict_proba(x)
    if not isinstance(probabilities, cp.ndarray):
        raise RuntimeError('cuML inference must return CUDA probabilities; CPU fallback rejected.')
    return cp.asnumpy(probabilities[:, classes.index(1)])


def verify_lightgbm_cuda(model):
    if model.booster_.params.get('device_type') != 'cuda':
        raise RuntimeError('LightGBM CUDA tree learner required; CPU/OpenCL fallback rejected.')


def gpu_smoke(backend: str):
    module = require_package(backend) if backend != 'random_forest' else None
    seed_everything(module if backend == 'torch' else None)
    if backend == 'torch':
        print(f'torch.cuda.is_available()={module.cuda.is_available()}')
        print(f'PyTorch CUDA runtime={module.version.cuda}')
        if not module.cuda.is_available():
            raise RuntimeError('PyTorch CUDA is unavailable; GPU required.')
        print(f'CUDA GPU={module.cuda.get_device_name(0)}')
        x = module.ones((2, 2), device='cuda')
        result = x @ x
        module.cuda.synchronize()
        if result.device.type != 'cuda' or not module.isfinite(result).all().item():
            raise RuntimeError('PyTorch CUDA tensor operation failed.')
        return
    # Report the active CUDA device even for an individually selected booster.
    cp = load_cupy()
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(128, 4)).astype(np.float32)
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    if backend == 'random_forest':
        classifier = load_cuml_rf()
        parameters = dict(model_parameters('random_forest'), n_estimators=2)
        model = classifier(**parameters)
        gx, gy = cp.asarray(x), cp.asarray(y, dtype=cp.int32)
        model.fit(gx, gy)
        probabilities = cuml_probability(cp, model, gx)
        if not np.isfinite(probabilities).all():
            raise RuntimeError('cuML CUDA smoke inference failed.')
        cp.cuda.get_current_stream().synchronize()
        print('cuML Random Forest CUDA smoke test passed.')
    elif backend == 'xgboost':
        print('XGBoost GPU configuration: tree_method=hist, device=cuda')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model = module.XGBClassifier(n_estimators=2, max_depth=2, tree_method='hist',
                device='cuda', random_state=SEED, objective='binary:logistic', eval_metric='logloss')
            model.fit(cp.asarray(x, dtype=cp.float32), cp.asarray(y, dtype=cp.int32))
            try:
                verify_xgboost_cuda(model)
            except Exception as exc:
                raise RuntimeError(str(exc) + ' ' + '; '.join(str(w.message) for w in caught)) from exc
        for warning in caught:
            print(f'XGBoost smoke warning: {warning.message}')
    elif backend == 'lightgbm':
        print('LightGBM GPU configuration: device_type=cuda; CUDA-enabled build required')
        model = module.LGBMClassifier(n_estimators=2, num_leaves=4, min_child_samples=2,
            min_data_in_bin=1, device_type='cuda', objective='binary', random_state=SEED,
            data_random_seed=SEED, feature_fraction_seed=SEED, bagging_seed=SEED, verbosity=1)
        model.fit(x, y)
        verify_lightgbm_cuda(model)
    else:
        raise ValueError(backend)


def preflight_backends(selected: tuple[str, ...]) -> dict[str, bool]:
    print(f'platform.system()={platform.system()}')
    print(f'Python version={sys.version}')
    try:
        if platform.system() != 'Linux':
            raise RuntimeError('Stage 4 GPU training requires Linux/WSL2; no Windows training backend.')
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError('Stage 4 requires Python 3.11.')
        require_package('pandas')
        require_package('numpy')
        require_package('scikit-learn', 'sklearn')
    except Exception as exc:
        print(f'NOT READY: {exc}\nPREFLIGHT_READY=False')
        return dict.fromkeys(selected, False)
    ready, checked = {}, {}
    for model in selected:
        backend = 'torch' if model in ('1d_cnn', 'cnn_lstm') else model
        if backend not in checked:
            try:
                gpu_smoke(backend)
                checked[backend] = (True, '')
            except Exception as exc:
                checked[backend] = (False, f'{type(exc).__name__}: {exc}')
        ready[model], error = checked[backend]
        print(f'{model}: {"READY" if ready[model] else "NOT READY"}' + (f' — {error}' if error else ''))
    print(f'PREFLIGHT_READY={all(ready.values())}')
    return ready


def evaluate_predictions(y_true, probabilities) -> dict[str, list[dict]]:
    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or set(y) != {0, 1}:
        raise ValueError('Evaluation targets must include both binary classes.')
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid evaluation probabilities: check count, finiteness and [0,1] bounds.')
    predicted = (p >= THRESHOLD).astype(int)
    bins, ece = [], 0.0
    edges = np.linspace(0, 1, 16)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p >= lo) & ((p <= hi) if i == 14 else (p < hi))
        count = int(mask.sum())
        mean = float(p[mask].mean()) if count else None
        observed = float(y[mask].mean()) if count else None
        gap = abs(mean - observed) if count else None
        if count:
            ece += count / len(y) * gap
        bins.append(dict(bin_index=i, lower_bound=lo, upper_bound=hi, count=count,
                         mean_probability=mean, observed_positive_fraction=observed, absolute_gap=gap))
    metrics = dict(accuracy=float(accuracy_score(y, predicted)),
        precision=float(precision_score(y, predicted, pos_label=1, zero_division=0)),
        recall=float(recall_score(y, predicted, pos_label=1, zero_division=0)),
        f1=float(f1_score(y, predicted, pos_label=1, zero_division=0)),
        weighted_f1=float(f1_score(y, predicted, average='weighted', zero_division=0)),
        mcc=float(matthews_corrcoef(y, predicted)), roc_auc=float(roc_auc_score(y, p)),
        auprc=float(average_precision_score(y, p)), ece=float(ece),
        brier_score=float(np.mean((p - y) ** 2)), threshold=THRESHOLD, seed=SEED)
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError('Non-finite evaluation metric.')
    matrix = confusion_matrix(y, predicted, labels=[0, 1])
    names = ['benign', 'malware']
    fpr, tpr, roc_thresholds = roc_curve(y, p, drop_intermediate=False)
    precision, recall, pr_thresholds = precision_recall_curve(y, p, drop_intermediate=False)
    return {
        'metrics.csv': [metrics],
        'confusion_matrix.csv': [dict(actual_class=a, predicted_class=b, count=int(matrix[i, j]))
            for i, a in enumerate(names) for j, b in enumerate(names)],
        'roc_curve.csv': [dict(fpr=a, tpr=b, threshold=c) for a, b, c in zip(fpr, tpr, roc_thresholds)],
        'pr_curve.csv': [dict(precision=a, recall=b, threshold=float(pr_thresholds[i]) if i < len(pr_thresholds) else None)
            for i, (a, b) in enumerate(zip(precision, recall))],
        'calibration_bins.csv': bins,
        'training_history.csv': [],
    }


def build_neural_model(name, torch):
    nn = torch.nn
    if name == '1d_cnn':
        return nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.20), nn.Linear(64, 1))
    if name != 'cnn_lstm':
        raise ValueError(name)

    class CNNLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(1, 64, kernel_size=3, padding=1),
                                      nn.BatchNorm1d(64), nn.ReLU())
            self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
            self.dropout = nn.Dropout(0.20)
            self.output = nn.Linear(64, 1)

        def forward(self, x):
            sequence = self.conv(x).transpose(1, 2)
            _, (hidden, _) = self.lstm(sequence)
            return self.output(self.dropout(hidden[-1]))

    return CNNLSTM()


def fit_neural_epochs(model, epoch_pass):
    """Run all 30 epochs, then restore the lowest-validation-loss checkpoint."""
    best_loss = float('inf')
    best_state, best_epoch = None, 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = float(epoch_pass(True))
        validation_loss = float(epoch_pass(False))
        if not np.isfinite([train_loss, validation_loss]).all():
            raise RuntimeError('Non-finite neural training/validation loss.')
        history.append(dict(step=epoch, train_loss=train_loss, validation_loss=validation_loss))
        print(f'epoch={epoch} train_loss={train_loss:.8f} validation_loss={validation_loss:.8f}')
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError('No valid neural checkpoint.')
    model.load_state_dict(best_state)
    return history, best_epoch


def train_neural(name: str, data: TrainingData):
    torch = require_package('torch')
    seed_everything(torch)
    if not torch.cuda.is_available():
        raise RuntimeError(f'{name}: CUDA required, no CPU fallback.')
    model = build_neural_model(name, torch).to('cuda')
    if any(p.device.type != 'cuda' for p in model.parameters()):
        raise RuntimeError(f'{name}: model parameters are not on CUDA.')
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(SEED)

    def loader(x, y, shuffle):
        values = x.to_numpy(dtype=np.float32, copy=True)
        if not np.isfinite(values).all():
            raise ValueError('Processed predictors overflow float32 neural input.')
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(values).unsqueeze(1),
                                                 torch.tensor(y, dtype=torch.float32))
        return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
            num_workers=0, drop_last=False, generator=generator, pin_memory=True)

    train_loader = loader(data.x_train, data.y_train, True)
    validation_loader = loader(data.x_validation, data.y_validation, False)

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

    history, best_epoch = fit_neural_epochs(model, epoch_pass)
    model.eval()

    def predict(frame):
        probabilities = []
        inference_loader = loader(frame, np.zeros(len(frame)), False)
        with torch.no_grad():
            for x, _ in inference_loader:
                probabilities.append(torch.sigmoid(model(x.to('cuda', non_blocking=True)).squeeze(-1)).cpu().numpy())
        return np.concatenate(probabilities)

    checkpoint = dict(model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        model=name, predictor_count=len(data.features), feature_names=data.features, seed=SEED,
        best_epoch=best_epoch, architecture=dict(channels=64, kernel_size=3, padding=1,
            dropout=0.20, lstm_hidden_size=64 if name == 'cnn_lstm' else None,
            lstm_num_layers=1 if name == 'cnn_lstm' else None))
    return predict, history, lambda path: torch.save(checkpoint, path)


def positive_probability(model, x, **kwargs):
    classes = list(model.classes_)
    if len(classes) != 2 or set(classes) != {0, 1}:
        raise ValueError('Model must expose classes 0 and 1.')
    return model.predict_proba(x, **kwargs)[:, classes.index(1)]


def xgboost_probability(cp, model, frame, best_iteration):
    x = gpu_features(cp, frame)
    verify_xgboost_cuda(model)
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message=r'(?s).*(mismatched devices|Falling back to prediction using DMatrix)')
        try:
            probabilities = positive_probability(model, x, iteration_range=(0, best_iteration))
        except Warning as exc:
            raise RuntimeError(f'XGBoost GPU inference required; device fallback rejected: {exc}') from exc
    # The sklearn wrapper may already return host probabilities after CUDA inference.
    return cp.asnumpy(probabilities) if isinstance(probabilities, cp.ndarray) else np.asarray(probabilities)


def train_model(name: str, data: TrainingData):
    seed_everything()
    if name in ('1d_cnn', 'cnn_lstm'):
        return train_neural(name, data)
    if name == 'random_forest':
        cp = load_cupy()
        classifier = load_cuml_rf()
        model = classifier(**model_parameters(name))
        model.fit(gpu_features(cp, data.x_train), cp.asarray(data.y_train, dtype=cp.int32))
        def predict(frame):
            return cuml_probability(cp, model, gpu_features(cp, frame))
        def save(path):
            with path.open('wb') as stream:
                pickle.dump(model, stream, protocol=5)
        return predict, [], save
    backend = require_package(name)
    eval_set = [(data.x_train, data.y_train), (data.x_validation, data.y_validation)]
    if name == 'xgboost':
        cp = load_cupy()
        x_train = gpu_features(cp, data.x_train)
        y_train = cp.asarray(data.y_train, dtype=cp.int32)
        eval_set = [(x_train, y_train),
                    (gpu_features(cp, data.x_validation), cp.asarray(data.y_validation, dtype=cp.int32))]
        model = backend.XGBClassifier(**model_parameters(name))
        model.fit(x_train, y_train, eval_set=eval_set, verbose=False)
        verify_xgboost_cuda(model)
        losses = model.evals_result()
        train_losses, validation_losses = losses['validation_0']['logloss'], losses['validation_1']['logloss']
        best_iteration = int(model.best_iteration) + 1
        predict = lambda frame: xgboost_probability(cp, model, frame, best_iteration)
        save = lambda path: model.save_model(path)
    elif name == 'lightgbm':
        model = backend.LGBMClassifier(**model_parameters(name))
        losses = {}
        model.fit(data.x_train, data.y_train, eval_set=eval_set, eval_names=['train', 'validation'],
            eval_metric='binary_logloss', callbacks=[backend.early_stopping(30, first_metric_only=True, verbose=False),
                                                   backend.record_evaluation(losses)])
        verify_lightgbm_cuda(model)
        train_losses, validation_losses = losses['train']['binary_logloss'], losses['validation']['binary_logloss']
        best_iteration = int(model.best_iteration_)
        predict = lambda frame: positive_probability(model, frame, num_iteration=best_iteration)
        save = lambda path: model.booster_.save_model(str(path), num_iteration=best_iteration)
    else:
        raise ValueError(name)
    if not 1 <= best_iteration <= 300:
        raise RuntimeError(f"{name}: invalid finalized boosting iteration.")
    if not train_losses or len(train_losses) != len(validation_losses):
        raise RuntimeError(f'{name}: missing TRAIN/validation iterative loss history.')
    history = [dict(step=i + 1, train_loss=float(a), validation_loss=float(b))
               for i, (a, b) in enumerate(zip(train_losses, validation_losses))]
    if not np.isfinite([[r['train_loss'], r['validation_loss']] for r in history]).all():
        raise RuntimeError(f'{name}: non-finite training history.')
    return predict, history, save


def publish_model(path: Path, save):
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite: {path}')
    with TemporaryDirectory(prefix='.model-', dir=path.parent) as temporary:
        staged = Path(temporary) / path.name
        save(staged)
        if not staged.is_file() or staged.stat().st_size == 0:
            raise RuntimeError(f'Model artifact was not saved: {path.name}')
        os.link(staged, path)


def write_results(directory: Path, tables: dict):
    if set(tables) != set(CSV_HEADERS):
        raise ValueError('Exactly six result tables are required.')
    # Validate keys before the shared writer, which otherwise ignores extras.
    for name, rows in tables.items():
        if any(set(row) != set(CSV_HEADERS[name]) for row in rows):
            raise ValueError(f'Invalid output schema: {name}')
    for name, rows in tables.items():
        write_csv_atomic(directory / name, CSV_HEADERS[name], rows)


def run(root: Path, model: str = 'all', execute: bool = False) -> int:
    selected = MODELS if model == 'all' else (model,)
    if any(name not in MODELS for name in selected):
        raise ValueError(f'Unknown model: {model}')
    root = root.resolve()
    now = datetime.now(timezone.utc)
    run_id, timestamp = now.strftime('%Y%m%dT%H%M%SZ'), now.isoformat().replace('+00:00', 'Z')
    model_dir = root / 'models' / DATASET / run_id
    result_dir = root / 'results' / 'models' / DATASET / run_id
    print(f'model_directory={model_dir}\nresult_directory={result_dir}\nPREFLIGHT_ONLY={not execute}')
    if execute:
        for directory in (model_dir, result_dir):
            if directory.exists():
                raise FileExistsError(f'Refusing to reuse run directory: {directory}')
    data = load_data(root)
    print(f'TRAIN rows={len(data.y_train)}, validation rows={len(data.y_validation)}, predictors={len(data.features)}, seed={SEED}')
    readiness = preflight_backends(selected)
    if not all(readiness.values()):
        if execute:
            raise RuntimeError('Selected models are NOT READY; real training was not started.')
        return 1
    if not execute:
        return 0
    model_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)
    tables = {name: [] for name in CSV_HEADERS}
    finalized = {}

    def record(name, split, y, probabilities):
        evaluated = evaluate_predictions(y, probabilities)
        evaluated['metrics.csv'][0]['device'] = DEVICES[name]
        for filename, rows in evaluated.items():
            tables[filename].extend(dict(row, dataset=DATASET, model=name, split=split,
                run_id=run_id, timestamp_utc=timestamp) for row in rows)

    try:
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
        # No model fitting, selection, or configuration is permitted below this boundary.
        x_test, y_test = load_test(root, data)
        for name, predict in finalized.items():
            try:
                record(name, 'test', y_test, predict(x_test))
            except Exception as exc:
                raise RuntimeError(f'{name} TEST evaluation failed; no retry or retraining: {exc}') from exc
    finally:
        if tables['metrics.csv']:
            write_results(result_dir, tables)
    print('VALIDATION=PASS\nTEST=PASS')
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--model', choices=('all',) + MODELS, default='all')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    try:
        return run(args.project_root, args.model, args.execute)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
