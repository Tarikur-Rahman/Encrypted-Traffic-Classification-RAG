"""Five GPU CIC classifiers; TEST stays unread until all five are finalized.

Dry run reads contracts only. Preflight imports/checks GPU backends without any
fitting or TEST access. Execution has a one-way TRAINING -> TEST -> COMPLETE
state machine; failed TEST attempts are consumed, never retried in that run.
No existing experiment directory is reused. Partial outputs and failure status
are retained. Neural models always run 30 epochs, then restore the earliest
minimum-validation-loss checkpoint. Boosters select iterations on validation.

Multiclass Brier = mean(sum((p-one_hot(y))**2, axis=1)), without division by four.
ECE uses top-label confidence and correctness, with 15 equal-width [0,1] bins,
left-inclusive and right-exclusive except the final bin includes 1. AUPRC is
the macro mean of four one-vs-rest average-precision scores, not trapezoidal PR
area. Probabilities and confusion cells always follow class IDs 0,1,2,3.
LightGBM uses the CUDA tree learner; its native prediction API is used after
training. cuML, neural and XGBoost prediction inputs stay on their CUDA backend.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import pickle
import random
from tempfile import TemporaryDirectory
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, log_loss, matthews_corrcoef, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve)

DATASET = 'CIC-Darknet2020'
MODELS = ('random_forest', '1d_cnn', 'xgboost', 'lightgbm', 'cnn_lstm')
LABELS = ('Non-Tor', 'NonVPN', 'Tor', 'VPN')
PROBABILITY_COLUMNS = ('class_0_non_tor', 'class_1_nonvpn', 'class_2_tor', 'class_3_vpn')
SEED, CLASS_COUNT, FEATURE_COUNT, BATCH_SIZE, EPOCHS = 42, 4, 62, 1024, 30
GROUP_COUNT = 79160
ROW_COUNTS = dict(train=95170, validation=31723, test=31723)
PROVENANCE = ['Label', 'Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Src Port', 'Dst Port', 'Label.1']
METADATA = ['target_id'] + PROVENANCE
ARTIFACTS = dict(zip(MODELS, ('random_forest.pkl', '1d_cnn.pt', 'xgboost.json', 'lightgbm.txt', 'cnn_lstm.pt')))
DEVICES = dict(zip(MODELS, ('cuml_cuda', 'cuda', 'cuda', 'lightgbm_cuda', 'cuda')))
POLICIES = dict(port_policy='EXCLUDED_PROVENANCE_ONLY', nonfinite_policy='COERCE_NUMERIC_REPLACE_INF_WITH_MISSING',
    imputation_policy='TRAIN_MEDIAN_ONLY', constant_feature_policy='REMOVE_TRAIN_ZERO_VARIANCE_ONLY',
    scaling_policy='TRAIN_STANDARD_SCALER_ONLY', resampling_policy='NONE', pca_policy='NONE', feature_selection_policy='NONE')
PREFIX = ['dataset', 'run_id', 'model', 'split']
METRICS = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1', 'mcc',
           'macro_ovr_roc_auc', 'macro_ovr_auprc', 'multiclass_log_loss', 'multiclass_brier_score', 'ece']
HEADERS = {
    'metrics.csv': PREFIX + ['device'] + METRICS + ['row_count'],
    'confusion_matrix.csv': PREFIX + ['actual_class_id', 'actual_class_label', 'predicted_class_id', 'predicted_class_label', 'count'],
    'training_history.csv': ['dataset', 'run_id', 'model', 'step', 'train_loss', 'validation_loss', 'is_best_validation_step'],
    'roc_curve.csv': PREFIX + ['class_id', 'class_label', 'fpr', 'tpr', 'threshold'],
    'pr_curve.csv': PREFIX + ['class_id', 'class_label', 'precision', 'recall', 'threshold'],
    'calibration_bins.csv': PREFIX + ['bin_index', 'lower_bound', 'upper_bound', 'count', 'mean_confidence', 'accuracy', 'absolute_gap'],
    'run_configuration.csv': ['dataset', 'run_id', 'model', 'key', 'value'],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path, *, text=False):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        header = next(csv.reader(stream), [])
    require(header and len(header) == len(set(header)), f'{path}: missing/duplicate columns.')
    return pd.read_csv(path, keep_default_na=False, float_precision='round_trip',
                       dtype=str if text else {name: 'string' for name in PROVENANCE})


@dataclass
class DataContract:
    features: list[str]
    manifest: pd.DataFrame


def load_contract(root):
    """Metadata only. In particular, never open any processed TEST file here."""
    directory = root / 'metadata/traffic'
    main = read_csv(directory / 'cic_phase1_contract.csv', text=True)
    expected = dict(dataset=DATASET, task='five_model_4class_benchmark',
        raw_source='data/raw/CIC-Darknet2020/Darknet.CSV', raw_rows=str(sum(ROW_COUNTS.values())), raw_columns='85',
        target_column='Label', auxiliary_label_column='Label.1', class_count='4', split_group='Flow ID',
        split_seed='42', initial_predictor_candidates='77', contract_status='FROZEN_FOR_MATERIALIZATION', **POLICIES)
    require(len(main) == 1 and all(main.iloc[0].get(k) == v for k, v in expected.items()), 'Invalid frozen CIC contract.')
    for split, count in ROW_COUNTS.items():
        require(int(main.iloc[0][f'{split}_rows']) == count, f'Frozen {split} row count differs.')
    mapping = read_csv(directory / 'cic_label_mapping.csv', text=True)
    require(mapping.to_dict('records') == [dict(class_id=str(i), raw_label=label, canonical_label=label,
        probability_column=PROBABILITY_COLUMNS[i]) for i, label in enumerate(LABELS)], 'Class mapping/probability order differs.')
    features = read_csv(directory / 'cic_feature_contract.csv', text=True)
    require(len(features) == 85 and features.column_name.is_unique
            and features.source_index.tolist() == list(map(str, range(85))), 'Invalid source feature ordering.')
    require(features.predictor_candidate.isin(['YES', 'NO']).all()
            and features.model_allowed.equals(features.predictor_candidate), 'Invalid predictor permission flags.')
    candidates = features.loc[features.predictor_candidate == 'YES', 'column_name'].tolist()
    excluded = features.loc[features.predictor_candidate == 'NO', 'column_name'].tolist()
    require(len(candidates) == 77 and set(excluded) == set(PROVENANCE), 'Identifier/label exclusion contract differs.')
    parameters = read_csv(directory / 'cic_preprocessing_parameters.csv', text=True)
    require(parameters.feature_name.tolist() == candidates and parameters.source_order.tolist() == list(map(str, range(77))),
            'Preprocessing candidate order differs.')
    require(parameters.retained_final.isin(['YES', 'NO']).all(), 'Invalid retained flags.')
    keep = parameters.retained_final.eq('YES')
    require(parameters.train_zero_variance.tolist() == np.where(keep, 'NO', 'YES').tolist(), 'Removal policy differs.')
    require(int(keep.sum()) == FEATURE_COUNT and parameters.loc[keep, 'final_predictor_order'].tolist()
            == list(map(str, range(FEATURE_COUNT))), 'Expected exactly 62 ordered retained predictors.')
    require(parameters.loc[~keep, ['final_predictor_order', 'train_scaler_mean', 'train_scaler_scale']].eq('').all().all(),
            'Removed-feature fields must be blank.')
    for name in ['train_median', 'train_scaler_mean', 'train_scaler_scale']:
        values = pd.to_numeric(parameters[name] if name == 'train_median' else parameters.loc[keep, name], errors='raise')
        require(np.isfinite(values).all() and (name != 'train_scaler_scale' or values.gt(0).all()), 'Invalid frozen preprocessing parameters.')
    retained = parameters.loc[keep, 'feature_name'].tolist()
    require(not set(retained) & set(METADATA), 'Forbidden predictor.')
    manifest = read_csv(directory / 'cic_split_manifest.csv', text=True)
    require(len(manifest) == GROUP_COUNT and manifest.flow_id.is_unique and manifest.flow_id.ne('').all(), 'Invalid Flow ID manifest.')
    require(manifest.dataset.eq(DATASET).all() and manifest.seed.eq('42').all()
            and manifest['split'].isin(['TRAIN', 'VALIDATION', 'TEST']).all(), 'Manifest identity differs.')
    require(manifest.split_method.eq(main.iloc[0]['split_method']).all(), 'Frozen split method differs.')
    count_columns = ['non_tor_rows', 'nonvpn_rows', 'tor_rows', 'vpn_rows']
    for column in ['group_row_count'] + count_columns:
        values = pd.to_numeric(manifest[column], errors='raise')
        require(np.isfinite(values).all() and values.ge(0).all() and values.eq(np.floor(values)).all(), 'Invalid group counts.')
        manifest[column] = values.astype(np.int64)
    require(manifest[count_columns].sum(axis=1).equals(manifest.group_row_count), 'Group count totals differ.')
    for split, count in ROW_COUNTS.items():
        part = manifest[manifest['split'] == split.upper()]
        require(int(part.group_row_count.sum()) == count and part[count_columns].sum().gt(0).all(), 'Frozen partition counts differ.')
    return DataContract(retained, manifest)


@dataclass
class SplitData:
    x: pd.DataFrame
    y: np.ndarray
    groups: set[str]


def read_partition(root, contract, split, *, guard=None):
    """Internal reader; TEST callers must go through RunGuard.load_test."""
    require(split in ROW_COUNTS, 'Unknown partition.')
    if split == 'test':
        require(guard is not None and guard.phase == 'TEST' and guard.ALL_MODELS_FINALIZED
                and guard.test_load_attempted and not guard.test_loaded,
                'TEST reader requires the all-model finalization gate.')
    frame = read_csv(root / 'data/processed' / DATASET / f'{split}.csv')
    require(len(frame) == ROW_COUNTS[split] and list(frame.columns) == contract.features + METADATA,
            f'{split}: row count or predictor order differs.')
    y = pd.to_numeric(frame.target_id, errors='raise').to_numpy()
    require(set(y) == {0, 1, 2, 3} and np.array_equal(frame.Label.map(dict(zip(LABELS, range(4)))), y),
            f'{split}: invalid target IDs/raw labels.')
    require(not frame[PROVENANCE].isna().any().any() and not frame[PROVENANCE].eq('').any().any(), 'Missing provenance.')
    x = frame[contract.features]
    require(all(pd.api.types.is_numeric_dtype(x[c]) and not pd.api.types.is_complex_dtype(x[c]) for c in x), 'Nonnumeric predictors.')
    require(np.isfinite(x.to_numpy(dtype=np.float64)).all(), f'{split}: nonfinite predictors.')
    with np.errstate(over='ignore'):
        require(np.isfinite(x.to_numpy(dtype=np.float32)).all(), f'{split}: float32 overflow.')
    expected = contract.manifest[contract.manifest['split'] == split.upper()].set_index('flow_id')
    actual = frame.groupby(['Flow ID', 'target_id']).size().unstack(fill_value=0).reindex(columns=range(4), fill_value=0)
    require(set(actual.index) == set(expected.index), f'{split}: Flow ID membership differs from manifest.')
    require(np.array_equal(actual.reindex(expected.index).to_numpy(),
        expected[['non_tor_rows', 'nonvpn_rows', 'tor_rows', 'vpn_rows']].to_numpy()), f'{split}: group/class counts differ.')
    return SplitData(x, y.astype(np.int64), set(frame['Flow ID']))


def load_training_data(root, contract):
    train = read_partition(root, contract, 'train')
    validation = read_partition(root, contract, 'validation')
    require(not train.groups & validation.groups, 'TRAIN/VALIDATION Flow ID overlap.')
    return train, validation


def validate_history(name, history):
    if name == 'random_forest':
        require(not history, 'Random Forest must not have fabricated iterative history.')
        return
    count = len(history)
    require(count == EPOCHS if name in ('1d_cnn', 'cnn_lstm') else 1 <= count <= 300, 'Invalid history length.')
    require([r['step'] for r in history] == list(range(1, count + 1)), 'History steps must be contiguous.')
    losses = np.array([[r['train_loss'], r['validation_loss']] for r in history])
    require(np.isfinite(losses).all() and (losses >= 0).all(), 'Invalid history losses.')
    flags = [r['is_best_validation_step'] for r in history]
    require(flags == ['YES' if i == int(np.argmin(losses[:, 1])) else 'NO' for i in range(count)], 'Best step must be earliest validation minimum.')


@dataclass
class RunGuard:
    phase: str = 'TRAINING'
    started: set = field(default_factory=set)
    finalized: dict = field(default_factory=dict)
    validation_models: set = field(default_factory=set)
    histories: dict = field(default_factory=dict)
    test_load_attempted: bool = False
    test_loaded: bool = False
    test_attempted: set = field(default_factory=set)

    @property
    def ALL_MODELS_FINALIZED(self):
        return set(self.finalized) == set(MODELS)

    def begin_training(self, name):
        require(self.phase == 'TRAINING' and not self.test_load_attempted and name in MODELS and name not in self.started,
                'Training/retraining is forbidden after TEST or a prior attempt.')
        self.started.add(name)

    def finalize(self, name, path, history):
        require(self.phase == 'TRAINING' and name in self.started and name in self.validation_models
                and name not in self.finalized, 'Missing validation or invalid finalization state.')
        require(path.is_file() and path.stat().st_size > 0, 'Missing finalized artifact.')
        validate_history(name, history)
        self.histories[name] = copy.deepcopy(history)
        self.finalized[name] = path

    def load_test(self, root, contract, train, validation):
        require(self.phase == 'TRAINING' and not self.test_load_attempted and self.ALL_MODELS_FINALIZED
                and self.validation_models == set(MODELS), 'TEST is locked until all five models and validation results are finalized.')
        for name in MODELS:
            require(self.finalized[name].is_file() and self.finalized[name].stat().st_size > 0, 'Final artifact missing at TEST gate.')
            validate_history(name, self.histories[name])
        self.test_load_attempted = True
        self.phase = 'TEST'
        test = read_partition(root, contract, 'test', guard=self)
        require(not test.groups & (train.groups | validation.groups), 'TEST Flow ID overlap.')
        self.test_loaded = True
        return test

    def evaluate_test(self, name, predict, test):
        require(self.phase == 'TEST' and self.test_loaded and self.ALL_MODELS_FINALIZED
                and name in MODELS and name not in self.test_attempted, 'TEST evaluation unavailable or already attempted.')
        self.test_attempted.add(name)  # Consume before inference: exceptions cannot trigger retries.
        return evaluate_predictions(test.y, predict(test.x))


def model_parameters(name):
    if name == 'random_forest':
        return dict(n_estimators=300, max_depth=None, max_features='sqrt', bootstrap=True,
                    random_state=SEED, n_streams=1, output_type='cupy')
    if name == 'xgboost':
        return dict(objective='multi:softprob', num_class=4, n_estimators=300, max_depth=6,
                    learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, tree_method='hist',
                    device='cuda', random_state=SEED, eval_metric='mlogloss', early_stopping_rounds=30)
    if name == 'lightgbm':
        return dict(objective='multiclass', num_class=4, n_estimators=300, num_leaves=31,
                    learning_rate=0.05, random_state=SEED, device_type='cuda', metric='multi_logloss',
                    data_random_seed=SEED, feature_fraction_seed=SEED, bagging_seed=SEED)
    require(name in ('1d_cnn', 'cnn_lstm'), 'Unknown model.')
    return dict(batch_size=1024, epochs=30, early_termination=False, loss='CrossEntropyLoss',
                optimizer='AdamW', learning_rate=0.001, weight_decay=0.0001, channels=64,
                kernel_size=3, padding=1, dropout=0.20, output_logits=4,
                lstm_hidden_size=64 if name == 'cnn_lstm' else None,
                lstm_num_layers=1 if name == 'cnn_lstm' else None)


def seed_everything(torch=None, cp=None):
    random.seed(SEED)
    np.random.seed(SEED)
    if cp is not None:
        cp.random.seed(SEED)
    if torch is not None:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def gpu_preflight():
    """Import/build/device checks only: no estimators are fitted or inferred."""
    try:
        cp = importlib.import_module('cupy')
        torch = importlib.import_module('torch')
        xgb = importlib.import_module('xgboost')
        lgb = importlib.import_module('lightgbm')
        rf = importlib.import_module('cuml.ensemble').RandomForestClassifier
        require(cp.cuda.runtime.getDeviceCount() > 0, 'CuPy cannot see CUDA.')
        probe = cp.ones(1, dtype=cp.float32) + 1
        cp.cuda.get_current_stream().synchronize()
        require(float(cp.asnumpy(probe)[0]) == 2, 'CuPy CUDA operation failed.')
        require(torch.cuda.is_available(), 'PyTorch CUDA unavailable.')
        require(torch.ones(1, device='cuda').device.type == 'cuda', 'PyTorch allocation is not CUDA.')
        torch.cuda.synchronize()
        require(xgb.build_info().get('USE_CUDA') is True, 'XGBoost lacks a CUDA build.')
        library = Path(lgb.basic._LIB._name).read_bytes()
        require(b'CUDASingleGPUTreeLearner' in library and b'CUDAMulticlassSoftmax' in library,
                'Cannot verify LightGBM CUDA multiclass build; no CPU/OpenCL fallback.')
        return dict(cp=cp, torch=torch, xgboost=xgb, lightgbm=lgb, rf=rf)
    except Exception as exc:
        raise RuntimeError(f'GPU preflight failed (no CPU fallback): {exc}') from exc


def float32_values(frame):
    with np.errstate(over='ignore'):
        values = frame.to_numpy(dtype=np.float32, copy=True)
    require(np.isfinite(values).all(), 'Invalid float32 model input.')
    return values


def ordered_probabilities(probabilities, classes):
    p = np.asarray(probabilities, dtype=float)
    classes = list(classes)
    require(len(classes) == 4 and set(classes) == {0, 1, 2, 3}, 'Classifier must expose all four classes.')
    require(p.ndim == 2 and p.shape[1] == 4, 'Expected four probability columns.')
    return p[:, [classes.index(i) for i in range(4)]]


def verify_xgboost_cuda(booster):
    configuration = json.loads(booster.save_config())
    require(str(configuration['learner']['generic_param']['device']).startswith('cuda'), 'XGBoost CPU fallback is forbidden.')


def xgboost_call(operation):
    with warnings.catch_warnings():
        warnings.filterwarnings('error', message=r'(?is).*(mismatch.*device|device.*mismatch|falling back.*dmatrix|fallback.*cpu|no visible gpu|not compiled with cuda|setting device to cpu).*')
        try:
            return operation()
        except Warning as exc:
            raise RuntimeError(f'Critical XGBoost device/fallback warning: {exc}') from exc


def build_neural_model(name, torch):
    nn = torch.nn
    if name == '1d_cnn':
        return nn.Sequential(nn.Conv1d(1, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.20), nn.Linear(64, 4))
    require(name == 'cnn_lstm', 'Unknown neural architecture.')
    class CNNLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv1d(1, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
            self.lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
            self.dropout = nn.Dropout(0.20)
            self.output = nn.Linear(64, 4)

        def forward(self, x):
            _, (hidden, _) = self.lstm(self.conv(x).transpose(1, 2))
            return self.output(self.dropout(hidden[-1]))
    return CNNLSTM()


def history_rows(train_losses, validation_losses):
    require(len(train_losses) == len(validation_losses) and len(train_losses) > 0, 'Missing iterative history.')
    best = int(np.argmin(validation_losses))
    return [dict(step=i + 1, train_loss=float(a), validation_loss=float(b),
                 is_best_validation_step='YES' if i == best else 'NO')
            for i, (a, b) in enumerate(zip(train_losses, validation_losses))]


def fit_neural_epochs(model, epoch_pass):
    train_losses, validation_losses = [], []
    best_loss, best_state = float('inf'), None
    for epoch in range(1, EPOCHS + 1):
        train_loss, validation_loss = float(epoch_pass(True)), float(epoch_pass(False))
        require(np.isfinite([train_loss, validation_loss]).all(), 'Nonfinite neural loss.')
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        if validation_loss < best_loss:
            best_loss, best_state = validation_loss, copy.deepcopy(model.state_dict())
        print(f'epoch={epoch} train_loss={train_loss:.8f} validation_loss={validation_loss:.8f}')
    require(best_state is not None, 'No valid checkpoint.')
    model.load_state_dict(best_state)  # Never restore or early-terminate inside the epoch loop.
    return history_rows(train_losses, validation_losses)


def train_neural(name, train, validation, features, torch):
    seed_everything(torch=torch)
    model = build_neural_model(name, torch).to('cuda')
    require(all(p.device.type == 'cuda' for p in model.parameters()), 'Neural model must reside on CUDA.')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    generator = torch.Generator().manual_seed(SEED)
    def loader(frame, y, shuffle):
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(float32_values(frame)).unsqueeze(1),
                                                 torch.tensor(y, dtype=torch.long))
        return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
            num_workers=0, drop_last=False, pin_memory=True, generator=generator)
    train_loader, validation_loader = loader(train.x, train.y, True), loader(validation.x, validation.y, False)
    def epoch_pass(training):
        model.train(training)
        total, count = 0.0, 0
        with torch.set_grad_enabled(training):
            for x, y in train_loader if training else validation_loader:
                x, y = x.to('cuda', non_blocking=True), y.to('cuda', non_blocking=True)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(x), y)
                if training:
                    loss.backward()
                    optimizer.step()
                total += float(loss.item()) * len(y)
                count += len(y)
        return total / count
    history = fit_neural_epochs(model, epoch_pass)
    model.eval()
    def predict(frame):
        chunks = []
        with torch.no_grad():
            for x, _ in loader(frame, np.zeros(len(frame), dtype=np.int64), False):
                chunks.append(torch.softmax(model(x.to('cuda', non_blocking=True)), dim=1).cpu().numpy())
        return np.concatenate(chunks)
    checkpoint = dict(model=name, model_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        feature_names=features, predictor_count=62, class_labels=LABELS, probability_columns=PROBABILITY_COLUMNS,
        seed=SEED, best_epoch=next(r['step'] for r in history if r['is_best_validation_step'] == 'YES'),
        hyperparameters=model_parameters(name))
    return predict, history, lambda path: torch.save(checkpoint, path)


def train_model(name, train, validation, features, backends, guard):
    guard.begin_training(name)
    cp = backends['cp']
    seed_everything(cp=cp)
    if name in ('1d_cnn', 'cnn_lstm'):
        return train_neural(name, train, validation, features, backends['torch'])
    if name == 'random_forest':
        model = backends['rf'](**model_parameters(name))
        model.fit(cp.asarray(float32_values(train.x)), cp.asarray(train.y, dtype=cp.int32))
        classes = cp.asnumpy(cp.asarray(model.classes_)).tolist()
        def predict(frame):
            p = model.predict_proba(cp.asarray(float32_values(frame)))
            require(isinstance(p, cp.ndarray), 'cuML inference must return CUDA probabilities.')
            return ordered_probabilities(cp.asnumpy(p), classes)
        def save(path):
            with path.open('wb') as stream:
                pickle.dump(model, stream, protocol=5)
        return predict, [], save
    if name == 'xgboost':
        backend = backends[name]
        model = backend.XGBClassifier(**model_parameters(name))
        x, y = cp.asarray(float32_values(train.x)), cp.asarray(train.y, dtype=cp.int32)
        xv, yv = cp.asarray(float32_values(validation.x)), cp.asarray(validation.y, dtype=cp.int32)
        xgboost_call(lambda: model.fit(x, y, eval_set=[(x, y), (xv, yv)], verbose=False))
        verify_xgboost_cuda(model.get_booster())
        losses = model.evals_result()
        history = history_rows(losses['validation_0']['mlogloss'], losses['validation_1']['mlogloss'])
        best = int(model.best_iteration) + 1
        require(best == next(r['step'] for r in history if r['is_best_validation_step'] == 'YES'), 'XGBoost best iteration differs.')
        booster = model.get_booster()[:best]  # Saved state contains only finalized rounds.
        booster.set_param({'device': 'cuda'})
        classes = list(model.classes_)
        def predict(frame):
            verify_xgboost_cuda(booster)
            p = xgboost_call(lambda: booster.inplace_predict(cp.asarray(float32_values(frame))))
            require(isinstance(p, cp.ndarray), 'XGBoost CUDA inference must return CuPy probabilities.')
            return ordered_probabilities(cp.asnumpy(p), classes)
        return predict, history, lambda path: booster.save_model(str(path))
    require(name == 'lightgbm', 'Unknown backend.')
    backend = backends[name]
    model = backend.LGBMClassifier(**model_parameters(name))
    losses = {}
    model.fit(train.x, train.y, eval_set=[(train.x, train.y), (validation.x, validation.y)],
        eval_names=['train', 'validation'], eval_metric='multi_logloss', callbacks=[
            backend.early_stopping(30, first_metric_only=True, verbose=False), backend.record_evaluation(losses)])
    require(model.booster_.params.get('device_type') == 'cuda', 'LightGBM CUDA tree learner required.')
    history = history_rows(losses['train']['multi_logloss'], losses['validation']['multi_logloss'])
    best = int(model.best_iteration_)
    require(best == next(r['step'] for r in history if r['is_best_validation_step'] == 'YES'), 'LightGBM best iteration differs.')
    # Freeze a separate prediction-only booster at the validation-selected state.
    booster = backend.Booster(model_str=model.booster_.model_to_string(num_iteration=best))
    classes = list(model.classes_)
    return (lambda frame: ordered_probabilities(booster.predict(frame, num_iteration=best), classes), history,
            lambda path: booster.save_model(str(path), num_iteration=best))


def evaluate_predictions(y_true, probabilities):
    y, p = np.asarray(y_true), np.asarray(probabilities, dtype=float)
    require(y.ndim == 1 and set(y) == {0, 1, 2, 3}, 'Evaluation requires all four classes.')
    require(p.shape == (len(y), 4) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            and np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0), 'Invalid four-class probabilities.')
    predicted = p.argmax(axis=1)
    onehot = np.eye(4)[y.astype(int)]
    confidence, correct = p.max(axis=1), predicted == y
    bins, ece = [], 0.0
    edges = np.linspace(0, 1, 16)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (confidence >= lo) & ((confidence <= hi) if i == 14 else (confidence < hi))
        count = int(mask.sum())
        mean = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        gap = abs(mean - accuracy) if count else None
        if count:
            ece += count / len(y) * gap
        bins.append(dict(bin_index=i, lower_bound=lo, upper_bound=hi, count=count,
                         mean_confidence=mean, accuracy=accuracy, absolute_gap=gap))
    metrics = dict(accuracy=float(accuracy_score(y, predicted)),
        macro_precision=float(precision_score(y, predicted, labels=range(4), average='macro', zero_division=0)),
        macro_recall=float(recall_score(y, predicted, labels=range(4), average='macro', zero_division=0)),
        macro_f1=float(f1_score(y, predicted, labels=range(4), average='macro', zero_division=0)),
        weighted_f1=float(f1_score(y, predicted, labels=range(4), average='weighted', zero_division=0)),
        mcc=float(matthews_corrcoef(y, predicted)),
        macro_ovr_roc_auc=float(np.mean([roc_auc_score(onehot[:, c], p[:, c]) for c in range(4)])),
        macro_ovr_auprc=float(np.mean([average_precision_score(onehot[:, c], p[:, c]) for c in range(4)])),
        multiclass_log_loss=float(log_loss(y, p, labels=range(4))),
        multiclass_brier_score=float(np.mean(np.sum((p - onehot) ** 2, axis=1))), ece=float(ece), row_count=len(y))
    require(np.isfinite(list(metrics.values())).all(), 'Nonfinite metric.')
    matrix = confusion_matrix(y, predicted, labels=range(4))
    roc, pr = [], []
    for c, label in enumerate(LABELS):
        fpr, tpr, thresholds = roc_curve(onehot[:, c], p[:, c], drop_intermediate=False)
        roc.extend(dict(class_id=c, class_label=label, fpr=float(a), tpr=float(b), threshold=float(t))
                   for a, b, t in zip(fpr, tpr, thresholds))
        precision, recall, thresholds = precision_recall_curve(onehot[:, c], p[:, c], drop_intermediate=False)
        pr.extend(dict(class_id=c, class_label=label, precision=float(a), recall=float(b),
                       threshold=float(thresholds[i]) if i < len(thresholds) else None)
                  for i, (a, b) in enumerate(zip(precision, recall)))
    return {'metrics.csv': [metrics], 'confusion_matrix.csv': [dict(actual_class_id=a, actual_class_label=LABELS[a],
        predicted_class_id=b, predicted_class_label=LABELS[b], count=int(matrix[a, b])) for a in range(4) for b in range(4)],
        'roc_curve.csv': roc, 'pr_curve.csv': pr, 'calibration_bins.csv': bins}


class ResultWriter:
    def __init__(self, directory, run_id):
        self.directory, self.run_id = directory, run_id
        for filename, header in HEADERS.items():
            with (directory / filename).open('x', encoding='utf-8', newline='') as stream:
                csv.writer(stream).writerow(header)

    def append(self, filename, rows):
        rows = [dict(dataset=DATASET, run_id=self.run_id, **row) for row in rows]
        require(all(set(row) == set(HEADERS[filename]) for row in rows), f'Invalid output schema: {filename}')
        with (self.directory / filename).open('a', encoding='utf-8', newline='') as stream:
            csv.DictWriter(stream, fieldnames=HEADERS[filename]).writerows(rows)

    def configuration(self, model, key, value):
        self.append('run_configuration.csv', [dict(model=model, key=key, value=value)])

    def evaluation(self, model, split, evaluated):
        for filename, rows in evaluated.items():
            self.append(filename, [dict(row, model=model, split=split,
                **({'device': DEVICES[model]} if filename == 'metrics.csv' else {})) for row in rows])


def publish_model(path, save):
    require(not path.exists(), 'Model artifact already exists.')
    with TemporaryDirectory(prefix='.model-', dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        save(temporary)
        require(temporary.is_file() and temporary.stat().st_size > 0, 'Empty model artifact.')
        os.link(temporary, path)  # Exclusive atomic publication; never replace another artifact.


def create_run(root, run_id):
    model_dir = root / 'models' / DATASET / run_id
    result_dir = root / 'results/models' / DATASET / run_id
    if model_dir.exists() or result_dir.exists():
        raise FileExistsError('Refusing to overwrite or resume an existing experimental run.')
    model_dir.mkdir(parents=True, exist_ok=False)
    result_dir.mkdir(parents=True, exist_ok=False)
    return model_dir, result_dir


def record_configuration(writer, contract, backends):
    global_values = dict(seed=42, class_count=4, class_mapping=dict(enumerate(LABELS)),
        probability_order=PROBABILITY_COLUMNS, predictor_count=62, feature_names=contract.features,
        train_rows=95170, validation_rows=31723, test_rows=31723, preprocessing='ALREADY_FROZEN',
        test_selection='FORBIDDEN', test_evaluations_per_model=1, model_roster=MODELS,
        brier_convention='mean(sum((p-one_hot_target)^2, classes))',
        ece_convention='top-label;15 equal-width [0,1] bins;last includes 1',
        auprc_convention='macro mean of four OVR average precision scores',
        lightgbm_early_stopping_rounds=30, lightgbm_inference='native prediction API',
        gpu_preflight='NO_FITTING;LightGBM CUDA build symbols checked;runtime training must still succeed',
        class_weights='NONE', resampling='NONE', additional_preprocessing='NONE')
    for key, value in global_values.items():
        writer.configuration('all', key, json.dumps(value) if isinstance(value, (dict, list, tuple)) else value)
    for name in MODELS:
        for key, value in model_parameters(name).items():
            writer.configuration(name, key, 'None' if value is None else value)
    for name in ('cp', 'torch', 'xgboost', 'lightgbm'):
        writer.configuration('all', f'{name}_version', getattr(backends[name], '__version__', 'unknown'))


def run(root, *, execute=False, preflight=False):
    require(not (execute and preflight), 'Choose execute or preflight, not both.')
    root = Path(root).resolve()
    contract = load_contract(root)
    if not execute and not preflight:
        print('DRY_RUN=PASS; configuration only; no processed partitions or GPU backends loaded.')
        return 0
    train, validation = load_training_data(root, contract)
    backends = gpu_preflight()
    if not execute:
        print('PREFLIGHT=PASS; no fitting or TEST access; LightGBM runtime training remains untested.')
        return 0
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    model_dir, result_dir = create_run(root, run_id)
    writer, guard = ResultWriter(result_dir, run_id), RunGuard()
    predictors = {}
    try:
        record_configuration(writer, contract, backends)
        writer.configuration('all', 'run_status', 'STARTED')
        for name in MODELS:
            predict, history, save = train_model(name, train, validation, contract.features, backends, guard)
            validate_history(name, history)
            evaluated = evaluate_predictions(validation.y, predict(validation.x))
            publish_model(model_dir / ARTIFACTS[name], save)
            writer.evaluation(name, 'validation', evaluated)
            writer.append('training_history.csv', [dict(row, model=name) for row in history])
            guard.validation_models.add(name)
            guard.finalize(name, model_dir / ARTIFACTS[name], history)
            predictors[name] = predict
        # Persisted validation rows and all five artifacts must precede TEST I/O.
        recorded = read_csv(result_dir / 'metrics.csv', text=True)
        require(len(recorded) == 5 and set(recorded.model) == set(MODELS)
                and recorded['split'].eq('validation').all(), 'Missing persisted validation metrics.')
        writer.configuration('all', 'ALL_MODELS_FINALIZED', guard.ALL_MODELS_FINALIZED)
        test = guard.load_test(root, contract, train, validation)
        for name in MODELS:
            writer.configuration(name, 'test_inference_attempt', 1)
            evaluated = guard.evaluate_test(name, predictors[name], test)
            writer.evaluation(name, 'test', evaluated)
        require(guard.test_attempted == set(MODELS), 'Missing TEST evaluation.')
        recorded = read_csv(result_dir / 'metrics.csv', text=True)
        require(len(recorded) == 10 and set(zip(recorded.model, recorded['split']))
                == {(name, split) for name in MODELS for split in ('validation', 'test')}, 'Expected exactly ten metric rows.')
        guard.phase = 'COMPLETE'
        writer.configuration('all', 'run_status', 'COMPLETE')
    except Exception as exc:
        guard.phase = 'FAILED'
        writer.configuration('all', 'run_status', 'FAILED_PROTOCOL_OR_EXECUTION')
        writer.configuration('all', 'failure', f'{type(exc).__name__}: {exc}')
        raise
    print(f'CIC_TRAINING=PASS; run_id={run_id}; models={model_dir}; results={result_dir}')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    return run(args.project_root, execute=args.execute, preflight=args.preflight)


if __name__ == '__main__':
    raise SystemExit(main())
