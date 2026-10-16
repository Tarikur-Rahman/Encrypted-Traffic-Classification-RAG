import numpy as np
from src.models.evaluation import evaluate
import json
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.models import ustc_train_models as stage4

def test_evaluate_multiclass():
    y=np.array([0,1,2,0,1,2])
    p=np.array([
        [0.9,0.05,0.05],[0.05,0.9,0.05],[0.05,0.05,0.9],
        [0.8,0.1,0.1],[0.1,0.8,0.1],[0.1,0.1,0.8],
    ])
    out=evaluate(y,p,[0,1,2])
    assert out["overall"]["macro_f1"] == 1.0
    assert len(out["predictions"]) == 6
    assert len(out["confusion_matrix_long"]) == 9


@pytest.fixture
def stage4_root(tmp_path):
    base = tmp_path / 'data/processed/USTC-TFC2016'
    base.mkdir(parents=True)
    features = [f'feature_{i}' for i in range(50)]
    pd.DataFrame([dict(feature_name=name, retained=True, final_feature_order=i + 1)
                  for i, name in reversed(list(enumerate(features)))]).to_csv(base / 'preprocessing_parameters.csv', index=False)
    for split in ['train', 'validation']:
        frame = pd.DataFrame({name: np.arange(12, dtype=float) * (i + 1) / 100
                              for i, name in reversed(list(enumerate(features)))})
        frame['target_binary'] = np.arange(12) % 2
        frame['binary_label'] = np.where(frame.target_binary == 0, 'benign', 'malware')
        frame['semantic_family'] = split + '_family_' + frame.binary_label
        frame['capture_group'] = split + '_capture_' + frame.binary_label
        frame['source_pcap'] = frame.capture_group + '.pcap'
        frame.to_csv(base / f'{split}.csv', index=False)
    # Deliberately no locked evaluation file is created.
    return tmp_path


def test_stage4_feature_order_and_no_test_dependency(stage4_root, monkeypatch):
    original_read, original_open = pd.read_csv, Path.open
    accessed = []
    def guard_read(path, *args, **kwargs):
        assert Path(path).name != 'test.csv'
        accessed.append(Path(path).name)
        return original_read(path, *args, **kwargs)
    def guard_open(path, *args, **kwargs):
        assert path.name != 'test.csv'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_csv', guard_read)
    monkeypatch.setattr(Path, 'open', guard_open)
    data = stage4.load_data(stage4_root)
    assert data.features == [f'feature_{i}' for i in range(50)]
    assert list(data.x_train.columns) == list(data.x_validation.columns) == data.features
    assert set(accessed) == {'train.csv', 'validation.csv', 'preprocessing_parameters.csv'}
    assert not set(stage4.METADATA) & set(data.features)
    np.testing.assert_allclose(data.x_train.feature_0, np.arange(12) / 100)


@pytest.mark.parametrize('mutation,match', [
    ('bad_target', 'target_binary'), ('bad_mapping', 'target mapping'),
    ('nonnumeric', 'numeric and finite'), ('nan', 'numeric and finite'),
    ('infinity', 'numeric and finite'), ('extra', 'exactly'), ('missing', 'exactly'),
    ('overlap', 'overlap'),
])
def test_stage4_rejects_bad_input(stage4_root, mutation, match):
    path = stage4_root / 'data/processed/USTC-TFC2016/validation.csv'
    frame = pd.read_csv(path)
    if mutation == 'bad_target':
        frame.loc[0, 'target_binary'] = 2
    elif mutation == 'bad_mapping':
        frame.loc[0, 'binary_label'] = 'malware'
    elif mutation == 'nonnumeric':
        frame['feature_0'] = 'text'
    elif mutation in ('nan', 'infinity'):
        frame.loc[0, 'feature_0'] = np.nan if mutation == 'nan' else np.inf
    elif mutation == 'extra':
        frame['src_ip'] = '192.0.2.1'
    elif mutation == 'missing':
        frame = frame.drop(columns='feature_0')
    elif mutation == 'overlap':
        frame.loc[0, 'semantic_family'] = 'train_family_benign'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        stage4.load_data(stage4_root)


@pytest.mark.parametrize('forbidden', stage4.METADATA + ['id', 'src_port', 'src_ip', 'application_name', 'dst2src_first_seen_ms'])
def test_stage4_rejects_forbidden_predictor_contract(stage4_root, forbidden):
    path = stage4_root / 'data/processed/USTC-TFC2016/preprocessing_parameters.csv'
    params = pd.read_csv(path)
    params.loc[0, 'feature_name'] = forbidden
    params.to_csv(path, index=False)
    with pytest.raises(ValueError, match='Forbidden predictor'):
        stage4.load_data(stage4_root)


def test_stage4_rejects_predictor_count_and_order(stage4_root):
    path = stage4_root / 'data/processed/USTC-TFC2016/preprocessing_parameters.csv'
    params = pd.read_csv(path)
    params.loc[0, 'final_feature_order'] = 1
    params.to_csv(path, index=False)
    with pytest.raises(ValueError, match='50 retained'):
        stage4.load_data(stage4_root)
    params.iloc[:-1].to_csv(path, index=False)
    with pytest.raises(ValueError, match='50 retained'):
        stage4.load_data(stage4_root)


def test_stage4_binary_metrics_curves_and_calibration():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.5, 0.5, 1.0])
    tables = stage4.evaluate_predictions(y, p)
    metrics = tables['metrics.csv'][0]
    assert metrics['accuracy'] == 0.75
    assert metrics['precision'] == pytest.approx(2 / 3)
    assert metrics['recall'] == 1
    assert metrics['f1'] == 0.8
    assert metrics['weighted_f1'] == pytest.approx((2 / 3 + 0.8) / 2)
    assert metrics['mcc'] == pytest.approx(1 / np.sqrt(3))
    assert metrics['roc_auc'] == 0.875
    assert metrics['auprc'] == pytest.approx(5 / 6)
    assert metrics['brier_score'] == 0.125
    assert metrics['ece'] == 0
    assert metrics['threshold'] == 0.5 and metrics['seed'] == 42
    assert 'auroc' not in metrics
    assert [row['count'] for row in tables['confusion_matrix.csv']] == [1, 1, 0, 2]
    assert [(r['actual_class'], r['predicted_class']) for r in tables['confusion_matrix.csv']] == [
        ('benign', 'benign'), ('benign', 'malware'), ('malware', 'benign'), ('malware', 'malware')]
    bins = tables['calibration_bins.csv']
    assert len(bins) == 15 and sum(r['count'] for r in bins) == 4
    assert bins[0]['count'] == 1 and bins[-1]['count'] == 1
    assert bins[1]['mean_probability'] is None
    assert len(tables['roc_curve.csv']) == 4
    assert np.isinf(tables['roc_curve.csv'][0]['threshold'])
    assert tables['pr_curve.csv'][-1] == {'precision': 1.0, 'recall': 0.0, 'threshold': None}
    # A deliberately miscalibrated constant probability checks the ECE formula.
    bad = stage4.evaluate_predictions(y, np.full(4, 0.9))
    assert bad['metrics.csv'][0]['ece'] == pytest.approx(0.4)
    assert bad['metrics.csv'][0]['brier_score'] == pytest.approx(0.41)
    edges = np.linspace(0, 1, 16)
    boundary = stage4.evaluate_predictions(np.arange(16) % 2, edges)['calibration_bins.csv']
    assert [r['count'] for r in boundary] == [1] * 14 + [2]


@pytest.mark.parametrize('probabilities', [[0.1], [np.nan, 0.5], [np.inf, 0.5], [-0.1, 0.5], [0.1, 1.1]])
def test_stage4_rejects_invalid_probabilities(probabilities):
    with pytest.raises(ValueError, match='probabilities'):
        stage4.evaluate_predictions([0, 1], probabilities)


@pytest.mark.parametrize('selection', ['all'] + list(stage4.MODELS))
def test_stage4_cli_selection_preflight_only(stage4_root, monkeypatch, selection):
    seen = []
    def ready(models):
        seen.extend(models)
        return dict.fromkeys(models, True)
    monkeypatch.setattr(stage4, 'preflight_backends', ready)
    monkeypatch.setattr(stage4, 'train_model', lambda *args: pytest.fail('Preflight must not train'))
    assert stage4.main(['--project-root', str(stage4_root), '--model', selection]) == 0
    assert seen == (list(stage4.MODELS) if selection == 'all' else [selection])
    assert not (stage4_root / 'models').exists()
    assert not (stage4_root / 'results').exists()


RESULT_SCHEMAS = {
    'metrics.csv': 'dataset model split device accuracy precision recall f1 weighted_f1 mcc roc_auc auprc ece brier_score threshold seed run_id timestamp_utc',
    'confusion_matrix.csv': 'dataset model split actual_class predicted_class count run_id timestamp_utc',
    'roc_curve.csv': 'dataset model split fpr tpr threshold run_id timestamp_utc',
    'pr_curve.csv': 'dataset model split precision recall threshold run_id timestamp_utc',
    'calibration_bins.csv': 'dataset model split bin_index lower_bound upper_bound count mean_probability observed_positive_fraction absolute_gap run_id timestamp_utc',
    'training_history.csv': 'dataset model step train_loss validation_loss run_id timestamp_utc',
}


@pytest.mark.parametrize('selection', ['all'] + list(stage4.MODELS))
def test_stage4_models_finalize_before_test_once(stage4_root, monkeypatch, selection):
    selected = stage4.MODELS if selection == 'all' else (selection,)
    create_test_split(stage4_root)
    events = []
    original_read = stage4.read_csv_unique
    def guarded_read(path):
        if path.name == 'test.csv':
            assert events == [(action, name) for name in selected for action in ('fit', 'validation', 'save')]
            events.append(('load', 'test'))
        return original_read(path)
    monkeypatch.setattr(stage4, 'read_csv_unique', guarded_read)
    seen = []
    def synthetic_model(name, data):
        seen.append((name, id(data)))
        events.append(('fit', name))
        history = [] if name == 'random_forest' else [dict(step=1, train_loss=0.4, validation_loss=0.5)]
        def predict(frame):
            split = 'validation' if frame is data.x_validation else 'test'
            events.append((split, name))
            return np.where(np.arange(len(frame)) % 2 == 1, 0.8, 0.2)
        def save(path):
            events.append(('save', name))
            path.write_bytes(b'synthetic model')
        return predict, history, save
    monkeypatch.setattr(stage4, 'preflight_backends', lambda models: dict.fromkeys(models, True))
    monkeypatch.setattr(stage4, 'train_model', synthetic_model)
    assert stage4.run(stage4_root, model=selection, execute=True) == 0
    assert [name for name, _ in seen] == list(selected)
    assert events == ([(action, name) for name in selected for action in ('fit', 'validation', 'save')]
                      + [('load', 'test')] + [('test', name) for name in selected])
    assert len({identity for _, identity in seen}) == 1
    model_dirs = list((stage4_root / 'models/USTC-TFC2016').iterdir())
    result_dirs = list((stage4_root / 'results/models/USTC-TFC2016').iterdir())
    assert len(model_dirs) == len(result_dirs) == 1
    assert model_dirs[0].name == result_dirs[0].name
    assert {p.name for p in model_dirs[0].iterdir()} == {stage4.ARTIFACTS[name] for name in selected}
    assert {p.name for p in result_dirs[0].iterdir()} == set(RESULT_SCHEMAS)
    for name, columns in RESULT_SCHEMAS.items():
        frame = pd.read_csv(result_dirs[0] / name)
        assert list(frame.columns) == columns.split()
        if not frame.empty:
            assert set(frame.run_id) == {result_dirs[0].name}
        if name != 'training_history.csv':
            assert set(frame.split) == {'validation', 'test'}
    metrics = pd.read_csv(result_dirs[0] / 'metrics.csv')
    assert set(metrics.seed) == {42} and set(metrics.threshold) == {0.5}
    assert len(metrics) == 2 * len(selected)
    assert metrics.groupby(['model', 'split']).size().eq(1).all()
    assert metrics.set_index('model').device.to_dict() == {name: stage4.DEVICES[name] for name in selected}
    assert set(metrics.device) <= {'cuml_cuda', 'cuda', 'lightgbm_cuda'}
    history = pd.read_csv(result_dirs[0] / 'training_history.csv')
    assert set(history.model) == set(selected) - {'random_forest'}
    pr = pd.read_csv(result_dirs[0] / 'pr_curve.csv')
    assert pr.groupby(['model', 'split']).tail(1).threshold.isna().all()


class FakeGPUArray(np.ndarray):
    """Distinct fake device type lets tests reject accidental host predictions."""


def fake_cupy():
    return SimpleNamespace(ndarray=FakeGPUArray, float32=np.float32, int32=np.int32,
        asarray=lambda a, dtype=None: np.asarray(a, dtype=dtype).view(FakeGPUArray),
        asnumpy=lambda a: np.asarray(a),
        cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)))


def create_test_split(root):
    base = root / 'data/processed/USTC-TFC2016'
    frame = pd.read_csv(base / 'validation.csv')
    for column in ('semantic_family', 'capture_group', 'source_pcap'):
        frame[column] = frame[column].str.replace('validation', 'test')
    frame['feature_0'] += 10  # Distinct TEST inputs, not a reused validation frame.
    frame.to_csv(base / 'test.csv', index=False)


def test_stage4_direct_cuml_fit_inference_and_pickle(stage4_root, monkeypatch):
    import sys
    from types import ModuleType
    data = stage4.load_data(stage4_root)
    cp = fake_cupy()
    events = []
    class GPUForest:
        classes_ = cp.asarray([1, 0])  # Check positive-class lookup, not fixed column 1.
        def __init__(self, **params):
            self.params = params
            events.append(('init', params))
        def fit(self, x, y):
            assert isinstance(x, FakeGPUArray) and isinstance(y, FakeGPUArray)
            assert x.dtype == np.float32 and y.dtype == np.int32 and x.shape[1] == 50
            np.testing.assert_array_equal(x, data.x_train.to_numpy(dtype=np.float32))
            events.append(('fit', self))
        def predict_proba(self, x):
            assert isinstance(x, FakeGPUArray) and x.dtype == np.float32
            events.append(('predict', self))
            return cp.asarray(np.tile([0.75, 0.25], (len(x), 1)))
    ensemble = ModuleType('cuml.ensemble')
    ensemble.RandomForestClassifier = GPUForest
    monkeypatch.setitem(sys.modules, 'cuml.ensemble', ensemble)
    monkeypatch.setattr(stage4, 'load_cupy', lambda: cp)
    def dependency(name, module):
        assert (name, module) == ('cuml-cu13', 'cuml')
    monkeypatch.setattr(stage4, 'require_package', dependency)
    def dump(model, stream, protocol):
        assert model is events[1][1] and protocol == 5
        stream.write(b'synthetic GPU pickle')
    monkeypatch.setattr(stage4.pickle, 'dump', dump)
    predict, history, save = stage4.train_model('random_forest', data)
    assert history == []
    assert np.all(predict(data.x_validation) == 0.75)
    assert np.all(predict(data.x_validation + 10) == 0.75)
    assert [action for action, _ in events] == ['init', 'fit', 'predict', 'predict']
    assert all(model is events[1][1] for _, model in events[1:])
    artifact = stage4_root / 'random_forest.pkl'
    stage4.publish_model(artifact, save)
    with pytest.raises(FileExistsError):
        stage4.publish_model(artifact, lambda path: pytest.fail('Must not overwrite'))


def test_stage4_cuml_missing_has_no_cpu_fallback(stage4_root, monkeypatch):
    data = stage4.load_data(stage4_root)
    monkeypatch.setattr(stage4, 'load_cupy', lambda: fake_cupy())
    def missing(*args):
        raise RuntimeError('Missing dependency: cuml-cu13==26.8.0')
    monkeypatch.setattr(stage4, 'require_package', missing)
    with pytest.raises(RuntimeError, match='cuml-cu13'):
        stage4.train_model('random_forest', data)
    host_model = SimpleNamespace(classes_=np.array([0, 1]), predict_proba=lambda x: np.array([[0.2, 0.8]]))
    with pytest.raises(RuntimeError, match='CPU fallback rejected'):
        stage4.cuml_probability(fake_cupy(), host_model, fake_cupy().asarray([[1]]))


@pytest.mark.parametrize('directory', ['models/USTC-TFC2016', 'results/models/USTC-TFC2016'])
def test_stage4_run_directory_no_overwrite(tmp_path, monkeypatch, directory):
    class Frozen:
        @staticmethod
        def now(tz):
            return datetime(2026, 1, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(stage4, 'datetime', Frozen)
    (tmp_path / directory / '20260101T000000Z').mkdir(parents=True)
    monkeypatch.setattr(stage4, 'load_data', lambda *args: pytest.fail('Must fail before loading data'))
    with pytest.raises(FileExistsError):
        stage4.run(tmp_path, execute=True)


@pytest.mark.parametrize('pandas_version', ['3.0.3', '2.2.3', '3.0.2', '3.0.4', '3.1.0'])
def test_stage4_linux_preflight_pandas_pin(monkeypatch, capsys, pandas_version):
    monkeypatch.setattr(stage4.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(stage4.sys, 'version_info', (3, 11, 0))
    monkeypatch.setattr(stage4, 'version', lambda name: pandas_version if name == 'pandas' else '0.0.0')
    monkeypatch.setattr(stage4.importlib, 'import_module', lambda name: SimpleNamespace())
    seen = []
    monkeypatch.setattr(stage4, 'gpu_smoke', seen.append)
    assert stage4.PINS['pandas'] == '3.0.3'
    compatible = pandas_version == '3.0.3'
    assert stage4.preflight_backends(('random_forest',)) == {'random_forest': compatible}
    assert seen == (['random_forest'] if compatible else [])
    output = capsys.readouterr().out
    assert f'pandas version={pandas_version}' in output
    assert f'PREFLIGHT_READY={compatible}' in output
    if not compatible:
        assert f'Incompatible pandas: requires 3.0.3, found {pandas_version}.' in output


def test_stage4_gpu_readiness_selection_and_fail_closed(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(stage4.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(stage4, 'require_package', lambda *args: SimpleNamespace())
    def smoke(name):
        seen.append(name)
        if name == 'lightgbm':
            raise RuntimeError('CUDA tree learner unavailable')
    monkeypatch.setattr(stage4, 'gpu_smoke', smoke)
    assert stage4.preflight_backends(('random_forest',)) == {'random_forest': True}
    assert seen == ['random_forest']
    seen.clear()
    readiness = stage4.preflight_backends(stage4.MODELS)
    assert seen == ['random_forest', 'torch', 'xgboost', 'lightgbm']
    assert readiness['lightgbm'] is False
    assert 'CUDA tree learner unavailable' in capsys.readouterr().out


def test_stage4_missing_gpu_blocks_training(stage4_root, monkeypatch):
    monkeypatch.setattr(stage4, 'preflight_backends', lambda models: dict.fromkeys(models, False))
    monkeypatch.setattr(stage4, 'train_model', lambda *args: pytest.fail('No training without readiness'))
    assert stage4.run(stage4_root, model='xgboost') == 1
    with pytest.raises(RuntimeError, match='NOT READY'):
        stage4.run(stage4_root, model='xgboost', execute=True)
    assert not (stage4_root / 'models').exists()


def test_stage4_xgboost_cpu_fallback_rejected():
    def model(device):
        return SimpleNamespace(get_booster=lambda: SimpleNamespace(
            save_config=lambda: json.dumps({'learner': {'generic_param': {'device': device}}})))
    with pytest.raises(RuntimeError, match='CPU fallback rejected'):
        stage4.verify_xgboost_cuda(model('cpu'))
    stage4.verify_xgboost_cuda(model('cuda:0'))


def test_stage4_backend_configs_and_seed():
    assert stage4.SEED == 42
    rf = stage4.model_parameters('random_forest')
    assert rf['max_depth'] is None and rf['max_features'] == 'sqrt' and rf['bootstrap'] is True
    assert rf['output_type'] == 'cupy' and rf['n_streams'] == 1
    assert stage4.PINS['cuml-cu13'] == '26.8.0'
    for model in ['random_forest', 'xgboost', 'lightgbm']:
        params = stage4.model_parameters(model)
        assert params['random_state'] == 42
        assert params['n_estimators'] == 300
        assert not {'class_weight', 'scale_pos_weight'} & params.keys()
    assert stage4.model_parameters('xgboost')['early_stopping_rounds'] == 30
    assert stage4.model_parameters('xgboost')['device'] == 'cuda'
    assert stage4.model_parameters('lightgbm')['device_type'] == 'cuda'
    stage4.seed_everything()
    import random
    first = random.random(), np.random.random()
    stage4.seed_everything()
    assert first == (random.random(), np.random.random())


def test_stage4_torch_cuda_seeding(monkeypatch):
    calls = []
    def call(name):
        return lambda seed: calls.append((name, seed))
    fake = SimpleNamespace(manual_seed=call('torch'),
        cuda=SimpleNamespace(is_available=lambda: True, manual_seed=call('cuda'), manual_seed_all=call('cuda_all')),
        backends=SimpleNamespace(cudnn=SimpleNamespace()), use_deterministic_algorithms=lambda *args, **kwargs: None)
    stage4.seed_everything(fake)
    assert calls == [('torch', 42), ('cuda', 42), ('cuda_all', 42)]
    assert fake.backends.cudnn.deterministic is True
    assert fake.backends.cudnn.benchmark is False


@pytest.mark.parametrize('losses,best_epoch', [
    ([0.7, 0.5] + [0.6] * 28, 2),
    ([1.0, 1.0 - 5e-7] + [1.1] * 28, 2),
    ([1 / i for i in range(1, 31)], 30),
    ([0.5] * 30, 1),
    ([0.5] + [0.6] * 28 + [0.4], 30),
])
def test_stage4_neural_exactly_30_epochs_and_best_checkpoint(losses, best_epoch):
    passes = []
    class Model:
        state = {'epoch': 0}
        def state_dict(self):
            return self.state
        def load_state_dict(self, state):
            assert passes == [mode for _ in range(30) for mode in (True, False)]
            assert self.state['epoch'] == 30
            self.state = state
    model = Model()
    def epoch_pass(training):
        passes.append(training)
        if training:
            model.state['epoch'] += 1
            return 0.4
        return losses[model.state['epoch'] - 1]
    history, best = stage4.fit_neural_epochs(model, epoch_pass)
    assert best == model.state['epoch'] == best_epoch
    assert history == [dict(step=i + 1, train_loss=0.4, validation_loss=loss)
                       for i, loss in enumerate(losses)]
    assert stage4.BATCH_SIZE == 1024 and stage4.MAX_EPOCHS == 30


def test_stage4_architecture_definitions():
    class Layer:
        def __init__(self, kind, *args, **kwargs):
            self.kind, self.args, self.kwargs = kind, args, kwargs
    kinds = ['Conv1d', 'ReLU', 'BatchNorm1d', 'AdaptiveAvgPool1d', 'Flatten', 'Dropout', 'Linear', 'Sequential', 'LSTM']
    nn = SimpleNamespace(Module=object, **{kind: (lambda *a, _kind=kind, **kw: Layer(_kind, *a, **kw)) for kind in kinds})
    cnn = stage4.build_neural_model('1d_cnn', SimpleNamespace(nn=nn))
    assert [layer.kind for layer in cnn.args] == ['Conv1d', 'ReLU', 'BatchNorm1d', 'Conv1d', 'ReLU', 'AdaptiveAvgPool1d', 'Flatten', 'Dropout', 'Linear']
    assert cnn.args[0].args == (1, 64) and cnn.args[3].args == (64, 64)
    assert cnn.args[0].kwargs == {'kernel_size': 3, 'padding': 1}
    assert cnn.args[-2].args == (0.20,) and cnn.args[-1].args == (64, 1)
    lstm = stage4.build_neural_model('cnn_lstm', SimpleNamespace(nn=nn))
    assert [layer.kind for layer in lstm.conv.args] == ['Conv1d', 'BatchNorm1d', 'ReLU']
    assert lstm.lstm.kwargs == dict(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
    assert lstm.dropout.args == (0.20,) and lstm.output.args == (64, 1)


def test_stage4_lightgbm_rejects_non_cuda_backends():
    for device in ('gpu', 'cpu', None):
        with pytest.raises(RuntimeError, match='CUDA tree learner required'):
            stage4.verify_lightgbm_cuda(SimpleNamespace(booster_=SimpleNamespace(params={'device_type': device})))
    stage4.verify_lightgbm_cuda(SimpleNamespace(booster_=SimpleNamespace(params={'device_type': 'cuda'})))


def test_stage4_preflight_rejects_windows(monkeypatch, capsys):
    monkeypatch.setattr(stage4.platform, 'system', lambda: 'Windows')
    monkeypatch.setattr(stage4, 'gpu_smoke', lambda *args: pytest.fail('No Windows model backend'))
    assert not any(stage4.preflight_backends(stage4.MODELS).values())
    output = capsys.readouterr().out
    assert 'Linux/WSL2' in output and 'PREFLIGHT_READY=False' in output


@pytest.mark.parametrize('backend', ['xgboost', 'lightgbm'])
def test_stage4_boosting_fit_uses_validation_early_stopping(stage4_root, monkeypatch, backend):
    data = stage4.load_data(stage4_root)
    cp = fake_cupy()
    monkeypatch.setattr(stage4, 'load_cupy', lambda: cp)
    events = {}
    class Booster:
        params = {'device_type': 'cuda'}
        def save_config(self):
            return json.dumps({'learner': {'generic_param': {'device': 'cuda:0'}}})
        def save_model(self, path, num_iteration):
            events['saved_iteration'] = num_iteration
            Path(path).write_text('booster')
    class Classifier:
        classes_ = np.array([0, 1])
        booster_ = Booster()
        best_iteration_ = 1
        best_iteration = 0
        def __init__(self, **kwargs):
            events['configuration'] = kwargs
        def fit(self, x, y, **kwargs):
            if backend == 'xgboost':
                assert isinstance(x, FakeGPUArray) and isinstance(y, FakeGPUArray)
                assert kwargs['eval_set'][0][0] is x and kwargs['eval_set'][0][1] is y
                for (gx, gy), (hx, hy) in zip(kwargs['eval_set'],
                        [(data.x_train, data.y_train), (data.x_validation, data.y_validation)]):
                    assert isinstance(gx, FakeGPUArray) and gx.dtype == np.float32
                    assert isinstance(gy, FakeGPUArray) and gy.dtype == np.int32
                    assert gx.shape[1] == 50
                    np.testing.assert_array_equal(gx, hx.to_numpy(dtype=np.float32))
                    np.testing.assert_array_equal(gy, hy)
            else:
                assert x is data.x_train and y is data.y_train
                assert kwargs['eval_set'][0][0] is data.x_train
                assert kwargs['eval_set'][1][0] is data.x_validation
                assert kwargs['eval_set'][1][1] is data.y_validation
            assert 'fit' not in events
            assert len(kwargs['eval_set']) == 2
            events['fit'] = kwargs
            if backend == 'lightgbm':
                for cb in kwargs['callbacks']:
                    if isinstance(cb, dict):
                        cb.update(train={'binary_logloss': [0.5, 0.4]}, validation={'binary_logloss': [0.6, 0.7]})
        def get_booster(self):
            return self.booster_
        def evals_result(self):
            return {'validation_0': {'logloss': [0.5, 0.4]}, 'validation_1': {'logloss': [0.6, 0.7]}}
        def predict_proba(self, x, **kwargs):
            assert kwargs == ({'iteration_range': (0, 1)} if backend == 'xgboost' else {'num_iteration': 1})
            if backend == 'xgboost':
                assert isinstance(x, FakeGPUArray) and x.dtype == np.float32
            events.setdefault('predictions', []).append(x)
            probabilities = np.tile([0.25, 0.75], (len(x), 1))
            return cp.asarray(probabilities) if backend == 'xgboost' else probabilities
        def save_model(self, path):
            Path(path).write_text('classifier')
    def stopping(rounds, **kwargs):
        events['stopping'] = (rounds, kwargs)
        return 'stop'
    fake = SimpleNamespace(XGBClassifier=Classifier, LGBMClassifier=Classifier,
                           early_stopping=stopping, record_evaluation=lambda results: results)
    monkeypatch.setattr(stage4, 'require_package', lambda name: fake)
    predict, history, save = stage4.train_model(backend, data)
    validation_probabilities = predict(data.x_validation)
    assert type(validation_probabilities) is np.ndarray
    assert np.all(validation_probabilities == 0.75)
    stage4.evaluate_predictions(data.y_validation, validation_probabilities)
    create_test_split(stage4_root)
    x_test, y_test = stage4.load_test(stage4_root, data)
    test_probabilities = predict(x_test)
    assert type(test_probabilities) is np.ndarray
    assert np.all(test_probabilities == 0.75)
    stage4.evaluate_predictions(y_test, test_probabilities)
    if backend == 'xgboost':
        np.testing.assert_array_equal(events['predictions'][0], data.x_validation.to_numpy(dtype=np.float32))
        np.testing.assert_array_equal(events['predictions'][1], x_test.to_numpy(dtype=np.float32))
    assert len(events['predictions']) == 2
    assert history == [dict(step=1, train_loss=0.5, validation_loss=0.6),
                       dict(step=2, train_loss=0.4, validation_loss=0.7)]
    assert events['configuration']['random_state'] == 42
    if backend == 'xgboost':
        assert events['configuration'] == dict(objective='binary:logistic', n_estimators=300,
            max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
            tree_method='hist', device='cuda', random_state=42, eval_metric='logloss',
            early_stopping_rounds=30)
        assert stage4.THRESHOLD == 0.5
    else:
        assert events['stopping'][0] == 30
        assert events['fit']['eval_names'] == ['train', 'validation']
        assert events['configuration']['device_type'] == 'cuda'
    save(stage4_root / stage4.ARTIFACTS[backend])
    if backend == 'lightgbm':
        assert events['saved_iteration'] == 1


@pytest.mark.parametrize('split', ['validation', 'test'])
@pytest.mark.parametrize('message', ['mismatched devices', 'Falling back to prediction using DMatrix'])
def test_stage4_xgboost_device_warning_rejected(stage4_root, split, message):
    data = stage4.load_data(stage4_root)
    create_test_split(stage4_root)
    frame = data.x_validation if split == 'validation' else stage4.load_test(stage4_root, data)[0]
    def predict(x, **kwargs):
        assert isinstance(x, FakeGPUArray)
        stage4.warnings.warn('XGBoost: ' + message, UserWarning)
        pytest.fail('Device fallback must stop inference')
    model = SimpleNamespace(classes_=np.array([0, 1]), predict_proba=predict,
        get_booster=lambda: SimpleNamespace(save_config=lambda: json.dumps(
            {'learner': {'generic_param': {'device': 'cuda:0'}}})))
    with pytest.raises(RuntimeError, match='device fallback rejected'):
        stage4.xgboost_probability(fake_cupy(), model, frame, 1)


def test_stage4_partial_failure_preserves_successful_results(stage4_root, monkeypatch):
    def model(name, data):
        if name != 'random_forest':
            raise RuntimeError('CUDA out of memory')
        return lambda frame: np.full(len(frame), 0.5), [], lambda path: path.write_bytes(b'model')
    monkeypatch.setattr(stage4, 'train_model', model)
    monkeypatch.setattr(stage4, 'preflight_backends', lambda models: dict.fromkeys(models, True))
    with pytest.raises(RuntimeError, match='CUDA out of memory'):
        stage4.run(stage4_root, execute=True)
    result_dir = next((stage4_root / 'results/models/USTC-TFC2016').iterdir())
    assert {p.name for p in result_dir.iterdir()} == set(RESULT_SCHEMAS)
    assert pd.read_csv(result_dir / 'metrics.csv').model.tolist() == ['random_forest']
    assert pd.read_csv(result_dir / 'training_history.csv').empty


def test_stage4_dependency_errors_and_torch_build_tag(monkeypatch):
    monkeypatch.setattr(stage4, 'version', lambda name: '2.14.0+cu130')
    sentinel = object()
    monkeypatch.setattr(stage4.importlib, 'import_module', lambda name: sentinel)
    assert stage4.require_package('torch') is sentinel
    monkeypatch.setattr(stage4, 'version', lambda name: '2.13.0')
    with pytest.raises(RuntimeError, match='Incompatible torch'):
        stage4.require_package('torch')
    def missing(name):
        raise stage4.PackageNotFoundError(name)
    monkeypatch.setattr(stage4, 'version', missing)
    with pytest.raises(RuntimeError, match='Missing dependency: torch==2.14.0'):
        stage4.require_package('torch')


@pytest.mark.parametrize('backend', ['random_forest', 'xgboost', 'lightgbm'])
def test_stage4_gpu_smoke_actually_fits_tiny_synthetic_data(monkeypatch, backend):
    events = []
    cp = fake_cupy()
    class Classifier:
        classes_ = cp.asarray([0, 1])
        booster_ = SimpleNamespace(params={'device_type': 'cuda'})
        def __init__(self, **params):
            assert params['n_estimators'] == 2 and params['random_state'] == 42
            if backend == 'random_forest':
                assert params['max_depth'] is None and params['output_type'] == 'cupy'
            else:
                assert params['device' if backend == 'xgboost' else 'device_type'] == 'cuda'
        def fit(self, x, y):
            assert x.shape == (128, 4) and set(y) == {0, 1}
            if backend == 'xgboost':
                assert isinstance(x, FakeGPUArray) and x.dtype == np.float32
                assert isinstance(y, FakeGPUArray) and y.dtype == np.int32
            if backend == 'random_forest':
                assert isinstance(x, FakeGPUArray) and y.dtype == np.int32
            events.append('fit')
        def predict_proba(self, x):
            events.append('predict')
            return cp.asarray(np.tile([0.5, 0.5], (len(x), 1)))
        def get_booster(self):
            return SimpleNamespace(save_config=lambda: json.dumps({'learner': {'generic_param': {'device': 'cuda:0'}}}))
    monkeypatch.setattr(stage4, 'require_package', lambda name: SimpleNamespace(XGBClassifier=Classifier, LGBMClassifier=Classifier))
    monkeypatch.setattr(stage4, 'load_cuml_rf', lambda: Classifier)
    monkeypatch.setattr(stage4, 'load_cupy', lambda: cp)
    stage4.gpu_smoke(backend)
    assert events == (['fit', 'predict'] if backend == 'random_forest' else ['fit'])


def test_stage4_torch_smoke_runs_cuda_operation_and_rejects_missing_cuda(monkeypatch):
    events = []
    class Tensor:
        device = SimpleNamespace(type='cuda')
        def __matmul__(self, other):
            events.append('matmul')
            return self
    def ones(shape, device):
        assert shape == (2, 2) and device == 'cuda'
        return Tensor()
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True,
        get_device_name=lambda index: 'Synthetic NVIDIA GPU', synchronize=lambda: events.append('sync')),
        version=SimpleNamespace(cuda='13.0'), ones=ones,
        isfinite=lambda result: SimpleNamespace(all=lambda: SimpleNamespace(item=lambda: True)))
    monkeypatch.setattr(stage4, 'require_package', lambda name: fake)
    monkeypatch.setattr(stage4, 'seed_everything', lambda *args: None)
    stage4.gpu_smoke('torch')
    assert events == ['matmul', 'sync']
    fake.cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match='CUDA is unavailable'):
        stage4.gpu_smoke('torch')


def test_stage4_cupy_unavailable_fails_clearly(monkeypatch):
    def unavailable(name):
        raise ImportError('CuPy missing')
    monkeypatch.setattr(stage4.importlib, 'import_module', unavailable)
    with pytest.raises(RuntimeError, match='CuPy CUDA unavailable; no CPU fallback'):
        stage4.load_cupy()


@pytest.mark.parametrize('name', ['1d_cnn', 'cnn_lstm'])
def test_stage4_neural_restored_model_uses_cuda_for_all_splits(stage4_root, monkeypatch, name):
    from contextlib import nullcontext
    data = stage4.load_data(stage4_root)
    events, loaders = [], []
    class Tensor:
        def __init__(self, values, device='cpu'):
            self.values = np.asarray(values)
            self.device = SimpleNamespace(type=device)
        def to(self, device, **kwargs):
            assert device == 'cuda'
            return Tensor(self.values, device)
        def unsqueeze(self, axis):
            return Tensor(np.expand_dims(self.values, axis), self.device.type)
        def squeeze(self, axis):
            return Tensor(np.squeeze(self.values, axis), self.device.type)
        def __len__(self):
            return len(self.values)
        def detach(self):
            return self
        def cpu(self):
            return Tensor(self.values)
        def numpy(self):
            return self.values
    class Model:
        training = True
        restored = False
        def to(self, device):
            assert device == 'cuda'
            return self
        def parameters(self):
            return [Tensor([1], 'cuda')]
        def train(self, training):
            self.training = training
        def eval(self):
            self.training = False
        def state_dict(self):
            return {'weight': Tensor([1], 'cuda')}
        def load_state_dict(self, state):
            assert not self.restored
            assert events == [(True, False, 0), (False, False, 0)] * 30
            self.restored = True
        def __call__(self, x):
            assert x.device.type == 'cuda' and x.values.shape == (12, 1, 50)
            events.append((self.training, self.restored, float(x.values[0, 0, 0])))
            return Tensor(np.zeros((len(x), 1)), 'cuda')
    model = Model()
    def loss(logits, y):
        assert logits.device.type == y.device.type == 'cuda'
        return SimpleNamespace(item=lambda: 0.5, backward=lambda: None)
    def loader(dataset, **kwargs):
        assert kwargs['batch_size'] == 1024 and kwargs['drop_last'] is False
        loaders.append(kwargs['shuffle'])
        return [dataset]
    def optimizer(parameters, **kwargs):
        assert kwargs == {'lr': 0.001, 'weight_decay': 0.0001}
        return SimpleNamespace(zero_grad=lambda **kwargs: None, step=lambda: None)
    saved = []
    fake = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True),
        nn=SimpleNamespace(BCEWithLogitsLoss=lambda: loss), optim=SimpleNamespace(AdamW=optimizer),
        Generator=lambda: SimpleNamespace(manual_seed=lambda seed: seed),
        from_numpy=Tensor, tensor=lambda values, **kwargs: Tensor(values), float32=np.float32,
        utils=SimpleNamespace(data=SimpleNamespace(TensorDataset=lambda *tensors: tensors, DataLoader=loader)),
        set_grad_enabled=lambda value: nullcontext(), no_grad=nullcontext,
        sigmoid=lambda tensor: Tensor(np.full(len(tensor), 0.5), tensor.device.type),
        save=lambda checkpoint, path: saved.append(checkpoint))
    monkeypatch.setattr(stage4, 'require_package', lambda name: fake)
    monkeypatch.setattr(stage4, 'seed_everything', lambda *args: None)
    monkeypatch.setattr(stage4, 'build_neural_model', lambda *args: model)
    predict, history, save = stage4.train_model(name, data)
    assert history == [dict(step=i, train_loss=0.5, validation_loss=0.5) for i in range(1, 31)]
    predict(data.x_validation)
    create_test_split(stage4_root)
    x_test, _ = stage4.load_test(stage4_root, data)
    predict(x_test)
    save(stage4_root / stage4.ARTIFACTS[name])
    assert events == [(True, False, 0), (False, False, 0)] * 30 + [(False, True, 0), (False, True, 10)]
    assert loaders == [True, False, False, False]
    assert saved[0]['best_epoch'] == 1 and saved[0]['seed'] == 42
    fake.cuda.is_available = lambda: False
    with pytest.raises(RuntimeError, match='CUDA required, no CPU fallback'):
        stage4.train_model(name, data)


def test_stage4_test_failure_never_retries_or_retrains(stage4_root, monkeypatch):
    create_test_split(stage4_root)
    events = []
    def train(name, data):
        events.append('fit')
        def predict(frame):
            if frame is not data.x_validation:
                events.append('test')
                raise RuntimeError('Synthetic TEST inference failure')
            events.append('validation')
            return np.full(len(frame), 0.5)
        return predict, [], lambda path: path.write_bytes(b'model')
    monkeypatch.setattr(stage4, 'preflight_backends', lambda names: dict.fromkeys(names, True))
    monkeypatch.setattr(stage4, 'train_model', train)
    with pytest.raises(RuntimeError, match='no retry or retraining'):
        stage4.run(stage4_root, model='random_forest', execute=True)
    assert events == ['fit', 'validation', 'test']


@pytest.mark.parametrize('column', ['semantic_family', 'capture_group', 'source_pcap'])
def test_stage4_test_overlap_rejected(stage4_root, column):
    data = stage4.load_data(stage4_root)
    create_test_split(stage4_root)
    path = stage4_root / 'data/processed/USTC-TFC2016/test.csv'
    frame = pd.read_csv(path)
    frame.loc[0, column] = next(iter(data.provenance[column]))
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='TRAIN/validation/TEST overlap'):
        stage4.load_test(stage4_root, data)


# Frozen-run integrity audit: all fixtures below are synthetic temporary data.
from src.evaluation import ustc_integrity_audit as integrity


@pytest.fixture
def integrity_evidence(tmp_path, monkeypatch):
    features = [c for c in integrity.CANDIDATES if c not in integrity.REMOVED]
    raw_rows, split_names = [], []
    for split, classes in integrity.FAMILIES.items():
        for label, families in classes.items():
            for family in sorted(families):
                for within in range(3):
                    index = len(raw_rows) + 1
                    row = {c: (0.0 if c in integrity.REMOVED else index * (i + 1) / 100)
                           for i, c in enumerate(integrity.CANDIDATES)}
                    row.update({c: 0 for c in integrity.EXCLUDED})
                    row.update(binary_label=label, semantic_family=family,
                               capture_group=family + '/capture', source_pcap=family + '/capture.pcap')
                    raw_rows.append(row)
                    split_names.append(split)
    flows = pd.DataFrame(raw_rows)
    mask = np.array(split_names) == 'train'
    parameters = []
    for c in integrity.CANDIDATES:
        keep = c in features
        values = flows.loc[mask, c].to_numpy()
        parameters.append(dict(feature_name=c, retained=keep, drop_reason='' if keep else 'zero_variance_train',
            train_median=np.median(values), train_mean=np.mean(values), train_std=np.std(values) if keep else 0,
            final_feature_order=features.index(c) + 1 if keep else None))
    parameters = pd.DataFrame(parameters, columns=integrity.PARAMETER_COLUMNS)
    p = parameters.set_index('feature_name')
    frames = {}
    for split in integrity.SPLITS:
        source = flows.loc[np.array(split_names) == split].reset_index(drop=True)
        frame = (source[features] - p.loc[features, 'train_mean']) / p.loc[features, 'train_std']
        frame['target_binary'] = source.binary_label.map({'benign': 0, 'malware': 1})
        frame[integrity.PROVENANCE] = source[integrity.PROVENANCE]
        frames[split] = frame
    monkeypatch.setattr(integrity, 'ROWS', {s: len(f) for s, f in frames.items()})
    results = {name: [] for name in integrity.RESULT_SCHEMAS}
    for model in integrity.ARTIFACTS:
        for split in ('validation', 'test'):
            y = frames[split].target_binary.to_numpy()
            tables = stage4.evaluate_predictions(y, np.where(y == 1, 0.8, 0.2))
            for name, rows in tables.items():
                if name == 'training_history.csv':
                    continue
                for row in rows:
                    result = dict(row, dataset=integrity.DATASET, model=model, split=split,
                                  run_id=integrity.RUN, timestamp_utc='2026-09-19T18:39:45Z')
                    if name == 'metrics.csv':
                        result['device'] = integrity.DEVICES[model]
                    results[name].append(result)
    for model, (count, best, loss) in integrity.HISTORY_REFERENCES.items():
        for step in range(1, count + 1):
            results['training_history.csv'].append(dict(dataset=integrity.DATASET, model=model, step=step,
                train_loss=0.2, validation_loss=loss if step == best else loss + 1,
                run_id=integrity.RUN, timestamp_utc='2026-09-19T18:39:45Z'))
    results = {name: pd.DataFrame(rows, columns=integrity.RESULT_SCHEMAS[name]) for name, rows in results.items()}
    base = tmp_path / 'data/processed' / integrity.DATASET
    base.mkdir(parents=True)
    for split, frame in frames.items():
        frame.to_csv(base / f'{split}.csv', index=False)
    parameters.to_csv(base / 'preprocessing_parameters.csv', index=False)
    interim = tmp_path / 'data/interim' / integrity.DATASET
    interim.mkdir(parents=True)
    flows.to_csv(interim / 'flows.csv', index=False)
    result_dir = tmp_path / 'results/models' / integrity.DATASET / integrity.RUN
    model_dir = tmp_path / 'models' / integrity.DATASET / integrity.RUN
    result_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    for name, frame in results.items():
        frame.to_csv(result_dir / name, index=False)
    for name in integrity.ARTIFACTS.values():
        (model_dir / name).write_bytes(b'opaque synthetic model -- must never be loaded')
    project = Path(__file__).resolve().parents[1]
    for relative in ('src/models/ustc_train_models.py', 'src/data/ustc_preprocess.py'):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text((project / relative).read_text(encoding='utf-8'), encoding='utf-8')
    return SimpleNamespace(root=tmp_path, frames=frames, flows=flows, parameters=parameters,
                           features=features, results=results, models=model_dir)


def integrity_failures(audit):
    return [r for r in audit.tables['ustc_integrity_audit_checklist.csv'] if r['status'] == 'FAIL']


@pytest.mark.parametrize('column,category', [('semantic_family', 'B'), ('capture_group', 'C'), ('source_pcap', 'C')])
def test_integrity_detects_provenance_overlap(integrity_evidence, column, category):
    e = integrity_evidence
    e.frames['test'].loc[0, column] = e.frames['train'].loc[0, column]
    audit = integrity.Audit()
    integrity.audit_provenance(audit, e.frames)
    assert any(r['category'] == category and 'overlap' in r['check_name'] for r in integrity_failures(audit))


@pytest.mark.parametrize('name', ['src_ip', 'src_port', 'flow_id', 'target_binary', 'target_copy',
                                  'split', 'semantic_family', 'capture_group', 'application_name', 'timestamp'])
def test_integrity_predictor_leakage_allowlist(name):
    assert integrity.predictor_inventory([name])[0]['status'] == 'UNSAFE'
    assert integrity.predictor_inventory(['bidirectional_packets'])[0]['status'] == 'SAFE'


def test_integrity_exact_rows_and_input_sanity(integrity_evidence):
    e = integrity_evidence
    e.frames['test'] = e.frames['test'].iloc[:-1].copy()
    e.frames['test'].loc[0, e.features[0]] = np.inf
    audit = integrity.Audit()
    integrity.audit_splits(audit, e.frames, e.features)
    names = {r['check_name'] for r in integrity_failures(audit)}
    assert 'test_rows' in names and 'total_rows' in names and 'test_no_positive_infinity' in names


def test_integrity_train_statistics_reject_global_preprocessing(integrity_evidence):
    e = integrity_evidence
    audit = integrity.Audit()
    integrity.audit_preprocessing(audit, e.parameters, e.flows, e.frames, e.features, list(e.flows.columns))
    assert not integrity_failures(audit)
    c = e.features[0]
    e.parameters.loc[e.parameters.feature_name == c, 'train_mean'] = e.flows[c].mean()
    changed = integrity.Audit()
    integrity.audit_preprocessing(changed, e.parameters, e.flows, e.frames, e.features, list(e.flows.columns))
    assert any(r['check_name'] == c + '_train_mean' for r in integrity_failures(changed))


def test_integrity_source_contracts_and_test_misuse(integrity_evidence):
    e = integrity_evidence
    pre = (e.root / 'src/data/ustc_preprocess.py').read_text()
    train = (e.root / 'src/models/ustc_train_models.py').read_text()
    actual = integrity.source_contracts(pre, train)
    assert all(actual.values()), actual
    changes = [
        ("train = clean.iloc[parts['train']]", 'train = clean', 'preprocessing_train_only', 'pre'),
        ("for split in ('train', 'validation'):", "for split in ('train', 'validation', 'test'):", 'initial_load_excludes_test', 'train'),
        ('loader(data.x_train, data.y_train, True)', 'loader(data.x_test, data.y_test, True)', 'fitting_inputs', 'train'),
        ('(data.x_validation, data.y_validation)', '(data.x_test, data.y_test)', 'fitting_inputs', 'train'),
        ('validation_loss < best_loss', 'train_loss < best_loss', 'validation_checkpoint_selection', 'train'),
        ('loss.backward()', 'loss.backward(); optimizer.step()', 'gradient_isolation', 'train'),
        ("record(name, 'test', y_test, predict(x_test))", "train_model(name, data)\n                record(name, 'test', y_test, predict(x_test))", 'finalized_before_test', 'train'),
        ('THRESHOLD = 0.5', 'THRESHOLD = 0.6', 'fixed_seed_threshold_contract', 'train'),
    ]
    for before, after, key, module in changes:
        original = pre if module == 'pre' else train
        assert before in original
        changed = original.replace(before, after)
        evidence = integrity.source_contracts(changed if module == 'pre' else pre, changed if module == 'train' else train)
        assert evidence[key] is False, (key, evidence)
    # Misleading comments cannot conceal executable misuse.
    changed = pre.replace("train = clean.iloc[parts['train']]", "train = clean  # fit using TRAIN only")
    assert not integrity.source_contracts(changed, train)['preprocessing_train_only']


def test_integrity_duplicate_evidence_without_cartesian_expansion(integrity_evidence):
    e = integrity_evidence
    left = e.frames['train'].iloc[:3].copy()
    right = e.frames['test'].iloc[:2].copy()
    left[e.features] = left.iloc[0][e.features].to_numpy()
    right[e.features] = left.iloc[0][e.features].to_numpy()
    report = integrity.duplicate_summary(left, right, e.features)
    assert report['shared_distinct_rows'] == 1
    assert (report['left_matching_rows'], report['right_matching_rows'], report['matching_row_pairs']) == (3, 2, 6)
    assert 'left_provenance' in report['examples'] and 'right_provenance' in report['examples']
    e.frames['train'], e.frames['test'] = left, right
    audit = integrity.Audit()
    integrity.audit_duplicates(audit, e.frames, e.features)
    assert not integrity_failures(audit)  # disjoint provenance: numeric matches are warnings
    assert any(r['status'] == 'WARN' for r in audit.tables['ustc_cross_split_duplicate_check.csv'])
    assert any('within_split' in r['check_name'] for r in audit.tables['ustc_integrity_audit_checklist.csv'])


def test_integrity_confusion_recomputation_and_result_corruption(integrity_evidence):
    m = integrity.confusion_metrics([1, 1, 0, 2])
    assert m['accuracy'] == .75 and m['precision'] == pytest.approx(2 / 3) and m['f1'] == .8
    e = integrity_evidence
    audit = integrity.Audit()
    integrity.audit_results(audit, e.results, e.frames)
    assert not integrity_failures(audit)
    e.results['metrics.csv'].loc[0, 'accuracy'] = .5
    bad = integrity.Audit()
    integrity.audit_results(bad, e.results, e.frames)
    assert any(r['category'] == 'J' and r['check_name'].endswith('_accuracy') for r in integrity_failures(bad))


@pytest.mark.parametrize('filename,column,row,value', [
    ('roc_curve.csv', 'threshold', 1, 1.1), ('roc_curve.csv', 'tpr', 1, np.nan),
    ('pr_curve.csv', 'precision', 0, np.inf), ('pr_curve.csv', 'threshold', 0, -.1),
    ('calibration_bins.csv', 'mean_probability', 3, 1.1), ('metrics.csv', 'brier_score', 0, .1),
])
def test_integrity_invalid_probability_or_inconsistent_metric(integrity_evidence, filename, column, row, value):
    e = integrity_evidence
    e.results[filename].loc[row, column] = value
    audit = integrity.Audit()
    integrity.audit_results(audit, e.results, e.frames)
    assert integrity_failures(audit)


def test_integrity_neural_history_requires_thirty_epochs(integrity_evidence):
    e = integrity_evidence
    audit = integrity.Audit()
    integrity.audit_histories(audit, e.results['training_history.csv'], e.models)
    assert not integrity_failures(audit)
    history = e.results['training_history.csv']
    history = history.loc[~((history.model == '1d_cnn') & (history.step == 30))]
    changed = integrity.Audit()
    integrity.audit_histories(changed, history, e.models)
    assert any(r['check_name'] == '1d_cnn_recorded_steps' for r in integrity_failures(changed))


@pytest.mark.parametrize('with_warnings', [False, True])
def test_integrity_end_to_end_read_only_overwrite_and_no_inference(integrity_evidence, monkeypatch, with_warnings):
    import pickle
    e = integrity_evidence
    if with_warnings:
        # Exercise held-out constants and numeric duplicates together, preserving
        # the synthetic frozen TRAIN statistics and all provenance identities.
        p = e.parameters.set_index('feature_name').loc['ip_version']
        for split in ('validation', 'test'):
            mask = e.flows.semantic_family.isin(e.frames[split].semantic_family)
            value = e.flows.loc[mask, 'ip_version'].iloc[0] if split == 'validation' else e.flows.ip_version.iloc[0]
            e.flows.loc[mask, 'ip_version'] = value
            e.frames[split]['ip_version'] = (value - p.train_mean) / p.train_std
        test_index = e.flows.index[e.flows.semantic_family.isin(e.frames['test'].semantic_family)][0]
        e.flows.loc[test_index, e.features] = e.flows.loc[0, e.features].to_numpy()
        e.frames['test'].loc[0, e.features] = e.frames['train'].loc[0, e.features].to_numpy()
        e.flows.to_csv(e.root / 'data/interim' / integrity.DATASET / 'flows.csv', index=False)
        for split, frame in e.frames.items():
            frame.to_csv(e.root / 'data/processed' / integrity.DATASET / f'{split}.csv', index=False)
    def prohibited(*args, **kwargs):
        pytest.fail('The integrity audit must never fit, preprocess, deserialize or infer')
    monkeypatch.setattr(stage4, 'train_model', prohibited)
    monkeypatch.setattr(stage4, 'train_neural', prohibited)
    monkeypatch.setattr(stage4, 'positive_probability', prohibited)
    monkeypatch.setattr(pickle, 'load', prohibited)
    from src.data import ustc_preprocess
    monkeypatch.setattr(ustc_preprocess, 'fit_transform', prohibited)
    from sklearn.preprocessing import StandardScaler
    monkeypatch.setattr(StandardScaler, 'fit', prohibited)
    original_open = Path.open
    def guarded_open(path, mode='r', *args, **kwargs):
        if path.parent == e.models:
            pytest.fail('Model artifact contents must never be opened')
        if any(flag in mode for flag in ('w', 'a', '+')) and ('data' in path.parts or integrity.RUN in path.parts):
            pytest.fail('Authoritative inputs must never be written')
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    assert integrity.run_audit(e.root) == 0
    output = e.root / 'results/audits/USTC-TFC2016/integrity_audit'
    assert {p.name for p in output.iterdir()} == set(integrity.OUTPUT_NAMES)
    metadata_path = output / 'ustc_integrity_audit_metadata.csv'
    metadata = pd.read_csv(metadata_path)
    assert metadata.overall_status.tolist() == ['PASS']
    assert metadata.finalization_authorized.tolist() == ['YES']
    findings = (output / integrity.FINDINGS).read_text(encoding='utf-8')
    assert 'NO DIRECT LEAKAGE EVIDENCE FOUND' in findings
    assert integrity.PROTOCOL_EVIDENCE in findings
    checklist = pd.read_csv(output / 'ustc_integrity_audit_checklist.csv').set_index('check_name')
    assert checklist.loc['historical_execution_evidence_limit', 'status'] == 'WARN'
    assert checklist.loc['near_perfect_tree_investigation', 'status'] == 'PASS'
    if with_warnings:
        assert checklist.loc['validation_constant_predictors_observed', 'status'] == 'WARN'
        assert checklist.loc['test_constant_predictors_observed', 'status'] == 'WARN'
        assert 'Numeric duplicates exist across independently sourced captures' in findings
        duplicates = pd.read_csv(output / 'ustc_cross_split_duplicate_check.csv')
        assert len(duplicates) == 6
        matched = duplicates.loc[duplicates.numeric_duplicate == 'YES']
        assert not matched.empty and matched.provenance_overlap.eq('NO').all()
        assert matched.leakage_interpretation.eq('WARN_ONLY').all()
        assert matched.status.eq('WARN').all()
    assert checklist.loc[checklist.category == 'F', 'status'].eq('PASS').all()
    unrelated = output / 'unrelated.txt'
    unrelated.write_text('keep me')
    (output / integrity.FINDINGS).write_text('stale audit findings')
    assert integrity.run_audit(e.root) == 0
    assert (output / integrity.FINDINGS).read_text(encoding='utf-8') == findings
    assert unrelated.read_text() == 'keep me'
    assert {p.name for p in output.iterdir()} == set(integrity.OUTPUT_NAMES) | {'unrelated.txt'}
    assert list(output.parent.iterdir()) == [output]
    # Each identity component independently prevents accidental overwrite.
    before = (output / integrity.FINDINGS).read_bytes()
    for key in ('dataset', 'task', 'authoritative_run'):
        changed = metadata.copy()
        changed.loc[0, key] = 'different'
        changed.to_csv(metadata_path, index=False)
        with pytest.raises(FileExistsError, match='identity differs'):
            integrity.run_audit(e.root)
        assert (output / integrity.FINDINGS).read_bytes() == before
        metadata.to_csv(metadata_path, index=False)


def test_integrity_default_cli_never_runs_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(integrity, 'run_audit', lambda *args: pytest.fail('Explicit execution required'))
    assert integrity.main(['--project-root', str(tmp_path)]) == 0
    assert list(tmp_path.iterdir()) == []


def test_integrity_missing_inputs_still_publish_failure(tmp_path):
    assert integrity.run_audit(tmp_path) == 1
    output = tmp_path / 'results/audits/USTC-TFC2016/integrity_audit'
    assert {p.name for p in output.iterdir()} == set(integrity.OUTPUT_NAMES)
    metadata = pd.read_csv(output / 'ustc_integrity_audit_metadata.csv')
    assert metadata.overall_status.tolist() == ['FAIL']
    assert metadata.finalization_authorized.tolist() == ['NO']


def test_integrity_duplicate_column_names_are_rejected(tmp_path):
    path = tmp_path / 'bad.csv'
    path.write_text('x,x\n1,2\n')
    with pytest.raises(ValueError, match='Duplicate column'):
        integrity.read_csv(path)


def test_integrity_publication_failure_cannot_leave_pass_metadata(tmp_path, monkeypatch):
    audit = integrity.Audit()
    audit.check('A', 'synthetic', True)
    findings = integrity.complete_report(audit, {}, [])
    directory = tmp_path / 'integrity_audit'
    integrity.publish_report(directory, audit, findings)
    # Test publisher behavior independently from the audit verdict.
    audit.tables['ustc_integrity_audit_metadata.csv'][0].update(overall_status='PASS', finalization_authorized='YES')
    original = integrity.os.replace
    def fail_during_publication(source, target):
        if Path(target).name == 'ustc_split_family_assignment.csv':
            raise OSError('Synthetic interrupted publication')
        return original(source, target)
    monkeypatch.setattr(integrity.os, 'replace', fail_during_publication)
    with pytest.raises(OSError, match='interrupted'):
        integrity.publish_report(directory, audit, findings)
    metadata = pd.read_csv(directory / 'ustc_integrity_audit_metadata.csv')
    assert metadata.overall_status.tolist() == ['FAIL']
    assert metadata.finalization_authorized.tolist() == ['NO']
    assert not list(directory.glob('*.tmp'))
    assert {p.name for p in directory.iterdir()} == set(integrity.OUTPUT_NAMES)


def test_integrity_unverified_existing_outputs_not_overwritten(tmp_path):
    directory = tmp_path / 'integrity_audit'
    directory.mkdir()
    findings = directory / integrity.FINDINGS
    findings.write_text('unverified audit')
    with pytest.raises(FileExistsError, match='without identity metadata'):
        integrity.check_destination(directory, dict(dataset=integrity.DATASET, task=integrity.TASK, authoritative_run=integrity.RUN))
    assert findings.read_text() == 'unverified audit'


def test_integrity_source_rejects_mutating_train_inputs(integrity_evidence):
    root = integrity_evidence.root
    pre = (root / 'src/data/ustc_preprocess.py').read_text()
    train = (root / 'src/models/ustc_train_models.py').read_text()
    mutated = train.replace('    seed_everything()\n', '    seed_everything()\n    data.x_train = data.x_validation\n')
    assert not integrity.source_contracts(pre, mutated)['fitting_inputs']
    mutated_pre = pre.replace('    medians = train.median()', '    medians = train.median()\n    medians.update(clean.median())')
    assert not integrity.source_contracts(mutated_pre, train)['preprocessing_train_only']


@pytest.mark.parametrize('split,status,check_name', [
    ('train', 'FAIL', 'train_final_predictors_nonconstant'),
    ('validation', 'WARN', 'validation_constant_predictors_observed'),
    ('test', 'WARN', 'test_constant_predictors_observed'),
])
def test_integrity_constant_policy_is_train_only(integrity_evidence, split, status, check_name):
    e = integrity_evidence
    e.frames[split]['ip_version'] = 0.0
    originals = {s: frame.copy(deep=True) for s, frame in e.frames.items()}
    audit = integrity.Audit()
    integrity.audit_splits(audit, e.frames, e.features)
    check = next(r for r in audit.tables['ustc_integrity_audit_checklist.csv'] if r['check_name'] == check_name)
    assert check['status'] == status and 'ip_version' in check['details']
    assert audit.passed == (split != 'train')
    assert len(e.features) == 50
    for s in integrity.SPLITS:
        pd.testing.assert_frame_equal(e.frames[s], originals[s])


def test_integrity_preprocessing_copies_readonly_pandas_arrays(integrity_evidence, monkeypatch):
    e = integrity_evidence
    flows_before, parameters_before = e.flows.copy(deep=True), e.parameters.copy(deep=True)
    original = pd.Series.to_numpy
    copies = []
    def readonly_unless_copied(series, *args, **kwargs):
        array = original(series, *args, **kwargs)
        if kwargs.get('copy', False):
            assert array.flags.writeable
            copies.append(series.name)
        else:
            array = array.view()
            array.setflags(write=False)
        return array
    monkeypatch.setattr(pd.Series, 'to_numpy', readonly_unless_copied)
    assert not e.flows[e.features[0]].to_numpy(dtype=float).flags.writeable
    audit = integrity.Audit()
    audit.section('F', lambda: integrity.audit_preprocessing(
        audit, e.parameters, e.flows, e.frames, e.features, list(e.flows.columns)))
    assert not integrity_failures(audit)
    assert set(copies) == set(integrity.CANDIDATES)
    assert all(r['status'] == 'PASS' for r in audit.tables['ustc_preprocessing_leakage_check.csv'])
    pd.testing.assert_frame_equal(e.flows, flows_before)
    pd.testing.assert_frame_equal(e.parameters, parameters_before)


@pytest.mark.parametrize('field,check_suffix', [
    ('train_median', 'train_median'), ('train_mean', 'train_mean'),
    ('train_std', 'train_scale_variance'), ('retained', 'train_only_retention'),
])
def test_integrity_independently_checks_each_saved_train_parameter(integrity_evidence, field, check_suffix):
    e = integrity_evidence
    feature = e.features[0]
    index = e.parameters.index[e.parameters.feature_name == feature][0]
    if field == 'retained':
        e.parameters.loc[index, field] = False
    else:
        e.parameters.loc[index, field] += 100
    audit = integrity.Audit()
    integrity.audit_preprocessing(audit, e.parameters, e.flows, e.frames, e.features, list(e.flows.columns))
    assert any(r['check_name'] == feature + '_' + check_suffix for r in integrity_failures(audit))


def test_integrity_recomputed_statistics_ignore_held_out_values(integrity_evidence):
    e = integrity_evidence
    feature = e.features[0]
    held_out = ~e.flows.semantic_family.isin(e.frames['train'].semantic_family)
    e.flows.loc[held_out, feature] = 1e12
    audit = integrity.Audit()
    integrity.audit_preprocessing(audit, e.parameters, e.flows, e.frames, e.features, list(e.flows.columns))
    checks = {r['check_name']: r['status'] for r in audit.tables['ustc_preprocessing_leakage_check.csv']}
    for suffix in ('train_median', 'train_mean', 'train_scale_variance', 'train_only_retention'):
        assert checks[feature + '_' + suffix] == 'PASS'
    # The independently saved held-out data now differs, and must still be detected.
    assert checks['validation_persisted_transform_matches'] == 'FAIL'
    assert checks['test_persisted_transform_matches'] == 'FAIL'


@pytest.mark.parametrize('overlap', [None, 'capture_group', 'source_pcap'])
def test_integrity_numeric_duplicates_classified_by_capture_identity(integrity_evidence, overlap):
    e = integrity_evidence
    e.frames['test'].loc[0, e.features] = e.frames['train'].loc[0, e.features].to_numpy()
    if overlap:
        e.frames['test'].loc[0, overlap] = e.frames['train'].loc[0, overlap]
    audit = integrity.Audit()
    integrity.audit_duplicates(audit, e.frames, e.features)
    rows = audit.tables['ustc_cross_split_duplicate_check.csv']
    assert len(rows) == 6
    pair = [r for r in rows if (r['left_split'], r['right_split']) == ('train', 'test')]
    assert len(pair) == 2
    for row in pair:
        assert row['numeric_duplicate'] == 'YES' and row['shared_distinct_rows'] >= 1
        assert 'left_provenance' in row['examples']
        assert row['provenance_overlap'] == ('YES' if overlap else 'NO')
        assert row['status'] == ('FAIL' if overlap else 'WARN')
        assert row['leakage_interpretation'] == ('CRITICAL_FAIL' if overlap else 'WARN_ONLY')
    assert audit.passed == (overlap is None)


@pytest.mark.parametrize('category', list('ABCDEFGHIJK'))
@pytest.mark.parametrize('failure_mode', ['failure', 'missing'])
def test_integrity_section_l_requires_completed_critical_checks(category, failure_mode):
    audit = integrity.Audit()
    for current in 'ABCDEFGHIJK':
        if current != category or failure_mode != 'missing':
            audit.check(current, 'synthetic_' + current, current != category)
    audit.check('G', 'historical_execution_evidence_limit', False, integrity.LIMITATION, warning=True)
    findings = integrity.complete_report(audit, {}, [])
    metadata = audit.tables['ustc_integrity_audit_metadata.csv'][0]
    assert metadata['overall_status'] == 'FAIL' and metadata['finalization_authorized'] == 'NO'
    assert any(r['category'] == 'L' for r in integrity_failures(audit))
    assert 'NO DIRECT LEAKAGE EVIDENCE FOUND' not in findings
    assert integrity.PROTOCOL_EVIDENCE not in findings


from src.evaluation import ustc_finalize_results as finalizer


@pytest.fixture
def finalization_evidence(integrity_evidence):
    e = integrity_evidence
    audit_dir = e.root / 'results/audits' / finalizer.DATASET / 'integrity_audit'
    audit_dir.mkdir(parents=True)
    pd.DataFrame([dict(dataset=finalizer.DATASET, task='integrity_audit', authoritative_run=finalizer.RUN,
        overall_status='PASS', finalization_authorized='YES', predictor_count=50, seed=42, threshold=0.5)]).to_csv(
        audit_dir / 'ustc_integrity_audit_metadata.csv', index=False)
    pd.DataFrame([dict(category='A', check_name='synthetic_pass', status='PASS', details='synthetic'),
                  dict(category='G', check_name='historical_execution_evidence_limit', status='WARN', details='synthetic')]
                 ).to_csv(audit_dir / 'ustc_integrity_audit_checklist.csv', index=False)
    for index, model in enumerate(finalizer.MODELS):
        for split in finalizer.SPLITS:
            counts = ([48000, 7 + index, 11 + index, 48983 - 2 * index] if split == 'validation'
                      else [48000, 13 + index, 17 + index, 50225 - 2 * index])
            for (actual, predicted), count in zip(finalizer.CLASS_PAIRS, counts):
                frame = e.results['confusion_matrix.csv']
                mask = ((frame.model == model) & (frame.split == split)
                        & (frame.actual_class == actual) & (frame.predicted_class == predicted))
                frame.loc[mask, 'count'] = count
            metrics = e.results['metrics.csv']
            mask = (metrics.model == model) & (metrics.split == split)
            for key, value in integrity.confusion_metrics(counts).items():
                metrics.loc[mask, key] = value
    result_dir = e.root / 'results/models' / finalizer.DATASET / finalizer.RUN
    for name in finalizer.INPUT_SCHEMAS:
        e.results[name].to_csv(result_dir / name, index=False)
    e.audit_dir, e.result_dir = audit_dir, result_dir
    e.destination = e.root / 'results/final' / finalizer.DATASET / finalizer.TRAIN_FILE
    return e


FINAL_TRAIN_COLUMNS = """dataset authoritative_run task model device step train_loss validation_loss
is_best_validation_step best_validation_step best_validation_loss predictor_count seed threshold""".split()
FINAL_EVALUATION_COLUMNS = """dataset authoritative_run model device accuracy precision recall f1 weighted_f1 mcc
roc_auc auprc ece brier_score true_negative false_positive false_negative true_positive total_rows total_errors
predictor_count seed threshold integrity_audit_status""".split()


def test_finalization_three_csv_values_and_read_only_access(finalization_evidence, monkeypatch):
    import pickle
    from src.data import ustc_preprocess
    from sklearn.preprocessing import StandardScaler
    e = finalization_evidence
    sources = [e.result_dir / name for name in finalizer.INPUT_SCHEMAS] + [
        e.audit_dir / 'ustc_integrity_audit_metadata.csv', e.audit_dir / 'ustc_integrity_audit_checklist.csv']
    originals = {path: path.read_bytes() for path in sources}
    def prohibited(*args, **kwargs):
        pytest.fail('Finalization must not train, infer, load models, or fit preprocessing')
    monkeypatch.setattr(stage4, 'train_model', prohibited)
    monkeypatch.setattr(stage4, 'train_neural', prohibited)
    monkeypatch.setattr(stage4, 'positive_probability', prohibited)
    monkeypatch.setattr(ustc_preprocess, 'fit_transform', prohibited)
    monkeypatch.setattr(StandardScaler, 'fit', prohibited)
    monkeypatch.setattr(pickle, 'load', prohibited)
    monkeypatch.setattr(pickle, 'loads', prohibited)
    original_open = Path.open
    reads = []
    def guarded_open(path, mode='r', *args, **kwargs):
        assert path in sources, f'Unauthorized file access: {path}'
        assert not any(flag in mode for flag in ('w', 'a', '+')), 'Only atomic final outputs may be written'
        reads.append(path)
        return original_open(path, mode, *args, **kwargs)
    with monkeypatch.context() as guarded:
        guarded.setattr(Path, 'open', guarded_open)
        assert finalizer.finalize(e.root) == e.destination.parent
    assert set(reads) == set(sources)
    for path, content in originals.items():
        assert path.read_bytes() == content
    assert {p.name for p in e.destination.parent.iterdir()} == {
        'ustc_train_results.csv', 'ustc_validation_results.csv', 'ustc_test_results.csv'}
    assert list(e.destination.parent.parent.iterdir()) == [e.destination.parent]
    history = pd.read_csv(e.destination)
    assert list(history.columns) == FINAL_TRAIN_COLUMNS
    assert len(history) == 606
    assert 'random_forest' not in set(history.model)
    assert not set(finalizer.METRICS) & set(history.columns)
    assert history.task.eq('final_results').all()
    for model, (count, best) in finalizer.HISTORY_EXPECTED.items():
        group = history.loc[history.model == model]
        assert len(group) == count and group.step.tolist() == list(range(1, count + 1))
        assert group.best_validation_step.eq(best).all()
        assert group.loc[group.is_best_validation_step == 'YES', 'step'].tolist() == [best]
        assert group.loc[group.step != best, 'is_best_validation_step'].eq('NO').all()
        np.testing.assert_allclose(group.best_validation_loss, integrity.HISTORY_REFERENCES[model][2])
    source_history = {(r['model'], int(r['step'])): r for r in finalizer.read_csv(e.result_dir / 'training_history.csv')}
    for row in finalizer.read_csv(e.destination):
        source = source_history[row['model'], int(row['step'])]
        assert row['train_loss'] == source['train_loss'] and row['validation_loss'] == source['validation_loss']
        assert row['best_validation_loss'] == source_history[row['model'], int(row['best_validation_step'])]['validation_loss']
    source_metrics = {(r['model'], r['split']): r for r in finalizer.read_csv(e.result_dir / 'metrics.csv')}
    for split, filename in finalizer.EVALUATION_FILES.items():
        path = e.destination.parent / filename
        frame = pd.read_csv(path)
        assert list(frame.columns) == FINAL_EVALUATION_COLUMNS
        assert len(frame) == 5 and frame.model.tolist() == list(finalizer.MODELS)
        assert frame.integrity_audit_status.eq('PASS').all()
        for index, model in enumerate(finalizer.MODELS):
            row = frame.loc[frame.model == model].iloc[0]
            counts = ([48000, 7 + index, 11 + index, 48983 - 2 * index] if split == 'validation'
                      else [48000, 13 + index, 17 + index, 50225 - 2 * index])
            assert row[finalizer.CONFUSION_FIELDS].tolist() == counts
            assert row.total_rows == finalizer.SPLIT_ROWS[split]
            assert row.total_errors == counts[1] + counts[2]
        for row in finalizer.read_csv(path):
            assert all(row[key] == source_metrics[row['model'], split][key] for key in finalizer.METRICS)
    for filename in finalizer.OUTPUT_SCHEMAS:
        frame = pd.read_csv(e.destination.parent / filename)
        for key, value in dict(dataset=finalizer.DATASET, authoritative_run=finalizer.RUN,
                predictor_count=50, seed=42, threshold=.5).items():
            assert frame[key].eq(value).all()


@pytest.mark.parametrize('field,value', [
    ('overall_status', 'FAIL'), ('finalization_authorized', 'NO'),
    ('dataset', 'different'), ('task', 'different'), ('authoritative_run', 'different'),
    ('predictor_count', 49), ('seed', 43), ('threshold', .6),
])
def test_finalization_metadata_gate_rejects_before_result_reads(finalization_evidence, monkeypatch, field, value):
    e = finalization_evidence
    path = e.audit_dir / 'ustc_integrity_audit_metadata.csv'
    frame = pd.read_csv(path)
    frame.loc[0, field] = value
    frame.to_csv(path, index=False)
    monkeypatch.setattr(finalizer, 'authoritative_rows', lambda *a: pytest.fail('Gate must precede source reads'))
    with pytest.raises(ValueError, match='Integrity gate'):
        finalizer.finalize(e.root)
    assert not e.destination.parent.exists()


@pytest.mark.parametrize('status', ['FAIL', '', 'UNKNOWN'])
def test_finalization_checklist_fail_or_invalid_status_rejected(finalization_evidence, status):
    e = finalization_evidence
    path = e.audit_dir / 'ustc_integrity_audit_checklist.csv'
    rows = finalizer.read_csv(path)
    rows[0]['status'] = status
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match='checklist'):
        finalizer.finalize(e.root)
    assert not e.destination.exists()


@pytest.mark.parametrize('mutation', [
    'missing_metric', 'duplicate_metric', 'unknown_model', 'unknown_split', 'wrong_metric_run',
    'nonfinite_metric', 'wrong_threshold', 'wrong_seed', 'wrong_device', 'metric_disagreement',
    'wrong_confusion_total', 'duplicate_confusion_cell', 'negative_count', 'fractional_count',
    'wrong_history_run', 'neural_29_epochs', 'cnn_wrong_best', 'lstm_wrong_best',
    'xgb_wrong_final', 'xgb_wrong_best', 'lightgbm_wrong_best', 'rf_history', 'nonfinite_history',
])
def test_finalization_invalid_authoritative_inputs_write_nothing(finalization_evidence, mutation):
    e = finalization_evidence
    metrics = e.results['metrics.csv']
    confusion = e.results['confusion_matrix.csv']
    history = e.results['training_history.csv']
    if mutation == 'missing_metric':
        e.results['metrics.csv'] = metrics.iloc[:-1]
    elif mutation == 'duplicate_metric':
        metrics.iloc[-1] = metrics.iloc[0]
    elif mutation in ('unknown_model', 'unknown_split', 'wrong_metric_run'):
        metrics.loc[0, {'unknown_model': 'model', 'unknown_split': 'split', 'wrong_metric_run': 'run_id'}[mutation]] = 'unexpected'
    elif mutation in ('nonfinite_metric', 'wrong_threshold', 'wrong_seed', 'wrong_device', 'metric_disagreement'):
        field, value = {'nonfinite_metric': ('roc_auc', np.inf), 'wrong_threshold': ('threshold', .6),
                        'wrong_seed': ('seed', 7), 'wrong_device': ('device', 'cpu'),
                        'metric_disagreement': ('accuracy', .2)}[mutation]
        metrics.loc[0, field] = value
    elif mutation in ('wrong_confusion_total', 'negative_count', 'fractional_count'):
        confusion['count'] = confusion['count'].astype(float)
        confusion.loc[0, 'count'] = {'wrong_confusion_total': 0, 'negative_count': -1, 'fractional_count': 1.5}[mutation]
    elif mutation == 'duplicate_confusion_cell':
        confusion.iloc[-1] = confusion.iloc[0]
    elif mutation == 'wrong_history_run':
        history.loc[0, 'run_id'] = 'unexpected'
    elif mutation in ('neural_29_epochs', 'xgb_wrong_final'):
        model = '1d_cnn' if mutation == 'neural_29_epochs' else 'xgboost'
        final_step = 30 if model == '1d_cnn' else 300
        e.results['training_history.csv'] = history.loc[~((history.model == model) & (history.step == final_step))]
    elif mutation.endswith('_wrong_best'):
        model = {'cnn_wrong_best': '1d_cnn', 'lstm_wrong_best': 'cnn_lstm',
                 'xgb_wrong_best': 'xgboost', 'lightgbm_wrong_best': 'lightgbm'}[mutation]
        history.loc[(history.model == model) & (history.step == 1), 'validation_loss'] = 0
    elif mutation == 'rf_history':
        history.loc[0, 'model'] = 'random_forest'
    elif mutation == 'nonfinite_history':
        history.loc[0, 'validation_loss'] = np.nan
    for name in finalizer.INPUT_SCHEMAS:
        e.results[name].to_csv(e.result_dir / name, index=False)
    with pytest.raises(ValueError):
        finalizer.finalize(e.root)
    assert not e.destination.parent.exists()


def test_finalization_missing_model_artifact_rejected(finalization_evidence, monkeypatch):
    e = finalization_evidence
    original = Path.is_file
    missing = e.models / 'random_forest.pkl'
    monkeypatch.setattr(Path, 'is_file', lambda path: False if path == missing else original(path))
    with pytest.raises(ValueError, match='missing model artifact'):
        finalizer.finalize(e.root)
    assert not e.destination.exists()


def test_finalization_same_task_overwrite_preserves_unrelated_files(finalization_evidence):
    e = finalization_evidence
    finalizer.finalize(e.root)
    unrelated = e.destination.parent / 'unrelated.txt'
    unrelated.write_text('preserve')
    metrics = e.results['metrics.csv']
    metrics.loc[metrics.split == 'test', 'roc_auc'] = np.linspace(.2, .9, 5)
    metrics.loc[metrics.split == 'validation', 'auprc'] = np.linspace(.4, .8, 5)
    # Reversed source rows must not turn TEST performance into ranking/selection.
    metrics.iloc[::-1].to_csv(e.result_dir / 'metrics.csv', index=False)
    e.results['training_history.csv']['train_loss'] = .12345678901234567
    e.results['training_history.csv'].to_csv(e.result_dir / 'training_history.csv', index=False)
    assert finalizer.finalize(e.root) == e.destination.parent
    test = pd.read_csv(e.destination.parent / 'ustc_test_results.csv')
    validation = pd.read_csv(e.destination.parent / 'ustc_validation_results.csv')
    train = pd.read_csv(e.destination)
    assert test.model.tolist() == validation.model.tolist() == list(finalizer.MODELS)
    np.testing.assert_allclose(test.roc_auc, np.linspace(.2, .9, 5))
    np.testing.assert_allclose(validation.auprc, np.linspace(.4, .8, 5))
    np.testing.assert_allclose(train.train_loss, .12345678901234567)
    assert unrelated.read_text() == 'preserve'
    assert {p.name for p in e.destination.parent.iterdir()} == set(finalizer.OUTPUT_SCHEMAS) | {'unrelated.txt'}


@pytest.mark.parametrize('field', ['dataset', 'task', 'authoritative_run'])
def test_finalization_different_identity_cannot_be_overwritten(finalization_evidence, field):
    e = finalization_evidence
    finalizer.finalize(e.root)
    frame = pd.read_csv(e.destination)
    frame.loc[9, field] = 'different'  # A single mixed-identity row also blocks overwrite.
    frame.to_csv(e.destination, index=False)
    existing = e.destination.read_bytes()
    with pytest.raises(FileExistsError, match='refusing overwrite'):
        finalizer.finalize(e.root)
    assert e.destination.read_bytes() == existing
    assert {p.name for p in e.destination.parent.iterdir()} == set(finalizer.OUTPUT_SCHEMAS)


def test_finalization_unverifiable_task_identity_cannot_be_overwritten(finalization_evidence):
    e = finalization_evidence
    finalizer.finalize(e.root)
    frame = pd.read_csv(e.destination).drop(columns='task')
    frame.to_csv(e.destination, index=False)
    existing = e.destination.read_bytes()
    with pytest.raises(ValueError, match='unexpected CSV schema'):
        finalizer.finalize(e.root)
    assert e.destination.read_bytes() == existing


def test_finalization_failed_atomic_replace_preserves_previous_csv(finalization_evidence, monkeypatch):
    e = finalization_evidence
    finalizer.finalize(e.root)
    existing = e.destination.read_bytes()
    def failed(*args):
        raise OSError('Synthetic replace failure')
    monkeypatch.setattr(finalizer.os, 'replace', failed)
    with pytest.raises(OSError, match='replace failure'):
        finalizer.finalize(e.root)
    assert e.destination.read_bytes() == existing
    assert {p.name for p in e.destination.parent.iterdir()} == set(finalizer.OUTPUT_SCHEMAS)


def test_finalization_default_cli_does_not_finalize(tmp_path, monkeypatch):
    monkeypatch.setattr(finalizer, 'finalize', lambda *args: pytest.fail('Explicit execution required'))
    assert finalizer.main(['--project-root', str(tmp_path)]) == 0
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('filename', ['ustc_validation_results.csv', 'ustc_test_results.csv'])
@pytest.mark.parametrize('field', ['dataset', 'authoritative_run'])
def test_finalization_evaluation_identity_mismatch_preserves_entire_set(finalization_evidence, filename, field):
    e = finalization_evidence
    finalizer.finalize(e.root)
    path = e.destination.parent / filename
    frame = pd.read_csv(path)
    frame.loc[0, field] = 'different'
    frame.to_csv(path, index=False)
    existing = {name: (e.destination.parent / name).read_bytes() for name in finalizer.OUTPUT_SCHEMAS}
    with pytest.raises(FileExistsError, match='refusing overwrite'):
        finalizer.finalize(e.root)
    assert {p.name for p in e.destination.parent.iterdir()} == set(finalizer.OUTPUT_SCHEMAS)
    for name, content in existing.items():
        assert (e.destination.parent / name).read_bytes() == content


def test_finalization_orphan_evaluation_csv_not_silently_adopted(finalization_evidence):
    import csv
    e = finalization_evidence
    tables = finalizer.build_results(e.root)
    e.destination.parent.mkdir(parents=True)
    path = e.destination.parent / 'ustc_validation_results.csv'
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=finalizer.EVALUATION_COLUMNS)
        writer.writeheader()
        writer.writerows(tables[path.name])
    existing = path.read_bytes()
    with pytest.raises(ValueError, match='no training-history task identity'):
        finalizer.finalize(e.root)
    assert path.read_bytes() == existing
    assert list(e.destination.parent.iterdir()) == [path]


def test_finalization_revoked_gate_before_publication_preserves_all_outputs(finalization_evidence, monkeypatch):
    e = finalization_evidence
    finalizer.finalize(e.root)
    existing = {name: (e.destination.parent / name).read_bytes() for name in finalizer.OUTPUT_SCHEMAS}
    original = finalizer.validate_gate
    calls = []
    def revoked(root):
        calls.append(root)
        if len(calls) == 2:
            raise ValueError('Integrity gate authorization revoked')
        return original(root)
    monkeypatch.setattr(finalizer, 'validate_gate', revoked)
    with pytest.raises(ValueError, match='authorization revoked'):
        finalizer.finalize(e.root)
    assert len(calls) == 2
    assert {p.name for p in e.destination.parent.iterdir()} == set(finalizer.OUTPUT_SCHEMAS)
    for name, content in existing.items():
        assert (e.destination.parent / name).read_bytes() == content


def test_finalization_missing_gate_cli_fails_without_outputs(tmp_path):
    assert finalizer.main(['--project-root', str(tmp_path), '--execute']) == 1
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def closure_evidence(finalization_evidence):
    e = finalization_evidence
    finalizer.finalize(e.root)  # Prepare only synthetic completed results.
    e.closure = e.root / 'metadata/traffic/ustc_experiment_closure.csv'
    return e


CLOSURE_EXPECTED_COLUMNS = '''dataset authoritative_run stage_status integrity_audit_status finalization_status
train_rows validation_rows test_rows predictor_count model_count seed threshold train_results_path
validation_results_path test_results_path model_artifact_directory authoritative_result_directory
integrity_audit_directory frozen_for_retraining frozen_for_test_reuse next_dataset'''.split()


def test_closure_writes_only_one_metadata_record_and_leaves_experiment_unchanged(closure_evidence, monkeypatch):
    import pickle
    from src.data import ustc_preprocess
    from sklearn.preprocessing import StandardScaler
    e = closure_evidence
    before = {path: path.read_bytes() for path in e.root.rglob('*') if path.is_file()}
    def prohibited(*args, **kwargs):
        pytest.fail('Closure must not finalize, train, infer, deserialize, or fit preprocessing')
    monkeypatch.setattr(finalizer, 'finalize', prohibited)
    monkeypatch.setattr(stage4, 'train_model', prohibited)
    monkeypatch.setattr(stage4, 'train_neural', prohibited)
    monkeypatch.setattr(stage4, 'positive_probability', prohibited)
    monkeypatch.setattr(ustc_preprocess, 'fit_transform', prohibited)
    monkeypatch.setattr(StandardScaler, 'fit', prohibited)
    monkeypatch.setattr(pickle, 'load', prohibited)
    monkeypatch.setattr(pickle, 'loads', prohibited)
    permitted = {e.destination.parent / name for name in finalizer.OUTPUT_SCHEMAS}
    permitted |= {e.result_dir / name for name in finalizer.INPUT_SCHEMAS}
    permitted |= {e.audit_dir / name for name in ('ustc_integrity_audit_metadata.csv', 'ustc_integrity_audit_checklist.csv')}
    original_open, original_replace = Path.open, finalizer.os.replace
    def guarded_open(path, mode='r', *args, **kwargs):
        assert path in permitted or path == e.closure, f'Unexpected closure input: {path}'
        assert not any(flag in mode for flag in ('w', 'a', '+'))
        return original_open(path, mode, *args, **kwargs)
    def guarded_replace(source, destination):
        assert Path(destination) == e.closure
        return original_replace(source, destination)
    with monkeypatch.context() as guard:
        guard.setattr(Path, 'open', guarded_open)
        guard.setattr(finalizer.os, 'replace', guarded_replace)
        assert finalizer.main(['--project-root', str(e.root), '--close', '--execute']) == 0
    assert {path for path in e.root.rglob('*') if path.is_file()} == set(before) | {e.closure}
    for path, content in before.items():
        assert path.read_bytes() == content
    rows = finalizer.read_csv(e.closure, schema=CLOSURE_EXPECTED_COLUMNS)
    assert rows == [dict(dataset='USTC-TFC2016', authoritative_run='20260919T183945Z', stage_status='CLOSED_FROZEN',
        integrity_audit_status='PASS', finalization_status='PASS', train_rows='367293', validation_rows='97001',
        test_rows='98255', predictor_count='50', model_count='5', seed='42', threshold='0.5',
        train_results_path='results/final/USTC-TFC2016/ustc_train_results.csv',
        validation_results_path='results/final/USTC-TFC2016/ustc_validation_results.csv',
        test_results_path='results/final/USTC-TFC2016/ustc_test_results.csv',
        model_artifact_directory='models/USTC-TFC2016/20260919T183945Z',
        authoritative_result_directory='results/models/USTC-TFC2016/20260919T183945Z',
        integrity_audit_directory='results/audits/USTC-TFC2016/integrity_audit',
        frozen_for_retraining='YES', frozen_for_test_reuse='YES', next_dataset='CIC-Darknet2020')]


@pytest.mark.parametrize('filename', list(finalizer.OUTPUT_SCHEMAS))
@pytest.mark.parametrize('mutation', ['missing_row', 'extra_row', 'wrong_run', 'wrong_dataset', 'corrupt_value'])
def test_closure_invalid_final_csv_cannot_create_frozen_record(closure_evidence, filename, mutation):
    e = closure_evidence
    path = e.destination.parent / filename
    rows = finalizer.read_csv(path)
    if mutation == 'missing_row':
        rows = rows[:-1]
    elif mutation == 'extra_row':
        rows.append(rows[0].copy())
    elif mutation == 'wrong_run':
        rows[0]['authoritative_run'] = 'different'
    elif mutation == 'wrong_dataset':
        rows[0]['dataset'] = 'different'
    else:
        rows[0]['seed'] = '43'
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Closure:'):
        finalizer.close_experiment(e.root)
    assert not e.closure.parent.exists()


@pytest.mark.parametrize('filename', list(finalizer.OUTPUT_SCHEMAS))
def test_closure_missing_final_csv_rejected(closure_evidence, monkeypatch, filename):
    e = closure_evidence
    original = finalizer.read_csv
    def missing(path, **kwargs):
        if Path(path) == e.destination.parent / filename:
            raise FileNotFoundError(path)
        return original(path, **kwargs)
    monkeypatch.setattr(finalizer, 'read_csv', missing)
    with pytest.raises(FileNotFoundError):
        finalizer.close_experiment(e.root)
    assert not e.closure.exists()


@pytest.mark.parametrize('filename', list(finalizer.ARTIFACTS.values()))
def test_closure_each_model_artifact_required(closure_evidence, monkeypatch, filename):
    e = closure_evidence
    original = Path.is_file
    monkeypatch.setattr(Path, 'is_file', lambda path: False if path == e.models / filename else original(path))
    with pytest.raises(ValueError, match='missing model artifact'):
        finalizer.close_experiment(e.root)
    assert not e.closure.exists()


@pytest.mark.parametrize('field,value', [('overall_status', 'FAIL'), ('finalization_authorized', 'NO'),
                                        ('dataset', 'different'), ('authoritative_run', 'different'), ('task', 'different')])
def test_closure_audit_gate_required(closure_evidence, field, value):
    e = closure_evidence
    path = e.audit_dir / 'ustc_integrity_audit_metadata.csv'
    rows = finalizer.read_csv(path)
    rows[0][field] = value
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match='Integrity gate'):
        finalizer.close_experiment(e.root)
    assert not e.closure.exists()


def test_closure_checklist_fail_rejected_even_when_metadata_passes(closure_evidence):
    e = closure_evidence
    path = e.audit_dir / 'ustc_integrity_audit_checklist.csv'
    rows = finalizer.read_csv(path)
    rows[0]['status'] = 'FAIL'
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match='checklist contains FAIL'):
        finalizer.close_experiment(e.root)
    assert not e.closure.exists()


@pytest.mark.parametrize('filename', list(finalizer.INPUT_SCHEMAS))
def test_closure_authoritative_run_must_match_every_source(closure_evidence, filename):
    e = closure_evidence
    path = e.result_dir / filename
    rows = finalizer.read_csv(path)
    rows[0]['run_id'] = 'different'
    pd.DataFrame(rows).to_csv(path, index=False)
    with pytest.raises(ValueError, match='authoritative dataset/run mismatch'):
        finalizer.close_experiment(e.root)
    assert not e.closure.exists()


def test_closure_same_identity_overwrites_only_canonical_metadata(closure_evidence):
    e = closure_evidence
    finalizer.close_experiment(e.root)
    unrelated = e.closure.parent / 'unrelated.csv'
    unrelated.write_text('preserve')
    rows = finalizer.read_csv(e.closure)
    rows[0]['next_dataset'] = 'stale metadata'
    pd.DataFrame(rows).to_csv(e.closure, index=False)
    before = {path: path.read_bytes() for path in e.root.rglob('*') if path.is_file() and path != e.closure}
    assert finalizer.close_experiment(e.root) == e.closure
    assert finalizer.read_csv(e.closure)[0]['next_dataset'] == 'CIC-Darknet2020'
    assert {path.name for path in e.closure.parent.iterdir()} == {'ustc_experiment_closure.csv', 'unrelated.csv'}
    for path, content in before.items():
        assert path.read_bytes() == content


@pytest.mark.parametrize('field', ['dataset', 'authoritative_run', 'stage_status'])
def test_closure_different_identity_preserved(closure_evidence, field):
    e = closure_evidence
    finalizer.close_experiment(e.root)
    rows = finalizer.read_csv(e.closure)
    rows[0][field] = 'different'
    pd.DataFrame(rows).to_csv(e.closure, index=False)
    before = e.closure.read_bytes()
    with pytest.raises(FileExistsError, match='refusing overwrite'):
        finalizer.close_experiment(e.root)
    assert e.closure.read_bytes() == before
    assert list(e.closure.parent.iterdir()) == [e.closure]


def test_closure_atomic_failure_preserves_previous_record(closure_evidence, monkeypatch):
    e = closure_evidence
    finalizer.close_experiment(e.root)
    before = e.closure.read_bytes()
    def fail(*args):
        raise OSError('Synthetic closure publication failure')
    monkeypatch.setattr(finalizer.os, 'replace', fail)
    with pytest.raises(OSError, match='publication failure'):
        finalizer.close_experiment(e.root)
    assert e.closure.read_bytes() == before
    assert list(e.closure.parent.iterdir()) == [e.closure]


def test_closure_requires_explicit_execute_and_never_calls_finalization(tmp_path, monkeypatch):
    monkeypatch.setattr(finalizer, 'finalize', lambda *args: pytest.fail('Closure must not call finalization'))
    monkeypatch.setattr(finalizer, 'close_experiment', lambda *args: pytest.fail('Closure must require --execute'))
    assert finalizer.main(['--project-root', str(tmp_path), '--close']) == 0
    assert list(tmp_path.iterdir()) == []


# CIC tests are synthetic and GPU-free unless explicitly checking an optional
# PyTorch architecture. No real dataset or experimental directory is used.
from src.models import cic_train_models as cic_models


@pytest.fixture
def cic_model_root(tmp_path, monkeypatch):
    m = cic_models
    monkeypatch.setattr(m, 'ROW_COUNTS', dict(train=8, validation=8, test=8))
    monkeypatch.setattr(m, 'GROUP_COUNT', 12)
    directory = tmp_path / 'metadata/traffic'
    directory.mkdir(parents=True)
    candidates = [f'feature_{i}' for i in range(77)]
    main = dict(dataset=m.DATASET, task='five_model_4class_benchmark',
        raw_source='data/raw/CIC-Darknet2020/Darknet.CSV', raw_rows=24, raw_columns=85,
        target_column='Label', auxiliary_label_column='Label.1', class_count=4,
        split_group='Flow ID', split_seed=42, initial_predictor_candidates=77,
        contract_status='FROZEN_FOR_MATERIALIZATION', split_method='fixture_frozen_split',
        train_rows=8, validation_rows=8, test_rows=8, **m.POLICIES)
    pd.DataFrame([main]).to_csv(directory / 'cic_phase1_contract.csv', index=False)
    pd.DataFrame([dict(class_id=i, raw_label=label, canonical_label=label, probability_column=m.PROBABILITY_COLUMNS[i])
        for i, label in enumerate(m.LABELS)]).to_csv(directory / 'cic_label_mapping.csv', index=False)
    pd.DataFrame([dict(source_index=i, column_name=name,
        predictor_candidate='YES' if i < 77 else 'NO', model_allowed='YES' if i < 77 else 'NO')
        for i, name in enumerate(candidates + m.PROVENANCE)]).to_csv(directory / 'cic_feature_contract.csv', index=False)
    pd.DataFrame([dict(source_order=i, feature_name=name, train_median=0,
        train_zero_variance='NO' if i < 62 else 'YES', retained_final='YES' if i < 62 else 'NO',
        final_predictor_order=i if i < 62 else '', train_scaler_mean=0 if i < 62 else '',
        train_scaler_scale=1 if i < 62 else '') for i, name in enumerate(candidates)]
        ).to_csv(directory / 'cic_preprocessing_parameters.csv', index=False)
    manifest = []
    data_dir = tmp_path / 'data/processed' / m.DATASET
    data_dir.mkdir(parents=True)
    for split in m.ROW_COUNTS:
        y = np.repeat(np.arange(4), 2)
        frame = pd.DataFrame({name: np.arange(8, dtype=float) + i / 100 for i, name in enumerate(candidates[:62])})
        frame['target_id'] = y
        frame['Label'] = [m.LABELS[c] for c in y]
        frame['Flow ID'] = [f'{split}-{c}' for c in y]
        for name in ['Src IP', 'Dst IP', 'Timestamp', 'Src Port', 'Dst Port']:
            frame[name] = '001'
        frame['Label.1'] = 'Unmodeled-Application'
        frame.to_csv(data_dir / f'{split}.csv', index=False)
        for c in range(4):
            manifest.append(dict(dataset=m.DATASET, flow_id=f'{split}-{c}', split=split.upper(),
                group_row_count=2, non_tor_rows=2 if c == 0 else 0, nonvpn_rows=2 if c == 1 else 0,
                tor_rows=2 if c == 2 else 0, vpn_rows=2 if c == 3 else 0, seed=42, split_method='fixture_frozen_split'))
    pd.DataFrame(manifest).to_csv(directory / 'cic_split_manifest.csv', index=False)
    return tmp_path


def cic_fake_history(name):
    if name == 'random_forest':
        return []
    n = 30 if name in ('1d_cnn', 'cnn_lstm') else 3
    return cic_models.history_rows([1.0] * n, [0.5] + [0.7] * (n - 1))


def test_cic_model_contract_and_phase_a_only(cic_model_root, monkeypatch):
    m = cic_models
    accessed = []
    real_read = m.read_csv
    def read(path, **kwargs):
        assert path.name != 'test.csv'
        accessed.append(path.name)
        return real_read(path, **kwargs)
    monkeypatch.setattr(m, 'read_csv', read)
    contract = m.load_contract(cic_model_root)
    assert contract.features == [f'feature_{i}' for i in range(62)]
    assert not set(contract.features) & set(m.METADATA)
    train, validation = m.load_training_data(cic_model_root, contract)
    assert train.x.shape == validation.x.shape == (8, 62)
    assert set(train.y) == set(validation.y) == {0, 1, 2, 3}
    assert not train.groups & validation.groups
    assert accessed[-2:] == ['train.csv', 'validation.csv']
    with pytest.raises(ValueError, match='TEST reader'):
        m.read_partition(cic_model_root, contract, 'test')


@pytest.mark.parametrize('problem', ['feature_count', 'mapping', 'probability_order', 'predictor_order', 'provenance'])
def test_cic_model_rejects_contract_changes(cic_model_root, problem):
    base = cic_model_root / 'metadata/traffic'
    filename = 'cic_preprocessing_parameters.csv' if problem in ('feature_count', 'predictor_order') else (
        'cic_feature_contract.csv' if problem == 'provenance' else 'cic_label_mapping.csv')
    frame = pd.read_csv(base / filename, keep_default_na=False)
    if problem == 'feature_count':
        frame.loc[0, 'retained_final'] = 'NO'
    elif problem == 'predictor_order':
        frame.loc[0, 'final_predictor_order'] = '61'
    elif problem == 'mapping':
        frame.loc[0, 'raw_label'] = 'non-tor'
    elif problem == 'probability_order':
        frame.loc[0, 'probability_column'] = 'class_3_vpn'
    else:
        frame.loc[frame.column_name == 'Label.1', ['model_allowed', 'predictor_candidate']] = 'YES'
    frame.to_csv(base / filename, index=False)
    with pytest.raises(ValueError):
        cic_models.load_contract(cic_model_root)


def test_cic_roster_and_hyperparameters():
    m = cic_models
    assert m.MODELS == ('random_forest', '1d_cnn', 'xgboost', 'lightgbm', 'cnn_lstm')
    assert (m.SEED, m.CLASS_COUNT, m.FEATURE_COUNT) == (42, 4, 62)
    assert m.LABELS == ('Non-Tor', 'NonVPN', 'Tor', 'VPN')
    assert m.PROBABILITY_COLUMNS == ('class_0_non_tor', 'class_1_nonvpn', 'class_2_tor', 'class_3_vpn')
    rf = m.model_parameters('random_forest')
    assert rf == dict(n_estimators=300, max_depth=None, max_features='sqrt', bootstrap=True,
                     random_state=42, n_streams=1, output_type='cupy')
    xgb = m.model_parameters('xgboost')
    assert xgb == dict(objective='multi:softprob', num_class=4, n_estimators=300, max_depth=6,
        learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, tree_method='hist', device='cuda',
        random_state=42, eval_metric='mlogloss', early_stopping_rounds=30)
    lgb = m.model_parameters('lightgbm')
    assert lgb['objective'] == 'multiclass' and lgb['num_class'] == 4 and lgb['device_type'] == 'cuda'
    assert lgb['n_estimators'] == 300 and lgb['num_leaves'] == 31 and lgb['learning_rate'] == 0.05
    for name in m.MODELS:
        assert 'class_weight' not in m.model_parameters(name)
    for name in ('1d_cnn', 'cnn_lstm'):
        params = m.model_parameters(name)
        assert (params['epochs'], params['batch_size'], params['output_logits']) == (30, 1024, 4)
        assert params['early_termination'] is False and params['loss'] == 'CrossEntropyLoss'


@pytest.mark.parametrize('name', ['1d_cnn', 'cnn_lstm'])
def test_cic_neural_architecture_four_logits(name):
    torch = pytest.importorskip('torch')
    model = cic_models.build_neural_model(name, torch)
    conv = [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv1d)]
    assert conv[0].in_channels == 1 and conv[0].out_channels == 64
    assert conv[0].kernel_size == (3,) and conv[0].padding == (1,)
    linear = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
    assert (linear[-1].in_features, linear[-1].out_features) == (64, 4)
    assert len(conv) == (2 if name == '1d_cnn' else 1)
    if name == 'cnn_lstm':
        assert model.lstm.input_size == model.lstm.hidden_size == 64
        assert model.lstm.num_layers == 1 and model.lstm.batch_first
    # Architecture inspection only; no training or forward pass is needed.


@pytest.mark.parametrize('name', ['1d_cnn', 'cnn_lstm'])
def test_cic_neural_thirty_epochs_then_restore(name):
    events = []
    class Model:
        epoch = 0
        def state_dict(self):
            return {'epoch': self.epoch}
        def load_state_dict(self, state):
            assert self.epoch == 30 and len(events) == 60
            self.restored = state['epoch']
    model = Model()
    def epoch_pass(training):
        if training:
            model.epoch += 1
        events.append(training)
        return 1.0 if training else (0.1 if model.epoch in (3, 5) else 0.8)
    history = cic_models.fit_neural_epochs(model, epoch_pass)
    cic_models.validate_history(name, history)
    assert len(history) == 30 and model.restored == 3
    assert [row['step'] for row in history if row['is_best_validation_step'] == 'YES'] == [3]


def test_cic_probability_reordering_and_multiclass_metrics():
    m = cic_models
    y = np.array([0, 1, 2, 3])
    perfect = np.eye(4)
    np.testing.assert_array_equal(m.ordered_probabilities(perfect[:, [3, 1, 0, 2]], [3, 1, 0, 2]), perfect)
    output = m.evaluate_predictions(y, perfect)
    metrics = output['metrics.csv'][0]
    for name in ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1', 'mcc',
                 'macro_ovr_roc_auc', 'macro_ovr_auprc']:
        assert metrics[name] == pytest.approx(1)
    assert metrics['multiclass_brier_score'] == 0 and metrics['ece'] == 0
    assert metrics['multiclass_log_loss'] == pytest.approx(0, abs=1e-12)
    assert len(output['confusion_matrix.csv']) == 16
    assert sum(row['count'] for row in output['confusion_matrix.csv']) == 4
    assert len(output['calibration_bins.csv']) == 15
    assert output['calibration_bins.csv'][-1]['count'] == 4
    uniform = m.evaluate_predictions(y, np.full((4, 4), 0.25))['metrics.csv'][0]
    assert uniform['macro_ovr_roc_auc'] == pytest.approx(0.5)
    assert uniform['macro_ovr_auprc'] == pytest.approx(0.25)
    assert uniform['multiclass_log_loss'] == pytest.approx(np.log(4))
    assert uniform['multiclass_brier_score'] == pytest.approx(0.75)
    assert uniform['ece'] == pytest.approx(0)
    wrong = np.roll(perfect, 1, axis=1)
    bad = m.evaluate_predictions(y, wrong)['metrics.csv'][0]
    assert bad['ece'] == pytest.approx(1) and bad['multiclass_brier_score'] == pytest.approx(2)
    for name in ['roc_curve.csv', 'pr_curve.csv']:
        assert {(row['class_id'], row['class_label']) for row in output[name]} == set(enumerate(m.LABELS))


@pytest.mark.parametrize('probabilities', [np.ones((4, 3)), np.full((4, 4), np.nan),
    np.full((4, 4), 0.3), np.tile([-0.1, 0.4, 0.4, 0.3], (4, 1))])
def test_cic_rejects_invalid_probabilities(probabilities):
    with pytest.raises(ValueError):
        cic_models.evaluate_predictions(np.arange(4), probabilities)


def test_cic_xgboost_fallback_critical():
    import warnings
    for message in ['Falling back to prediction using DMatrix due to mismatched devices.',
                    'No visible GPU is found, setting device to CPU.']:
        with pytest.raises(RuntimeError, match='Critical XGBoost'):
            cic_models.xgboost_call(lambda: warnings.warn(message))
    booster = SimpleNamespace(save_config=lambda: json.dumps({'learner': {'generic_param': {'device': 'cpu'}}}))
    with pytest.raises(ValueError, match='CPU fallback'):
        cic_models.verify_xgboost_cuda(booster)


def test_cic_test_gate_and_once_only(cic_model_root):
    m = cic_models
    contract = m.load_contract(cic_model_root)
    train, validation = m.load_training_data(cic_model_root, contract)
    guard = m.RunGuard()
    with pytest.raises(ValueError, match='locked'):
        guard.load_test(cic_model_root, contract, train, validation)
    for name in m.MODELS:
        guard.begin_training(name)
        guard.validation_models.add(name)
        artifact = cic_model_root / m.ARTIFACTS[name]
        artifact.write_bytes(b'synthetic finalized model')
        guard.finalize(name, artifact, cic_fake_history(name))
    test = guard.load_test(cic_model_root, contract, train, validation)
    assert guard.ALL_MODELS_FINALIZED and guard.test_loaded
    with pytest.raises(ValueError):
        guard.load_test(cic_model_root, contract, train, validation)
    with pytest.raises(ValueError, match='Training/retraining'):
        guard.begin_training('random_forest')
    calls = []
    def predict(x):
        calls.append(len(x))
        return np.eye(4)[test.y]
    guard.evaluate_test('random_forest', predict, test)
    with pytest.raises(ValueError, match='already attempted'):
        guard.evaluate_test('random_forest', predict, test)
    assert calls == [8]
    def fail(x):
        raise RuntimeError('synthetic failure')
    with pytest.raises(RuntimeError, match='synthetic failure'):
        guard.evaluate_test('xgboost', fail, test)
    with pytest.raises(ValueError, match='already attempted'):
        guard.evaluate_test('xgboost', predict, test)


def test_cic_run_two_phases_outputs_and_no_overwrite(cic_model_root, monkeypatch):
    m = cic_models
    events = []
    original_read = m.read_partition
    def read(root, contract, split, **kwargs):
        if split == 'test':
            assert events == list(m.MODELS)
            assert kwargs['guard'].ALL_MODELS_FINALIZED
        return original_read(root, contract, split, **kwargs)
    monkeypatch.setattr(m, 'read_partition', read)
    monkeypatch.setattr(m, 'gpu_preflight', lambda: dict.fromkeys(['cp', 'torch', 'xgboost', 'lightgbm'], SimpleNamespace(__version__='fixture')))
    prediction_calls = []
    def train_model(name, train, validation, features, backends, guard):
        guard.begin_training(name)
        events.append(name)
        def predict(frame):
            prediction_calls.append((name, guard.phase))
            return np.eye(4)[np.repeat(np.arange(4), 2)]
        return predict, cic_fake_history(name), lambda path: path.write_bytes(b'finalized fixture')
    monkeypatch.setattr(m, 'train_model', train_model)
    fixed = datetime(2026, 9, 20, 12, 30, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(m, 'datetime', SimpleNamespace(now=lambda tz: fixed))
    assert m.run(cic_model_root, execute=True) == 0
    base = cic_model_root / 'results/models/CIC-Darknet2020/20260920T123000Z'
    models = cic_model_root / 'models/CIC-Darknet2020/20260920T123000Z'
    assert {p.name for p in models.iterdir()} == set(m.ARTIFACTS.values())
    assert {p.name for p in base.iterdir()} == set(m.HEADERS)
    metrics = pd.read_csv(base / 'metrics.csv')
    assert len(metrics) == 10 and metrics['split'].tolist() == ['validation'] * 5 + ['test'] * 5
    history = pd.read_csv(base / 'training_history.csv')
    assert not history.model.eq('random_forest').any()
    assert history.groupby('model').size().to_dict() == {'1d_cnn': 30, 'cnn_lstm': 30, 'xgboost': 3, 'lightgbm': 3}
    assert prediction_calls == [(name, 'TRAINING') for name in m.MODELS] + [(name, 'TEST') for name in m.MODELS]
    before = {p: p.read_bytes() for p in base.iterdir()}
    with pytest.raises(FileExistsError):
        m.create_run(cic_model_root, '20260920T123000Z')
    assert before == {p: p.read_bytes() for p in base.iterdir()}


def test_cic_dry_run_no_partitions_or_gpu(cic_model_root, monkeypatch):
    monkeypatch.setattr(cic_models, 'load_training_data', lambda *args: pytest.fail('Dry run read partitions'))
    monkeypatch.setattr(cic_models, 'gpu_preflight', lambda: pytest.fail('Dry run used CUDA'))
    assert cic_models.main(['--project-root', str(cic_model_root)]) == 0
    assert not (cic_model_root / 'models').exists()


def test_cic_preflight_no_fitting_or_test(cic_model_root, monkeypatch):
    monkeypatch.setattr(cic_models, 'gpu_preflight', lambda: {})
    monkeypatch.setattr(cic_models, 'train_model', lambda *args: pytest.fail('Preflight fitted a model'))
    monkeypatch.setattr(cic_models.RunGuard, 'load_test', lambda *args: pytest.fail('Preflight loaded TEST'))
    assert cic_models.main(['--project-root', str(cic_model_root), '--preflight']) == 0
    assert not (cic_model_root / 'models').exists()


def test_cic_model_module_no_digest_or_cross_dataset_dependency():
    import ast
    source = Path(cic_models.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    imports, calls = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
        elif isinstance(node, ast.Call):
            calls.append(node.func.attr if isinstance(node.func, ast.Attribute)
                         else node.func.id if isinstance(node.func, ast.Name) else '')
    assert not any('ustc' in name.lower() or 'hash' in name.lower() for name in imports)
    assert not set(calls) & {'hash', 'sha256', 'sha256_file', 'fit_transform'}
    assert 'USTC-TFC2016' not in source and 'THRESHOLD' not in source


def test_cic_imbalanced_macro_metrics_match_fixed_ovr_definition():
    from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
    y = np.array([0, 0, 0, 0, 1, 1, 2, 3])
    p = np.array([[.7, .1, .1, .1], [.2, .5, .2, .1], [.4, .2, .2, .2], [.6, .1, .2, .1],
                  [.1, .7, .1, .1], [.5, .2, .2, .1], [.1, .2, .6, .1], [.4, .1, .1, .4]])
    metrics = cic_models.evaluate_predictions(y, p)['metrics.csv'][0]
    onehot = np.eye(4)[y]
    assert metrics['macro_ovr_roc_auc'] == pytest.approx(np.mean([
        roc_auc_score(onehot[:, i], p[:, i]) for i in range(4)]))
    assert metrics['macro_ovr_auprc'] == pytest.approx(np.mean([
        average_precision_score(onehot[:, i], p[:, i]) for i in range(4)]))
    assert metrics['multiclass_log_loss'] == pytest.approx(log_loss(y, p, labels=[0, 1, 2, 3]))
    assert metrics['multiclass_brier_score'] == pytest.approx(np.square(p - onehot).sum(axis=1).mean())


@pytest.mark.parametrize('problem', ['missing_artifact', 'missing_validation', 'short_neural_history'])
def test_cic_finalization_gate_rechecks_evidence(cic_model_root, monkeypatch, problem):
    m = cic_models
    contract = m.load_contract(cic_model_root)
    train, validation = m.load_training_data(cic_model_root, contract)
    guard = m.RunGuard()
    for name in m.MODELS:
        path = cic_model_root / m.ARTIFACTS[name]
        path.write_bytes(b'fixture')
        guard.begin_training(name)
        guard.validation_models.add(name)
        guard.finalize(name, path, cic_fake_history(name))
    if problem == 'missing_artifact':
        guard.finalized['xgboost'].unlink()
    elif problem == 'missing_validation':
        guard.validation_models.remove('lightgbm')
    else:
        guard.histories['cnn_lstm'].pop()
    monkeypatch.setattr(m, 'read_partition', lambda *args, **kwargs: pytest.fail('TEST read before complete evidence'))
    with pytest.raises(ValueError):
        guard.load_test(cic_model_root, contract, train, validation)


def test_cic_rejects_fabricated_rf_history():
    with pytest.raises(ValueError, match='fabricated'):
        cic_models.validate_history('random_forest', cic_fake_history('xgboost'))


def test_cic_failed_run_preserved_without_test(cic_model_root, monkeypatch):
    m = cic_models
    monkeypatch.setattr(m, 'gpu_preflight', lambda: dict.fromkeys(['cp', 'torch', 'xgboost', 'lightgbm'], SimpleNamespace(__version__='fixture')))
    monkeypatch.setattr(m.RunGuard, 'load_test', lambda *args: pytest.fail('Failed phase A accessed TEST'))
    def fake_train(name, train, validation, features, backends, guard):
        guard.begin_training(name)
        if name == '1d_cnn':
            raise RuntimeError('synthetic GPU failure')
        return lambda x: np.eye(4)[validation.y], [], lambda path: path.write_bytes(b'finalized RF')
    monkeypatch.setattr(m, 'train_model', fake_train)
    monkeypatch.setattr(m, 'datetime', SimpleNamespace(now=lambda tz: datetime(2026, 9, 20, tzinfo=timezone.utc)))
    with pytest.raises(RuntimeError, match='synthetic GPU failure'):
        m.run(cic_model_root, execute=True)
    directory = cic_model_root / 'results/models/CIC-Darknet2020/20260920T000000Z'
    metrics = pd.read_csv(directory / 'metrics.csv')
    assert metrics.model.tolist() == ['random_forest'] and metrics['split'].tolist() == ['validation']
    configuration = pd.read_csv(directory / 'run_configuration.csv')
    assert configuration.loc[configuration.key == 'run_status', 'value'].iloc[-1] == 'FAILED_PROTOCOL_OR_EXECUTION'
    assert (cic_model_root / 'models/CIC-Darknet2020/20260920T000000Z/random_forest.pkl').exists()


def test_cic_gpu_preflight_source_never_fits_or_predicts():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(cic_models.gpu_preflight))
    methods = [node.func.attr for node in ast.walk(tree)
               if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert not set(methods) & {'fit', 'predict', 'predict_proba', 'forward'}


# Consolidated CIC finalization: saved synthetic evidence only, never models.
from src.evaluation import cic_finalize_results as cic_final


def write_cic_evidence(root, name, rows):
    path = root / cic_final.RESULT_DIR / name
    pd.DataFrame(rows, columns=cic_final.INPUT_HEADERS[name]).to_csv(path, index=False)


@pytest.fixture
def cic_final_evidence(cic_model_root, monkeypatch):
    f = cic_final
    root = cic_model_root
    monkeypatch.setattr(f, 'ROWS', dict(train=8, validation=8, test=8))
    monkeypatch.setattr(f, 'GROUP_COUNT', 12)
    monkeypatch.setattr(f, 'CONFLICT_COUNT', 0)
    manifest_path = root / 'metadata/traffic/cic_split_manifest.csv'
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    manifest['has_label_conflict'] = 'NO'
    manifest.to_csv(manifest_path, index=False)
    models, results = root / f.MODEL_DIR, root / f.RESULT_DIR
    models.mkdir(parents=True)
    results.mkdir(parents=True)
    for name in f.ARTIFACTS.values():
        (models / name).write_bytes(b'not a loadable model; existence evidence only')
    tables = {name: [] for name in f.INPUT_HEADERS}
    for model in f.MODELS:
        for split in f.SPLITS:
            prefix = dict(dataset=f.DATASET, run_id=f.RUN, model=model, split=split)
            values = dict.fromkeys(f.METRICS, '1.0000000000000000')
            values.update(multiclass_log_loss='0.0000000000000000', multiclass_brier_score='0', ece='0')
            tables['metrics.csv'].append(dict(prefix, device=f.DEVICES[model], **values, row_count=8))
            for a in range(4):
                for b in range(4):
                    tables['confusion_matrix.csv'].append(dict(prefix, actual_class_id=a, actual_class_label=f.LABELS[a],
                        predicted_class_id=b, predicted_class_label=f.LABELS[b], count=2 if a == b else 0))
                for x, y, threshold in [(0, 0, 'inf'), (0, 1, 1), (1, 1, 0)]:
                    tables['roc_curve.csv'].append(dict(prefix, class_id=a, class_label=f.LABELS[a], fpr=x, tpr=y, threshold=threshold))
                for p, r, threshold in [(0.25, 1, 0), (1, 1, 1), (1, 0, '')]:
                    tables['pr_curve.csv'].append(dict(prefix, class_id=a, class_label=f.LABELS[a], precision=p, recall=r, threshold=threshold))
            for i in range(15):
                tables['calibration_bins.csv'].append(dict(prefix, bin_index=i, lower_bound=i/15, upper_bound=(i+1)/15,
                    count=8 if i == 14 else 0, mean_confidence=1 if i == 14 else '',
                    accuracy=1 if i == 14 else '', absolute_gap=0 if i == 14 else ''))
        if model != 'random_forest':
            for i in range(1, 31 if model in ('1d_cnn', 'cnn_lstm') else 4):
                tables['training_history.csv'].append(dict(dataset=f.DATASET, run_id=f.RUN, model=model, step=i,
                    train_loss='0.40000000000000002', validation_loss='0.12500000000000000' if i == 2 else '0.5',
                    is_best_validation_step='YES' if i == 2 else 'NO'))
    globals_ = dict(seed=42, class_count=4, predictor_count=62, preprocessing='ALREADY_FROZEN',
        test_selection='FORBIDDEN', test_evaluations_per_model=1, class_weights='NONE', resampling='NONE',
        additional_preprocessing='NONE', train_rows=8, validation_rows=8, test_rows=8,
        class_mapping=json.dumps(dict(enumerate(f.LABELS))), probability_order=json.dumps(f.PROBABILITY_COLUMNS),
        model_roster=json.dumps(f.MODELS), feature_names=json.dumps([f'feature_{i}' for i in range(62)]),
        lightgbm_early_stopping_rounds=30)
    def config(model, key, value):
        tables['run_configuration.csv'].append(dict(dataset=f.DATASET, run_id=f.RUN, model=model, key=key, value=value))
    for key, value in globals_.items():
        config('all', key, value)
    for model in f.MODELS:
        for key, value in cic_models.model_parameters(model).items():
            config(model, key, 'None' if value is None else value)
    config('all', 'run_status', 'STARTED')
    config('all', 'ALL_MODELS_FINALIZED', 'True')
    for model in f.MODELS:
        config(model, 'test_inference_attempt', 1)
    config('all', 'run_status', 'COMPLETE')
    for name, rows in tables.items():
        write_cic_evidence(root, name, rows)
    return SimpleNamespace(root=root, tables=tables, models=models, results=results)


def test_cic_final_dry_audit_is_read_only(cic_final_evidence, monkeypatch):
    e, f = cic_final_evidence, cic_final
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in e.root.rglob('*') if p.is_file()}
    original_open = Path.open
    def guarded_open(path, *args, **kwargs):
        assert not path.is_relative_to(e.models), 'Model contents must never be opened'
        assert 'USTC-TFC2016' not in str(path)
        assert 'data/raw' not in path.as_posix(), 'Finalizer cannot re-fit from raw data'
        return original_open(path, *args, **kwargs)
    with monkeypatch.context() as scope:
        scope.setattr(Path, 'open', guarded_open)
        report = f.finalize(e.root)
    assert report.passed
    assert all(row['status'] == 'PASS' for row in report.checklist)
    note = next(row for row in report.checklist if row['check'] == 'probability_contract_guard')
    assert note['details'] == f.PROBABILITY_NOTE and note['status'] == 'PASS'
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in e.root.rglob('*') if p.is_file()}
    assert not (e.root / f.FINAL_DIR).exists() and not (e.root / f.CLOSURE).exists()


def test_cic_final_outputs_and_closure(cic_final_evidence):
    e, f = cic_final_evidence, cic_final
    before = {p: p.read_bytes() for p in e.root.rglob('*') if p.is_file()}
    assert f.finalize(e.root, execute=True).passed
    for name, columns in f.FINAL_HEADERS.items():
        assert list(pd.read_csv(e.root / f.FINAL_DIR / name).columns) == columns
    train = f.read_rows(e.root / f.FINAL_DIR / 'cic_train_results.csv')
    assert len(train) == 66
    assert {r['model'] for r in train} == {'1d_cnn', 'cnn_lstm', 'xgboost', 'lightgbm'}
    assert all(r['train_loss'] == '0.40000000000000002' for r in train)
    assert all(r['best_validation_step'] == '2' and r['best_validation_loss'] == '0.12500000000000000' for r in train)
    for split in f.SPLITS:
        rows = f.read_rows(e.root / f.FINAL_DIR / f'cic_{split}_results.csv')
        assert len(rows) == 5 and [r['model'] for r in rows] == list(f.MODELS)
        assert all(r['accuracy'] == '1.0000000000000000' and r['task'] == 'privacy_network_4class' for r in rows)
    per_class = pd.read_csv(e.root / f.FINAL_DIR / 'cic_per_class_results.csv')
    assert len(per_class) == 40 and per_class.support.eq(2).all()
    assert per_class[['precision', 'recall', 'f1', 'roc_auc_ovr', 'average_precision_ovr']].eq(1).all().all()
    assert not (e.root / f.CLOSURE).exists()
    assert f.finalize(e.root, execute=True, close=True).passed
    closure = f.read_rows(e.root / f.CLOSURE, schema=f.CLOSURE_COLUMNS)
    assert len(closure) == 1
    for key, value in dict(dataset=f.DATASET, authoritative_run=f.RUN, stage_status='CLOSED_FROZEN',
        integrity_audit_status='PASS', finalization_status='PASS', frozen_for_retraining='YES', frozen_for_test_reuse='YES',
        phase_i_status='COMPLETE', next_phase='PROVENANCE_AWARE_RAG', predictor_count='62', class_count='4',
        initial_predictor_count='77', removed_train_zero_variance='15', model_count='5', seed='42').items():
        assert closure[0][key] == value
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize('problem', ['run', 'dataset', 'missing_row', 'duplicate_row', 'unknown_model', 'bad_split',
    'nan', 'range', 'negative_logloss', 'negative_brier', 'bad_rows', 'accuracy'])
def test_cic_final_metric_gates(cic_final_evidence, problem):
    e, f = cic_final_evidence, cic_final
    rows = e.tables['metrics.csv']
    if problem == 'missing_row':
        rows.pop()
    elif problem == 'duplicate_row':
        rows[-1] = rows[0].copy()
    else:
        field, value = {'run': ('run_id', 'other'), 'dataset': ('dataset', 'other'),
            'unknown_model': ('model', 'dummy'), 'bad_split': ('split', 'train'),
            'nan': ('mcc', 'nan'), 'range': ('macro_f1', 1.1), 'negative_logloss': ('multiclass_log_loss', -1),
            'negative_brier': ('multiclass_brier_score', -1), 'bad_rows': ('row_count', 7), 'accuracy': ('accuracy', .5)}[problem]
        rows[0][field] = value
    write_cic_evidence(e.root, 'metrics.csv', rows)
    report = f.finalize(e.root, execute=True)
    assert not report.passed and any(r['status'] == 'FAIL' for r in report.checklist)
    assert not (e.root / f.FINAL_DIR).exists() and not (e.root / f.CLOSURE).exists()
    assert (e.root / f.CHECKLIST).exists()


@pytest.mark.parametrize('problem', ['missing', 'empty'])
def test_cic_final_artifact_gate(cic_final_evidence, problem):
    e = cic_final_evidence
    path = e.models / 'random_forest.pkl'
    if problem == 'missing':
        path.unlink()
    else:
        path.write_bytes(b'')
    report = cic_final.audit(e.root)
    assert not report.passed and report.checklist[-1]['check'] == 'five_model_artifacts'


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'negative', 'fractional', 'total', 'label'])
def test_cic_final_confusion_gates(cic_final_evidence, problem):
    e = cic_final_evidence
    rows = e.tables['confusion_matrix.csv']
    if problem == 'missing':
        rows.pop()
    elif problem == 'duplicate':
        rows.append(rows[0].copy())
    else:
        field, value = {'negative': ('count', -1), 'fractional': ('count', 1.5),
                        'total': ('count', 3), 'label': ('actual_class_label', 'VPN')}[problem]
        rows[0][field] = value
    write_cic_evidence(e.root, 'confusion_matrix.csv', rows)
    assert not cic_final.audit(e.root).passed


@pytest.mark.parametrize('kind,problem', [('roc', 'missing_class'), ('roc', 'order'), ('roc', 'endpoint'),
    ('roc', 'nan'), ('roc', 'macro'), ('pr', 'missing_class'), ('pr', 'order'), ('pr', 'endpoint'), ('pr', 'macro')])
def test_cic_final_curve_gates(cic_final_evidence, kind, problem):
    e = cic_final_evidence
    rows = e.tables[f'{kind}_curve.csv']
    if problem == 'missing_class':
        rows[:] = [r for r in rows if not (r['model'] == 'random_forest' and r['split'] == 'validation' and r['class_id'] == 0)]
    elif problem == 'order':
        rows[1]['threshold'] = 2 if kind == 'roc' else -1
    elif problem == 'endpoint':
        rows[2]['fpr' if kind == 'roc' else 'recall'] = .5
    elif problem == 'nan':
        rows[1]['tpr'] = 'nan'
    else:
        metric = 'macro_ovr_roc_auc' if kind == 'roc' else 'macro_ovr_auprc'
        e.tables['metrics.csv'][0][metric] = .5
        write_cic_evidence(e.root, 'metrics.csv', e.tables['metrics.csv'])
    write_cic_evidence(e.root, f'{kind}_curve.csv', rows)
    assert not cic_final.audit(e.root).passed


def test_cic_final_ap_is_not_trapezoidal_pr_area(cic_final_evidence):
    e, f = cic_final_evidence, cic_final
    # Recall 1->.5->0, left precisions .25,.5: AP=.375, trapz=.5625.
    rows = []
    for model in f.MODELS:
        for split in f.SPLITS:
            for c, label in enumerate(f.LABELS):
                for p, r, t in [(0.25, 1, 0), (.5, .5, .5), (1, 0, '')]:
                    rows.append(dict(dataset=f.DATASET, run_id=f.RUN, model=model, split=split,
                        class_id=c, class_label=label, precision=p, recall=r, threshold=t))
    write_cic_evidence(e.root, 'pr_curve.csv', rows)
    for row in e.tables['metrics.csv']:
        row['macro_ovr_auprc'] = .375
    write_cic_evidence(e.root, 'metrics.csv', e.tables['metrics.csv'])
    report = f.audit(e.root)
    assert report.passed
    assert all(r['average_precision_ovr'] == pytest.approx(.375) for r in report.finals['cic_per_class_results.csv'])


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'count', 'bounds', 'gap', 'ece'])
def test_cic_final_calibration_gates(cic_final_evidence, problem):
    e = cic_final_evidence
    rows = e.tables['calibration_bins.csv']
    if problem == 'missing':
        rows.pop()
    elif problem == 'duplicate':
        rows.append(rows[0].copy())
    elif problem == 'count':
        rows[14]['count'] = 7
    elif problem == 'bounds':
        rows[0]['lower_bound'] = .1
    elif problem == 'gap':
        rows[14]['absolute_gap'] = .1
    else:
        e.tables['metrics.csv'][0]['ece'] = .1
        write_cic_evidence(e.root, 'metrics.csv', e.tables['metrics.csv'])
    write_cic_evidence(e.root, 'calibration_bins.csv', rows)
    assert not cic_final.audit(e.root).passed


@pytest.mark.parametrize('problem', ['neural_short', 'neural_gap', 'multiple_yes', 'no_yes', 'wrong_best',
    'rf', 'booster_long', 'missing_booster', 'nonfinite'])
def test_cic_final_history_gates(cic_final_evidence, problem):
    e = cic_final_evidence
    rows = e.tables['training_history.csv']
    if problem == 'neural_short':
        rows.pop(0)
    elif problem == 'neural_gap':
        rows[0]['step'] = 9
    elif problem == 'multiple_yes':
        rows[0]['is_best_validation_step'] = 'YES'
    elif problem == 'no_yes':
        for row in rows:
            if row['model'] == '1d_cnn':
                row['is_best_validation_step'] = 'NO'
    elif problem == 'wrong_best':
        rows[0]['validation_loss'] = 0
    elif problem == 'rf':
        rows.append(dict(rows[0], model='random_forest'))
    elif problem == 'booster_long':
        rows.extend(dict(rows[0], model='xgboost', step=i) for i in range(4, 302))
    elif problem == 'missing_booster':
        rows[:] = [r for r in rows if r['model'] != 'lightgbm']
    else:
        rows[0]['train_loss'] = 'inf'
    write_cic_evidence(e.root, 'training_history.csv', rows)
    assert not cic_final.audit(e.root).passed


@pytest.mark.parametrize('problem', ['selection', 'repeat_test', 'premature_test', 'failed_run'])
def test_cic_final_test_isolation_config_gate(cic_final_evidence, problem):
    e = cic_final_evidence
    rows = e.tables['run_configuration.csv']
    if problem == 'selection':
        next(r for r in rows if r['key'] == 'test_selection')['value'] = 'ALLOWED'
    elif problem == 'repeat_test':
        rows.append(next(r.copy() for r in rows if r['key'] == 'test_inference_attempt'))
    elif problem == 'premature_test':
        index = next(i for i, r in enumerate(rows) if r['key'] == 'test_inference_attempt')
        rows.insert(0, rows.pop(index))
    else:
        rows[-1]['value'] = 'FAILED_PROTOCOL_OR_EXECUTION'
    write_cic_evidence(e.root, 'run_configuration.csv', rows)
    assert not cic_final.audit(e.root).passed


def test_cic_final_closure_requires_verified_existing_finals(cic_final_evidence):
    e, f = cic_final_evidence, cic_final
    assert not f.finalize(e.root, close=True).passed
    assert not (e.root / f.CLOSURE).exists()
    assert f.finalize(e.root, execute=True).passed
    path = e.root / f.FINAL_DIR / 'cic_test_results.csv'
    rows = f.read_rows(path)
    rows[0]['accuracy'] = '0.2'
    pd.DataFrame(rows).to_csv(path, index=False)
    report = f.finalize(e.root, close=True, execute=True)
    assert not report.passed and not (e.root / f.CLOSURE).exists()
    assert f.read_rows(path)[0]['accuracy'] == '0.2'


def test_cic_final_canonical_overwrite_preserves_unrelated_and_run(cic_final_evidence):
    e, f = cic_final_evidence, cic_final
    marker = e.root / 'metadata/traffic/ustc_experiment_closure.csv'
    marker.write_text('protected original closure', encoding='utf-8')
    assert f.finalize(e.root, execute=True).passed
    unrelated = e.root / f.FINAL_DIR / 'unrelated.csv'
    unrelated.write_text('preserve', encoding='utf-8')
    assert f.finalize(e.root, execute=True, close=True).passed
    snapshot = {p: p.read_bytes() for p in e.root.rglob('*') if p.is_file()}
    assert f.finalize(e.root, execute=True).passed
    assert f.finalize(e.root, execute=True, close=True).passed
    assert snapshot == {p: p.read_bytes() for p in e.root.rglob('*') if p.is_file()}
    assert {p.name for p in (e.root / f.FINAL_DIR).iterdir()} == set(f.FINAL_HEADERS) | {'unrelated.csv'}
    assert marker.read_text(encoding='utf-8') == 'protected original closure'


def test_cic_final_rejects_other_output_identity(cic_final_evidence):
    e, f = cic_final_evidence, cic_final
    assert f.finalize(e.root, execute=True).passed
    path = e.root / f.FINAL_DIR / 'cic_train_results.csv'
    rows = f.read_rows(path)
    rows[0]['authoritative_run'] = 'different'
    pd.DataFrame(rows).to_csv(path, index=False)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='another/unknown run'):
        f.finalize(e.root, execute=True)
    assert path.read_bytes() == before


def test_cic_final_cli_wrong_run_and_dry_close_never_write(cic_final_evidence, capsys):
    e, f = cic_final_evidence, cic_final
    assert f.main(['--project-root', str(e.root), '--run-id', 'wrong', '--execute']) == 1
    assert 'CIC_FINALIZATION=BLOCKED' in capsys.readouterr().out
    assert not (e.root / f.CHECKLIST).exists()
    assert f.main(['--project-root', str(e.root), '--run-id', f.RUN]) == 0
    assert not (e.root / f.FINAL_DIR).exists()


def test_cic_final_publication_failure_restores_existing(cic_final_evidence, monkeypatch):
    e, f = cic_final_evidence, cic_final
    assert f.finalize(e.root, execute=True).passed
    before = {p: p.read_bytes() for p in e.root.rglob('*') if p.is_file()}
    replace = f.os.replace
    failed = False
    def fail_once(src, dst):
        nonlocal failed
        if not failed and Path(dst).name == 'cic_validation_results.csv':
            failed = True
            raise OSError('synthetic write failure')
        return replace(src, dst)
    monkeypatch.setattr(f.os, 'replace', fail_once)
    with pytest.raises(OSError, match='synthetic write failure'):
        f.finalize(e.root, execute=True)
    assert before == {p: p.read_bytes() for p in e.root.rglob('*') if p.is_file()}


def test_cic_finalizer_dependency_boundary():
    import ast
    source = Path(cic_final.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    imports, calls = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
        elif isinstance(node, ast.Call):
            calls.append(node.func.attr if isinstance(node.func, ast.Attribute)
                         else node.func.id if isinstance(node.func, ast.Name) else '')
    assert set(imports) <= {'__future__', 'argparse', 'collections', 'csv', 'dataclasses', 'json', 'math',
                           'os', 'pathlib', 'shutil', 'tempfile'}
    assert not set(calls) & {'predict', 'predict_proba', 'fit', 'train', 'load_model', 'load', 'sha256', 'hash', 'sha256_file'}
    assert 'USTC-TFC2016' not in source
