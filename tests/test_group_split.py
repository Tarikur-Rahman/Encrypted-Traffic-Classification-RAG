import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
from itertools import combinations

from src.data import ustc_preprocess as ustc


def test_group_split_no_overlap(tmp_path):
    data = tmp_path / "data.csv"
    rows = []

    for g in range(30):
        for i in range(4):
            rows.append({
                "capture_group": f"g{g}",
                "target_binary": g % 2,
                "x": g + i,
            })

    pd.DataFrame(rows).to_csv(data, index=False)

    repo_root = Path(__file__).resolve().parents[1]

    cmd = [
        sys.executable,
        "-m",
        "src.data.group_split",
        "--project-root",
        str(tmp_path),
        "--input",
        str(data),
        "--target",
        "target_binary",
        "--group",
        "capture_group",
    ]

    p = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert '"train_test": 0' in p.stdout
    assert '"train_validation": 0' in p.stdout
    assert '"validation_test": 0' in p.stdout
    assert "PREFLIGHT_ONLY=True" in p.stdout


@pytest.fixture
def ustc_frame():
    rows = []
    for label in ['benign', 'malware']:
        for family in range(10):
            for row in range(6 + family):
                capture = row % 5 if label == 'benign' and family == 0 else 0
                source = f'{label}/family{family}/capture{capture}.pcap'
                rows.append(dict(binary_label=label, semantic_family=f'{label}_family{family}',
                    capture_group=source, source_pcap=source, id=len(rows),
                    src_port=443, src_ip='192.0.2.1', application_name='excluded',
                    bidirectional_first_seen_ms=123456,
                    packets=float(row + family), duration=float(2 * row + family), constant=5.0))
    return pd.DataFrame(rows)


def candidates(frame):
    return ustc.inspect_input(frame, expected_rows=len(frame), expected_predictors=3)


def test_ustc_family_split_is_deterministic_and_disjoint(ustc_frame):
    frame = ustc_frame
    predictors = candidates(frame)
    assert predictors == ['packets', 'duration', 'constant']
    assignment = ustc.assign_families(frame)
    shuffled = frame.sample(frac=1, random_state=123).reset_index(drop=True)
    pd.testing.assert_frame_equal(assignment, ustc.assign_families(shuffled))
    parts = ustc.split_rows(frame, assignment)
    assert np.array_equal(np.sort(np.concatenate(list(parts.values()))), np.arange(len(frame)))
    for split, expected in [('train', 6), ('validation', 2), ('test', 2)]:
        sub = frame.iloc[parts[split]]
        assert sub.groupby('binary_label').semantic_family.nunique().to_dict() == {'benign': expected, 'malware': expected}
        assert sub.semantic_family.nunique() == 2 * expected
    for left, right in combinations(parts, 2):
        for column in ['semantic_family', 'capture_group', 'source_pcap']:
            assert not set(frame.iloc[parts[left]][column]) & set(frame.iloc[parts[right]][column])
    multi_capture_family = assignment[assignment.semantic_family == 'benign_family0'].iloc[0]
    assert multi_capture_family.capture_group_count == 5
    assert len(frame.iloc[parts[multi_capture_family['split']]].query("semantic_family == 'benign_family0'")) == 6
    # Independent exhaustive check of the flow-balance objective.
    for label in ['benign', 'malware']:
        sub = assignment[assignment.binary_label == label]
        sizes = sub.flow_count.tolist()
        total = sum(sizes)
        def score(counts):
            return sum((5 * count - ratio * total) ** 2 for count, ratio in zip(counts, [3, 1, 1]))
        scores = []
        for test_ids in combinations(range(10), 2):
            for val_ids in combinations([i for i in range(10) if i not in test_ids], 2):
                test = sum(sizes[i] for i in test_ids)
                validation = sum(sizes[i] for i in val_ids)
                scores.append(score([total - validation - test, validation, test]))
        actual = [int(sub.loc[sub['split'] == split, 'flow_count'].sum()) for split in ustc.SPLITS]
        assert score(actual) == min(scores)


def test_ustc_train_only_parameters_and_transformation(ustc_frame):
    frame = ustc_frame
    parts = ustc.split_rows(frame, ustc.assign_families(frame))
    train = parts['train']
    heldout = np.concatenate([parts['validation'], parts['test']])
    frame.loc[train, 'packets'] = np.resize([1.0, np.nan, 3.0, np.inf, -np.inf, 9.0], len(train))
    frame.loc[heldout, 'packets'] = np.resize([1000.0, np.nan, -np.inf], len(heldout))
    frame.loc[heldout, 'constant'] = np.arange(len(heldout)) + 1000.0
    output, parameters, diagnostics = ustc.fit_transform(frame, candidates(frame), parts)
    param = {r['feature_name']: r for r in parameters}
    finite_train = frame.loc[train, 'packets'].replace([np.inf, -np.inf], np.nan).to_numpy()
    median = np.nanmedian(finite_train)
    imputed_train = np.where(np.isnan(finite_train), median, finite_train)
    assert param['packets']['train_median'] == median
    assert param['packets']['train_mean'] == pytest.approx(imputed_train.mean())
    assert param['packets']['train_std'] == pytest.approx(imputed_train.std(ddof=0))
    assert param['constant']['retained'] is False
    assert param['constant']['drop_reason'] == 'zero_variance_train'
    assert param['constant']['final_feature_order'] is None
    for split, processed in output.items():
        assert list(processed.columns) == ['packets', 'duration', 'target_binary', 'binary_label',
                                           'semantic_family', 'capture_group', 'source_pcap']
        raw = frame.iloc[parts[split]].packets.to_numpy()
        imputed = np.where(np.isfinite(raw), raw, median)
        np.testing.assert_allclose(processed.packets, (imputed - imputed_train.mean()) / imputed_train.std(ddof=0))
        assert np.isfinite(processed[['packets', 'duration']].to_numpy()).all()
        assert processed.target_binary.tolist() == frame.iloc[parts[split]].binary_label.map({'benign': 0, 'malware': 1}).tolist()
    np.testing.assert_allclose(output['train'][['packets', 'duration']].mean(), 0, atol=1e-12)
    np.testing.assert_allclose(output['train'][['packets', 'duration']].std(ddof=0), 1, atol=1e-12)
    changed = frame.copy()
    changed.loc[heldout, ['packets', 'duration', 'constant']] = 1e12
    _, changed_parameters, _ = ustc.fit_transform(changed, candidates(changed), parts)
    assert parameters == changed_parameters
    info = {r['feature_name']: r for r in diagnostics}['packets']
    assert info['missing_before'] == int(frame.packets.isna().sum())
    assert info['infinite_before'] == int(np.isinf(frame.packets).sum())
    assert info['train_missing_before'] == int(np.isnan(finite_train).sum())


def synthetic_input(tmp_path, frame, monkeypatch):
    path = tmp_path / 'data/interim/USTC-TFC2016/flows.csv'
    path.parent.mkdir(parents=True)
    frame.to_csv(path, index=False)
    monkeypatch.setattr(ustc, 'EXPECTED_ROWS', len(frame))
    monkeypatch.setattr(ustc, 'EXPECTED_PREDICTORS', 3)
    return path


def test_ustc_preflight_never_splits_fits_or_writes(tmp_path, ustc_frame, monkeypatch, capsys):
    path = synthetic_input(tmp_path, ustc_frame, monkeypatch)
    before = path.read_bytes()
    monkeypatch.setattr(ustc, 'assign_families', lambda *args: pytest.fail('Preflight attempted a split'))
    monkeypatch.setattr(ustc, 'fit_transform', lambda *args: pytest.fail('Preflight fitted preprocessing'))
    assert ustc.main(['--project-root', str(tmp_path)]) == 0
    report = capsys.readouterr().out
    assert 'PREFLIGHT_ONLY=True' in report
    assert 'candidate_predictor_count=3' in report
    assert 'semantic_family_count=20' in report
    assert '6/2/2' in report
    assert not (tmp_path / 'data/processed').exists()
    assert not (tmp_path / 'results').exists()
    assert path.read_bytes() == before


def test_ustc_processed_outputs_and_parameter_replay(tmp_path, ustc_frame, monkeypatch):
    path = synthetic_input(tmp_path, ustc_frame, monkeypatch)
    before = path.read_bytes(), path.stat().st_mtime_ns
    output, results = ustc.preprocess(tmp_path, execute=True)
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)
    assert {p.name for p in output.iterdir()} == {'train.csv', 'validation.csv', 'test.csv', 'preprocessing_parameters.csv'}
    headers = {
        'split_distribution.csv': 'dataset split class flow_count percentage_within_split percentage_of_class run_id timestamp_utc',
        'split_family_assignment.csv': 'dataset binary_label semantic_family split flow_count capture_group_count run_id timestamp_utc',
        'feature_processing.csv': 'dataset feature_name original_dtype missing_before infinite_before train_missing_before retained drop_reason final_feature_order run_id timestamp_utc',
        'processing_summary.csv': 'dataset input_rows train_rows validation_rows test_rows input_predictor_count final_predictor_count zero_variance_removed_count train_family_count validation_family_count test_family_count run_id timestamp_utc',
    }
    assert {p.name for p in results.iterdir()} == set(headers)
    for name, expected in headers.items():
        table = pd.read_csv(results / name)
        assert list(table.columns) == expected.split()
        assert set(table.run_id) == {results.name}
    parameters = pd.read_csv(output / 'preprocessing_parameters.csv')
    assert list(parameters.columns) == ['feature_name', 'retained', 'drop_reason', 'train_median', 'train_mean', 'train_std', 'final_feature_order']
    kept = parameters[parameters.retained].sort_values('final_feature_order')
    assert kept.final_feature_order.tolist() == [1, 2]
    assignment = pd.read_csv(results / 'split_family_assignment.csv').set_index('semantic_family')['split']
    total = 0
    for split in ustc.SPLITS:
        processed = pd.read_csv(output / f'{split}.csv')
        assert list(processed.columns) == ['packets', 'duration', 'target_binary', 'binary_label', 'semantic_family', 'capture_group', 'source_pcap']
        total += len(processed)
        raw = ustc_frame[ustc_frame.semantic_family.map(assignment) == split]
        for row in kept.itertuples():
            replay = (raw[row.feature_name].fillna(row.train_median) - row.train_mean) / row.train_std
            np.testing.assert_allclose(processed[row.feature_name], replay)
    assert total == len(ustc_frame)
    dist = pd.read_csv(results / 'split_distribution.csv')
    assert dist.flow_count.sum() == len(ustc_frame)
    np.testing.assert_allclose(dist.groupby('split').percentage_within_split.sum(), 100)
    np.testing.assert_allclose(dist.groupby('class').percentage_of_class.sum(), 100)
    summary = pd.read_csv(results / 'processing_summary.csv').iloc[0]
    assert summary.input_rows == summary.train_rows + summary.validation_rows + summary.test_rows
    assert [summary.train_family_count, summary.validation_family_count, summary.test_family_count] == [12, 4, 4]
    assert [summary.input_predictor_count, summary.final_predictor_count, summary.zero_variance_removed_count] == [3, 2, 1]


@pytest.mark.parametrize('existing', ['train.csv', 'validation.csv', 'test.csv', 'preprocessing_parameters.csv'])
def test_ustc_no_overwrite_before_read_or_fit(tmp_path, monkeypatch, existing):
    output = tmp_path / 'data/processed/USTC-TFC2016'
    output.mkdir(parents=True)
    (output / existing).write_bytes(b'existing artifact')
    monkeypatch.setattr(ustc, 'fit_transform', lambda *args: pytest.fail('Should fail before fit'))
    with pytest.raises(FileExistsError, match='overwrite'):
        ustc.preprocess(tmp_path, execute=True)
    assert (output / existing).read_bytes() == b'existing artifact'
    assert list(output.iterdir()) == [output / existing]
    assert not (tmp_path / 'results').exists()


def test_ustc_rejects_invalid_predictors_and_training_data(ustc_frame):
    frame = ustc_frame.copy()
    frame['packets'] = 'unexpected string'
    with pytest.raises(ValueError, match='real numeric'):
        candidates(frame)
    frame = ustc_frame.copy()
    parts = ustc.split_rows(frame, ustc.assign_families(frame))
    frame.loc[parts['train'], 'packets'] = np.inf
    with pytest.raises(ValueError, match='TRAIN median'):
        ustc.fit_transform(frame, candidates(frame), parts)
    with pytest.raises(ValueError, match='Stage-2'):
        ustc.fit_transform(ustc_frame, ['id'], parts)
    frame[['packets', 'duration', 'constant']] = 1.0
    with pytest.raises(ValueError, match='No predictors remain'):
        ustc.fit_transform(frame, candidates(frame), parts)


def test_ustc_validates_input_contract_and_row_disjointness(ustc_frame):
    with pytest.raises(ValueError, match='562549'):
        ustc.inspect_input(ustc_frame, expected_rows=562549, expected_predictors=3)
    with pytest.raises(ValueError, match='60 predictor'):
        ustc.inspect_input(ustc_frame, expected_rows=len(ustc_frame), expected_predictors=60)
    invalid = ustc_frame.copy()
    invalid.loc[0, 'binary_label'] = 'unknown'
    with pytest.raises(ValueError, match='benign and malware'):
        candidates(invalid)
    invalid = ustc_frame.copy()
    invalid.loc[0, 'semantic_family'] = 'malware_family0'
    with pytest.raises(ValueError, match='one binary class'):
        candidates(invalid)
    invalid = ustc_frame.copy()
    invalid.loc[0, 'capture_group'] = invalid.iloc[-1].capture_group
    with pytest.raises(ValueError, match='Inconsistent provenance'):
        candidates(invalid)
    parts = ustc.split_rows(ustc_frame, ustc.assign_families(ustc_frame))
    parts['test'] = np.append(parts['test'], parts['train'][0])
    with pytest.raises(ValueError, match='disjoint'):
        ustc.validate_split(ustc_frame, parts)
