import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.common import audit, io
from src.data.audit_raw import audit_raw, provenance
from src.data.audit_cic_schema import audit_cic_schema
from src.data import cic_define_contract as cic
from src.data import cic_preprocess as cic_prep

TAIL = 'run_id timestamp_utc'
RAW_HEADERS = {
    'raw_file_manifest.csv': 'dataset relative_path extension size_bytes authoritative_binary_label',
    'dataset_summary.csv': 'dataset file_count total_bytes pcap_count csv_count benign_pcap_count malware_pcap_count',
    'extension_summary.csv': 'dataset extension file_count total_bytes',
    'pcap_inventory.csv': 'dataset binary_label semantic_family capture_group relative_path size_bytes',
    'family_inventory.csv': 'dataset binary_label semantic_family pcap_count total_bytes',
    'ustc_audit_checks.csv': 'check_name observed_value expected_value status details',
}
CIC_HEADERS = {
    'cic_dataset_summary.csv': 'dataset source_file rows columns size_bytes candidate_label_column_count',
    'schema_columns.csv': 'dataset source_file column_index column_name dtype missing_count missing_fraction unique_count is_candidate_label',
    'label_value_counts.csv': 'dataset source_file label_column label_value count fraction',
    'missing_values.csv': 'dataset source_file column_name missing_count missing_fraction',
    'duplicate_column_names.csv': 'dataset source_file collision_type normalized_name column_name column_index',
    'cic_candidate_label_summary.csv': 'dataset source_file label_column unique_value_count missing_count missing_fraction',
}

def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))

def verify_outputs(out, headers):
    assert {p.name for p in out.iterdir()} == set(headers) | {'artifact_index.csv'}
    for name, columns in headers.items():
        frame = pd.read_csv(out / name)
        assert list(frame.columns) == (columns + ' ' + TAIL).split()
    index = pd.read_csv(out / 'artifact_index.csv')
    assert list(index.columns) == ['relative_path', 'size_bytes']
    assert set(index.relative_path) == set(headers)
    assert len(index) == len(headers)
    assert index.size_bytes.dtype.kind in 'iu'
    for row in index.itertuples():
        path = out / row.relative_path
        assert row.size_bytes == path.stat().st_size

def make_file(base, name, content):
    path = base / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path

@pytest.mark.parametrize('relative,label,family', [
    ('Benign/FTP.pcap', 'benign', 'FTP'),
    ('Benign/SMB/SMB/SMB-1.pcap', 'benign', 'SMB'),
    ('Benign/Weibo/Weibo/Weibo-4.pcap', 'benign', 'Weibo'),
    ('Malware/Cridex/Cridex.pcap', 'malware', 'Cridex'),
    ('Malware/Zeus.pcap', 'malware', 'Zeus'),
    ('Other/Benign/Fake.pcap', '', ''),
    ('benign/Fake.pcap', '', ''),
    ('Benign/Malware/Thing.pcap', 'benign', 'Malware'),
])
def test_provenance(relative, label, family, tmp_path):
    assert provenance(tmp_path / relative, tmp_path)[:2] == (label, family)

def test_ustc_inventory_and_immutability(tmp_path):
    base = tmp_path / 'data/raw/USTC-TFC2016'
    paths = ['Benign/FTP.pcap', 'Benign/SMB/SMB/SMB-1.pcap', 'Benign/SMB/SMB/SMB-2.pcap',
             'Benign/SMB/other/SMB-1.pcap', 'Malware/Zeus.pcap', 'Other/Benign/unknown.pcap', 'Benign/readme.txt']
    for i, name in enumerate(paths):
        make_file(base, name, str(i).encode())
    before = {p.relative_to(base): (p.read_bytes(), p.stat().st_mtime_ns) for p in base.rglob('*') if p.is_file()}
    legacy = make_file(tmp_path, 'metadata/audits/old.json', b'legacy')
    out = audit_raw(tmp_path, 'USTC-TFC2016')
    verify_outputs(out, RAW_HEADERS)
    assert before == {p.relative_to(base): (p.read_bytes(), p.stat().st_mtime_ns) for p in base.rglob('*') if p.is_file()}
    assert legacy.read_bytes() == b'legacy'
    inventory = pd.read_csv(out / 'pcap_inventory.csv')
    assert inventory.capture_group.nunique() == 6
    assert inventory[inventory.semantic_family == 'SMB'].capture_group.nunique() == 3
    summary = pd.read_csv(out / 'dataset_summary.csv').iloc[0]
    assert summary.benign_pcap_count == 4
    checks = {r['check_name']: r for r in read(out / 'ustc_audit_checks.csv')}
    assert checks['raw_file_count']['status'] == 'FAIL'
    assert checks['missing_binary_label_pcap_count']['observed_value'] == '1'
    assert checks['duplicate_capture_group_count']['status'] == 'PASS'

def test_complete_release_checks_derived(tmp_path):
    base = tmp_path / 'data/raw/USTC-TFC2016'
    for label in ['Benign', 'Malware']:
        for family in range(10):
            count = 5 if label == 'Benign' and family == 0 else 1
            for capture in range(count):
                name = f'{label}/family{family}/capture{capture}.pcap'
                make_file(base, name, name.encode())
    make_file(base, 'readme.txt', b'doc')
    make_file(base, 'license.txt', b'license')
    out = audit_raw(tmp_path, 'USTC-TFC2016')
    checks = read(out / 'ustc_audit_checks.csv')
    assert {r['status'] for r in checks} == {'PASS'}
    assert [r['check_name'] for r in checks] == [
        'raw_file_count', 'pcap_count', 'benign_pcap_count', 'malware_pcap_count',
        'semantic_family_count', 'benign_family_count', 'malware_family_count',
        'duplicate_capture_group_count', 'missing_binary_label_pcap_count',
    ]

def test_cic_exact_labels_and_missing(tmp_path):
    base = tmp_path / 'data/raw/CIC-Darknet2020'
    values = ['AUDIO-STREAMING', 'Audio-Streaming', 'AUDIO-STREAMING', '001', 'NA', 'NULL', '<NA>', ' spaced ', 'a,b', 'line\nbreak', 'é', '']
    path = base / 'source.csv'
    base.mkdir(parents=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['Label', 'number'])
        writer.writerows((value, i) for i, value in enumerate(values))
    before = path.read_bytes(), path.stat().st_mtime_ns
    out = audit_cic_schema(tmp_path)
    verify_outputs(out, CIC_HEADERS)
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)
    counts = read(out / 'label_value_counts.csv')
    assert {r['label_value'] for r in counts} == set(values)
    assert next(r['count'] for r in counts if r['label_value'] == 'AUDIO-STREAMING') == '2'
    assert next(r['count'] for r in counts if r['label_value'] == 'Audio-Streaming') == '1'
    assert sum(int(r['count']) for r in counts) == len(values)
    assert sum(float(r['fraction']) for r in counts) == pytest.approx(1)
    missing = read(out / 'missing_values.csv')
    assert [r['missing_count'] for r in missing] == ['1', '0']
    assert pd.read_csv(out / 'label_value_counts.csv')['count'].dtype.kind in 'iu'
    raw_out = audit_raw(tmp_path, 'CIC-Darknet2020')
    verify_outputs(raw_out, {k: v for k, v in RAW_HEADERS.items() if k in ['raw_file_manifest.csv', 'dataset_summary.csv', 'extension_summary.csv']})

def test_original_duplicate_headers(tmp_path):
    make_file(tmp_path / 'data/raw/CIC-Darknet2020', 'data.csv', b'Label,Label,LABEL,x\nOne,Two,Three,1\n')
    out = audit_cic_schema(tmp_path)
    schema = read(out / 'schema_columns.csv')
    assert [r['column_name'] for r in schema] == ['Label', 'Label.1', 'LABEL', 'x']
    assert [r['column_index'] for r in schema] == ['0', '1', '2', '3']
    collisions = read(out / 'duplicate_column_names.csv')
    assert len([r for r in collisions if r['collision_type'] == 'exact']) == 2
    assert len([r for r in collisions if r['collision_type'] == 'case_insensitive']) == 3
    assert [(r['column_name'], r['normalized_name'], r['column_index']) for r in collisions
            if r['collision_type'] == 'exact'] == [('Label', 'Label', '0'), ('Label.1', 'Label', '1')]
    assert [r['label_value'] for r in read(out / 'label_value_counts.csv')] == ['One', 'Two', 'Three']

@pytest.mark.parametrize('second_header', ['Label.1', 'Label'])
def test_cic_label_column_identity(tmp_path, second_header):
    base = tmp_path / 'data/raw/CIC-Darknet2020'
    path = make_file(base, 'data.csv', b'')
    frame = pd.DataFrame([
        ['Non-Tor', 'AUDIO-STREAMING'],
        ['Non-Tor', 'Audio-Streaming'],
        ['VPN', 'Browsing'],
        ['VPN', 'Audio-Streaming'],
    ], columns=['Label', second_header])
    frame.to_csv(path, index=False)
    before = path.read_bytes()
    out = audit_cic_schema(tmp_path)
    verify_outputs(out, CIC_HEADERS)
    assert path.read_bytes() == before
    counts = read(out / 'label_value_counts.csv')
    assert sorted((r['label_column'], r['label_value'], int(r['count']), float(r['fraction']))
                  for r in counts) == sorted([
        ('Label', 'Non-Tor', 2, 0.5), ('Label', 'VPN', 2, 0.5),
        ('Label.1', 'AUDIO-STREAMING', 1, 0.25),
        ('Label.1', 'Audio-Streaming', 2, 0.5),
        ('Label.1', 'Browsing', 1, 0.25),
    ])
    summary = read(out / 'cic_candidate_label_summary.csv')
    assert [(r['label_column'], int(r['unique_value_count'])) for r in summary] == [
        ('Label', 2), ('Label.1', 3)]
    schema = read(out / 'schema_columns.csv')
    assert [(r['column_name'], int(r['column_index'])) for r in schema] == [
        ('Label', 0), ('Label.1', 1)]
    assert [r['column_name'] for r in read(out / 'missing_values.csv')] == ['Label', 'Label.1']

def test_existing_suffixes_follow_pandas_identity(tmp_path):
    path = make_file(tmp_path / 'data/raw/CIC-Darknet2020', 'data.csv',
                     b'Label,Label,Label.1\n001,002,003\n')
    expected = list(pd.read_csv(path, nrows=0).columns)
    out = audit_cic_schema(tmp_path)
    assert [r['column_name'] for r in read(out / 'schema_columns.csv')] == expected
    assert [r['label_column'] for r in read(out / 'cic_candidate_label_summary.csv')] == expected
    counts = read(out / 'label_value_counts.csv')
    assert [r['label_column'] for r in counts] == expected
    assert [r['label_value'] for r in counts] == ['001', '002', '003']

def test_run_collision_never_overwrites(tmp_path, monkeypatch):
    class Frozen:
        @staticmethod
        def now(tz):
            return datetime(2026, 9, 19, 1, 2, 3, tzinfo=timezone.utc)
    monkeypatch.setattr(audit, 'datetime', Frozen)
    first = audit.AuditRun(tmp_path, 'dataset', 'raw_audit')
    first.write('test.csv', ['value'], [{'value': 'original'}])
    original = (first.path / 'test.csv').read_bytes()
    with pytest.raises(FileExistsError):
        audit.AuditRun(tmp_path, 'dataset', 'raw_audit')
    with pytest.raises(FileExistsError):
        first.write('test.csv', ['value'], [{'value': 'changed'}])
    assert first.path.name == '20260919T010203Z'
    assert (first.path / 'test.csv').read_bytes() == original

def test_atomic_publication_race(tmp_path, monkeypatch):
    path = tmp_path / 'out.csv'
    real_link = io.os.link
    def raced_link(src, dst):
        Path(dst).write_bytes(b'other writer')
        real_link(src, dst)
    monkeypatch.setattr(io.os, 'link', raced_link)
    with pytest.raises(FileExistsError):
        io.write_csv_atomic(path, ['x'], [{'x': 1}])
    assert path.read_bytes() == b'other writer'
    assert list(tmp_path.iterdir()) == [path]

@pytest.mark.parametrize('content', [b'Label,x\n', b'x\n1\n'])
def test_empty_or_no_candidate(tmp_path, content):
    make_file(tmp_path / 'data/raw/CIC-Darknet2020', 'data.csv', content)
    verify_outputs(audit_cic_schema(tmp_path), CIC_HEADERS)

def test_malformed_csv_has_no_completion_index(tmp_path):
    make_file(tmp_path / 'data/raw/CIC-Darknet2020', 'data.csv', b'Label,x\na,1,extra\n')
    with pytest.raises(ValueError, match='fields'):
        audit_cic_schema(tmp_path)
    assert not list((tmp_path / 'results').rglob('artifact_index.csv'))


# CIC-1 execution tests use only synthetic CSVs under tmp_path. Production
# source gates remain fixed; no test invokes contract generation on real data.
CIC_CONTRACT_HEADERS = {
    'cic_label_mapping.csv': ['class_id', 'raw_label', 'canonical_label', 'probability_column'],
    'cic_feature_contract.csv': ['source_index', 'column_name', 'role', 'predictor_candidate',
        'model_allowed', 'preprocessing_policy', 'reason'],
    'cic_split_manifest.csv': ['dataset', 'flow_id', 'split', 'group_row_count', 'non_tor_rows',
        'nonvpn_rows', 'tor_rows', 'vpn_rows', 'has_label_conflict', 'seed', 'split_method'],
}


@pytest.fixture
def cic_root(tmp_path, monkeypatch):
    rows = []
    for class_id, label in enumerate(cic.LABELS):
        for group in range(50):
            identity = f'{class_id}-{group:03}'
            if class_id == 0 and group < 2:
                identity = ['001', '1'][group]
            rows.extend((identity, label) for _ in range(3))
    for group in range(10):
        rows.extend([(f' mixed-{group} ', 'Non-Tor'), (f' mixed-{group} ', 'NonVPN')])
    frame = pd.DataFrame({
        'Flow ID': [r[0] for r in rows], 'Src IP': ['10.0.0.1', '10.0.0.2'] * (len(rows) // 2),
        'Src Port': '00443', 'Dst IP': ['10.0.0.2', '10.0.0.1'] * (len(rows) // 2),
        'Dst Port': '080', 'Protocol': 6, 'Timestamp': '2020-01-01',
        'Flow Bytes/s': ['NaN', 'inf', '-inf'] + ['3.5'] * (len(rows) - 3),
        'Flow Packets/s': 1.0,
        **{f'feature_{i}': i for i in range(74)},
        'Label': [r[1] for r in rows],
        'Label.1': ['AUDIO-STREAMING', 'Audio-Streaming'] * (len(rows) // 2),
    })
    source = tmp_path / cic.RAW_SOURCE
    source.parent.mkdir(parents=True)
    # Exercise the actual duplicate raw Label header, not only an explicit .1.
    raw = frame.copy()
    raw.columns = [name if name != 'Label.1' else 'Label' for name in raw.columns]
    raw.to_csv(source, index=False)
    closure = tmp_path / cic.CLOSURE
    closure.parent.mkdir(parents=True)
    pd.DataFrame([dict(dataset='USTC-TFC2016', stage_status='CLOSED_FROZEN',
                       frozen_for_retraining='YES', frozen_for_test_reuse='YES')]).to_csv(closure, index=False)
    monkeypatch.setattr(cic, 'EXPECTED_ROWS', len(frame))
    monkeypatch.setattr(cic, 'EXPECTED_COUNTS', frame.Label.value_counts().to_dict())
    monkeypatch.setattr(cic, 'EXPECTED_GROUPS', 210)
    monkeypatch.setattr(cic, 'EXPECTED_SIZE', source.stat().st_size)
    return tmp_path, frame


def test_cic_production_source_gates():
    assert (cic.EXPECTED_ROWS, cic.EXPECTED_COLUMNS, cic.EXPECTED_SIZE) == (158616, 85, 73325208)
    assert cic.EXPECTED_COUNTS == {'Non-Tor': 110442, 'NonVPN': 23863, 'Tor': 1392, 'VPN': 22919}
    assert (cic.EXPECTED_GROUPS, cic.EXPECTED_CONFLICTS, cic.SEED) == (79160, 10, 42)
    with pytest.raises(ValueError, match='158616 source rows'):
        cic.inspect_source(pd.DataFrame({'Label': ['Non-Tor']}))


@pytest.mark.parametrize('field,value', [
    ('dataset', 'other'), ('stage_status', 'OPEN'),
    ('frozen_for_retraining', 'NO'), ('frozen_for_test_reuse', 'NO'),
])
def test_cic_ustc_freeze_gate(cic_root, monkeypatch, field, value):
    root, _ = cic_root
    closure = pd.read_csv(root / cic.CLOSURE)
    closure.loc[0, field] = value
    closure.to_csv(root / cic.CLOSURE, index=False)
    monkeypatch.setattr(cic, 'read_source', lambda *_: pytest.fail('Source read before USTC gate'))
    with pytest.raises(ValueError, match='USTC protection'):
        cic.define_contract(root, execute=True)
    assert not (root / cic.OUTPUT_DIRECTORY / cic.MAIN_FILE).exists()


def test_cic_missing_duplicate_closure_and_preflight(cic_root, monkeypatch, capsys):
    root, _ = cic_root
    monkeypatch.setattr(cic, 'read_source', lambda *_: pytest.fail('Preflight read source'))
    assert cic.main(['--project-root', str(root)]) == 0
    assert 'PREFLIGHT_ONLY=True' in capsys.readouterr().out
    closure = pd.read_csv(root / cic.CLOSURE)
    pd.concat([closure, closure]).to_csv(root / cic.CLOSURE, index=False)
    with pytest.raises(ValueError, match='USTC protection'):
        cic.define_contract(root, execute=True)
    (root / cic.CLOSURE).unlink()
    with pytest.raises(FileNotFoundError):
        cic.define_contract(root, execute=True)


@pytest.mark.parametrize('value,error', [('', 'Missing Label'), (None, 'Missing Label'),
    ('non-tor', 'four authorized'), (' Non-Tor', 'four authorized'), ('unknown', 'four authorized'),
    ('VPN', 'class counts')])
def test_cic_rejects_target_changes(cic_root, value, error):
    _, frame = cic_root
    frame.loc[0, 'Label'] = value
    with pytest.raises(ValueError, match=error):
        cic.inspect_source(frame)


@pytest.mark.parametrize('problem,error', [
    ('rows', 'source rows'), ('columns', '85 distinct'), ('duplicate', '85 distinct'),
    ('missing_column', 'required label/provenance'), ('empty_group', 'Missing provenance'),
    ('numeric_group', 'exact source string'), ('nonnumeric', 'nonnumeric'),
    ('groups', 'distinct Flow ID'), ('conflicts', 'mixed-label'),
])
def test_cic_rejects_invalid_source(cic_root, monkeypatch, problem, error):
    _, frame = cic_root
    if problem == 'rows':
        frame = frame.iloc[:-1]
    elif problem == 'columns':
        frame = frame.drop(columns='feature_0')
    elif problem == 'duplicate':
        frame = frame.rename(columns={'feature_0': 'feature_1'})
    elif problem == 'missing_column':
        frame = frame.rename(columns={'Label.1': 'Other'})
    elif problem == 'empty_group':
        frame.loc[0, 'Flow ID'] = ''
    elif problem == 'numeric_group':
        frame['Flow ID'] = frame['Flow ID'].astype(object)
        frame.loc[0, 'Flow ID'] = 1
    elif problem == 'nonnumeric':
        frame.loc[0, 'Flow Bytes/s'] = 'not-a-number-feature'
    elif problem == 'groups':
        monkeypatch.setattr(cic, 'EXPECTED_GROUPS', 211)
    else:
        monkeypatch.setattr(cic, 'EXPECTED_CONFLICTS', 11)
    with pytest.raises(ValueError, match=error):
        cic.inspect_source(frame)


def test_cic_source_size_and_csv_width(cic_root, monkeypatch):
    root, _ = cic_root
    source = root / cic.RAW_SOURCE
    with source.open('a', encoding='utf-8') as stream:
        stream.write('bad,row\n')
    with pytest.raises(ValueError, match='source size_bytes'):
        cic.read_source(root)
    monkeypatch.setattr(cic, 'EXPECTED_SIZE', source.stat().st_size)
    with pytest.raises(ValueError, match='Malformed source CSV'):
        cic.read_source(root)


def test_cic_contract_synthetic_execution(cic_root, monkeypatch):
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from src.models import evaluation
    from src.common import hashing

    root, original = cic_root
    def forbidden(*args, **kwargs):
        pytest.fail('CIC-1 attempted fitting, inference, evaluation, or a digest')
    for cls, method in [(SimpleImputer, 'fit'), (StandardScaler, 'fit'),
                        (RandomForestClassifier, 'fit'), (RandomForestClassifier, 'predict'),
                        (RandomForestClassifier, 'predict_proba')]:
        monkeypatch.setattr(cls, method, forbidden)
    monkeypatch.setattr(evaluation, 'evaluate', forbidden)
    monkeypatch.setattr(hashing, 'sha256_file', forbidden)
    legacy = make_file(root, 'results/audits/CIC-Darknet2020/schema_audit/20260919T114031Z/legacy.csv', b'preserve')
    unrelated = make_file(root, 'metadata/traffic/unrelated.csv', b'unchanged')
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
    source = cic.read_source(root)
    assert list(source.columns) == list(original.columns)
    assert {'001', '1', ' mixed-0 '}.issubset(set(source['Flow ID']))
    assert set(source['Label.1']) == {'AUDIO-STREAMING', 'Audio-Streaming'}
    source_before = source.copy(deep=True)
    # Exercise the public CLI as well as the in-memory source immutability gate.
    monkeypatch.setattr(cic, 'read_source', lambda _: source)
    assert cic.main(['--project-root', str(root), '--execute']) == 0
    pd.testing.assert_frame_equal(source, source_before)
    directory = root / cic.OUTPUT_DIRECTORY
    new = {p.relative_to(directory).as_posix() for p in directory.iterdir() if p not in before}
    assert new == set(CIC_CONTRACT_HEADERS) | {'cic_phase1_contract.csv'}
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    assert legacy.read_bytes() == b'preserve' and unrelated.read_bytes() == b'unchanged'
    assert not (root / 'data/processed').exists() and not (root / 'models').exists()
    for name, fields in CIC_CONTRACT_HEADERS.items():
        assert list(pd.read_csv(directory / name).columns) == fields
    mapping = pd.read_csv(directory / cic.MAPPING_FILE)
    assert mapping.to_dict('records') == [
        {'class_id': i, 'raw_label': label, 'canonical_label': label, 'probability_column': probability}
        for i, (label, probability) in enumerate(zip(
            ['Non-Tor', 'NonVPN', 'Tor', 'VPN'],
            ['class_0_non_tor', 'class_1_nonvpn', 'class_2_tor', 'class_3_vpn']))]
    features = pd.read_csv(directory / cic.FEATURE_FILE)
    assert features.source_index.tolist() == list(range(85))
    assert features.column_name.tolist() == original.columns.tolist()
    allowed = features.loc[features.model_allowed == 'YES', 'column_name'].tolist()
    assert allowed == [c for c in original if c not in cic.EXCLUDED]
    assert len(allowed) == 77
    roles = features.set_index('column_name')
    assert roles.loc[list(cic.EXCLUDED), 'model_allowed'].eq('NO').all()
    assert roles.loc[list(cic.PROVENANCE), 'role'].eq('provenance_only').all()
    assert roles.loc['Label.1', 'role'] == 'auxiliary_label'
    assert 'split_group' in roles.loc['Flow ID', 'reason']
    # Constants remain candidates; no fitted removal list or fitted parameters.
    assert 'feature_0' in allowed
    manifest = pd.read_csv(directory / cic.MANIFEST_FILE, dtype={'flow_id': str})
    assert manifest.flow_id.is_unique and len(manifest) == 210
    assert set(manifest.flow_id) == set(original['Flow ID'])
    assert manifest.group_row_count.sum() == len(original)
    assert manifest.has_label_conflict.eq('YES').sum() == 10
    assert manifest.seed.eq(42).all()
    assigned = original.merge(manifest, left_on='Flow ID', right_on='flow_id', validate='many_to_one')
    assert len(assigned) == len(original) and assigned.split.notna().all()
    assert assigned.groupby('Flow ID').split.nunique().eq(1).all()
    for split, fraction in zip(cic.SPLITS, [0.6, 0.2, 0.2]):
        part = assigned[assigned.split == split]
        assert set(part.Label) == set(cic.LABELS)
        assert len(part) / len(original) == pytest.approx(fraction, abs=0.04)
        for label in cic.LABELS:
            assert part.Label.eq(label).sum() / original.Label.eq(label).sum() == pytest.approx(fraction, abs=0.04)
    main = pd.read_csv(directory / cic.MAIN_FILE)
    assert len(main) == 1
    row = main.iloc[0]
    assert row.raw_rows == len(original) and row.raw_columns == 85
    assert row.initial_predictor_candidates == 77 and row.split_seed == 42
    assert row.contract_status == 'FROZEN_FOR_MATERIALIZATION'
    assert row.next_stage == 'CIC_PREPROCESSING_MATERIALIZATION'
    for key, value in cic.POLICIES.items():
        assert row[key] == value
    assert all(row[k] == 1 for k in main.columns if k.startswith('endpoint_pair_overlap_'))
    _, diagnostics = cic.inspect_source(source)
    assert diagnostics['Flow Bytes/s'] == {'missing': 1, 'positive_inf': 1, 'negative_inf': 1}
    assert all(not any(key in name for key in ('median_value', 'train_mean', 'train_std', 'scale_'))
               for table in [main, features, manifest, mapping] for name in table.columns)


def test_cic_best_assignment_and_lexical_ties():
    from fractions import Fraction
    from itertools import permutations
    import numpy as np
    assert cic.choose_folds(np.ones((5, 4), dtype=int)) == (Fraction(0), (0, 1, 2), 3, 4)
    counts = np.array([[100, 15, 3, 19], [80, 30, 2, 22], [115, 20, 1, 10],
                       [90, 11, 4, 30], [105, 24, 5, 17]])
    score, train, validation, test = cic.choose_folds(counts)
    candidates = []
    for v, t in permutations(range(5), 2):
        tr = tuple(i for i in range(5) if i not in (v, t))
        allocations = np.array([counts[list(tr)].sum(axis=0), counts[v], counts[t]])
        target = np.array([0.6, 0.2, 0.2])
        objective = ((allocations.sum(axis=1) / counts.sum() - target) ** 2).sum()
        objective += ((allocations / counts.sum(axis=0) - target[:, None]) ** 2).sum() / 4
        candidates.append((objective, tr, v, t))
    assert len(candidates) == 20
    expected = min(candidates)
    assert float(score) == pytest.approx(expected[0])
    assert (train, validation, test) == expected[1:]


def test_cic_determinism_and_endpoint_diagnostic_only(cic_root):
    _, frame = cic_root
    first, _ = cic.build_contract(frame)
    changed = frame.copy()
    changed['Src IP'] = [f'source-{i}' for i in range(len(frame))]
    changed['Dst IP'] = [f'destination-{i}' for i in range(len(frame))]
    changed['feature_0'] = range(len(frame))
    second, _ = cic.build_contract(changed)
    pd.testing.assert_frame_equal(first[cic.MANIFEST_FILE], second[cic.MANIFEST_FILE])
    assert all(value == 0 for value in cic.endpoint_overlap(changed, second[cic.MANIFEST_FILE]).values())


@pytest.mark.parametrize('problem', ['duplicate', 'missing', 'unassigned', 'counts', 'conflicts', 'class_absent'])
def test_cic_manifest_acceptance(cic_root, problem):
    _, frame = cic_root
    manifest, _ = cic.split_manifest(frame)
    if problem == 'duplicate':
        manifest = pd.concat([manifest, manifest.iloc[:1]])
    elif problem == 'missing':
        manifest = manifest.iloc[1:]
    elif problem == 'unassigned':
        manifest.loc[0, 'split'] = ''
    elif problem == 'counts':
        manifest.loc[0, 'group_row_count'] += 1
    elif problem == 'conflicts':
        manifest.loc[manifest.has_label_conflict == 'YES', 'has_label_conflict'] = 'NO'
    else:
        manifest.loc[manifest.tor_rows > 0, 'split'] = 'TRAIN'
    with pytest.raises(ValueError):
        cic.validate_manifest(frame, manifest)


def test_cic_rejects_cross_fold_groups(cic_root, monkeypatch):
    import numpy as np
    _, frame = cic_root
    class BrokenSplitter:
        def __init__(self, **kwargs):
            assert kwargs == dict(n_splits=5, shuffle=True, random_state=42)
        def split(self, x, y, groups):
            for fold in range(5):
                yield np.array([], dtype=int), np.arange(fold, len(y), 5)
    monkeypatch.setattr(cic, 'StratifiedGroupKFold', BrokenSplitter)
    with pytest.raises(ValueError, match='crosses folds'):
        cic.split_manifest(frame)


def test_cic_canonical_overwrite_and_identity(cic_root, monkeypatch):
    root, _ = cic_root
    cic.define_contract(root, execute=True)
    directory = root / cic.OUTPUT_DIRECTORY
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    cic.define_contract(root, execute=True)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}
    for field in ['dataset', 'task', 'raw_source']:
        contract = pd.read_csv(directory / cic.MAIN_FILE)
        contract.loc[0, field] = 'different_identity'
        contract.to_csv(directory / cic.MAIN_FILE, index=False)
        snapshot = {p: p.read_bytes() for p in directory.iterdir()}
        with pytest.raises(ValueError, match='identity differs'):
            cic.define_contract(root, execute=True)
        assert snapshot == {p: p.read_bytes() for p in directory.iterdir()}
        (directory / cic.MAIN_FILE).write_bytes(before[cic.MAIN_FILE])
    (directory / cic.MAIN_FILE).unlink()
    with pytest.raises(ValueError, match='ownership requires review'):
        cic.define_contract(root, execute=True)


def test_cic_publication_failure_restores_previous(cic_root, monkeypatch):
    root, frame = cic_root
    tables, _ = cic.build_contract(frame)
    cic.publish(root, tables)
    directory = root / cic.OUTPUT_DIRECTORY
    before = {p: p.read_bytes() for p in directory.iterdir()}
    real_replace = cic.os.replace
    failed = False
    def fail_once(src, dst):
        nonlocal failed
        if not failed and Path(dst).name == cic.MANIFEST_FILE:
            failed = True
            raise OSError('synthetic publication failure')
        return real_replace(src, dst)
    monkeypatch.setattr(cic.os, 'replace', fail_once)
    with pytest.raises(OSError, match='synthetic publication failure'):
        cic.publish(root, tables)
    assert before == {p: p.read_bytes() for p in directory.iterdir()}


def test_cic_module_has_no_training_or_digest_dependencies():
    import ast
    tree = ast.parse(Path(cic.__file__).read_text(encoding='utf-8'))
    imports = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
        elif isinstance(node, ast.Call):
            calls.append(node.func.attr if isinstance(node.func, ast.Attribute)
                         else node.func.id if isinstance(node.func, ast.Name) else '')
    assert set(imports) <= {'__future__', 'argparse', 'csv', 'fractions', 'itertools', 'os',
        'pathlib', 'tempfile', 'numpy', 'pandas', 'sklearn', 'sklearn.model_selection'}
    assert not set(calls) & {'fit', 'fit_transform', 'train', 'predict', 'predict_proba',
                             'evaluate', 'hash', 'sha256', 'sha256_file', 'median', 'std'}


@pytest.fixture
def cic_preprocessing_root(cic_root, monkeypatch):
    import numpy as np
    root, frame = cic_root
    cic.define_contract(root, execute=True)
    manifest = pd.read_csv(root / cic.OUTPUT_DIRECTORY / cic.MANIFEST_FILE, dtype={'flow_id': str})
    assignments = frame['Flow ID'].map(manifest.set_index('flow_id')['split'])
    train = np.flatnonzero(assignments.eq('TRAIN'))
    holdout = np.flatnonzero(assignments.ne('TRAIN'))
    frame['Flow Bytes/s'] = (np.arange(len(frame)) % 17).astype(float)
    frame.loc[train[:3], 'Flow Bytes/s'] = [np.nan, np.inf, -np.inf]
    frame.loc[holdout[:3], 'Flow Bytes/s'] = [np.nan, np.inf, -np.inf]
    frame['Flow Packets/s'] = (np.arange(len(frame)) % 7).astype(float)
    # Constant only on holdouts: must remain a predictor.
    frame['feature_0'] = 999.0
    frame.loc[train, 'feature_0'] = np.arange(len(train)) % 19
    # Constant only on TRAIN: must be removed even though holdouts vary.
    frame['feature_1'] = np.arange(len(frame)).astype(float)
    frame.loc[train, 'feature_1'] = 7.0
    frame['feature_2'] = (np.arange(len(frame)) % 11).astype(float)
    frame.loc[train[:8], 'feature_2'] = np.nan
    frame.loc[holdout, 'feature_2'] = 1e9 + np.arange(len(holdout))
    raw = frame.copy()
    raw.columns = [name if name != 'Label.1' else 'Label' for name in raw]
    raw.to_csv(root / cic.RAW_SOURCE, index=False)
    monkeypatch.setattr(cic, 'EXPECTED_SIZE', (root / cic.RAW_SOURCE).stat().st_size)
    return root


def test_cic_preprocessing_frozen_reuse_train_only(cic_preprocessing_root, monkeypatch):
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from src.models import evaluation
    from src.common import hashing
    root = cic_preprocessing_root
    untouched = make_file(root, 'data/raw/USTC-TFC2016/sentinel.txt', b'USTC untouched')
    unrelated = make_file(root, 'data/processed/CIC-Darknet2020/unrelated.txt', b'preserve')
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
    def forbidden(*args, **kwargs):
        pytest.fail('Forbidden splitting, USTC access, model execution, evaluation, or digest')
    for name in ['split_manifest', 'choose_folds', 'protect_ustc']:
        monkeypatch.setattr(cic, name, forbidden)
    for name in ['fit', 'predict', 'predict_proba']:
        monkeypatch.setattr(RandomForestClassifier, name, forbidden)
    monkeypatch.setattr(evaluation, 'evaluate', forbidden)
    monkeypatch.setattr(hashing, 'sha256_file', forbidden)
    raw, candidates, parts = cic_prep.load_inputs(root)
    expected_train = cic_prep.numeric_candidates(raw, candidates).iloc[parts['TRAIN']]
    expected_medians = expected_train.median()
    expected_imputed = expected_train.fillna(expected_medians)
    final = [name for name in candidates if expected_imputed[name].nunique() > 1]
    assert final == ['Flow Bytes/s', 'Flow Packets/s', 'feature_0', 'feature_2']
    calls = []
    real_fit = cic_prep.StandardScaler.fit
    def checked_fit(self, x, *args, **kwargs):
        pd.testing.assert_frame_equal(x, expected_imputed[final])
        calls.append(len(x))
        return real_fit(self, x, *args, **kwargs)
    monkeypatch.setattr(cic_prep.StandardScaler, 'fit', checked_fit)
    assert cic_prep.main(['--project-root', str(root), '--execute']) == 0
    assert calls == [len(parts['TRAIN'])]
    parameters = pd.read_csv(root / cic_prep.PARAMETERS, keep_default_na=False)
    assert list(parameters.columns) == [
        'source_order', 'feature_name', 'train_median', 'train_zero_variance',
        'retained_final', 'final_predictor_order', 'train_scaler_mean', 'train_scaler_scale']
    assert parameters.source_order.tolist() == list(range(77))
    assert parameters.feature_name.tolist() == candidates
    np.testing.assert_allclose(parameters.train_median, expected_medians)
    p = parameters.set_index('feature_name')
    assert p.loc['feature_0', 'retained_final'] == 'YES'
    assert p.loc['feature_1', 'retained_final'] == 'NO'
    assert p.loc['feature_1', 'train_zero_variance'] == 'YES'
    assert p.loc['feature_1', ['final_predictor_order', 'train_scaler_mean', 'train_scaler_scale']].eq('').all()
    assert p.loc['feature_2', 'train_median'] != cic_prep.numeric_candidates(raw, candidates).feature_2.median()
    assert parameters.loc[parameters.retained_final == 'YES', 'final_predictor_order'].tolist() == ['0', '1', '2', '3']
    all_groups = set()
    total = 0
    for split in cic.SPLITS:
        table = cic_prep.read_table(root / cic_prep.OUTPUTS[split])
        assert list(table.columns) == final + cic_prep.METADATA_COLUMNS
        assert np.isfinite(table[final].to_numpy()).all()
        assert set(table.target_id) == {'0', '1', '2', '3'}
        assert not all_groups & set(table['Flow ID'])
        all_groups.update(table['Flow ID'])
        assert table['Flow ID'].tolist() == raw.iloc[parts[split]]['Flow ID'].tolist()
        assert set(table['Label.1']) == {'AUDIO-STREAMING', 'Audio-Streaming'}
        total += len(table)
    assert total == len(raw)
    assert not set(final) & set(cic.EXCLUDED)
    # The independent saved-output audit must never fit or rewrite anything.
    snapshot = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
    monkeypatch.setattr(cic_prep.StandardScaler, 'fit', forbidden)
    monkeypatch.setattr(cic_prep.StandardScaler, 'transform', forbidden)
    summary = cic_prep.verify_materialization(root, raw, candidates, parts)
    assert summary['preprocessing_leakage'] == 'PASS'
    assert summary['removed_train_zero_variance'] == 73 and summary['final_predictors'] == 4
    assert snapshot == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in snapshot}
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    assert untouched.read_bytes() == b'USTC untouched' and unrelated.read_bytes() == b'preserve'
    assert set(snapshot) - set(before) == {root / p for p in (*cic_prep.OUTPUTS.values(), cic_prep.PARAMETERS)}


@pytest.mark.parametrize('problem', ['status', 'policy', 'mapping', 'feature', 'manifest', 'missing'])
def test_cic_preprocessing_requires_frozen_contract(cic_preprocessing_root, monkeypatch, problem):
    root = cic_preprocessing_root
    directory = root / cic.OUTPUT_DIRECTORY
    filename = {'status': cic.MAIN_FILE, 'policy': cic.MAIN_FILE, 'mapping': cic.MAPPING_FILE,
                'feature': cic.FEATURE_FILE, 'manifest': cic.MANIFEST_FILE, 'missing': cic.MAIN_FILE}[problem]
    path = directory / filename
    frame = pd.read_csv(path, keep_default_na=False)
    if problem == 'status':
        frame.loc[0, 'contract_status'] = 'DRAFT'
    elif problem == 'policy':
        frame.loc[0, 'imputation_policy'] = 'ALL_DATA_MEDIAN'
    elif problem == 'mapping':
        frame.loc[0, 'probability_column'] = 'wrong_class'
    elif problem == 'feature':
        frame.loc[frame.column_name == 'Label.1', 'model_allowed'] = 'YES'
    elif problem == 'manifest':
        frame = pd.concat([frame, frame.iloc[:1]])
    if problem == 'missing':
        path.unlink()
    else:
        frame.to_csv(path, index=False)
    monkeypatch.setattr(cic_prep, 'fit_preprocessing', lambda *_: pytest.fail('Fit before frozen input validation'))
    with pytest.raises((ValueError, FileNotFoundError)):
        cic_prep.preprocess(root, execute=True)
    assert not (root / cic_prep.PARAMETERS).exists()


def test_cic_preprocessing_holdouts_cannot_change_parameters(cic_preprocessing_root):
    import numpy as np
    root = cic_preprocessing_root
    raw, candidates, parts = cic_prep.load_inputs(root)
    outputs, first = cic_prep.fit_preprocessing(raw, candidates, parts)
    changed = raw.copy()
    holdout = np.concatenate([parts['VALIDATION'], parts['TEST']])
    for name in candidates:
        changed[name] = pd.to_numeric(changed[name], errors='coerce').astype(float)
        changed.loc[holdout, name] = 1e12 + np.arange(len(holdout))
    altered, second = cic_prep.fit_preprocessing(changed, candidates, parts)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(outputs['TRAIN'], altered['TRAIN'])


@pytest.mark.parametrize('problem', ['all_missing', 'all_constant'])
def test_cic_preprocessing_rejects_unusable_train(cic_preprocessing_root, problem):
    import numpy as np
    raw, candidates, parts = cic_prep.load_inputs(cic_preprocessing_root)
    if problem == 'all_missing':
        raw.loc[parts['TRAIN'], 'Flow Bytes/s'] = np.nan
        message = 'finite TRAIN median'
    else:
        for name in candidates:
            raw[name] = pd.to_numeric(raw[name], errors='coerce')
        raw.loc[parts['TRAIN'], candidates] = 1
        message = 'No predictors remain'
    with pytest.raises(ValueError, match=message):
        cic_prep.fit_preprocessing(raw, candidates, parts)


def test_cic_preprocessing_canonical_rerun(cic_preprocessing_root):
    root = cic_preprocessing_root
    assert cic_prep.preprocess(root) is None
    assert not (root / cic_prep.PARAMETERS).exists()
    cic_prep.preprocess(root, execute=True)
    snapshot = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    cic_prep.preprocess(root, execute=True)
    assert snapshot == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert {p.name for p in (root / cic_prep.PROCESSED).iterdir()} == {'train.csv', 'validation.csv', 'test.csv'}
    (root / cic_prep.OUTPUTS['TEST']).unlink()
    with pytest.raises(ValueError, match='Partial existing'):
        cic_prep.preprocess(root, execute=True)


@pytest.mark.parametrize('problem', ['median', 'zero', 'scale', 'provenance', 'infinite'])
def test_cic_preprocessing_independent_audit_detects_corruption(cic_preprocessing_root, problem):
    root = cic_preprocessing_root
    cic_prep.preprocess(root, execute=True)
    raw, candidates, parts = cic_prep.load_inputs(root)
    if problem in ('median', 'zero', 'scale'):
        path = root / cic_prep.PARAMETERS
        table = pd.read_csv(path, keep_default_na=False)
        row = table.index[table.retained_final == 'YES'][0]
        field, value = {'median': ('train_median', 999), 'zero': ('train_zero_variance', 'YES'),
                        'scale': ('train_scaler_scale', '999')}[problem]
        table.loc[row, field] = value
    else:
        path = root / cic_prep.OUTPUTS['VALIDATION']
        table = cic_prep.read_table(path)
        if problem == 'provenance':
            table.loc[0, 'Flow ID'] = 'wrong-source'
        else:
            table.loc[0, 'Flow Bytes/s'] = float('inf')
    table.to_csv(path, index=False)
    snapshot = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):
        cic_prep.verify_materialization(root, raw, candidates, parts)
    with pytest.raises(ValueError):
        cic_prep.preprocess(root, execute=True)
    assert snapshot == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_cic_preprocessing_rejects_changed_split_identity(cic_preprocessing_root):
    root = cic_preprocessing_root
    cic_prep.preprocess(root, execute=True)
    directory = root / cic.OUTPUT_DIRECTORY
    manifest = pd.read_csv(directory / cic.MANIFEST_FILE, dtype={'flow_id': str})
    index = manifest.index[manifest.split == 'TRAIN'][0]
    manifest.loc[index, 'split'] = 'VALIDATION'
    manifest.to_csv(directory / cic.MANIFEST_FILE, index=False)
    main = pd.read_csv(directory / cic.MAIN_FILE, keep_default_na=False)
    for split in cic.SPLITS:
        part = manifest[manifest.split == split]
        main.loc[0, f'{split.lower()}_rows'] = part.group_row_count.sum()
        for name in cic.CLASS_COUNT_COLUMNS:
            main.loc[0, f'{split.lower()}_{name}'] = part[name].sum()
    main.to_csv(directory / cic.MAIN_FILE, index=False)
    snapshot = {root / path: (root / path).read_bytes() for path in (*cic_prep.OUTPUTS.values(), cic_prep.PARAMETERS)}
    with pytest.raises(ValueError):
        cic_prep.preprocess(root, execute=True)
    assert snapshot == {path: path.read_bytes() for path in snapshot}


def test_cic_preprocessing_publication_rollback(cic_preprocessing_root, monkeypatch):
    root = cic_preprocessing_root
    cic_prep.preprocess(root, execute=True)
    raw, candidates, parts = cic_prep.load_inputs(root)
    outputs, parameters = cic_prep.fit_preprocessing(raw, candidates, parts)
    snapshot = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    real_replace = cic_prep.os.replace
    failed = False
    def fail_once(src, dst):
        nonlocal failed
        if not failed and Path(dst).name == 'test.csv':
            failed = True
            raise OSError('publication interrupted')
        return real_replace(src, dst)
    monkeypatch.setattr(cic_prep.os, 'replace', fail_once)
    with pytest.raises(OSError, match='publication interrupted'):
        cic_prep.publish(root, outputs, parameters, raw, candidates, parts)
    assert snapshot == {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_cic_preprocessing_dependency_boundary():
    import ast
    tree = ast.parse(Path(cic_prep.__file__).read_text(encoding='utf-8'))
    imports, calls = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
        elif isinstance(node, ast.Call):
            calls.append(node.func.attr if isinstance(node.func, ast.Attribute)
                         else node.func.id if isinstance(node.func, ast.Name) else '')
    assert set(imports) <= {'__future__', 'argparse', 'csv', 'os', 'pathlib', 'shutil', 'tempfile',
                            'numpy', 'pandas', 'sklearn.preprocessing', 'src.data'}
    assert not set(calls) & {'train', 'predict', 'predict_proba', 'evaluate', 'hash', 'sha256',
                             'sha256_file', 'split_manifest', 'choose_folds', 'protect_ustc'}


# USTC preparation uses synthetic capture paths and a stand-in flow iterator;
# these tests never load NFStream or process packets.
from src.data import ustc_prepare_dataset as prepare

PREPARATION_HEADERS = {
    'dataset_summary.csv': 'dataset total_flows benign_flows malware_flows semantic_family_count capture_group_count native_feature_count predictor_candidate_count',
    'class_distribution.csv': 'dataset class flow_count percentage',
    'family_distribution.csv': 'dataset binary_label semantic_family flow_count percentage_of_dataset percentage_within_binary_class',
    'capture_distribution.csv': 'dataset binary_label semantic_family capture_group source_pcap flow_count percentage',
    'feature_summary.csv': 'dataset feature_name dtype missing_count missing_percentage unique_count role',
    'processing_summary.csv': 'dataset source_pcap_count output_flow_count native_column_count predictor_candidate_count provenance_metadata_count excluded_identifier_count excluded_dpi_count',
}

@pytest.fixture
def preparation_root(tmp_path):
    base = tmp_path / 'data/raw/USTC-TFC2016'
    for label in ['Benign', 'Malware']:
        for family in range(10):
            names = ([f'{label}/SMB/SMB/SMB-{i}.pcap' for i in range(5)]
                     if label == 'Benign' and family == 0 else [f'{label}/Family{family}.pcap'])
            for name in names:
                make_file(base, name, b'fixture only, not a PCAP')
    return tmp_path


def test_preparation_preflight_no_reads_or_outputs(preparation_root, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError('Preflight must not read packets or load NFStream')
    monkeypatch.setattr(prepare, 'load_streamer', forbidden)
    monkeypatch.setattr(Path, 'open', forbidden)
    assert prepare.main(['--project-root', str(preparation_root)]) == 0
    report = capsys.readouterr().out
    for text in ['pcap_count=24', 'benign_pcap_count=14', 'malware_pcap_count=10',
                 'semantic_family_count=20', 'PREFLIGHT_ONLY=True', 'flows.csv', 'result_directory=']:
        assert text in report
    assert not (preparation_root / 'results').exists()
    assert not (preparation_root / 'data/interim').exists()
    captures = prepare.discover_pcaps(preparation_root)
    assert len({r['capture_group'] for r in captures}) == 24
    smb = [r for r in captures if r['semantic_family'] == 'SMB']
    assert len(smb) == 5
    assert len({r['capture_group'] for r in smb}) == 5
    assert all(r['binary_label'] == 'benign' for r in smb)


def test_preparation_rejects_wrong_structure_before_extraction(tmp_path, monkeypatch):
    make_file(tmp_path / 'data/raw/USTC-TFC2016', 'Benign/FTP.pcap', b'fixture')
    monkeypatch.setattr(prepare, 'load_streamer', lambda: pytest.fail('Extraction attempted'))
    with pytest.raises(ValueError, match='pcap_count=1'):
        prepare.prepare_dataset(tmp_path, execute=True)
    assert not (tmp_path / 'results').exists()
    make_file(tmp_path / 'data/raw/USTC-TFC2016', 'Other/Malware/bad.pcap', b'fixture')
    with pytest.raises(ValueError, match='authoritative'):
        prepare.discover_pcaps(tmp_path)


def test_preparation_feature_roles():
    expected = {
        'target': ['binary_label'],
        'provenance_metadata': ['semantic_family', 'capture_group', 'source_pcap'],
        'excluded_identifier': ['id', 'expiration_id', 'src_ip', 'dst_ip', 'src_mac', 'dst_mac',
            'src_oui', 'dst_oui', 'src_port', 'dst_port', 'bidirectional_first_seen_ms',
            'bidirectional_last_seen_ms', 'src2dst_first_seen_ms', 'dst2src_last_seen_ms', 'system_process_pid'],
        'excluded_dpi': ['application_name', 'application_category_name', 'application_confidence',
            'application_is_guessed', 'requested_server_name', 'client_fingerprint',
            'server_fingerprint', 'user_agent', 'content_type', 'ndpi_protocol', 'tls_version'],
        'predictor_candidate': ['protocol', 'ip_version', 'bidirectional_duration_ms',
            'bidirectional_packets', 'src2dst_mean_ps', 'dst2src_stddev_piat_ms'],
    }
    for role, columns in expected.items():
        for column in columns:
            assert prepare.feature_role(column) == role


def test_preparation_execution_synthetic(preparation_root, monkeypatch):
    base = preparation_root / 'data/raw/USTC-TFC2016'
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in base.rglob('*.pcap')}
    calls = []
    def fake_streamer(**kwargs):
        calls.append(kwargs)
        return iter([
            dict(id=0, src_ip='10.0.0.1', src_port=443, bidirectional_first_seen_ms=12345,
                 bidirectional_packets=3, bidirectional_mean_ps=100.5, application_name='Unexpected'),
            dict(id=1, src_ip='10.0.0.2', src_port=80, bidirectional_first_seen_ms=12346,
                 bidirectional_packets=7, bidirectional_mean_ps=float('nan'), application_name='Unexpected'),
        ])
    monkeypatch.setattr(prepare, 'load_streamer', lambda: fake_streamer)
    output, results = prepare.prepare_dataset(preparation_root, execute=True)
    assert len(calls) == 24
    for call in calls:
        assert Path(call['source']).is_relative_to(base)
        assert call['n_dissections'] == 0
        assert call['statistical_analysis'] is True
        assert call['splt_analysis'] == call['system_visibility_mode'] == call['max_nflows'] == 0
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    frame = pd.read_csv(output)
    assert len(frame) == 48
    assert list(frame.columns[-4:]) == prepare.METADATA
    assert frame[prepare.METADATA].notna().all().all()
    assert frame.capture_group.nunique() == 24
    assert list(output.parent.iterdir()) == [output]
    assert {p.name for p in results.iterdir()} == set(PREPARATION_HEADERS)
    for name, header in PREPARATION_HEADERS.items():
        table = pd.read_csv(results / name)
        assert list(table.columns) == (header + ' ' + TAIL).split()
        assert set(table.run_id) == {results.name}
        assert set(table.dataset) == {'USTC-TFC2016'}
    summary = pd.read_csv(results / 'dataset_summary.csv').iloc[0]
    assert (summary.total_flows, summary.benign_flows, summary.malware_flows) == (48, 28, 20)
    assert (summary.semantic_family_count, summary.capture_group_count) == (20, 24)
    assert (summary.native_feature_count, summary.predictor_candidate_count) == (7, 2)
    for name in ['class_distribution.csv', 'family_distribution.csv', 'capture_distribution.csv']:
        table = pd.read_csv(results / name)
        assert table.flow_count.sum() == 48
        pct = 'percentage_of_dataset' if name == 'family_distribution.csv' else 'percentage'
        assert table[pct].sum() == pytest.approx(100)
        if name == 'family_distribution.csv':
            assert table.groupby('binary_label').percentage_within_binary_class.sum().tolist() == pytest.approx([100, 100])
    features = pd.read_csv(results / 'feature_summary.csv').set_index('feature_name')
    assert features.loc['bidirectional_mean_ps', 'missing_count'] == 24
    assert features.loc['bidirectional_mean_ps', 'missing_percentage'] == 50
    assert features.loc['bidirectional_packets', 'unique_count'] == 2
    assert features.loc['application_name', 'role'] == 'excluded_dpi'
    processing = pd.read_csv(results / 'processing_summary.csv').iloc[0]
    assert (processing.provenance_metadata_count, processing.excluded_identifier_count, processing.excluded_dpi_count) == (3, 4, 1)
    existing = output.read_bytes()
    monkeypatch.setattr(prepare, 'load_streamer', lambda: pytest.fail('Existing output must block extraction'))
    with pytest.raises(FileExistsError):
        prepare.prepare_dataset(preparation_root, execute=True)
    assert output.read_bytes() == existing


@pytest.mark.parametrize('mode', ['empty', 'schema_change', 'metadata_collision'])
def test_preparation_failure_does_not_publish_flows(preparation_root, monkeypatch, mode):
    def fake_streamer(**kwargs):
        if mode == 'empty':
            return iter([])
        if mode == 'metadata_collision':
            return iter([dict(binary_label='bad')])
        return iter([dict(id=0), dict(id=1, extra=1)])
    monkeypatch.setattr(prepare, 'load_streamer', lambda: fake_streamer)
    with pytest.raises(ValueError):
        prepare.prepare_dataset(preparation_root, execute=True)
    assert not (preparation_root / 'data/interim/USTC-TFC2016/flows.csv').exists()


def test_preparation_summary_zero_capture_and_invalid_metadata():
    captures = [dict(binary_label=label, semantic_family=family, capture_group=group, source_pcap=group)
                for label, family, group in [('benign', 'FTP', 'Benign/FTP.pcap'), ('malware', 'Zeus', 'Malware/Zeus.pcap')]]
    frame = pd.DataFrame([dict(captures[0], bidirectional_packets=5)])
    tables = prepare.summarize(frame, captures)
    assert tables['capture_distribution.csv'][1]['flow_count'] == 0
    assert tables['family_distribution.csv'][1]['percentage_within_binary_class'] == 0
    frame.loc[0, 'capture_group'] = ''
    with pytest.raises(ValueError, match='complete provenance'):
        prepare.summarize(frame, captures)
    frame.loc[0, 'capture_group'] = 'wrong'
    with pytest.raises(ValueError, match='source inventory'):
        prepare.summarize(frame, captures)


def test_preparation_dependency_version(monkeypatch):
    monkeypatch.setattr(prepare, 'version', lambda name: '6.5.0')
    with pytest.raises(RuntimeError, match='6.6.0'):
        prepare.load_streamer()
