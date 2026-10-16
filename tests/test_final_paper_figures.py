"""Lightweight source/registry policy tests. No figures, inference, retrieval or LLMs."""
import ast
import csv
import json
from pathlib import Path
import re
import pytest

from src.figures.final_paper.common import (
    BASE, CATEGORIES, DATASETS, FIELDS, FORMATS, MODELS, OUTPUT, RUNS,
    Spec, Unsupported, guard, paths, registry,
)
from src.figures.final_paper.artifact_inventory import Inventory, Sources, classify_metadata
from src.figures.final_paper.generate_final_paper_figures import main, prepare, write_csv, read_csv


def csv_file(root, relative, records):
    path = root/relative
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return path


def frozen_fixture(root):
    test_path = f'results/final/{DATASETS[0]}/ustc_test_results.csv'
    csv_file(root,'metadata/traffic/ustc_experiment_closure.csv', [dict(
        dataset=DATASETS[0],authoritative_run=RUNS[0],stage_status='CLOSED_FROZEN',
        finalization_status='PASS',integrity_audit_status='PASS',test_rows=8,test_results_path=test_path)])
    final = [dict(dataset=DATASETS[0],authoritative_run=RUNS[0],model=model,
                  integrity_audit_status='PASS',total_rows=8,accuracy=.75,mcc=.5,f1=.75) for model in MODELS]
    csv_file(root,test_path,final)
    run = [dict(dataset=DATASETS[0],run_id=RUNS[0],model=model,split=split,
                accuracy=.75 if split == 'test' else .99,mcc=.5,f1=.75)
           for split in ('validation','test') for model in MODELS]
    csv_file(root,f'results/models/{DATASETS[0]}/{RUNS[0]}/metrics.csv',run)
    return final,run


def test_registry_unique_and_deterministic():
    first,second = registry(),registry()
    assert first == second
    assert len({s.figure_id for s in first}) == len(first)
    assert all(s.category in CATEGORIES for s in first)


def test_three_formats_and_stable_canonical_paths(tmp_path):
    for spec in registry():
        output = paths(tmp_path,spec)
        assert set(output) == {f+'_path' for f in FORMATS}
        for fmt in FORMATS:
            path = output[fmt+'_path']
            assert path == tmp_path/OUTPUT/spec.figure_set/spec.category/fmt/(spec.figure_id+'.'+fmt)
            assert path.stem == spec.figure_id
            assert not re.search(r'\d{8}T\d{6}|final_final|_copy',str(path.relative_to(tmp_path)))


def test_correct_model_roster_and_selected_lightgbm():
    assert tuple(MODELS.values()) == ('Random Forest','1D-CNN','XGBoost','LightGBM','CNN-LSTM')
    specs = {s.figure_id:s for s in registry()}
    assert specs['phase1_model_comparison'].split == 'TEST'
    assert specs['phase1_best_model_confusions'].model == 'lightgbm'


@pytest.mark.parametrize('split',['VALIDATION','DEVELOPMENT_VALIDATION_SUBSET'])
@pytest.mark.parametrize('category',['phase1_test','phase2_test','final'])
def test_validation_cannot_enter_test_or_final(split,category):
    with pytest.raises(Unsupported):
        guard(Spec('bad','Bad',category,split,'stage'),split)


def test_registry_validation_is_supplementary():
    for spec in registry():
        if spec.split in ('VALIDATION','DEVELOPMENT_VALIDATION_SUBSET'):
            assert spec.category == 'supplementary_validation'
        guard(spec,spec.split)


@pytest.mark.parametrize('row,expected',[
    ({'source_split':'test'},'TEST'),
    ({'split':'validation'},'VALIDATION'),
    ({'evaluation_scope':'stratified_development_validation_subset'},'DEVELOPMENT_VALIDATION_SUBSET'),
    ({'source_split':'test','rag_partition':'development_validation'},'UNKNOWN'),
    ({'source_split':'test','evaluation_scope':'stratified_development_validation_subset'},'UNKNOWN'),
    ({'source_split':'test','split':'validation'},'UNKNOWN'),
    ({'source_split':'test','evaluation_scope':'validation'},'UNKNOWN'),
    ({'filename':'final_test_results.csv'},'UNKNOWN'),
    ({},'UNKNOWN'),
])
def test_split_is_provenance_not_filename(row,expected):
    assert classify_metadata(row) == expected


def test_unknown_sources_fail_closed():
    with pytest.raises(Unsupported):
        guard(Spec('unknown','Unknown','phase2_test','TEST','stage'),'UNKNOWN')


def test_missing_source_has_actionable_reason(tmp_path):
    source = Sources(tmp_path)
    with pytest.raises(Unsupported,match='Missing canonical artifact'):
        list(source.rows('metadata/traffic/ustc_experiment_closure.csv'))
    assert 'metadata/traffic/ustc_experiment_closure.csv' in source.used


def test_frozen_test_metrics_ignore_validation_rows(tmp_path):
    frozen_fixture(tmp_path)
    result = Inventory(tmp_path).metrics(0)
    assert set(result) == set(MODELS)
    assert all(float(r['accuracy']) == .75 for r in result.values())


def test_test_filename_cannot_replace_explicit_test_rows(tmp_path):
    _,run = frozen_fixture(tmp_path)
    for row in run:
        row['split'] = 'validation'
    csv_file(tmp_path,f'results/models/{DATASETS[0]}/{RUNS[0]}/metrics.csv',run)
    with pytest.raises(Unsupported,match='No explicit TEST'):
        Inventory(tmp_path).metrics(0)


def test_wrong_authoritative_run_rejected(tmp_path):
    final,_ = frozen_fixture(tmp_path)
    final[0]['authoritative_run'] = 'unfrozen_run'
    csv_file(tmp_path,f'results/final/{DATASETS[0]}/ustc_test_results.csv',final)
    with pytest.raises(Unsupported,match='identity mismatch'):
        Inventory(tmp_path).metrics(0)


def test_phase2_subset_blocks_test_figure(tmp_path):
    csv_file(tmp_path,BASE+'llm_execution_configuration.csv',[
        {'field':'evaluation_scope','value':'stratified_development_validation_subset'}])
    spec = Spec('legacy_block','Legacy development chain','final','TEST','blocked')
    with pytest.raises(Unsupported,match='BLOCKED_NO_CANONICAL_TEST_ARTIFACT'):
        prepare(spec,Inventory(tmp_path))
    assert not (tmp_path/OUTPUT).exists()


def test_source_hash_detects_changes(tmp_path):
    name = 'metadata/traffic/example.csv'
    path = csv_file(tmp_path,name,[{'a':'1'}])
    source = Sources(tmp_path)
    assert list(source.rows(name))[0]['a'] == '1'
    path.write_text('a\n2\n',encoding='utf-8')
    with pytest.raises(Unsupported,match='Scientific source changed'):
        source.unchanged()


def test_no_raw_traffic_or_models_are_read(tmp_path):
    source = Sources(tmp_path)
    for name in ('data/processed/USTC-TFC2016/test.csv','models/test.joblib','data/raw/Darknet.csv'):
        with pytest.raises(Unsupported,match='allowlist'):
            source.path(name)


def test_manifest_roundtrip_schema(tmp_path):
    record = dict.fromkeys(FIELDS,'')
    record.update(figure_id='F8_phase2_test_evidence_fusion',status='BLOCKED_NO_CANONICAL_TEST_ARTIFACT',
                  source_artifact=json.dumps([BASE+'llm_execution_configuration.csv']))
    write_csv(tmp_path,'figure_manifest.csv',FIELDS,[record])
    assert read_csv(tmp_path,'figure_manifest.csv') == [record]
    assert set(FIELDS) >= {'figure_id','figure_title','category','paper_role','source_dataset','source_split',
                           'source_artifact','scientific_scope','status','skip_reason','png_path','svg_path','pdf_path'}


def test_list_does_not_read_sources_or_write_outputs(tmp_path,capsys):
    assert main(['--project-root',str(tmp_path),'--list']) == 0
    first = capsys.readouterr().out
    assert main(['--project-root',str(tmp_path),'--list']) == 0
    assert capsys.readouterr().out == first
    assert list(tmp_path.iterdir()) == []


def test_matplotlib_only_no_scientific_execution_imports():
    folder = Path(__file__).resolve().parents[1]/'src/figures/final_paper'
    forbidden = {'seaborn','plotly','bokeh','altair','torch','lightgbm','xgboost','transformers','requests','httpx'}
    for path in sorted(folder.glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                names = [node.module or '']
            else:
                continue
            assert not any(name.split('.')[0] in forbidden for name in names),path
            assert not any('train_models' in name or name.startswith('src.models') for name in names),path


def test_path_traversal_rejected(tmp_path):
    with pytest.raises(Unsupported):
        paths(tmp_path,Spec('../escape','Escape','final','TEST','p1'))


@pytest.mark.parametrize('value,score,percentage', [
    ('0.9999', '0.9999', '99.99%'),
    ('0.99999', '0.9999', '99.99%'),
    ('0.999999999999999999999999999999999999', '0.9999', '99.99%'),
    (0.9999999999999999, '0.9999', '99.99%'),
    (1.0, '1.0000', '100.00%'),
    (0, '0.0000', '0.00%'),
    ('0.123459', '0.1234', '12.34%'),
    ('-0.99999', '-0.9999', '-99.99%'),
])
def test_safe_format_boundaries(value, score, percentage):
    from src.figures.final_paper.common import safe_format
    assert safe_format(value) == score
    assert safe_format(value, 2, percentage=True) == percentage
    assert safe_format(value, 2, percentage=True, include_value=True) == f'{percentage} ({score})'


def test_safe_format_precision_and_percent_units():
    from decimal import Decimal, localcontext
    from src.figures.final_paper.common import safe_format, safe_share
    with localcontext() as ctx:
        ctx.prec = 3  # Helper must not depend on ambient Decimal precision.
        for places in (0, 1, 2, 4, 8):
            value = Decimal('0.999999999999999999999999999999999999')
            assert Decimal(safe_format(value, places)) < 1
            assert Decimal(safe_format(value, places, percentage=True)[:-1]) < 100
        assert safe_share(10**40-1, 10**40) == '99.99%'
        assert safe_share(10**40, 10**40) == '100.00%'
        assert safe_format('99.99999', 2, percentage=True, percent_input=True) == '99.99%'
        assert safe_format('100', 2, percentage=True, percent_input=True) == '100.00%'
        assert safe_format('-0.00001', 2) == '0.00'


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity'])
def test_safe_format_rejects_nonfinite(value):
    from src.figures.final_paper.common import safe_format
    with pytest.raises(ValueError, match='nonfinite'):
        safe_format(value)


def test_normalized_confusion_labels_use_original_counts(monkeypatch):
    from types import SimpleNamespace
    from src.figures.final_paper import phase1_test_figures as module
    cells = [[999999, 1], [0, 1000000]]
    captured = {}
    monkeypatch.setattr(module, 'heatmap', lambda *args, **kwargs: captured.update(kwargs))
    module.confusion(SimpleNamespace(set=lambda **kwargs: None),
                     SimpleNamespace(matrix=lambda *args: cells), 0, 'lightgbm', True)
    assert captured['cell_labels'] == [['99.99%', '0.00%'], ['0.00%', '100.00%']]
    assert cells == [[999999, 1], [0, 1000000]]


def test_pie_labels_ignore_rounded_matplotlib_percentages():
    from types import SimpleNamespace
    from src.figures.final_paper import consolidated_dataset_figures as module
    texts = []
    def pie(values, **kwargs):
        assert kwargs['autopct'] is None
        return [SimpleNamespace(theta1=0, theta2=360) for _ in values], []
    ax = SimpleNamespace(pie=pie, text=lambda x, y, label, **kw: texts.append(label),
                         set_title=lambda *args: None)
    class Axes:
        def __getitem__(self, index):
            return ax
    inv = SimpleNamespace(closure=lambda i: dict(train_rows=99999999, validation_rows=1, test_rows=0))
    plt = SimpleNamespace(subplots=lambda *args, **kw: (object(), Axes()))
    module.build(next(s for s in module.specs() if s.detail == 'dataset_split_pies'), inv, plt)
    assert texts == ['99.99%', '0.00%', '0.00%'] * 2


@pytest.mark.parametrize('figure_set', ['detailed', 'consolidated'])
def test_safe_tick_labels_preserve_fixed_labels(monkeypatch, tmp_path, figure_set):
    import sys
    from types import SimpleNamespace
    from src.figures.final_paper.style import safe_tick_labels
    class Scalar:
        pass
    class Axis:
        def __init__(self, formatter, bounds):
            self.formatter, self.bounds = formatter, bounds
        def get_view_interval(self): return self.bounds
        def get_scale(self): return 'linear'
        def get_major_formatter(self): return self.formatter
        def get_minor_formatter(self): return None
        def set_major_formatter(self, formatter): self.formatter = formatter
    monkeypatch.setitem(sys.modules, 'matplotlib.ticker', SimpleNamespace(
        ScalarFormatter=Scalar, FuncFormatter=lambda function: function))
    numeric, fixed = Axis(Scalar(), (0, 1.04)), Axis('class labels', (0, 1))
    counts = Axis(Scalar(), (0, 1000))
    safe_tick_labels(SimpleNamespace(axes=[SimpleNamespace(xaxis=fixed, yaxis=numeric),
                                          SimpleNamespace(xaxis=fixed, yaxis=counts)]))
    assert numeric.formatter(.99999, 0) == '0.99'
    assert numeric.formatter(1, 0) == '1.00'
    assert fixed.formatter == 'class labels'
    assert isinstance(counts.formatter, Scalar)
    from src.figures.final_paper.style import save
    fig = SimpleNamespace(axes=[SimpleNamespace(xaxis=fixed, yaxis=numeric)])
    fig.savefig = lambda path, **kw: path.write_text(numeric.formatter(.99999, 0), encoding='utf-8')
    spec = registry(figure_set)[0]
    outputs = save(fig, tmp_path, spec)
    assert set(outputs) == {'png_path', 'svg_path', 'pdf_path'}
    assert all((tmp_path/path).read_text(encoding='utf-8') == '0.99' for path in outputs.values())
    assert all('/'+figure_set+'/' in path for path in outputs.values())



def test_dual_registry_counts_and_retained_definitions():
    from src.figures.final_paper.common import legacy_registry, UNAVAILABLE_DETAILED
    assert len(registry('detailed')) == 109
    assert len(registry('consolidated')) == 12
    assert len(registry()) == 121
    assert {s.figure_id for s in legacy_registry()} == {s.figure_id for s in registry('detailed')} | set(UNAVAILABLE_DETAILED)
    assert {s.figure_set for s in registry('detailed')} == {'detailed'}
    assert {s.figure_set for s in registry('consolidated')} == {'consolidated'}
    assert 'P1_ustc_test_confidence' in UNAVAILABLE_DETAILED


@pytest.mark.parametrize('selected,other', [('consolidated','detailed'), ('detailed','consolidated')])
def test_selective_execute_preserves_other_set_exports_and_manifests(tmp_path, monkeypatch, selected, other):
    from types import SimpleNamespace
    from src.figures.final_paper import generate_final_paper_figures as cli
    from src.figures.final_paper import style
    import hashlib
    class FakeInventory:
        def __init__(self, root):
            self.s = SimpleNamespace(used=set(), unchanged=lambda:None)
        def catalog(self): return []
    monkeypatch.setattr(cli, 'Inventory', FakeInventory)
    monkeypatch.setattr(cli, 'prepare', lambda spec, inv: spec.split)
    monkeypatch.setattr(cli, 'audit_rows', lambda *args: [])
    monkeypatch.setattr(style, 'plotting', lambda root: SimpleNamespace(close=lambda *args:None))
    monkeypatch.setattr(cli, 'renderer', lambda spec: lambda *args: (SimpleNamespace(suptitle=lambda *args, **kw:None), 'Fixture'))
    def save(fig, root, spec):
        output = paths(root, spec)
        for path in output.values():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture export')
        return {key:path.relative_to(root).as_posix() for key,path in output.items()}
    monkeypatch.setattr(style, 'save', save)
    cli.execute(tmp_path, True)  # Default executes both sets.
    manifest = cli.read_csv(tmp_path, 'figure_manifest.csv')
    assert len(manifest) == 121
    assert all(r['status'] == 'GENERATED' for r in manifest)
    assert len(list((tmp_path/OUTPUT).rglob('*.png'))) == 121
    retained_paths = list((tmp_path/OUTPUT/other).rglob('*')) + [tmp_path/OUTPUT/'manifests'/(other+'_figure_manifest.csv')]
    before = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in retained_paths if p.is_file()}
    records = cli.execute(tmp_path, True, selected)
    assert len(records) == (109 if selected == 'detailed' else 12)
    assert before == {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before}
    assert len(cli.read_csv(tmp_path, 'figure_manifest.csv')) == 121
    assert len(list((tmp_path/OUTPUT).rglob('*.png'))) == 121


def test_selected_first_run_records_other_set_as_pending(tmp_path):
    from src.figures.final_paper import generate_final_paper_figures as cli
    rows = [cli.pending_row(s) for s in registry('consolidated')]
    for row in rows:
        row.update(status='SKIPPED_UNSUPPORTED', skip_reason='Fixture unavailable')
    combined = cli.publish_records(tmp_path, rows, [], 'consolidated', True)
    assert len(combined) == 121
    assert sum(r['status'] == 'NOT_GENERATED' for r in combined) == 109
    assert len(cli.read_csv(tmp_path, 'detailed_figure_manifest.csv')) == 109
    assert len(cli.read_csv(tmp_path, 'consolidated_figure_manifest.csv')) == 12


@pytest.mark.parametrize('figure_set,count', [('all',121), ('detailed',109), ('consolidated',12)])
def test_list_selection(tmp_path, capsys, figure_set, count):
    assert main(['--project-root',str(tmp_path),'--list','--set',figure_set]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == count
    if figure_set != 'all':
        assert all(line.startswith(figure_set+'\t') for line in lines)
    assert not list(tmp_path.iterdir())


def test_validation_fails_pending_sets_and_checks_counts(tmp_path, capsys):
    from src.figures.final_paper import generate_final_paper_figures as cli
    cli.publish_records(tmp_path, [], [], 'all', True)
    assert cli.validate(tmp_path) == 1
    output = capsys.readouterr().out
    assert 'MISSING_FORMATS=363' in output
    checks = {r['check']:r for r in cli.read_csv(tmp_path,'figure_validation_checklist.csv')}
    assert checks['detailed expected logical count']['status'] == 'PASS'
    assert checks['consolidated expected logical count']['status'] == 'PASS'
    assert checks['PNG/SVG/PDF completeness and non-empty files']['status'] == 'FAIL'
    rows = cli.read_csv(tmp_path,'figure_manifest.csv')
    rows[-1]['figure_set'] = 'detailed'
    cli.write_csv(tmp_path,'figure_manifest.csv',FIELDS,rows)
    assert cli.validate(tmp_path) == 1


def test_validation_rejects_duplicate_logical_ids(tmp_path):
    from src.figures.final_paper import generate_final_paper_figures as cli
    cli.publish_records(tmp_path, [], [], 'all', True)
    rows = cli.read_csv(tmp_path,'figure_manifest.csv')
    cli.write_csv(tmp_path,'figure_manifest.csv',FIELDS,rows + [rows[0]])
    assert cli.validate(tmp_path) == 1
