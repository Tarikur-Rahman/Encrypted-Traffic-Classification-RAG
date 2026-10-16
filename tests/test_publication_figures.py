"""Lightweight publication-suite tests. Run explicitly; use temporary fixtures only."""
import ast
import csv
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.figures import common, io, style, manifests
from src.figures import generate_publication_figures as cli

MODULES = ('common','io','style','manifests','dataset_figures','phase1_figures',
           'phase2_figures','results_figures','validation_figures',
           'final_summary_figures','generate_publication_figures')
SOURCE = 'results/rag/interface/interface_summary.csv'


def test_all_modules_import_and_compile():
    for name in MODULES:
        module = importlib.import_module('src.figures.'+name)
        path = Path(module.__file__)
        compile(ast.parse(path.read_text(encoding='utf-8-sig')),str(path),'exec')


def test_unique_complete_inventory():
    module = importlib.import_module('src.figures.generate_publication_figures')
    ids = tuple(s.figure_id for s in module.INVENTORY)
    assert len(ids) == len(set(ids)) == 66
    assert tuple(s.number for s in module.INVENTORY) == tuple(range(1,67))
    assert ids[0] == 'fig01_complete_methodology_architecture'
    assert ids[37] == 'fig38_phase2_final_summary'
    assert ids[-1] == 'fig66_finalization_summary'
    assert {s.category for s in module.INVENTORY} == set(common.SECTIONS)
    assert all(s.figure_id.startswith(f'fig{s.number:02d}_') for s in module.INVENTORY)
    for number in range(58,65):
        assert module.INVENTORY[number-1].category == 'final'


def test_list_is_read_only_and_needs_no_outputs(tmp_path,capsys,monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('Read-only mode attempted rendering or scientific reads')
    monkeypatch.setattr(cli,'plotting',forbidden)
    monkeypatch.setattr(io.Sources,'rows',forbidden)
    for _ in range(2):
        assert cli.main(['--project-root',str(tmp_path),'--list']) == 0
        lines = capsys.readouterr().out.splitlines()
        assert [r.split('\t')[0] for r in lines] == [s.figure_id for s in common.INVENTORY]
    assert not list(tmp_path.iterdir())


def test_sources_stream_and_preserve_inputs(tmp_path):
    path = tmp_path/SOURCE
    path.parent.mkdir(parents=True)
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=('id','text'))
        writer.writeheader()
        for i in range(4):
            writer.writerow({'id':i,'text':'a,b\n'+'x'*150000})
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    source = io.Sources(tmp_path)
    rows = source.rows(SOURCE)
    assert not source.used
    assert iter(rows) is rows
    assert next(rows)['id'] == '0'
    assert [r['id'] for r in rows] == ['1','2','3']
    source.unchanged()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('path',[
    'results/final/USTC-TFC2016/ustc_test_results.csv',
    'results/models/USTC-TFC2016/20260919T183945Z/roc_curve.csv',
    'data/raw/CIC-Darknet2020/Darknet.CSV',
    'models/arbitrary.joblib', '../outside.csv',
])
def test_scientific_allowlist_rejects_test_mixed_split_and_models_before_open(tmp_path,path,monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('Prohibited source was opened')
    monkeypatch.setattr(Path,'open',forbidden)
    with pytest.raises(common.Unsupported,match='allowlist'):
        next(io.Sources(tmp_path).rows(path))


def test_nonvalidation_rows_and_missing_fields_are_unsupported():
    with pytest.raises(common.Unsupported,match='TEST is prohibited'):
        io.development({'source_split':'test','rag_partition':'development_validation'})
    with pytest.raises(common.Unsupported,match='Missing saved field'):
        io.SavedRow({},SOURCE)['absent']


@pytest.mark.parametrize('prefix,reason,label',[
    ('validator','POSITIVE_EVIDENCE','Positive-evidence fallback'),
    ('validator','SEMANTIC_VALIDATION','Semantic-validation fallback'),
    ('runtime','UNPARSEABLE_MODEL_OUTPUT','Unparseable-output fallback'),
    ('runtime','TRUNCATED_COMPLETION','Truncated-completion fallback'),
    ('runtime','TRANSPORT_RETRIES_EXHAUSTED','Transport fallback'),
])
def test_fallbacks_remain_distinct_from_native_insufficient(prefix,reason,label):
    row = {'evidence_status':'INSUFFICIENT',prefix+'_fallback_applied':'YES',prefix+'_fallback_reason':reason}
    assert io.outcome(row) == label
    assert io.outcome({'evidence_status':'INSUFFICIENT'}) == 'Model-native responses'


def test_output_paths_are_canonical(tmp_path):
    for spec in common.INVENTORY:
        for fmt in common.FORMATS:
            assert style.output_path(tmp_path,spec,fmt) == tmp_path/'artifacts'/'figures'/spec.category/fmt/(spec.figure_id+'.'+fmt)
    with pytest.raises(common.Unsupported):
        style.output_path(tmp_path,common.Spec(2,'ustc_split_distribution_1'),'png')
    assert not list(tmp_path.iterdir())


def prepare_skipped_output(root):
    for section in common.SECTIONS:
        for fmt in common.FORMATS:
            (root/common.OUTPUT/section/fmt).mkdir(parents=True)
    rows, logs = [], []
    for spec in common.INVENTORY:
        row = dict.fromkeys(common.MANIFEST_FIELDS,'')
        dataset,scope = common.scope_for(spec)
        row.update(figure_id=spec.figure_id,figure_number=spec.number,section=spec.category,title=spec.title,
                   status='SKIPPED_UNSUPPORTED',source_artifacts='[]',dataset_scope=dataset,
                   evaluation_scope=scope,notes='Fixture: saved data absent',reason='Saved data absent')
        rows.append(row)
        logs.append(dict(figure_id=spec.figure_id,status=row['status'],error_type='Unsupported',message=row['notes']))
    manifests.write_csv(root,manifests.MANIFEST,common.MANIFEST_FIELDS,rows)
    summary = manifests.summary_record(rows)
    manifests.write_csv(root,manifests.SUMMARY,tuple(summary),[summary])
    manifests.write_csv(root,manifests.LOG,manifests.LOG_FIELDS,logs)
    report = manifests.validate(root,check_report=False,quiet=True)
    manifests.write_validation_report(root,report)
    return rows,logs


def test_manifest_summary_validate_modes_are_read_only(tmp_path,monkeypatch):
    prepare_skipped_output(tmp_path)
    before = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr(cli,'plotting',lambda *args:pytest.fail('Read-only mode rendered'))
    assert cli.main(['--project-root',str(tmp_path),'--summary']) == 0
    assert cli.main(['--project-root',str(tmp_path),'--validate']) == 0
    after = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob('*') if p.is_file()}
    assert before == after


@pytest.mark.parametrize('extra',['fig02_ustc_split_distribution_1.png',
                                  'fig02_ustc_split_distribution_20260928.svg','unknown.pdf'])
def test_validation_rejects_extra_outputs(tmp_path,extra):
    prepare_skipped_output(tmp_path)
    (tmp_path/common.OUTPUT/'dataset'/'png'/extra).write_bytes(b'extra')
    with pytest.raises(common.Unsupported,match='Unmanifested'):
        manifests.validate(tmp_path)


def test_validation_rejects_manifest_duplicates_and_summary_drift(tmp_path):
    rows,_ = prepare_skipped_output(tmp_path)
    rows[-1]['figure_id'] = rows[0]['figure_id']
    manifests.write_csv(tmp_path,manifests.MANIFEST,common.MANIFEST_FIELDS,rows)
    with pytest.raises(common.Unsupported,match='Manifest inventory'):
        manifests.validate(tmp_path)
    rows[-1]['figure_id'] = common.INVENTORY[-1].figure_id
    manifests.write_csv(tmp_path,manifests.MANIFEST,common.MANIFEST_FIELDS,rows)
    summary = manifests.summary_record(rows)
    summary['generated_figures'] = 1
    manifests.write_csv(tmp_path,manifests.SUMMARY,tuple(summary),[summary])
    with pytest.raises(common.Unsupported,match='summary disagrees'):
        manifests.read_summary(tmp_path)


def test_unsupported_and_technical_failure_are_isolated(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'plotting',lambda root:SimpleNamespace(close=lambda *args:None))
    called = []
    def build(spec,*args):
        called.append(spec.figure_id)
        if spec.number == 1:
            raise RuntimeError('render failure fixture')
        raise common.Unsupported('missing saved artifact fixture')
    monkeypatch.setattr(cli,'build',build)
    assert cli.execute(tmp_path) == 2
    assert called == [s.figure_id for s in common.INVENTORY]
    rows = manifests.read_manifest(tmp_path)
    assert rows[0]['status'] == 'SKIPPED_ERROR'
    assert all(r['status'] == 'SKIPPED_UNSUPPORTED' for r in rows[1:])
    assert all(r['reason'] for r in rows)
    assert not list(tmp_path.rglob('*.png'))
    assert manifests.read_summary(tmp_path)['failed_figures'] == 1


def test_missing_scientific_artifacts_skip_without_rendering(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'plotting',lambda root:SimpleNamespace(close=lambda *args:None))
    assert cli.execute(tmp_path) == 0
    assert all(r['status']=='SKIPPED_UNSUPPORTED' for r in manifests.read_manifest(tmp_path))


def test_three_formats_overwrite_solid_style_and_manifest(tmp_path):
    pytest.importorskip('matplotlib')
    rows,logs = prepare_skipped_output(tmp_path)
    plt = style.plotting(tmp_path)
    spec = common.INVENTORY[1]
    source = tmp_path/SOURCE
    source.parent.mkdir(parents=True)
    source.write_text('count\n2\n',encoding='utf-8')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    for _ in range(2):
        fig = style.bars(plt,{'A':2,'B':3})
        assert all(p.get_alpha() in (None,1.0) and not p.get_hatch() for ax in fig.axes for p in ax.patches)
        paths = style.save_publication_figure(fig,tmp_path,spec)
        assert not plt.fignum_exists(fig.number)
    files = [p for p in (tmp_path/common.OUTPUT).rglob('*') if p.suffix in ('.png','.svg','.pdf')]
    assert len(files) == 3 and all(p.stem == spec.figure_id for p in files)
    from PIL import Image
    with Image.open(tmp_path/paths['png_path']) as image:
        assert all(abs(v-600)<1 for v in image.info['dpi'])
    svg = (tmp_path/paths['svg_path']).read_text(encoding='utf-8')
    assert '<svg' in svg and '<text' in svg and '<image' not in svg
    assert (tmp_path/paths['pdf_path']).read_bytes().startswith(b'%PDF')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    rows[1].update(paths,status='GENERATED',reason='',notes='Fixture saved counts',source_artifacts=json.dumps([SOURCE]))
    logs[1].update(status='GENERATED',error_type='',message=rows[1]['notes'])
    manifests.write_csv(tmp_path,manifests.MANIFEST,common.MANIFEST_FIELDS,rows)
    summary = manifests.summary_record(rows)
    manifests.write_csv(tmp_path,manifests.SUMMARY,tuple(summary),[summary])
    manifests.write_csv(tmp_path,manifests.LOG,manifests.LOG_FIELDS,logs)
    report = manifests.validate(tmp_path,check_report=False,quiet=True)
    manifests.write_validation_report(tmp_path,report)
    assert manifests.validate(tmp_path)['generated_figures'] == 1
    (tmp_path/paths['svg_path']).write_bytes(b'')
    with pytest.raises(common.Unsupported,match='Missing/empty'):
        manifests.validate(tmp_path)


CONSOLIDATED_IDS = (
    'dataset_overview', 'dataset_split_pies', 'flow_extraction_combined',
    'phase1_workflow', 'phase1_combined_roc', 'phase1_model_comparison',
    'phase1_best_model_confusions', 'phase1_final_test_summary',
    'phase2_overall_pipeline', 'phase2_stage_summary',
    'phase2_stage8_to_stage10_summary', 'phase2_final_test_summary',
)


def test_consolidated_inventory_exact_and_no_legacy_duplicates():
    from src.figures.final_paper.common import registry, legacy_registry
    assert tuple(s.figure_id for s in registry('consolidated')) == CONSOLIDATED_IDS
    assert not set(CONSOLIDATED_IDS) & {s.figure_id for s in legacy_registry()}
    for name in ('consolidated_dataset_figures','consolidated_phase1_figures','consolidated_phase2_figures'):
        importlib.import_module('src.figures.final_paper.'+name)


def test_consolidated_phase2_scope_boundaries():
    from src.figures.final_paper.consolidated_phase2_figures import specs, members, scope
    byid = {s.detail:s for s in specs()}
    full = byid['stage_summary']
    assert full.split == 'TEST'
    assert [s.detail for s in members(full)] == list(map(str,range(1,8)))
    assert 'full held-out TEST' in scope(full)
    for name in ('stage8_to_stage10_summary','final_test_summary'):
        assert 'Stratified held-out TEST subset' in scope(byid[name])
        assert 'not full-TEST population metrics' in scope(byid[name])


def test_combined_roc_requires_every_model_and_class():
    from src.figures.final_paper.artifact_inventory import Inventory
    from src.figures.final_paper.common import MODELS, CLASSES, Unsupported
    inv = Inventory(Path('.'))
    rows = [dict(model=m,class_id=str(c)) for m in MODELS for c in range(len(CLASSES[1])) for _ in range(2)]
    inv.run_rows = lambda *args: rows
    assert inv.roc_rows(1) == rows
    rows.pop()
    with pytest.raises(Unsupported,match='Missing saved TEST ROC'):
        inv.roc_rows(1)


def test_combined_roc_uses_saved_coordinates_and_rejects_invalid_values():
    from src.figures.final_paper.phase1_test_figures import combined_roc
    from src.figures.final_paper.common import MODELS, Unsupported
    rows = [dict(model=m,fpr=x,tpr=y) for m in MODELS for x,y in [(0,0),(.2,.7),(1,1)]]
    lines = []
    ax = SimpleNamespace(plot=lambda x,y,**kw:lines.append((x,y,kw)), set=lambda **kw:None)
    inv = SimpleNamespace(roc_rows=lambda i:rows)
    combined_roc(ax,inv,0)
    assert len(lines) == 6
    assert all(x == [0,.2,1] and y == [0,.7,1] for x,y,_ in lines[:5])
    assert [kw['label'] for _,_,kw in lines[:5]] == list(MODELS.values())
    rows[1]['fpr'] = 2
    with pytest.raises(Unsupported,match='outside'):
        combined_roc(ax,inv,0)
