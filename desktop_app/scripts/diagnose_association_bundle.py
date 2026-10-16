"""Capture fail-closed V5f association-bundle identity diagnostics.

This script only reads the research bundle and writes diagnostics beneath
desktop_app/runtime. It never edits checked-in metadata or research artifacts.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import traceback
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def diff(expected, actual, prefix=''):
    keys = sorted(set(expected) | set(actual)) if isinstance(expected, dict) and isinstance(actual, dict) else []
    output = {}
    for key in keys:
        name = f'{prefix}.{key}' if prefix else key
        if key not in expected:
            output[name] = {'expected': '<missing>', 'actual': actual[key]}
        elif key not in actual:
            output[name] = {'expected': expected[key], 'actual': '<missing>'}
        elif isinstance(expected[key], dict) and isinstance(actual[key], dict):
            output.update(diff(expected[key], actual[key], name))
        elif expected[key] != actual[key]:
            output[name] = {'expected': expected[key], 'actual': actual[key]}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, default=Path('desktop_app/runtime/diagnostics/association_bundle_failure'))
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bundle_rel = Path('data/raw/rag/evidence_sources_v2/v5f_validation/bundle.json')
    bundle_path = root / bundle_rel
    bundle = json.loads(bundle_path.read_text(encoding='utf-8'))
    statistics_path = root / 'data/raw/rag/evidence_sources_v2/v5f_validation/statistics.json'
    chunks_path = root / 'data/processed/rag/v5f_validation/evidence_chunks.csv'
    index_path = root / 'artifacts/rag/v5f_validation/index/dense_embeddings.npy'

    runtime = {
        'sys.executable': sys.executable,
        'sys.version': sys.version,
        'python': platform.python_version(),
        'pandas': None,
        'numpy': None,
        'lightgbm': None,
    }
    try:
        import pandas
        runtime['pandas'] = pandas.__version__
    except Exception as exc:
        runtime['pandas_error'] = repr(exc)
    try:
        import numpy
        runtime['numpy'] = numpy.__version__
    except Exception as exc:
        runtime['numpy_error'] = repr(exc)
    try:
        import lightgbm
        runtime['lightgbm'] = lightgbm.__version__
    except Exception as exc:
        runtime['lightgbm_error'] = repr(exc)

    environment = {name: os.environ.get(name) for name in ('PATH', 'PYTHONPATH', 'VIRTUAL_ENV', 'CONDA_PREFIX')}
    expected = dict(bundle.get('provenance', {}))
    actual = {}
    failure = None
    try:
        from src.rag import build_semantic_association_v5 as method
        actual = method.freeze(root)
        try:
            records = method.verify_bundle(root)
            verification = {'status': 'PASS', 'record_count': len(records)}
        except Exception as exc:
            failure = ''.join(traceback.format_exception(exc)).strip()
            verification = {'status': 'FAIL', 'error': str(exc)}
    except Exception as exc:
        failure = ''.join(traceback.format_exception(exc)).strip()
        verification = {'status': 'FAIL', 'error': str(exc)}

    contract = {}
    try:
        from src.rag.final_test import contracts
        contract = {
            'configuration_id': contracts.RUN_ID,
            'contract_id': contracts.RUN_ID,
            'validation_profile': contracts.DEVELOPMENT_ID,
            'model_run_id': contracts.RUN_ID,
            'model': contracts.FROZEN['model'],
            'max_model_len': contracts.FROZEN['max_model_len'],
            'max_completion_tokens': contracts.FROZEN['max_completion_tokens'],
            'fusion_policy': contracts.FROZEN['fusion_policy'],
            'bundle_sha256': contracts.BUNDLE_SHA256,
        }
    except Exception as exc:
        contract = {'error': repr(exc)}

    artifact_paths = {
        'association_bundle': str(bundle_path),
        'evidence_corpus': str(chunks_path),
        'evidence_index': str(index_path),
        'statistics': str(statistics_path),
        'configuration': str(root / 'configs/rag_semantic_association_v5.json'),
        'metadata_manifest': str(root / 'metadata/rag/v5f_validation/stage02_corpus/evidence_source_manifest_v2.csv'),
    }
    artifact_identity = {}
    for name, value in artifact_paths.items():
        path = Path(value)
        artifact_identity[name] = {
            'path': value,
            'exists': path.is_file(),
            'size': path.stat().st_size if path.is_file() else None,
            'sha256': sha256(path) if path.is_file() else None,
        }
    record_count = None
    if statistics_path.is_file():
        record_count = len(json.loads(statistics_path.read_text(encoding='utf-8')))

    expected_identity = {
        'bundle_identifier': expected.get('version'),
        'validation_profile': 'v5f_validation',
        'bundle_sha256': sha256(bundle_path),
        'bundle_record_count': record_count,
        'provenance': expected,
        'runtime': {
            'python': expected.get('environment', {}).get('python'),
            'pandas': expected.get('environment', {}).get('pandas'),
            'numpy': expected.get('environment', {}).get('numpy'),
            'lightgbm': '4.7.0',
        },
        'contract': contract,
        'artifact_identity': artifact_identity,
        'project_root': str(root),
    }
    actual_identity = {
        'bundle_identifier': actual.get('version'),
        'validation_profile': 'v5f_validation',
        'bundle_sha256': sha256(bundle_path),
        'bundle_record_count': record_count,
        'provenance': actual,
        'runtime': {key: runtime.get(key) for key in ('python', 'pandas', 'numpy', 'lightgbm')},
        'contract': contract,
        'artifact_identity': artifact_identity,
        'project_root': str(root),
    }
    identity_diff = diff(expected_identity, actual_identity)
    project = {
        'resolved_project_root': str(root),
        'cwd': str(Path.cwd().resolve()),
        'bundle_path': str(bundle_path),
        'desktop_app_path': str(root / 'desktop_app'),
        'metadata_paths': artifact_paths,
    }
    (output / 'runtime_identity.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    (output / 'expected_bundle_identity.json').write_text(json.dumps(expected_identity, indent=2, default=str), encoding='utf-8')
    (output / 'actual_bundle_identity.json').write_text(json.dumps(actual_identity, indent=2, default=str), encoding='utf-8')
    (output / 'identity_diff.json').write_text(json.dumps({'field_diff': identity_diff, 'verification': verification}, indent=2, default=str), encoding='utf-8')
    (output / 'backend_environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    (output / 'project_root.json').write_text(json.dumps(project, indent=2), encoding='utf-8')
    (output / 'failure.txt').write_text(failure or 'No failure: bundle verification passed.\n', encoding='utf-8')
    print(json.dumps({'verification': verification, 'runtime': runtime, 'field_diff': identity_diff, 'output': str(output)}, indent=2, default=str))
    return 0 if verification['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
