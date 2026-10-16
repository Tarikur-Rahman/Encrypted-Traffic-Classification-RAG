"""USTC preparation only; default invocation is a read-only structural preflight.

Native flow columns are retained unchanged, including excluded columns. Only
feature_summary rows with role=predictor_candidate may later become predictors.
Capture IDs are release-relative PCAP paths, consistent with the raw audit.
Distributions include zero-flow captures/families from the source inventory;
percentages use extracted flow counts and are on a 0..100 scale. Unique counts
exclude missing values. Extraction streams to a temporary CSV; descriptive
analysis subsequently loads that flow table into memory with pandas.

The six result CSVs are published before flows.csv, the completion output.
Failures may leave a partial result directory, which is never reused. No
cross-file transaction is claimed. Concurrent publication cannot overwrite
flows.csv. No packet copies, feature transformations, or final target mapping.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from itertools import chain
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def _register_npcap_dll_directory_if_available():
    """Register early in Windows spawn children without blocking preflight."""
    if os.name != 'nt':
        return None
    windir = os.environ.get('WINDIR')
    if not windir:
        return None
    npcap_dir = Path(windir) / 'System32' / 'Npcap'
    try:
        if npcap_dir.is_dir() and all((npcap_dir / name).is_file() for name in ('wpcap.dll', 'Packet.dll')):
            return os.add_dll_directory(str(npcap_dir))
    except OSError:
        # Execute-time validation reports prerequisites and registration errors.
        pass
    return None


# Windows spawn re-executes this module before unpickling NFStream workers.
# Keep the handle alive for this process's lifetime, including in each child.
_NPCAP_DLL_HANDLE = _register_npcap_dll_directory_if_available()

import pandas as pd

from src.common.io import write_csv_atomic
from src.data.audit_raw import provenance

DATASET = 'USTC-TFC2016'
METADATA = ['binary_label', 'semantic_family', 'capture_group', 'source_pcap']
NFSTREAM_OPTIONS = dict(
    decode_tunnels=True, snapshot_length=1536, idle_timeout=120,
    active_timeout=1800, accounting_mode=0, n_dissections=0,
    statistical_analysis=True, splt_analysis=0, n_meters=0,
    max_nflows=0, performance_report=0, system_visibility_mode=0,
)
TAIL = ['run_id', 'timestamp_utc']
RESULT_HEADERS = {
    'dataset_summary.csv': ['dataset', 'total_flows', 'benign_flows', 'malware_flows', 'semantic_family_count', 'capture_group_count', 'native_feature_count', 'predictor_candidate_count'] + TAIL,
    'class_distribution.csv': ['dataset', 'class', 'flow_count', 'percentage'] + TAIL,
    'family_distribution.csv': ['dataset', 'binary_label', 'semantic_family', 'flow_count', 'percentage_of_dataset', 'percentage_within_binary_class'] + TAIL,
    'capture_distribution.csv': ['dataset', 'binary_label', 'semantic_family', 'capture_group', 'source_pcap', 'flow_count', 'percentage'] + TAIL,
    'feature_summary.csv': ['dataset', 'feature_name', 'dtype', 'missing_count', 'missing_percentage', 'unique_count', 'role'] + TAIL,
    'processing_summary.csv': ['dataset', 'source_pcap_count', 'output_flow_count', 'native_column_count', 'predictor_candidate_count', 'provenance_metadata_count', 'excluded_identifier_count', 'excluded_dpi_count'] + TAIL,
}


def discover_pcaps(root: Path) -> list[dict]:
    """Inspect filenames and paths only; never open packet contents."""
    base = root / 'data' / 'raw' / DATASET
    if not base.is_dir():
        raise FileNotFoundError(base)
    captures = []
    resolved = set()
    for path in sorted(p for p in base.rglob('*') if p.is_file() and p.suffix.lower() in {'.pcap', '.pcapng'}):
        physical_path = path.resolve()
        if not physical_path.is_relative_to(base.resolve()) or physical_path in resolved:
            raise ValueError(f'External or repeated source path: {path}')
        resolved.add(physical_path)
        label, family, group = provenance(path, base)
        if not label or not family:
            raise ValueError(f'Missing authoritative release metadata: {path}')
        captures.append(dict(binary_label=label, semantic_family=family,
                             capture_group=group, source_pcap=path.relative_to(base).as_posix()))
    return captures


def validate_structure(captures: list[dict]) -> dict:
    families = {(r['binary_label'], r['semantic_family']) for r in captures}
    counts = dict(pcap_count=len(captures),
                  benign_pcap_count=sum(r['binary_label'] == 'benign' for r in captures),
                  malware_pcap_count=sum(r['binary_label'] == 'malware' for r in captures),
                  semantic_family_count=len(families),
                  benign_family_count=sum(label == 'benign' for label, _ in families),
                  malware_family_count=sum(label == 'malware' for label, _ in families))
    expected = dict(pcap_count=24, benign_pcap_count=14, malware_pcap_count=10,
                    semantic_family_count=20, benign_family_count=10, malware_family_count=10)
    failures = [f'{key}={counts[key]} (expected {value})' for key, value in expected.items() if counts[key] != value]
    if len({r['capture_group'] for r in captures}) != len(captures):
        failures.append('capture_group identifiers are not unique')
    if failures:
        raise ValueError('USTC structure validation failed: ' + '; '.join(failures))
    return counts


def feature_role(name: str) -> str:
    if name == 'binary_label':
        return 'target'
    if name in METADATA[1:]:
        return 'provenance_metadata'
    lower = name.lower()
    if (lower.startswith(('application', 'dpi', 'ndpi', 'http_', 'tls_', 'ssl_', 'dns_'))
            or any(token in lower for token in ('fingerprint', 'server_name', 'user_agent', 'content_type'))):
        return 'excluded_dpi'
    if (lower in {'id', 'expiration_id', 'src_ip', 'dst_ip', 'src_mac', 'dst_mac',
                  'src_oui', 'dst_oui', 'src_port', 'dst_port', 'vlan_id'}
            or 'first_seen' in lower or 'last_seen' in lower
            or lower.startswith('system_')
            or any(token in lower for token in ('ip_address', 'filename', 'source_file', 'row_id'))):
        return 'excluded_identifier'
    return 'predictor_candidate'


def load_streamer():
    """Lazy import keeps preflight usable without the native dependency."""
    global _NPCAP_DLL_HANDLE
    try:
        installed = version('nfstream')
    except PackageNotFoundError as exc:
        raise RuntimeError('Execution requires nfstream==6.6.0; it is not installed.') from exc
    if installed != '6.6.0':
        raise RuntimeError(f'Execution requires nfstream==6.6.0; found {installed}.')
    if os.name == 'nt':
        windir = os.environ.get('WINDIR')
        if not windir:
            raise RuntimeError('Npcap is required for NFStream on Windows; WINDIR is not set.')
        npcap_dir = Path(windir) / 'System32' / 'Npcap'
        if not npcap_dir.is_dir():
            raise RuntimeError(f'Npcap is required for NFStream on Windows; directory not found: {npcap_dir}')
        missing = [name for name in ('wpcap.dll', 'Packet.dll') if not (npcap_dir / name).is_file()]
        if missing:
            raise RuntimeError(
                f'Npcap is required for NFStream on Windows; missing {", ".join(missing)} in {npcap_dir}.'
            )
        if _NPCAP_DLL_HANDLE is None:
            try:
                _NPCAP_DLL_HANDLE = os.add_dll_directory(str(npcap_dir))
            except OSError as exc:
                raise RuntimeError(
                    f'Npcap is required for NFStream on Windows; unable to add DLL directory: {npcap_dir}'
                ) from exc
    from nfstream import NFStreamer
    return NFStreamer


def flow_records(root: Path, captures: list[dict], streamer_factory):
    columns = None
    for capture in captures:
        path = root / 'data' / 'raw' / DATASET / capture['source_pcap']
        count = 0
        for flow in streamer_factory(source=str(path), **NFSTREAM_OPTIONS):
            keys, values = list(flow.keys()), list(flow.values())
            if not keys or len(keys) != len(set(keys)) or len(keys) != len(values) or set(keys) & set(METADATA):
                raise ValueError(f'Invalid or metadata-conflicting native columns in {path}')
            if columns is None:
                columns = keys
            if keys != columns:
                raise ValueError(f'NFStream column schema changed in {path}')
            yield {key: None if pd.isna(value) else value for key, value in zip(keys, values)} | capture
            count += 1
        print(f"{capture['source_pcap']}: {count} flows")


def summarize(frame: pd.DataFrame, captures: list[dict]) -> dict:
    if frame.empty:
        raise ValueError('Extraction produced no flows.')
    if not set(METADATA).issubset(frame.columns):
        raise ValueError('Missing required flow metadata columns.')
    if frame[METADATA].isna().any().any() or frame[METADATA].eq('').any().any():
        raise ValueError('Every flow must have complete provenance metadata.')
    actual = set(frame[METADATA].itertuples(index=False, name=None))
    allowed = {tuple(r[c] for c in METADATA) for r in captures}
    if not actual.issubset(allowed):
        raise ValueError('Flow metadata does not match the source inventory.')
    total = len(frame)
    native_count = len(frame.columns) - len(METADATA)
    roles = Counter(feature_role(c) for c in frame.columns)
    class_counts = frame.groupby('binary_label').size()
    family_counts = frame.groupby(['binary_label', 'semantic_family']).size()
    capture_counts = frame.groupby('capture_group').size()
    families = sorted({(r['binary_label'], r['semantic_family']) for r in captures})
    def percentage(count, denominator):
        return 100.0 * count / denominator if denominator else 0.0
    tables = {
        'dataset_summary.csv': [dict(total_flows=total,
            benign_flows=int(class_counts.get('benign', 0)), malware_flows=int(class_counts.get('malware', 0)),
            semantic_family_count=len(families), capture_group_count=len(captures),
            native_feature_count=native_count, predictor_candidate_count=roles['predictor_candidate'])],
        'class_distribution.csv': [{'class': label, 'flow_count': int(class_counts.get(label, 0)),
            'percentage': percentage(class_counts.get(label, 0), total)} for label in ['benign', 'malware']],
        'family_distribution.csv': [dict(binary_label=label, semantic_family=family,
            flow_count=int(family_counts.get((label, family), 0)),
            percentage_of_dataset=percentage(family_counts.get((label, family), 0), total),
            percentage_within_binary_class=percentage(family_counts.get((label, family), 0), class_counts.get(label, 0)))
            for label, family in families],
        'capture_distribution.csv': [dict(r, flow_count=int(capture_counts.get(r['capture_group'], 0)),
            percentage=percentage(capture_counts.get(r['capture_group'], 0), total)) for r in captures],
        'feature_summary.csv': [dict(feature_name=c, dtype=str(frame[c].dtype),
            missing_count=int(frame[c].isna().sum()), missing_percentage=percentage(frame[c].isna().sum(), total),
            unique_count=int(frame[c].nunique(dropna=True)), role=feature_role(c)) for c in frame.columns],
        'processing_summary.csv': [dict(source_pcap_count=len(captures), output_flow_count=total,
            native_column_count=native_count, predictor_candidate_count=roles['predictor_candidate'],
            provenance_metadata_count=roles['provenance_metadata'],
            excluded_identifier_count=roles['excluded_identifier'], excluded_dpi_count=roles['excluded_dpi'])],
    }
    for name in ['class_distribution.csv', 'family_distribution.csv', 'capture_distribution.csv']:
        if sum(row['flow_count'] for row in tables[name]) != total:
            raise ValueError(f'Flow count reconciliation failed: {name}')
    return tables


def prepare_dataset(root: Path, execute: bool = False) -> tuple[Path, Path]:
    root = root.resolve()
    captures = discover_pcaps(root)
    counts = validate_structure(captures)
    now = datetime.now(timezone.utc)
    run_id = now.strftime('%Y%m%dT%H%M%SZ')
    timestamp = now.isoformat().replace('+00:00', 'Z')
    output = root / 'data' / 'interim' / DATASET / 'flows.csv'
    result_dir = root / 'results' / 'datasets' / DATASET / run_id
    for name, value in counts.items():
        print(f'{name}={value}')
    print(f'flow_output={output}\nresult_directory={result_dir}')
    print('PREFLIGHT_ONLY=' + str(not execute))
    if not execute:
        return output, result_dir
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    if result_dir.exists():
        raise FileExistsError(f'Refusing to reuse: {result_dir}')
    streamer_factory = load_streamer()
    result_dir.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.ustc-prepare-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'flows.csv'
        records = flow_records(root, captures, streamer_factory)
        first = next(records, None)
        if first is None:
            raise ValueError('Extraction produced no flows.')
        write_csv_atomic(staged, list(first), chain([first], records))
        frame = pd.read_csv(staged, low_memory=False, keep_default_na=False, na_values=[''],
                            dtype={c: 'string' for c in METADATA})
        tables = summarize(frame, captures)
        for name, rows in tables.items():
            write_csv_atomic(result_dir / name, RESULT_HEADERS[name], (
                dict(row, dataset=DATASET, run_id=run_id, timestamp_utc=timestamp) for row in rows))
        if {p.name for p in result_dir.iterdir()} != set(RESULT_HEADERS):
            raise RuntimeError('Expected exactly six dataset-result CSVs.')
        # Exclusive, atomic publication of the completed flow dataset.
        os.link(staged, output)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError('Missing or empty completed flow dataset.')
    print(f'VALIDATION=PASS\noutput_flow_count={len(frame)}')
    return output, result_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    prepare_dataset(args.project_root, execute=args.execute)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
