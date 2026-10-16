"""Read-only CSV/JSONL access with seek indexes kept exclusively in the app.

Indexes contain offsets, not research computations. CSV offsets respect quoted
multiline cells. Source metadata invalidates an index on change.
"""
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from threading import RLock
from backend.config import ROOT, RUNTIME

csv.field_size_limit(100_000_000)
LOCK = RLock()

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        if str(path).endswith('.jsonl'):
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        else:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError('CSV headers must be present and unique.')
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise ValueError('The CSV contains an incomplete or malformed record.')
                yield row

def identity(path):
    stat = path.stat()
    return [str(path.resolve()), stat.st_size, stat.st_mtime_ns]

def select(relative, key, value):
    path = ROOT / relative
    if not path.is_file():
        return []
    fingerprint = identity(path)
    name = hashlib.sha256((str(path) + key).encode()).hexdigest()[:24]
    target = RUNTIME / ('index-' + name + '.json')
    with LOCK:
        cached = json.loads(target.read_text()) if target.exists() else {}
        if cached.get('source') != fingerprint:
            offsets = {}
            with path.open(encoding='utf-8-sig', newline='') as stream:
                is_json = path.suffix == '.jsonl'
                reader = None if is_json else csv.reader(iter(stream.readline, ''))
                header = [] if is_json else next(reader)
                while True:
                    offset = stream.tell()
                    if is_json:
                        line = stream.readline()
                        if not line:
                            break
                        row = json.loads(line)
                    else:
                        cells = next(reader, None)
                        if cells is None:
                            break
                        row = dict(zip(header, cells))
                    offsets.setdefault(str(row.get(key, '')), []).append(offset)
            if identity(path) != fingerprint:
                raise ValueError('The research artifact changed while being read.')
            cached = dict(source=fingerprint, header=header, offsets=offsets)
            temporary = target.with_suffix('.tmp')
            temporary.write_text(json.dumps(cached), encoding='utf-8')
            temporary.replace(target)
    result = []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for offset in cached['offsets'].get(str(value), []):
            stream.seek(offset)
            result.append(json.loads(stream.readline()) if path.suffix == '.jsonl' else
                          dict(zip(cached['header'], next(csv.reader(iter(stream.readline, ''))))))
    return result

def decode(value, fallback=None):
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback

def checklist(relative):
    path = ROOT / relative
    if not path.exists():
        return {'status': 'UNAVAILABLE', 'path': relative, 'checks': []}
    checks = list(rows(path))
    return {'status': 'PASS' if checks and all(r.get('status') == 'PASS' for r in checks) else 'REVIEW',
            'path': relative, 'checks': checks}
