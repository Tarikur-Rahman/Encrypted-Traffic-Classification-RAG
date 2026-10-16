import json
import threading
from datetime import datetime, timezone
from uuid import uuid4
from backend.config import RUNTIME
from backend.utils.paths import output_path

DIRECTORY = RUNTIME / 'sessions'
DIRECTORY.mkdir(exist_ok=True)
LOCK = threading.RLock()
CACHE = {}
STAGES = [('input','Input validation'),('extraction','Flow reconstruction / NFStream'),('schema','Schema validation'),('preprocessing','Frozen preprocessing'),
 ('classification','Phase-I classification'),('representation','Traffic representation + provenance'),
 ('traffic_provenance','Traffic provenance'),('comparison','Five-model comparison'),('query','Structured security query'),('corpus','Cybersecurity evidence corpus'),('index','Evidence indexing'),
 ('retrieval','Evidence retrieval'),('provenance','Provenance verification'),
 ('reranking','Provenance-aware reranking'),('context','Trusted context construction'),
 ('reasoning','Constrained LLM reasoning'),('fusion','Evidence-supported decision fusion'),('decision','Final classification')]

def save(session):
    with LOCK:
        path = output_path(DIRECTORY / (session['analysis_id']+'.json'))
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(session, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        temp.replace(path)

def create(source, completed_only=True):
    identifier = uuid4().hex[:12].upper()
    session = dict(analysis_id=identifier, timestamp=datetime.now(timezone.utc).isoformat(), input=source,
        completed_only=completed_only, status='created', stages=[dict(id=k,label=v,status='pending',message='') for k,v in STAGES],
        flows=[], phase2={}, error=None, audit_events=[])
    CACHE[identifier] = session
    save(session)
    return session

def get(identifier):
    if len(identifier) != 12 or any(c not in '0123456789ABCDEF' for c in identifier):
        raise KeyError('Unknown analysis.')
    with LOCK:
        if identifier not in CACHE:
            path = DIRECTORY / (identifier+'.json')
            if not path.exists():
                raise KeyError('Unknown analysis.')
            CACHE[identifier] = json.loads(path.read_text(encoding='utf-8'))
            if CACHE[identifier]['status'] == 'running':
                CACHE[identifier].update(status='interrupted', error='The backend stopped during processing. Start a new analysis or retry the selected flow.')
        return CACHE[identifier]

def stage(session, key, status, message):
    with LOCK:
        record = next(r for r in session['stages'] if r['id'] == key)
        record.update(status=status, message=message)
        session['audit_events'].append(dict(timestamp=datetime.now(timezone.utc).isoformat(), stage=key, status=status, message=message))
