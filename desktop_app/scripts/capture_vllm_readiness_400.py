"""Capture the current vLLM readiness-probe response without changing runtime code."""
import json
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ENDPOINT = 'http://127.0.0.1:8000/v1/chat/completions'
TARGET = Path(__file__).resolve().parents[1] / 'runtime/diagnostics/vllm_400/readiness_probe_400.json'


def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    body = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    request = urllib.request.Request(ENDPOINT, data=body,
        headers={'Content-Type': 'application/json'}, method='POST')
    record = {
        'captured_at': datetime.now(timezone.utc).isoformat(),
        'caller': 'backend.external_phase2.llm_runtime.check_llm_runtime',
        'function': 'check_llm_runtime',
        'endpoint': ENDPOINT,
        'method': 'POST',
        'request_payload': payload,
        'request_body_bytes': body.decode('utf-8'),
    }
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=5) as response:
            record.update(response_status=response.status,
                response_headers=dict(response.headers),
                response_body=response.read(1024 * 1024).decode('utf-8', errors='replace'))
    except urllib.error.HTTPError as exc:
        record.update(response_status=exc.code,
            response_headers=dict(exc.headers),
            response_body=exc.read(1024 * 1024).decode('utf-8', errors='replace'))
        exc.close()
    except Exception as exc:
        record.update(response_status=None, response_body=None,
            error=repr(exc), traceback=traceback.format_exc())
    TARGET.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record, indent=2))
    return 0 if record.get('response_status') in (400, 422) else 1


if __name__ == '__main__':
    raise SystemExit(main())
