from __future__ import annotations
import json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from src.common.io import write_json_atomic

def main() -> int:
    root = Path(__file__).resolve().parents[2]
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    except Exception:
        freeze = []
    payload = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "pip_freeze": freeze,
    }
    out = root / "metadata/environment/environment.json"
    write_json_atomic(out, payload)
    print(out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
