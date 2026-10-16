"""Common support for immutable CSV audit runs."""
from datetime import datetime, timezone
from pathlib import Path
from src.common.io import write_csv_atomic

class AuditRun:
    def __init__(self, root: Path, dataset: str, kind: str):
        now = datetime.now(timezone.utc)
        self.run_id = now.strftime('%Y%m%dT%H%M%SZ')
        self.timestamp_utc = now.isoformat().replace('+00:00', 'Z')
        self.path = root / 'results' / 'audits' / dataset / kind / self.run_id
        # A same-second collision fails closed; retry in a later second.
        self.path.mkdir(parents=True, exist_ok=False)

    def write(self, name, columns, rows):
        write_csv_atomic(self.path / name, columns, (
            dict(row, run_id=self.run_id, timestamp_utc=self.timestamp_utc) for row in rows))

    def finish(self):
        write_csv_atomic(self.path / 'artifact_index.csv', ['relative_path', 'size_bytes'], [
            dict(relative_path=p.name, size_bytes=p.stat().st_size)
            for p in sorted(self.path.glob('*.csv')) if p.name != 'artifact_index.csv'])
        return self.path
