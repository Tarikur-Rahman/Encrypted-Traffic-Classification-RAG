from pathlib import Path
from backend.config import APP, RUNTIME

def output_path(path):
    resolved = Path(path).resolve()
    if resolved == APP or not resolved.is_relative_to(APP):
        raise ValueError('Refusing to write outside desktop_app.')
    return resolved

def analysis_directory(identifier):
    if not identifier.isalnum():
        raise ValueError('Invalid analysis identity.')
    directory = output_path(RUNTIME / 'analyses' / identifier)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
