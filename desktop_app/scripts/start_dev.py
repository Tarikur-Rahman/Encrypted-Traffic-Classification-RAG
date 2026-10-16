"""Convenience launcher using system Node or the isolated local runtime."""
from pathlib import Path
import os
import shutil
import subprocess

app = Path(__file__).resolve().parents[1]
local = app / 'tools/node-v22.22.0-win-x64/node.exe'
node = str(local) if local.exists() else shutil.which('node')
if not node:
    raise SystemExit('Install Node.js 22 or newer, then run npm install inside desktop_app.')
environment = dict(os.environ)
environment.pop('ELECTRON_RUN_AS_NODE', None)
raise SystemExit(subprocess.call([node, 'scripts/start.mjs', '--dev'], cwd=app, env=environment))
