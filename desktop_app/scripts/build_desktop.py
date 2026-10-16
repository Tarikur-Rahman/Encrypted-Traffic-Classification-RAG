"""Build a Windows portable executable without changing the research tree."""
from pathlib import Path
import os
import shutil
import subprocess

app = Path(__file__).resolve().parents[1]
local = app / 'tools/node-v22.22.0-win-x64'
environment = dict(os.environ)
if local.exists():
    environment['PATH'] = str(local) + os.pathsep + environment.get('PATH','')
environment['npm_config_cache'] = str(app/'runtime/npm-cache')
environment['ELECTRON_CACHE'] = str(app/'runtime/electron-cache')
environment['ELECTRON_BUILDER_CACHE'] = str(app/'runtime/builder-cache')
node = str(local/'node.exe') if local.exists() else shutil.which('node')
if not node:
    raise SystemExit('Node.js is required.')
cli = local/'node_modules/npm/bin/npm-cli.js' if local.exists() else Path(shutil.which('npm')).parent/'node_modules/npm/bin/npm-cli.js'
raise SystemExit(subprocess.call([node,str(cli),'run','package'],cwd=app,env=environment))
