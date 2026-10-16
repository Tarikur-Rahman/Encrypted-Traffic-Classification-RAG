$ErrorActionPreference = 'Stop'
$scriptPath = Join-Path $PSScriptRoot 'start-local-llm.sh'
$linuxPath = (& wsl.exe -d Ubuntu-24.04 -- wslpath -a $scriptPath).Trim()
# This launcher owns no unrelated server. Logs/caches stay in desktop_app/runtime.
& wsl.exe -d Ubuntu-24.04 -- bash $linuxPath
