@echo off
setlocal
cd /d "%~dp0"
set "ELECTRON_RUN_AS_NODE="
if exist "tools\node-v22.22.0-win-x64\node.exe" (
  "tools\node-v22.22.0-win-x64\node.exe" "scripts\start.mjs"
) else (
  node "scripts\start.mjs"
)
