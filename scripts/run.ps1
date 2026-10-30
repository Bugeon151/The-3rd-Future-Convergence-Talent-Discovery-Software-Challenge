# Runs the NEAR-ZERO X dashboard (sim scenario by default) at http://localhost:5000
$Root = Split-Path -Parent $PSScriptRoot
& (Join-Path $Root ".venv\Scripts\python.exe") -m nearzero.demo --mode sim --scenario demo_avoided --port 5000