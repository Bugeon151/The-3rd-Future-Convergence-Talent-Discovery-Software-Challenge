# Runs the unit + integration test suite (pytest).
$Root = Split-Path -Parent $PSScriptRoot
$Tmp = Join-Path $env:TEMP "opencode\nx_tests"
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
& (Join-Path $Root ".venv\Scripts\python.exe") -m pytest --basetemp "$Tmp" -q tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }