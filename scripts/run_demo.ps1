# Headless transcript of the demo scenario (risk timeline + counterfactual table).
param(
    [string]$Scenario = "demo_avoided",
    [int]$Steps = 80
)
$Root = Split-Path -Parent $PSScriptRoot
& (Join-Path $Root ".venv\Scripts\python.exe") -m nearzero.demo --headless --scenario $Scenario --steps $Steps