# Runs the full evaluation suite: baselines A/B/C vs NEAR-ZERO X (D),
# ablation, and the runtime benchmark. Writes results/ artifacts.
$Root = Split-Path -Parent $PSScriptRoot
& (Join-Path $Root ".venv\Scripts\python.exe") -m nearzero.eval.all