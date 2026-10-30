# Bootstraps the GPU-enabled virtual environment for NEAR-ZERO X.
# Requires: Python 3.10-3.13, an NVIDIA GPU, CUDA 12.x/13.x driver.
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $Root ".venv"
$Py = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Py)) {
    Write-Host "[1/3] creating virtualenv..."
    python -m venv $Venv
}

Write-Host "[2/3] installing core deps (PyTorch CUDA build)..."
& $Py -m pip install --upgrade pip
& $Py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw "torch cu130 install failed" }
& $Py -m pip install -r (Join-Path $Root "requirements.txt")

Write-Host "[3/3] verifying CUDA..."
& $Py -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available())"

Write-Host "Done. Use scripts/run.ps1 (dashboard), scripts/run_demo.ps1 (headless demo), scripts/run_eval.ps1 (benchmarks)."