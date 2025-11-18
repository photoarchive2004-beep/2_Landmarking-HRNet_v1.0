$ErrorActionPreference = "Stop"

# HRNet environment installer (no YOLO, no ultralytics)
$LM_ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$envDir = Join-Path $LM_ROOT ".venv_lm"
$pythonExe = Join-Path $envDir "Scripts\\python.exe"

Write-Host "=== Installing HRNet env in: $envDir ==="

if (Test-Path $pythonExe) {
    Write-Host "Virtual environment already exists. Nothing to install."
    Write-Host "Using Python: $pythonExe"
    Write-Host "=== HRNet env is ready. ==="
    return
}

Write-Host "Creating virtual environment .venv_lm ..."
python -m venv "$envDir"

if (-not (Test-Path $pythonExe)) {
    Write-Host "ERROR: Failed to create virtual environment at $envDir"
    exit 1
}

Write-Host "Using Python: $pythonExe"
Write-Host "Upgrading pip/setuptools/wheel ..."
& "$pythonExe" -m pip install --upgrade pip setuptools wheel

Write-Host "Installing core packages for HRNet (numpy, pillow, opencv-python, pyyaml, pandas, matplotlib, scipy, torch, torchvision)..."
& "$pythonExe" -m pip install numpy pillow opencv-python pyyaml pandas matplotlib scipy
& "$pythonExe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Write-Host "=== HRNet env installation finished successfully. ==="
