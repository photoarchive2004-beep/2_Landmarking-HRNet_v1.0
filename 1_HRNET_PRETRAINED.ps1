$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $ScriptDir

$VenvPath = Join-Path $ScriptDir ".venv_hrnet"
$PythonExe = Join-Path $VenvPath "Scripts\python.exe"
$ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"

if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment not found at $VenvPath. Run 0_HRNET_SETUP_ENV.ps1 first."
}

if (Test-Path $ActivateScript) {
    Write-Host "[INFO] Activating virtual environment ..."
    & $ActivateScript
} else {
    Write-Host "[WARN] Activate.ps1 not found. Continuing without session activation."
}

Write-Host "[INFO] Running HRNet pretrained backbone test ..."

& $PythonExe .\scripts\test_hrnet_pretrained.py
if ($LASTEXITCODE -ne 0) {
    throw "test_hrnet_pretrained.py exited with code $LASTEXITCODE"
}

if (Test-Path (Join-Path $ScriptDir ".git")) {
    git status -sb
    git add scripts\test_hrnet_pretrained.py scripts\train_hrnet.py 1_HRNET_PRETRAINED.ps1
    git commit -m "HRNet: load ImageNet pretrained HRNet-W32 backbone" --allow-empty
    git push origin main
}
