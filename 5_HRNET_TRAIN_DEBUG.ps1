$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$logsDir = Join-Path $scriptDir "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logsDir ("hrnet_train_debug_{0}.log" -f $timestamp)

Start-Transcript -Path $logPath | Out-Null

try {
    Write-Host "=== HRNet debug training ==="
    Write-Host "Project root: $scriptDir"

    $activateScript = Join-Path $scriptDir ".venv_lm\\Scripts\\Activate.ps1"
    if (-not (Test-Path $activateScript)) {
        throw "Virtual environment activator not found: $activateScript"
    }
    Write-Host "Activating virtual environment via $activateScript"
    & $activateScript

    Write-Host "Running python -m scripts.train_hrnet --debug ..."
    python -m scripts.train_hrnet --debug
    if ($LASTEXITCODE -ne 0) {
        Write-Error "train_hrnet debug mode failed with exit code $LASTEXITCODE"
        return
    }

    Write-Host "Running git status/add/commit/push ..."
    git status
    git add scripts/train_hrnet.py 5_HRNET_TRAIN_DEBUG.ps1
    git commit -m "HRNet: add debug training mode"
    git push origin main
}
catch {
    Write-Error $_
    throw
}
finally {
    Stop-Transcript | Out-Null
}
