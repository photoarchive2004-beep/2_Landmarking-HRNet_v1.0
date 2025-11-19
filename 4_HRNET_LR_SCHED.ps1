$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$logsDir = Join-Path $scriptDir "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logsDir ("hrnet_lr_sched_{0}.log" -f $timestamp)

Start-Transcript -Path $logPath | Out-Null

try {
    Write-Host "=== HRNet LR scheduler diagnostic ==="
    Write-Host "Project root: $scriptDir"

    $activateScript = Join-Path $scriptDir ".venv_lm\\Scripts\\Activate.ps1"
    if (-not (Test-Path $activateScript)) {
        throw "Virtual environment activator not found: $activateScript"
    }
    Write-Host "Activating virtual environment via $activateScript"
    & $activateScript

    $pythonExe = Join-Path $scriptDir ".venv_lm\\Scripts\\python.exe"
    if (-not (Test-Path $pythonExe)) {
        throw "Python executable not found: $pythonExe"
    }

    Write-Host "Running python -m scripts.test_hrnet_lr_scheduler ..."
    & $pythonExe -m scripts.test_hrnet_lr_scheduler
    if ($LASTEXITCODE -ne 0) {
        throw "scripts.test_hrnet_lr_scheduler failed with exit code $LASTEXITCODE"
    }

    Write-Host "Running git status/add/commit/push ..."
    git status
    git add scripts/train_hrnet.py scripts/test_hrnet_lr_scheduler.py 4_HRNET_LR_SCHED.ps1
    git commit -m "HRNet: add LR scheduler and scheduler test"
    git push origin main
}
catch {
    Write-Error $_
    throw
}
finally {
    Stop-Transcript | Out-Null
}
