param()

$ErrorActionPreference = "Stop"

# Корень проекта
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs" | Out-Null
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = ".\logs\hrnet_config_check_{0}.log" -f $ts

Start-Transcript -Path $logPath -Force

Write-Host "=== HRNet CONFIG CHECK ==="
Write-Host "Project root: $root"
Write-Host "Log file: $logPath"
Write-Host ""

# Активация окружения
$venv = ".\.venv_lm\Scripts\Activate.ps1"
if (Test-Path $venv) {
    Write-Host "[INFO] Activating .venv_lm ..."
    . $venv
} else {
    Write-Host "[WARN] .venv_lm not found, используется системный python."
}

Write-Host "[INFO] Using python:" (Get-Command python).Source
Write-Host ""

Write-Host "[STEP] Running python -m scripts.test_hrnet_config_consistency ..."
python -m scripts.test_hrnet_config_consistency
$exit = $LASTEXITCODE

if ($exit -ne 0) {
    Write-Host ""
    Write-Host "ERROR: test_hrnet_config_consistency exited with code $exit"
    Write-Host "See log: $logPath"
    Stop-Transcript
    exit $exit
}

Write-Host ""
Write-Host "[STEP] Git commit & push (новый тест и этот скрипт)..."
git status
git add scripts\test_hrnet_config_consistency.py 11_HRNET_CONFIG_CHECK.ps1
git commit -m "HRNet: add config consistency check (manual)" -ErrorAction SilentlyContinue | Out-Null
git push origin main

Write-Host ""
Write-Host "=== HRNet CONFIG CHECK DONE ==="
Write-Host "Log file: $logPath"

Stop-Transcript
