# 6_HRNET_TRAIN_FULL.ps1
# Полноценное обучение HRNet (без --debug, с max_epochs из train_hrnet.py)

param()

# Переходим в папку проекта (куда положен этот скрипт)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Папка для логов
if (-not (Test-Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs" | Out-Null
}

# Файл лога
$logPath = ".\logs\hrnet_train_full_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss")
Start-Transcript -Path $logPath -Force

Write-Host "=== HRNet FULL TRAINING ==="
Write-Host "Project root: $scriptDir"
Write-Host "Log file: $logPath"
Write-Host ""

# Активация окружения
$venvActivate = ".\.venv_lm\Scripts\Activate.ps1"
if (Test-Path $venvActivate) {
    Write-Host "Activating virtual environment via $venvActivate"
    . $venvActivate
} else {
    Write-Host "WARNING: .venv_lm not found, using system python."
}

Write-Host ""
Write-Host "Running: python -m scripts.train_hrnet"
python -m scripts.train_hrnet

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: train_hrnet finished with exit code $LASTEXITCODE"
    Write-Host "See log: $logPath"
    Stop-Transcript
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "train_hrnet finished successfully."
Write-Host "Now committing config + logs (без весов модели)..."

# Git-операции: фиксируем конфиг и логи, но не модели
git status
git add config\hrnet_config.yaml
git add logs\hrnet_train_full_*.log

git commit -m "HRNet: full training run logs" -ErrorAction SilentlyContinue
git push origin main

Write-Host ""
Write-Host "=== HRNet FULL TRAINING DONE ==="
Write-Host "Log file: $logPath"

Stop-Transcript
