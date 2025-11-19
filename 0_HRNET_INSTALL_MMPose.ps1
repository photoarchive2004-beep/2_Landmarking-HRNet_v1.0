# 0_HRNET_INSTALL_MMPose.ps1
# Установка mmpose/timm/opencv-python в .venv_lm для HRNet

param()

$ErrorActionPreference = "Stop"

# Корень проекта
$root = "D:\GM\tools\2_Landmarking-HRNet_v1.0"
Set-Location $root

# Логи
if (-not (Test-Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs" | Out-Null
}
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = ".\logs\hrnet_install_mmpose_$ts.log"

Start-Transcript -Path $logPath -Force

Write-Host "=== HRNet: install mmpose/timm/opencv into .venv_lm ==="
Write-Host "Project root: $root"
Write-Host "Log file: $logPath"
Write-Host ""

# Создать виртуальное окружение, если его ещё нет
if (-not (Test-Path ".\.venv_lm")) {
    Write-Host "[INFO] .venv_lm not found, creating via 'py -3 -m venv .venv_lm' ..."
    py -3 -m venv .venv_lm
} else {
    Write-Host "[INFO] .venv_lm already exists."
}

$activatePath = ".\.venv_lm\Scripts\Activate.ps1"
if (-not (Test-Path $activatePath)) {
    Write-Error "[FATAL] Cannot find Activate script: $activatePath"
    Stop-Transcript
    exit 1
}

Write-Host "[INFO] Activating virtualenv .venv_lm ..."
. $activatePath

Write-Host "[INFO] Using python:" (Get-Command python).Source
Write-Host ""

Write-Host "[STEP] Upgrading pip ..."
python -m pip install --upgrade pip

Write-Host ""
Write-Host "[STEP] Installing packages: mmpose, timm, opencv-python ..."
python -m pip install mmpose timm opencv-python

Write-Host ""
Write-Host "[STEP] pip show mmpose ..."
python -m pip show mmpose

Write-Host ""
Write-Host "=== HRNet env install DONE ==="

# Зафиксировать сам скрипт в git
git status
git add 0_HRNET_INSTALL_MMPose.ps1 | Out-Null
git commit -m "HRNet: add mmpose install helper script" -ErrorAction SilentlyContinue | Out-Null
git push origin main

Stop-Transcript
