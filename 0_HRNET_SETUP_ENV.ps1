# 0_HRNET_SETUP_ENV.ps1
# Настройка окружения для HRNet: установка mmpose/timm/opencv в .venv_lm
# и проверка, что MMPose импортируется.

param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Перейти в корень проекта
Set-Location "D:\GM\tools\2_Landmarking-HRNet_v1.0"

# Логи
if (-not (Test-Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs" | Out-Null
}
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = ".\logs\hrnet_setup_env_$ts.log"

Start-Transcript -Path $logPath -Force
Write-Host "=== HRNet: setup .venv_lm environment (mmpose/timm/opencv) ==="
Write-Host "Project root: $(Get-Location)"
Write-Host "Log file: $logPath"
Write-Host ""

# Создать venv, если его ещё нет
if (-not (Test-Path ".\.venv_lm")) {
    Write-Host "[INFO] .venv_lm not found, creating virtualenv via 'py -3 -m venv .venv_lm' ..."
    py -3 -m venv .venv_lm
}

$activatePath = ".\.venv_lm\Scripts\Activate.ps1"
if (-not (Test-Path $activatePath)) {
    Write-Error "[FATAL] Cannot find Activate script: $activatePath"
    Stop-Transcript
    exit 1
}

Write-Host "[INFO] Activating virtualenv .venv_lm ..."
. $activatePath

Write-Host "[INFO] Python in venv:" (Get-Command python).Source
Write-Host ""

# Обновить pip и поставить зависимости
Write-Host "[STEP] Upgrading pip ..."
python -m pip install --upgrade pip

Write-Host "[STEP] Installing HRNet dependencies (mmpose, timm, opencv-python) ..."
python -m pip install mmpose timm opencv-python

Write-Host ""
Write-Host "[STEP] Quick import test inside .venv_lm ..."
python - << 'PYCODE'
import importlib

mods = ["mmpose", "timm", "cv2"]
ok = True
for name in mods:
    try:
        importlib.import_module(name)
        print(f"[OK] Imported {name}")
    except Exception as e:
        ok = False
        print(f"[FAIL] Cannot import {name}: {e!r}")

if not ok:
    raise SystemExit(1)

print("[INFO] Basic imports are OK.")
PYCODE

Write-Host ""
Write-Host "=== HRNet env setup DONE ==="

# git: зафиксировать сам скрипт (если ещё не в репо)
git status
git add 0_HRNET_SETUP_ENV.ps1 | Out-Null
git commit -m "HRNet: add env setup script for mmpose/timm" -ErrorAction SilentlyContinue | Out-Null
git push origin main

Stop-Transcript
