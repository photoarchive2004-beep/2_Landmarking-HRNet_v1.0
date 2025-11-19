# 0_HRNET_INSTALL_MMSTACK.ps1
# Установка стека OpenMMLab (openmim + mmengine + mmcv + mmpose) в .venv_lm

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
$logPath = ".\logs\hrnet_install_mmstack_$ts.log"

Start-Transcript -Path $logPath -Force

Write-Host "=== HRNet: install OpenMMLab stack into .venv_lm ==="
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
Write-Host "[STEP] Installing openmim ..."
python -m pip install -U openmim

Write-Host ""
Write-Host "[STEP] Installing mmengine, mmcv и mmpose через mim ..."
python -m mim install "mmengine>=0.7.1" "mmcv>=2.0.0" "mmpose"

Write-Host ""
Write-Host "[STEP] Проверяем, что всё импортируется ..."
python -c "import importlib; [importlib.import_module(m) for m in ['mmengine','mmcv','mmpose']]; print('[INFO] All OpenMMLab core modules imported successfully.')"

Write-Host ""
Write-Host '=== HRNet OpenMMLab stack install DONE ==='

# Зафиксировать сам скрипт (если его ещё нет в репо)
git status
git add 0_HRNET_INSTALL_MMSTACK.ps1 | Out-Null
git commit -m 'HRNet: add OpenMMLab stack install script' -ErrorAction SilentlyContinue | Out-Null
git push origin main

Stop-Transcript
