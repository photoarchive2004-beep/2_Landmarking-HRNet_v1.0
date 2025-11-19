# 7_HRNET_INSTALL_TORCH_CUDA.ps1
# Установка CUDA-версии torch/torchvision/torchaudio в .venv_lm

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
$logPath = ".\logs\hrnet_install_torch_cuda_$ts.log"

Start-Transcript -Path $logPath -Force

Write-Host "=== HRNet: install CUDA-enabled PyTorch into .venv_lm ==="
Write-Host "Project root: $root"
Write-Host "Log file: $logPath"
Write-Host ""

# Активируем виртуальное окружение
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

Write-Host "[STEP] Uninstalling existing CPU torch/torchvision/torchaudio (if any) ..."
python -m pip uninstall -y torch torchvision torchaudio

Write-Host ""
Write-Host "[STEP] Installing CUDA-enabled torch/torchvision/torchaudio from PyTorch index ..."
Write-Host "      (может качаться несколько ГБ, это нормально)"
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host ""
Write-Host "[STEP] Checking torch CUDA availability ..."

# Пишем маленький .py-файл и запускаем его
$checkScript = @"
import torch

print("torch version:", torch.__version__)
print("cuda.is_available:", torch.cuda.is_available())
print("cuda.device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"[GPU {i}] name={torch.cuda.get_device_name(i)}")
"@

$checkPath = ".\scripts\_check_torch_cuda_after_install.py"
$checkScript | Set-Content -Path $checkPath -Encoding UTF8

python $checkPath

Remove-Item $checkPath -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "=== HRNet: CUDA PyTorch install DONE ==="
Write-Host "See log:" $logPath

# Зафиксируем сам скрипт в git (чтобы не потерять)
git status
git add 7_HRNET_INSTALL_TORCH_CUDA.ps1 | Out-Null
git commit -m "HRNet: add CUDA torch install script" -ErrorAction SilentlyContinue | Out-Null
git push origin main

Stop-Transcript
