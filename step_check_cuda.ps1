# ============================================
# step_check_cuda.ps1
# Проверка и установка PyTorch с поддержкой CUDA в .venv_lm
# Проект: D:\GM\tools\2_Landmarking-Yolo_v1.0
# ============================================

$ErrorActionPreference = "Stop"

# 1) Переходим в LM_ROOT
Set-Location "D:\GM\tools\2_Landmarking-Yolo_v1.0"

# 2) Готовим лог PowerShell-транскрипта
$logsDir = Join-Path (Get-Location) "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$transcriptPath = Join-Path $logsDir ("step_check_cuda_" + $timestamp + ".log")

Start-Transcript -Path $transcriptPath -Force

Write-Host "=== Checking CUDA for YOLO training ==="
Write-Host "LM_ROOT = $(Get-Location)"

# 3) Находим Python из .venv_lm
$venvPython = Join-Path (Get-Location) ".venv_lm\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[ERR] .venv_lm\Scripts\python.exe not found. Cannot proceed." -ForegroundColor Red
    Write-Host "Сначала нужно установить окружение через 0_INSTALL_ENV.ps1."
    Stop-Transcript
    exit 1
}

Write-Host "Using Python from venv: $venvPython"
Write-Host ""

# 4) Текущая проверка torch + CUDA
Write-Host "=== Current torch / CUDA state in .venv_lm ==="

& $venvPython - <<EOF
import sys

def main():
    print("Python:", sys.version.replace("\\n"," "))
    try:
        import torch
    except ModuleNotFoundError:
        print("torch: NOT INSTALLED")
        return
    except Exception as exc:
        print("torch: IMPORT_ERROR:", exc)
        return

    print("torch.__version__:", torch.__version__)
    cuda_avail = torch.cuda.is_available()
    print("torch.cuda.is_available():", cuda_avail)
    try:
        print("torch.version.cuda:", torch.version.cuda)
    except Exception as exc:
        print("torch.version.cuda: ERROR:", exc)
    if cuda_avail:
        try:
            name = torch.cuda.get_device_name(0)
        except Exception as exc:
            name = f"<error: {exc}>"
        print("CUDA device 0:", name)

if __name__ == "__main__":
    main()
EOF

Write-Host ""
Write-Host "=== Installing / upgrading PyTorch with CUDA (cu121) ==="
Write-Host "Если уже стоит правильная версия, pip просто переустановит её поверх."

# 5) Обновляем pip и ставим torch с поддержкой CUDA
& $venvPython -m pip install --upgrade pip

# Официальная команда установки PyTorch+CUDA для Windows (cu121).
# См. инструкции PyTorch: она ставит сборку с поддержкой GPU. 
& $venvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host ""
Write-Host "=== Re-check torch / CUDA state after install ==="

& $venvPython - <<EOF
import sys

def main():
    print("Python:", sys.version.replace("\\n"," "))
    try:
        import torch
    except ModuleNotFoundError:
        print("torch: NOT INSTALLED AFTER INSTALL!")
        return
    except Exception as exc:
        print("torch: IMPORT_ERROR AFTER INSTALL:", exc)
        return

    print("torch.__version__:", torch.__version__)
    cuda_avail = torch.cuda.is_available()
    print("torch.cuda.is_available():", cuda_avail)
    try:
        print("torch.version.cuda:", torch.version.cuda)
    except Exception as exc:
        print("torch.version.cuda: ERROR:", exc)
    if cuda_avail:
        try:
            name = torch.cuda.get_device_name(0)
        except Exception as exc:
            name = f"<error: {exc}>"
        print("CUDA device 0:", name)

if __name__ == "__main__":
    main()
EOF

# 6) Проверяем git-состояние (ожидаем, что изменений в репо нет)
Write-Host ""
Write-Host "=== Git status (no repo changes expected) ==="
$gitStatus = git status --porcelain
if ([string]::IsNullOrWhiteSpace($gitStatus)) {
    Write-Host "Working tree clean. No changes to commit."
} else {
    Write-Host "There are some local changes in the repo:"
    Write-Host $gitStatus
    Write-Host "Они не связаны с установкой PyTorch (pip не трогает файлы репозитория)."
}

Write-Host ""
Write-Host "=== Done. See transcript for torch / CUDA info: $transcriptPath ==="

Stop-Transcript
