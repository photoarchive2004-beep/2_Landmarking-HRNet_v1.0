# 1_RUN_STEP2.ps1
# Обёртка для Шага 2: прогон test_hrnet_pretrained (как модуля) + git add/commit/push

param()

# Переходим в папку проекта (куда положен этот скрипт)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Папка для логов
if (-not (Test-Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs" | Out-Null
}

# Старт логирования
$logPath = ".\logs\step2_hrnet_pretrained_{0}.log" -f (Get-Date -Format "yyyyMMdd_HH-mm-ss")
Start-Transcript -Path $logPath -Force

Write-Host "=== STEP 2: HRNet pretrained weights test ==="
Write-Host "Project path: $scriptDir"
Write-Host "Log file: $logPath"
Write-Host ""

# === Активация окружения ===
# Если есть отдельный скрипт активации venv для HRNet — можно вызвать его здесь.
# Пока оставляем комментарий, чтобы не ломать текущую схему.
Write-Host ">>> NOTE: виртуальное окружение должно быть уже активировано (или используется системный python)."
Write-Host ""

# Запускаем тест предобученных весов КАК МОДУЛЬ ПАКЕТА scripts
if (Test-Path ".\scripts\test_hrnet_pretrained.py") {
    Write-Host ">>> python -m scripts.test_hrnet_pretrained"
    python -m scripts.test_hrnet_pretrained
} else {
    Write-Host "ERROR: scripts\test_hrnet_pretrained.py не найден."
    Stop-Transcript
    exit 1
}

# Проверяем код возврата Python
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: test_hrnet_pretrained завершился с кодом $LASTEXITCODE"
    Write-Host "Смотри лог: $logPath"
    Stop-Transcript
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "test_hrnet_pretrained завершился без ошибок (код возврата 0)."
Write-Host "Теперь git add/commit/push изменений..."

# Git-операции (без reset/push --force)
git status
git add .

git commit -m "HRNet: step 2 pretrained backbone test"
if ($LASTEXITCODE -ne 0) {
    Write-Host "git commit вернул код $LASTEXITCODE (скорее всего, нет изменений для коммита)."
}

git push origin main

Write-Host ""
Write-Host "=== STEP 2 DONE. Check log: $logPath ==="

Stop-Transcript
