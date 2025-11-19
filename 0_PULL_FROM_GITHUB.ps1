# 0_PULL_FROM_GITHUB.ps1
# Стянуть последние изменения Codex из GitHub для 2_Landmarking-HRNet_v1.0

param()

# Переходим в папку проекта (где лежит этот скрипт)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Папка для логов
if (-not (Test-Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs" | Out-Null
}

# Старт логирования
$logPath = ".\logs\pull_codex_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss")
Start-Transcript -Path $logPath -Force

Write-Host "=== PULL FROM GITHUB: 2_Landmarking-HRNet_v1.0 ==="
Write-Host "Project path: $scriptDir"
Write-Host "Log file: $logPath"
Write-Host ""

# Показываем текущий статус
Write-Host ">>> git status"
git status

Write-Host ""
Write-Host ">>> git pull origin main"
git pull origin main

Write-Host ""
Write-Host ">>> git log -5 --oneline"
git log -5 --oneline

Write-Host ""
Write-Host "Содержимое scripts после pull (если папка есть):"
if (Test-Path ".\scripts") {
    Get-ChildItem .\scripts
} else {
    Write-Host "Папка .\scripts пока отсутствует."
}

Write-Host ""
Write-Host "=== PULL DONE. Check log: $logPath ==="

Stop-Transcript
