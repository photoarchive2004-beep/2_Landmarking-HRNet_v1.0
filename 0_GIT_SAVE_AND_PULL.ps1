# 0_GIT_SAVE_AND_PULL.ps1
# Сохраняем все локальные изменения в отдельный коммит и тянем origin/main

param()

$ErrorActionPreference = "Stop"

$root = "D:\GM\tools\2_Landmarking-HRNet_v1.0"
Set-Location $root

if (-not (Test-Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs" | Out-Null
}
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = ".\logs\git_save_and_pull_$ts.log"

Start-Transcript -Path $logPath -Force

Write-Host "=== GIT SAVE AND PULL (HRNet) ==="
Write-Host "Project root: $root"
Write-Host "Log file: $logPath"
Write-Host ""

Write-Host ">>> git status"
git status

Write-Host ""
Write-Host ">>> git add -A"
git add -A

Write-Host ""
Write-Host ">>> git commit"
git commit -m "local HRNet work before Codex sync" -ErrorAction SilentlyContinue

Write-Host ""
Write-Host ">>> git pull origin main"
git pull origin main

Write-Host ""
Write-Host "=== DONE. See log:" $logPath

Stop-Transcript
