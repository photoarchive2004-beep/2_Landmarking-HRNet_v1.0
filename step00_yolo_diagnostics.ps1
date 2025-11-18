param(
    [string]$LM_ROOT = "D:\GM\tools\2_Landmarking-Yolo_v1.0"
)

Write-Host "=== GM Landmarking YOLO diagnostic script (Step 1) ==="
Write-Host "LM_ROOT = $LM_ROOT"
Write-Host ""

# Проверка существования корня проекта
if (-not (Test-Path -Path $LM_ROOT -PathType Container)) {
    Write-Host "ERROR: LM_ROOT path '$LM_ROOT' does not exist. Please check the path and try again." -ForegroundColor Red
    exit 1
}

# Подготовка папки logs и имён файлов
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logsDir   = Join-Path $LM_ROOT "logs"

if (-not (Test-Path -Path $logsDir -PathType Container)) {
    Write-Host "Creating logs directory: $logsDir"
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

$treeFile = Join-Path $logsDir ("tree_{0}.txt" -f $timestamp)
$gitFile  = Join-Path $logsDir ("git_status_{0}.txt" -f $timestamp)

Write-Host "Tree log will be saved to: $treeFile"
Write-Host "Git status log will be saved to: $gitFile"
Write-Host ""

# Переходим в корень проекта
Push-Location $LM_ROOT
try {
    # ---------------------------
    # 1. Снимок дерева файлов
    # ---------------------------
    "=== Directory tree for '$LM_ROOT' ===" | Out-File -FilePath $treeFile -Encoding UTF8
    "" | Out-File -FilePath $treeFile -Append -Encoding UTF8
    try {
        tree /F /A | Out-File -FilePath $treeFile -Append -Encoding UTF8
    } catch {
        "ERROR: Failed to execute 'tree /F /A'. Error details:" | Out-File -FilePath $treeFile -Append -Encoding UTF8
        $_ | Out-File -FilePath $treeFile -Append -Encoding UTF8
    }

    # ---------------------------
    # 2. Git-состояние
    # ---------------------------
    "=== Git diagnostics for '$LM_ROOT' ===" | Out-File -FilePath $gitFile -Encoding UTF8
    "" | Out-File -FilePath $gitFile -Append -Encoding UTF8

    ">>> git status" | Out-File -FilePath $gitFile -Append -Encoding UTF8
    "-----------------------------" | Out-File -FilePath $gitFile -Append -Encoding UTF8
    git status 2>&1 | Out-File -FilePath $gitFile -Append -Encoding UTF8
    "" | Out-File -FilePath $gitFile -Append -Encoding UTF8

    ">>> git remote -v" | Out-File -FilePath $gitFile -Append -Encoding UTF8
    "-----------------------------" | Out-File -FilePath $gitFile -Append -Encoding UTF8
    git remote -v 2>&1 | Out-File -FilePath $gitFile -Append -Encoding UTF8
    "" | Out-File -FilePath $gitFile -Append -Encoding UTF8

    ">>> git branch" | Out-File -FilePath $gitFile -Append -Encoding UTF8
    "-----------------------------" | Out-File -FilePath $gitFile -Append -Encoding UTF8
    git branch 2>&1 | Out-File -FilePath $gitFile -Append -Encoding UTF8
    "" | Out-File -FilePath $gitFile -Append -Encoding UTF8

    ">>> git log -n 5" | Out-File -FilePath $gitFile -Append -Encoding UTF8
    "-----------------------------" | Out-File -FilePath $gitFile -Append -Encoding UTF8
    git log -n 5 2>&1 | Out-File -FilePath $gitFile -Append -Encoding UTF8
    "" | Out-File -FilePath $gitFile -Append -Encoding UTF8

} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Diagnostics completed successfully." -ForegroundColor Green
Write-Host "Tree file: $treeFile"
Write-Host "Git status file: $gitFile"
Write-Host ""
Write-Host "You can now attach these files in chat for further analysis."
exit 0
