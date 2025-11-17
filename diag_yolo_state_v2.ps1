param()

$ErrorActionPreference = "Stop"

$root   = "D:\GM\tools\2_Landmarking-Yolo_v1.0"
Set-Location $root

$logDir = Join-Path $root "logs"
if (!(Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$diagLog = Join-Path $logDir ("diag_yolo_state_v2_{0}.txt" -f $stamp)

Start-Transcript -Path $diagLog -Force

Write-Host "=== GM Landmarking-YOLO: diagnostic state v2 ==="
Write-Host "LM_ROOT = $root"
Write-Host ""

# 1) Git status
Write-Host "=== git status ==="
try {
    git status
} catch {
    Write-Host "[WARN] git status failed: $($_.Exception.Message)"
}
Write-Host ""

# 2) Top-level files
Write-Host "=== Top-level files ==="
Get-ChildItem -File | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
Write-Host ""

# 3) scripts/
Write-Host "=== scripts/ ==="
if (Test-Path "scripts") {
    Get-ChildItem "scripts" -File | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
} else {
    Write-Host "[WARN] scripts folder not found"
}
Write-Host ""

# 4) Check train_yolo.py
Write-Host "=== scripts\\train_yolo.py (first lines) ==="
if (Test-Path "scripts\train_yolo.py") {
    Get-Content "scripts\train_yolo.py" -First 80
} else {
    Write-Host "scripts\train_yolo.py not found"
}
Write-Host ""

# 5) Check trainer_menu.py
Write-Host "=== scripts\\trainer_menu.py (first lines) ==="
if (Test-Path "scripts\trainer_menu.py") {
    Get-Content "scripts\trainer_menu.py" -First 120
} else {
    Write-Host "scripts\trainer_menu.py not found"
}
Write-Host ""

# 6) 2_TRAIN-INFER_YOLO.bat
Write-Host "=== 2_TRAIN-INFER_YOLO.bat ==="
if (Test-Path "2_TRAIN-INFER_YOLO.bat") {
    Get-Content "2_TRAIN-INFER_YOLO.bat"
} else {
    Write-Host "2_TRAIN-INFER_YOLO.bat not found"
}
Write-Host ""

Stop-Transcript
Write-Host ""
Write-Host "Diagnostic log saved to: $diagLog"
