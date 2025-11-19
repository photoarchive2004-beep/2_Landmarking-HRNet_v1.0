# === HRNet final self-test for 2_Landmarking-HRNet_v1.0 ===

$LM_ROOT = 'D:\GM\tools\2_Landmarking-HRNet_v1.0'

if (-not (Test-Path $LM_ROOT)) {
    Write-Host "ERROR: LM_ROOT not found: $LM_ROOT"
    exit 1
}

Set-Location $LM_ROOT

# Ensure logs dir exists
$logsDir = Join-Path $LM_ROOT 'logs'
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$logPath   = Join-Path $logsDir ("hrnet_selftest_{0}.log" -f $timestamp)

Start-Transcript -Path $logPath -Force
Write-Host "=== HRNet self-test ==="
Write-Host "LM_ROOT = $LM_ROOT"
Write-Host ""

# --- Detect Python ---
Write-Host "Step 0: detecting Python ..."

$pyCandidates = @(
    (Join-Path $LM_ROOT ".venv_lm\Scripts\python.exe"),
    (Join-Path $LM_ROOT ".venv\Scripts\python.exe"),
    "python"
)

$PY = $null
foreach ($cand in $pyCandidates) {
    if (Get-Command $cand -ErrorAction SilentlyContinue) {
        $PY = $cand
        break
    }
}

if (-not $PY) {
    Write-Host "ERROR: Python not found (.venv_lm, .venv, or system)."
    Stop-Transcript
    exit 1
}

Write-Host "Using Python: $PY"
Write-Host ""

function Run-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )

    Write-Host "==== $Name ===="
    try {
        & $Action
        $code = $LASTEXITCODE
        if ($code -eq $null) { $code = 0 }
        Write-Host "Exit code: $code"
    }
    catch {
        Write-Host "EXCEPTION in step '$Name': $_"
    }
    Write-Host ""
}

# Step 1: py_compile scripts
Run-Step "Step 1: python -m py_compile scripts\*.py" {
    if (Test-Path ".\scripts") {
        $pyFiles = Get-ChildItem -Path ".\scripts" -Filter "*.py" -File
        if ($pyFiles.Count -gt 0) {
            & $PY -m py_compile $pyFiles.FullName
        }
        else {
            Write-Host "WARNING: no .py files in .\scripts, skipping."
        }
    }
    else {
        Write-Host "WARNING: scripts folder not found, skipping."
    }
}

# Step 2: init_structure.py
Run-Step "Step 2: python scripts\init_structure.py" {
    $scriptPath = ".\scripts\init_structure.py"
    if (Test-Path $scriptPath) {
        & $PY $scriptPath
    }
    else {
        Write-Host "WARNING: $scriptPath not found, skipping."
    }
}

# Step 3: rebuild_localities_status.py --dry-run
Run-Step "Step 3: python scripts\rebuild_localities_status.py --dry-run" {
    $scriptPath = ".\scripts\rebuild_localities_status.py"
    if (Test-Path $scriptPath) {
        & $PY $scriptPath --dry-run
    }
    else {
        Write-Host "WARNING: $scriptPath not found, skipping."
    }
}

# Step 4: trainer_menu.py --help  (вместо несуществующего --selftest)
Run-Step "Step 4: python scripts\trainer_menu.py --help" {
    $scriptPath = ".\scripts\trainer_menu.py"
    if (Test-Path $scriptPath) {
        & $PY $scriptPath --help
    }
    else {
        Write-Host "WARNING: $scriptPath not found, skipping."
    }
}

Write-Host "=== HRNet self-test finished ==="
Write-Host "Log saved to: $logPath"
Stop-Transcript
