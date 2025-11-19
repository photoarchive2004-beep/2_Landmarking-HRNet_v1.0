param(
    [string]$VenvName = ".venv_hrnet",
    [switch]$EnableGitSync
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -Path $ScriptDir

$LogsDir = Join-Path $ScriptDir "logs"
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir | Out-Null
}
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$TranscriptPath = Join-Path $LogsDir "setup_hrnet_env_$Timestamp.log"
Start-Transcript -Path $TranscriptPath -Append | Out-Null

try {
    Write-Host "=== HRNet environment setup ==="

    $VenvPath = Join-Path $ScriptDir $VenvName
    $PythonExe = Join-Path $VenvPath "Scripts\python.exe"
    $ActivateScript = Join-Path $VenvPath "Scripts\Activate.ps1"

    if (-not (Test-Path $VenvPath)) {
        Write-Host "[INFO] Creating virtual environment at $VenvPath"
        $BasePython = $null
        foreach ($Candidate in @("python", "py")) {
            if (Get-Command $Candidate -ErrorAction SilentlyContinue) {
                $BasePython = $Candidate
                break
            }
        }
        if (-not $BasePython) {
            throw "Python interpreter not found in PATH."
        }
        & $BasePython -m venv $VenvPath
    } else {
        Write-Host "[INFO] Reusing existing virtual environment at $VenvPath"
    }

    if (-not (Test-Path $PythonExe)) {
        throw "python.exe not found inside $VenvPath"
    }

    if (Test-Path $ActivateScript) {
        Write-Host "[INFO] Activating virtual environment ..."
        & $ActivateScript
    } else {
        Write-Host "[WARN] Activate.ps1 not found. Continuing without session activation."
    }

    Write-Host "[INFO] Upgrading pip/setuptools/wheel ..."
    & $PythonExe -m pip install --upgrade pip setuptools wheel

    Write-Host "[INFO] Installing core scientific stack ..."
    & $PythonExe -m pip install numpy pillow opencv-python pandas scipy matplotlib pyyaml tqdm timm

    Write-Host "[INFO] Installing PyTorch (CUDA 11.8 wheels) ..."
    & $PythonExe -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

    Write-Host "[INFO] Installing OpenMMLab packages (mmengine/mmcv/mmpose) ..."
    & $PythonExe -m pip install mmengine mmcv mmpose

    Write-Host "[INFO] Running scripts\\test_hrnet_import.py ..."
    & $PythonExe .\scripts\test_hrnet_import.py
    if ($LASTEXITCODE -ne 0) {
        throw "test_hrnet_import.py exited with code $LASTEXITCODE"
    }

    if ($EnableGitSync) {
        Write-Host "[INFO] Synchronizing changes with git ..."
        git add 0_HRNET_SETUP_ENV.ps1 scripts/test_hrnet_import.py
        git commit -m "Add HRNet env setup automation" --allow-empty
        git push origin main
    } else {
        Write-Host "[INFO] Git sync skipped (pass -EnableGitSync to enable)."
    }
}
finally {
    Stop-Transcript | Out-Null
    Write-Host "[INFO] Transcript saved to $TranscriptPath"
}
