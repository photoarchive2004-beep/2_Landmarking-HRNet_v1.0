param(
    [switch]$GitSync
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

Write-Host "=== HRNet augmentations diagnostic ==="
Write-Host "Project root: $scriptDir"

$pythonCandidates = @(
    (Join-Path $scriptDir ".venv_lm\Scripts\python.exe"),
    (Join-Path $scriptDir ".venv_hrnet\Scripts\python.exe"),
    (Join-Path $scriptDir ".venv\Scripts\python.exe"),
    "python",
    "py"
)

$PythonExe = $null
foreach ($candidate in $pythonCandidates) {
    if (Test-Path $candidate) {
        $PythonExe = $candidate
        break
    }
    try {
        $cmd = Get-Command $candidate -ErrorAction Stop
        $PythonExe = $cmd.Source
        break
    }
    catch {
    }
}

if (-not $PythonExe) {
    throw "Python interpreter not found."
}

Write-Host "Using python: $PythonExe"

if ($PythonExe -like "*\Scripts\python.exe") {
    $activateScript = Join-Path (Split-Path $PythonExe -Parent) "Activate.ps1"
    if (Test-Path $activateScript) {
        Write-Host "Activating virtual environment via $activateScript"
        & $activateScript
    }
}

Write-Host "Running python -m scripts.test_hrnet_augment ..."
& $PythonExe -m scripts.test_hrnet_augment
if ($LASTEXITCODE -ne 0) {
    throw "scripts.test_hrnet_augment failed with exit code $LASTEXITCODE"
}

if ($GitSync) {
    Write-Host "Running git status/add/commit/push ..."
    git status
    git add .
    git commit -m "HRNet: add/train augmentations diagnostics"
    git push origin main
}

Write-Host "=== Done ==="
