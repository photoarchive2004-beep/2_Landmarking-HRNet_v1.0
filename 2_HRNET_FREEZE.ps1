param(
    [switch]$GitSync
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "=== HRNet freeze-low-stages diagnostic ==="
Write-Host "Project root: $ScriptDir"

$pythonCandidates = @(
    (Join-Path $ScriptDir ".venv_hrnet\Scripts\python.exe"),
    (Join-Path $ScriptDir ".venv_lm\Scripts\python.exe"),
    (Join-Path $ScriptDir ".venv\Scripts\python.exe"),
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

Write-Host "Running python -m scripts.test_hrnet_freeze ..."
& $PythonExe -m scripts.test_hrnet_freeze
if ($LASTEXITCODE -ne 0) {
    throw "scripts.test_hrnet_freeze failed with exit code $LASTEXITCODE"
}

if ($GitSync) {
    Write-Host "Running git status/add/commit/push ..."
    git status
    git add .
    git commit -m "HRNet: freeze low stages and test trainable params"
    git push origin main
}

Write-Host "=== Done ==="
