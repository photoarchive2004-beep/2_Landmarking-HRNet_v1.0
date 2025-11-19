$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

$logsDir = Join-Path $scriptDir "logs"
if (-not (Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logsDir ("hrnet_infer_debug_{0}.log" -f $timestamp)

Start-Transcript -Path $logPath | Out-Null
$exitCode = 0

try {
    $activateScript = Join-Path $scriptDir ".venv_lm\\Scripts\\Activate.ps1"
    if (Test-Path $activateScript) {
        Write-Host "Activating virtual environment via $activateScript"
        & $activateScript
    } else {
        Write-Warning "Virtual environment activator not found: $activateScript"
    }

    Write-Host "Running python -m scripts.infer_hrnet --mode debug ..."
    python -m scripts.infer_hrnet --mode debug
    if ($LASTEXITCODE -ne 0) {
        $exitCode = $LASTEXITCODE
        Write-Error "HRNet inference (debug) failed with exit code $LASTEXITCODE"
    }
}
catch {
    $exitCode = 1
    Write-Error $_
}
finally {
    Stop-Transcript | Out-Null
    exit $exitCode
}
