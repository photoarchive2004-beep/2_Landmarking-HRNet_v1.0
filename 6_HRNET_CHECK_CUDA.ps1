param()

$ErrorActionPreference = "Stop"

$root = "D:\GM\tools\2_Landmarking-HRNet_v1.0"
Set-Location $root

if (-not (Test-Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs" | Out-Null
}
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = ".\logs\hrnet_check_cuda_$ts.log"

Start-Transcript -Path $logPath -Force

Write-Host "=== HRNet: torch CUDA diagnostic ==="
Write-Host "Project root: $root"
Write-Host "Log file: $logPath"
Write-Host ""

$activatePath = ".\.venv_lm\Scripts\Activate.ps1"
if (-not (Test-Path $activatePath)) {
    Write-Error "[FATAL] Cannot find Activate script: $activatePath"
    Stop-Transcript
    exit 1
}

Write-Host "[INFO] Activating virtualenv .venv_lm ..."
. $activatePath
Write-Host "[INFO] Using python:" (Get-Command python).Source
Write-Host ""

$diagPath = ".\scripts\check_cuda_torch.py"
@"
import torch

print("torch version:", torch.__version__)
print("cuda.is_available:", torch.cuda.is_available())
print("cuda.device_count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(idx)
        print(f"[GPU {idx}] name={name}")
"@ | Set-Content -Path $diagPath -Encoding UTF8

Write-Host "[STEP] Running CUDA diagnostic ..."
python $diagPath

Write-Host ""
Write-Host "=== HRNet CUDA diagnostic DONE ==="
Write-Host "See log:" $logPath

Stop-Transcript
