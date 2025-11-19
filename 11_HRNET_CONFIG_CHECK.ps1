$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

if (-not (Test-Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs" | Out-Null
}

$logPath = ".\logs\hrnet_config_check_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
Start-Transcript -Path $logPath -Force

$venv = ".\.venv_lm\Scripts\Activate.ps1"
if (Test-Path $venv) { . $venv }

python -m scripts.test_hrnet_config_consistency

if ($LASTEXITCODE -eq 0) {
    git status
    git add scripts\train_hrnet.py scripts\hrnet_config_utils.py scripts\test_hrnet_config_consistency.py 11_HRNET_CONFIG_CHECK.ps1 config\hrnet_config.yaml
    git commit -m "HRNet: sync num_keypoints and max_epochs with config" -ErrorAction SilentlyContinue
    git push origin main
}

Stop-Transcript
