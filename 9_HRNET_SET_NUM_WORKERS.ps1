# 9_HRNET_SET_NUM_WORKERS.ps1
# Установить train.num_workers = 12 в config/hrnet_config.yaml

param()

$ErrorActionPreference = "Stop"

$root = "D:\GM\tools\2_Landmarking-HRNet_v1.0"
Set-Location $root

if (-not (Test-Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs" | Out-Null
}
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = ".\logs\hrnet_set_num_workers_$ts.log"

Start-Transcript -Path $logPath -Force

Write-Host "=== HRNet: set train.num_workers = 12 ==="
Write-Host "Project root: $root"
Write-Host "Log file: $logPath"
Write-Host ""

# Активируем .venv_lm, если есть
$activatePath = ".\.venv_lm\Scripts\Activate.ps1"
if (Test-Path $activatePath) {
    Write-Host "[INFO] Activating virtualenv .venv_lm ..."
    . $activatePath
} else {
    Write-Host "[WARN] .venv_lm not found, используем системный python."
}

Write-Host "[INFO] Using python:" (Get-Command python).Source
Write-Host ""

Write-Host "[STEP] Ensuring PyYAML is installed ..."
python -m pip install pyyaml

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: pip install pyyaml failed, code $LASTEXITCODE"
    Stop-Transcript
    exit $LASTEXITCODE
}

# Пишем вспомогательный скрипт
$pyPath = Join-Path $root "scripts\_set_num_workers.py"
$pyCode = @"
from pathlib import Path
import yaml

root = Path(__file__).resolve().parents[1]
cfg_path = root / "config" / "hrnet_config.yaml"

print(f"[INFO] Config path: {cfg_path}")

text = cfg_path.read_text(encoding="utf-8")
cfg = yaml.safe_load(text) or {}
if not isinstance(cfg, dict):
    raise SystemExit(f"Unexpected root type in hrnet_config.yaml: {type(cfg)!r}")

train = cfg.setdefault("train", {})
old = train.get("num_workers")
train["num_workers"] = 12

print(f"[INFO] train.num_workers: {old!r} -> {train['num_workers']!r}")

cfg_path.write_text(
    yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)
"@

$pyCode | Set-Content -Path $pyPath -Encoding UTF8

Write-Host ""
Write-Host "[STEP] Running Python helper to update num_workers ..."
python $pyPath
$exit = $LASTEXITCODE

Remove-Item $pyPath -ErrorAction SilentlyContinue

if ($exit -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Python helper failed with code $exit"
    Write-Host "See log: $logPath"
    Stop-Transcript
    exit $exit
}

Write-Host ""
Write-Host "[STEP] Git commit & push (config + этот скрипт)..."
git status
git add config\hrnet_config.yaml 9_HRNET_SET_NUM_WORKERS.ps1
git commit -m "HRNet: set train.num_workers=12" -ErrorAction SilentlyContinue | Out-Null
git push origin main

Write-Host ""
Write-Host "=== DONE: train.num_workers = 12 ==="
Write-Host "Log file: $logPath"

Stop-Transcript
