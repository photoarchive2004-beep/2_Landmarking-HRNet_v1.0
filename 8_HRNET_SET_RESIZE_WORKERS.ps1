# 8_HRNET_SET_RESIZE_WORKERS.ps1
# Установить resize до 1024 и num_workers=12 в config/hrnet_config.yaml

param()

$ErrorActionPreference = "Stop"

$root = "D:\GM\tools\2_Landmarking-HRNet_v1.0"
Set-Location $root

if (-not (Test-Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs" | Out-Null
}
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = ".\logs\hrnet_set_resize_workers_$ts.log"

Start-Transcript -Path $logPath -Force

Write-Host "=== HRNet: set resize.long_side=1024 and train.num_workers=12 ==="
Write-Host "Project root: $root"
Write-Host "Log file: $logPath"
Write-Host ""

$cfgPath = Join-Path $root "config\hrnet_config.yaml"
if (-not (Test-Path $cfgPath)) {
    Write-Error "Файл конфига не найден: $cfgPath"
    Stop-Transcript
    exit 1
}

Write-Host "[STEP] Updating config/hrnet_config.yaml via Python ..."

python - << 'PYCODE'
from pathlib import Path
import yaml

root = Path(r"D:\GM\tools\2_Landmarking-HRNet_v1.0")
cfg_path = root / "config" / "hrnet_config.yaml"

text = cfg_path.read_text(encoding="utf-8")
cfg = yaml.safe_load(text) or {}
if not isinstance(cfg, dict):
    raise SystemExit(f"Unexpected root type in hrnet_config.yaml: {type(cfg)!r}")

# Блок resize
resize = cfg.setdefault("resize", {})
resize["enabled"] = True
resize["long_side"] = 1024
resize.setdefault("keep_aspect_ratio", True)

# На всякий случай дублируем старый ключ совместимости
cfg["resize_long_side"] = 1024

# Блок train
train = cfg.setdefault("train", {})
train["num_workers"] = 12

cfg_path.write_text(
    yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
    encoding="utf-8",
)

print("Updated: resize.long_side=1024, resize.enabled=True, train.num_workers=12")
PYCODE

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Python обновление конфига завершилось с кодом $LASTEXITCODE"
    Write-Host "Смотри лог: $logPath"
    Stop-Transcript
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[STEP] Git commit & push (config + этот скрипт)..."
git status
git add config\hrnet_config.yaml 8_HRNET_SET_RESIZE_WORKERS.ps1
git commit -m "HRNet: set resize to 1024 and num_workers=12" -ErrorAction SilentlyContinue | Out-Null
git push origin main

Write-Host ""
Write-Host "=== DONE. Config updated ==="
Write-Host "Log file: $logPath"

Stop-Transcript
