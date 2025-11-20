# 11_HRNET_CONFIG_CHECK.ps1 и scripts\test_hrnet_config_consistency.py
# Создание диагностического шага для проверки num_keypoints и max_epochs

param()

$ErrorActionPreference = "Stop"

$root = "D:\GM\tools\2_Landmarking-HRNet_v1.0"
Set-Location $root

if (-not (Test-Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs" | Out-Null
}

# 1) Пишем Python-скрипт в scripts\test_hrnet_config_consistency.py
if (-not (Test-Path ".\scripts")) {
    New-Item -ItemType Directory -Path ".\scripts" | Out-Null
}

$pyPath = Join-Path $root "scripts\test_hrnet_config_consistency.py"
$pyCode = @"
\"\"\"Проверка согласованности LM_number.txt, hrnet_config.yaml и настроек train_hrnet.\"\"\"
from __future__ import annotations

import sys
from typing import Any, Dict

from scripts import hrnet_config_utils


def main() -> int:
    print("=== HRNet config consistency check ===")

    # 1. Читаем число ландмарок из LM_number.txt
    try:
        n_txt = hrnet_config_utils.read_num_keypoints()
        print(f"LM_number.txt num_keypoints: {n_txt}")
    except Exception as e:
        print("[ERR] Не удалось прочитать LM_number.txt:", e)
        return 1

    # 2. Загружаем конфиг
    try:
        cfg: Dict[str, Any] = hrnet_config_utils.load_hrnet_config()
    except Exception as e:
        print("[ERR] Не удалось загрузить hrnet_config.yaml:", e)
        return 1

    model_cfg = cfg.get("model", {}) or {}
    train_cfg = cfg.get("train", {}) or {}

    n_cfg = model_cfg.get("num_keypoints", None)
    max_epochs = train_cfg.get("max_epochs", None)

    print(f"config model.num_keypoints: {n_cfg}")
    print(f"train.max_epochs: {max_epochs}")

    # Простые предупреждения
    if n_cfg != n_txt:
        print("[WARN] LM_number.txt и config.model.num_keypoints не совпадают.")

    if max_epochs is None:
        print("[WARN] В конфиге не задан train.max_epochs (используется дефолт в коде).")

    print("=== DONE config consistency check ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"@

$pyCode | Set-Content -Path $pyPath -Encoding UTF8

# 2) Пишем сам 11_HRNET_CONFIG_CHECK.ps1 (если его ещё нет или хотим обновить)
$psSelfPath = Join-Path $root "11_HRNET_CONFIG_CHECK.ps1"
$psSelfCode = @"
param()

\$ErrorActionPreference = "Stop"

\$root = "D:\GM\tools\2_Landmarking-HRNet_v1.0"
Set-Location \$root

if (-not (Test-Path ".\logs")) {
    New-Item -ItemType Directory -Path ".\logs" | Out-Null
}

\$ts = Get-Date -Format "yyyyMMdd_HHmmss"
\$logPath = ".\logs\hrnet_config_check_\$ts.log"

Start-Transcript -Path \$logPath -Force

Write-Host "=== HRNet CONFIG CHECK ==="
Write-Host "Project root: \$root"
Write-Host "Log file: \$logPath"
Write-Host ""

# Активация окружения
\$venv = ".\.venv_lm\Scripts\Activate.ps1"
if (Test-Path \$venv) {
    Write-Host "[INFO] Activating .venv_lm ..."
    . \$venv
} else {
    Write-Host "[WARN] .venv_lm not found, используется системный python."
}

Write-Host "[INFO] Using python:" (Get-Command python).Source
Write-Host ""

Write-Host "[STEP] Running python -m scripts.test_hrnet_config_consistency ..."
python -m scripts.test_hrnet_config_consistency
\$exit = \$LASTEXITCODE

if (\$exit -ne 0) {
    Write-Host ""
    Write-Host "ERROR: test_hrnet_config_consistency exited with code \$exit"
    Write-Host "See log: \$logPath"
    Stop-Transcript
    exit \$exit
}

Write-Host ""
Write-Host "[STEP] Git commit & push (новый тест и этот скрипт)..."
git status
git add scripts\test_hrnet_config_consistency.py 11_HRNET_CONFIG_CHECK.ps1
git commit -m "HRNet: add config consistency check" -ErrorAction SilentlyContinue | Out-Null
git push origin main

Write-Host ""
Write-Host "=== HRNet CONFIG CHECK DONE ==="
Write-Host "Log file: \$logPath"

Stop-Transcript
"@

$psSelfCode | Set-Content -Path $psSelfPath -Encoding UTF8

Write-Host "Созданы/обновлены:"
Write-Host "  $pyPath"
Write-Host "  $psSelfPath"

Write-Host ""
Write-Host "Запускаем 11_HRNET_CONFIG_CHECK.ps1 ..."
& $psSelfPath
