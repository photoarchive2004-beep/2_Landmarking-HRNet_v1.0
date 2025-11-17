param()

$ErrorActionPreference = "Stop"

$root = "D:\GM\tools\2_Landmarking-Yolo_v1.0"
Set-Location $root

$logsDir = Join-Path $root "logs"
if (!(Test-Path $logsDir)) {
    New-Item -ItemType Directory -Path $logsDir | Out-Null
}
$stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logsDir "fix_yolo_trainer_imports_$stamp.log"

"=== Fix YOLO trainer imports ($stamp) ===" | Tee-Object -FilePath $logPath

try {
    $trainerPath = Join-Path $root "scripts\trainer_menu.py"
    if (!(Test-Path $trainerPath)) {
        throw "trainer_menu.py not found: $trainerPath"
    }

    # Бэкап
    $backupPath = "$trainerPath.bak_$stamp"
    Copy-Item $trainerPath $backupPath -Force
    "Backup created: $backupPath" | Tee-Object -FilePath $logPath -Append

    # Читаем построчно
    $lines = Get-Content $trainerPath -Encoding UTF8

    $hasSys = $false
    $hasSubproc = $false

    foreach ($l in $lines) {
        if ($l -match '^\s*import\s+sys\b')        { $hasSys = $true }
        if ($l -match '^\s*import\s+subprocess\b') { $hasSubproc = $true }
    }

    if ($hasSys -and $hasSubproc) {
        "sys/subprocess already imported, nothing to change." | Tee-Object -FilePath $logPath -Append
    }
    else {
        $newLines = @()
        foreach ($l in $lines) {
            $newLines += $l
            if ($l -match '^\s*import\s+os\b') {
                if (-not $hasSys) {
                    $newLines += 'import sys'
                    "Inserted: import sys" | Tee-Object -FilePath $logPath -Append
                }
                if (-not $hasSubproc) {
                    $newLines += 'import subprocess'
                    "Inserted: import subprocess" | Tee-Object -FilePath $logPath -Append
                }
            }
        }
        $newLines | Set-Content -Path $trainerPath -Encoding UTF8
        "trainer_menu.py updated." | Tee-Object -FilePath $logPath -Append
    }

    # Проверка синтаксиса
    $py = ".\.venv_lm\Scripts\python.exe"
    if (!(Test-Path $py)) {
        throw "Virtual env python not found: $py"
    }

    & $py -m py_compile "scripts\trainer_menu.py" 2>&1 | Tee-Object -FilePath $logPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "py_compile failed for trainer_menu.py, see log."
    }
    "Python syntax OK." | Tee-Object -FilePath $logPath -Append

    # git add / commit / push
    git status --short | Tee-Object -FilePath $logPath -Append

    git add "scripts/trainer_menu.py" 2>&1 | Tee-Object -FilePath $logPath -Append
    git commit -m "YOLO: fix trainer imports (sys, subprocess for action 1)" 2>&1 | Tee-Object -FilePath $logPath -Append
    git push -u origin main 2>&1 | Tee-Object -FilePath $logPath -Append
    "Git push done." | Tee-Object -FilePath $logPath -Append

    "=== Fix completed successfully ===" | Tee-Object -FilePath $logPath -Append
}
catch {
    "ERROR: $($_.Exception.Message)" | Tee-Object -FilePath $logPath -Append
    throw
}
