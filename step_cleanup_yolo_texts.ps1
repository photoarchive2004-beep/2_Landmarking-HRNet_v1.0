param()

# === Общие настройки ===
$LM_ROOT = "D:\GM\tools\2_Landmarking-HRNet_v1.0"
Set-Location $LM_ROOT

$logDir = Join-Path $LM_ROOT "logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}
$ts      = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path $logDir "cleanup_yolo_texts_$ts.log"

Start-Transcript -Path $logPath -Force
Write-Host "=== HRNet project: cleanup YOLO labels ==="
Write-Host "LM_ROOT = $LM_ROOT"
Write-Host ""

# === Проверка git ===
$doGit = $false
if (-not (Test-Path ".git")) {
    Write-Host "WARNING: .git folder not found – git sync will be skipped."
} else {
    $branch = git rev-parse --abbrev-ref HEAD 2>$null
    if ($LASTEXITCODE -eq 0 -and $branch -eq "main") {
        $remote = git remote get-url origin 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Git: branch=$branch, origin=$remote"
            $doGit = $true
        } else {
            Write-Host "WARNING: cannot read git remote, git push will be skipped."
        }
    } else {
        Write-Host "WARNING: current git branch is '$branch' (expected 'main'), git push will be skipped."
    }
}

# === Таблица замен только для нужных надписей про YOLO ===
$replacements = @(
    @{ old = "=== GM Landmarking: YOLO Trainer (v1.0) ==="; new = "=== GM Landmarking: HRNet Trainer (v1.0) ===" },
    @{ old = "GM Landmarking: YOLO Trainer (v1.0)";       new = "GM Landmarking: HRNet Trainer (v1.0)" },
    @{ old = "Train / finetune YOLO model on MANUAL localities"; new = "Train / finetune HRNet model on MANUAL localities" },
    @{ old = "Autolabel locality with current YOLO model";       new = "Autolabel locality with current HRNet model" },
    @{ old = "Show current YOLO model info";                      new = "Show current HRNet model info" },
    @{ old = "Show / edit YOLO config";                           new = "Show / edit HRNet config" }
)

function Update-FileText {
    param(
        [string]$Path
    )

    Write-Host "Processing $Path"

    try {
        $text = Get-Content -LiteralPath $Path -Raw -ErrorAction Stop
    }
    catch {
        Write-Host "  WARNING: cannot read file, skipping. $_"
        return
    }

    if ($null -eq $text -or $text.Length -eq 0) {
        Write-Host "  INFO: empty file, skipping."
        return
    }

    $original = $text
    foreach ($pair in $replacements) {
        $old = [string]$pair.old
        $new = [string]$pair.new
        if ($text.Contains($old)) {
            $text = $text.Replace($old, $new)
        }
    }

    if ($text -ne $original) {
        try {
            Set-Content -LiteralPath $Path -Value $text -Encoding UTF8
            Write-Host "  Updated."
        }
        catch {
            Write-Host "  ERROR: failed to write file. $_"
        }
    }
    else {
        Write-Host "  No changes."
    }
}

# === Обрабатываем только текстовые файлы проекта, исключая виртуальные окружения ===
$patterns = @("*.bat","*.ps1","*.py","*.txt","*.yaml","*.yml")
$files = Get-ChildItem -Recurse -File -Include $patterns |
    Where-Object {
        $_.FullName -notmatch "\\\.venv" -and
        $_.FullName -notmatch "\\\.venv_lm"
    }

foreach ($f in $files) {
    Update-FileText -Path $f.FullName
}

# === Создаём / обновляем 2_TRAIN-INFER_HRNet.bat ===
$yoloBat  = Join-Path $LM_ROOT "2_TRAIN-INFER_YOLO.bat"
$hrnetBat = Join-Path $LM_ROOT "2_TRAIN-INFER_HRNet.bat"

if (Test-Path $yoloBat) {
    if (-not (Test-Path $hrnetBat)) {
        Copy-Item $yoloBat $hrnetBat
        Write-Host "Created 2_TRAIN-INFER_HRNet.bat from YOLO version."
    }
}

if (Test-Path $hrnetBat) {
    Update-FileText -Path $hrnetBat
} else {
    Write-Host "WARNING: 2_TRAIN-INFER_HRNet.bat not found."
}

# === Git add / commit / push ===
if ($doGit) {
    Write-Host ""
    Write-Host "=== Git status before commit ==="
    git status

    git add -A
    git commit -m "Rename trainer texts from YOLO to HRNet"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "git commit finished with code $LASTEXITCODE (maybe nothing to commit)."
    }

    git push -u origin main
    if ($LASTEXITCODE -ne 0) {
        Write-Host "git push failed with code $LASTEXITCODE. Repository on GitHub is NOT up to date."
    } else {
        Write-Host "git push OK, repository should be in sync with GitHub."
    }
} else {
    Write-Host "Git push skipped due to earlier warnings."
}

Stop-Transcript
Write-Host "Log saved to $logPath"
