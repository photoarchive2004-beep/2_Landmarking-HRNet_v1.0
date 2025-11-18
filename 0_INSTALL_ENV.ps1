param(
    [string]$VenvName = ".venv_lm"
)

# 0_INSTALL_ENV.ps1 — установка окружения для HRNet-модуля авторазметки
# Проект: 2_Landmarking-HRNet_v1.0
# Логика:
#   - Если виртуальное окружение уже существует — просто выходим.
#   - Иначе создаём .venv_lm и ставим базовые пакеты для HRNet (БЕЗ YOLO).

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvPath   = Join-Path $ScriptDir $VenvName
$VenvPython = Join-Path $VenvPath "Scripts\\python.exe"

Write-Host "=== HRNet env setup in: $VenvPath ==="

if (Test-Path $VenvPath) {
    Write-Host "[INFO] Virtual environment already exists. Nothing to do."
    return
}

# ---- Поиск базового Python ----
$pyCandidates = @("python", "py")
$PY = $null
foreach ($cand in $pyCandidates) {
    if (Get-Command $cand -ErrorAction SilentlyContinue) {
        $PY = $cand
        break
    }
}

if (-not $PY) {
    Write-Host "[ERROR] Could not find Python in PATH."
    exit 1
}

Write-Host "[INFO] Using base Python: $PY"
Write-Host "[INFO] Creating virtual environment $VenvName ..."

& $PY -m venv $VenvPath
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to create virtual environment."
    exit $LASTEXITCODE
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] python.exe not found inside venv."
    exit 1
}

Write-Host "[INFO] Upgrading pip/setuptools/wheel ..."
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to upgrade pip/setuptools/wheel."
    exit $LASTEXITCODE
}

# ВАЖНО: ставим только базовые пакеты для HRNet/torch.
# НИКАКОГО ultralytics, YOLO и т.п.
Write-Host "[INFO] Installing HRNet dependencies (no YOLO) ..."
& $VenvPython -m pip install `
    numpy `
    pillow `
    opencv-python `
    pyyaml `
    pandas `
    matplotlib `
    scipy `
    torch `
    torchvision `
    torchaudio

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install HRNet dependencies."
    exit $LASTEXITCODE
}

Write-Host "[INFO] HRNet environment is ready."
