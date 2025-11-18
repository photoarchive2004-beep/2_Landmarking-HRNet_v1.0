@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Determine module root (LM_ROOT) = folder of this BAT
set "LM_ROOT=%~dp0"
cd /d "%LM_ROOT%"

echo ==============================================
echo   GM Landmarking: HRNet Trainer (v1.0)
echo ==============================================
echo.

REM Ensure cfg directory exists
if not exist "cfg" mkdir "cfg"

set "BASE_FILE=cfg\last_base.txt"

REM Ask user for base localities folder via PowerShell dialog
powershell -ExecutionPolicy Bypass -NoProfile ^
  -Command "Add-Type -AssemblyName System.Windows.Forms; $initial = ''; if (Test-Path 'cfg\\last_base.txt') { $initial = Get-Content 'cfg\\last_base.txt' -ErrorAction SilentlyContinue }; $dlg = New-Object System.Windows.Forms.FolderBrowserDialog; if ($initial) { $dlg.SelectedPath = $initial }; $dlg.Description = 'Select base localities folder'; if ($dlg.ShowDialog() -eq 'OK') { $dlg.SelectedPath | Out-File -Encoding ascii 'cfg\\last_base.txt' }"

if not exist "%BASE_FILE%" (
    echo Base localities not selected. Exiting.
    goto :EOF
)

set /p BASE_LOCALITIES=<"%BASE_FILE%"
if "%BASE_LOCALITIES%"=="" (
    echo Base localities not selected. Exiting.
    goto :EOF
)

echo Using base localities: "%BASE_LOCALITIES%"
echo.

REM Optional: run environment setup if available
if exist "0_INSTALL_ENV.ps1" (
    echo Running 0_INSTALL_ENV.ps1 ...
    powershell -ExecutionPolicy Bypass -NoProfile -File "0_INSTALL_ENV.ps1"
    echo.
)

REM Run initialization and status rebuild
echo Initializing project structure...
python scripts\init_structure.py
echo.

echo Rebuilding localities status...
python scripts\rebuild_localities_status.py
echo.

REM Launch HRNet trainer menu
echo Starting HRNet trainer menu...
python scripts\trainer_menu.py --root "%LM_ROOT%" --base-localities "%BASE_LOCALITIES%"

echo.
echo HRNet trainer finished. Press any key to exit.
pause >nul

endlocal

