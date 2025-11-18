@echo off
setlocal enabledelayedexpansion

REM Root of the landmarking module
set "LM_ROOT=%~dp0"
cd /d "%LM_ROOT%"

echo === GM Landmarking: HRNet Trainer ===

python ".\scripts\trainer_entry.py"

endlocal
