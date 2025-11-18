@echo off
setlocal EnableExtensions

rem Определяем корень проекта
set "LM_ROOT=%~dp0"
if "%LM_ROOT:~-1%"=="\\" set "LM_ROOT=%LM_ROOT:~0,-1%"

cd /d "%LM_ROOT%"
python ".\scripts\annotator_entry.py"

endlocal
