@echo off
setlocal
set "LAB_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
if not exist "%LAB_PYTHON%" (
  echo Project environment was not found. Run uv sync --locked in the project root.
  pause
  exit /b 1
)
set "PYTHONPATH=%~dp0..\..\src;%PYTHONPATH%"
"%LAB_PYTHON%" -B -m decision_lab grasp
if errorlevel 1 pause
