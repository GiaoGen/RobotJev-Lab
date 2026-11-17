@echo off
setlocal
set "LAB_PYTHON=%~dp0..\..\.venv\Scripts\python.exe"
if not exist "%LAB_PYTHON%" (
  echo Project environment was not found. Run uv sync --locked in the project root.
  pause
  exit /b 1
)
set "PYTHONPATH=%~dp0..\..\src;%PYTHONPATH%"
set "LAB_CONFIG=%~1"
if not defined LAB_CONFIG set "LAB_CONFIG=%~dp0..\..\configs\tasks\panda_pick_place.json"
"%LAB_PYTHON%" -B -m decision_lab pick_place --config "%LAB_CONFIG%"
if errorlevel 1 pause
