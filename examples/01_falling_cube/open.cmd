@echo off
setlocal
rem Use the locked project environment.
set "LESSON_PYTHONW=%~dp0..\..\.venv\Scripts\pythonw.exe"
if not exist "%LESSON_PYTHONW%" (
  echo Project environment was not found. Run uv sync --locked in the project root.
  pause
  exit /b 1
)
start "" "%LESSON_PYTHONW%" -m mujoco.studio.viewer --gfx=opengl_software --hot_reload=model --model "%~dp0scene.xml"
