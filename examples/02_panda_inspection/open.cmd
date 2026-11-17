@echo off
setlocal
rem Load the pinned official Panda scene using the locked project environment.
set "LESSON_PYTHONW=%~dp0..\..\.venv\Scripts\pythonw.exe"
set "PANDA_SCENE=%~dp0..\..\assets\robots\franka_emika_panda\scene.xml"
if not exist "%LESSON_PYTHONW%" (
  echo Project environment was not found. Run uv sync --locked in the project root.
  pause
  exit /b 1
)
if not exist "%PANDA_SCENE%" (
  echo The official Panda scene was not found at %PANDA_SCENE%
  pause
  exit /b 1
)
start "" "%LESSON_PYTHONW%" -m mujoco.studio.viewer --gfx=opengl_software --model "%PANDA_SCENE%"
