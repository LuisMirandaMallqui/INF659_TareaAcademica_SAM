@echo off
REM Evaluacion final en test y bolivia. Sin argumentos es una PRUEBA SEGURA sobre valid (no toca test).
REM   tools\lanzar_evaluacion_test.bat                                  -> prueba en valid
REM   tools\lanzar_evaluacion_test.bat --abrir-test                     -> candidato SAM en test y bolivia
REM   tools\lanzar_evaluacion_test.bat --abrir-test --modelos sam,unet  -> + U-Net como linea base
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."

set "ENV_DIR=%USERPROFILE%\inf659_env"
if exist "%ENV_DIR%\venv\Scripts\python.exe" (
    if not defined INF659_DATA set "INF659_DATA=%ENV_DIR%\data"
    if not defined INF659_CACHE set "INF659_CACHE=%ENV_DIR%\cache"
    if not defined INF659_MODEL set "INF659_MODEL=%ENV_DIR%\models\sam-vit-base"
    set "PY=%ENV_DIR%\venv\Scripts\python.exe"
) else if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

title Evaluacion final (test y bolivia)
echo Python: %PY%
"%PY%" tools\evaluar_test.py %*
echo.
pause
