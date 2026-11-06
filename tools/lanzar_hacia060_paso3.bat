@echo off
REM Hacia IoU 0.60 - paso 3 (3a U-Net, 3b Tversky). Doble clic para abrir una ventana con el avance.
REM Argumento opcional:  --rapido  (prueba de humo de pocos minutos)
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
if not defined INF659_THREADS set INF659_THREADS=8

title Hacia IoU 0.60 - paso 3 (U-Net y Tversky)
echo Python: %PY%
"%PY%" tools\hacia060_paso3.py %*
echo.
echo Fin. La bitacora quedo actualizada en docs\bitacora_hacia_060.md
pause
