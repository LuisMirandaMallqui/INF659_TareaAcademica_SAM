@echo off
REM Hacia IoU 0.60 - paso 1. Doble clic para abrir una ventana con el avance del entrenamiento.
REM Argumentos opcionales, p. ej.:  lanzar_hacia060.bat --rapido   o   --configs db3_otsu,db3_sin
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0.."

REM Entorno ya instalado en esta PC (datos, pesos, caches y venv con GPU). Si no existe, se usan
REM las rutas por defecto del repo (data/, models/, cache/) y el python del .venv o del PATH.
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

title Hacia IoU 0.60 - paso 1
echo Python: %PY%
"%PY%" tools\hacia060_paso1.py %*
echo.
echo Fin. La bitacora quedo actualizada en docs\bitacora_hacia_060.md
pause
