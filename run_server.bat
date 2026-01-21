@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Ambiente virtual nao encontrado.
    echo Rode install_deps.bat primeiro.
    pause
    exit /b 1
)

call venv\Scripts\activate
if errorlevel 1 (
    echo Falha ao ativar o ambiente virtual.
    pause
    exit /b 1
)

venv\Scripts\python.exe -m streamlit run server.py --server.maxUploadSize=4096

pause
exit /b 0

