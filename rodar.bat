@echo off
title Deep Audio Cleaner Server

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo Criando ambiente virtual...
    py -3 -m venv venv 2>nul
    if errorlevel 1 (
        python -m venv venv
    )
)

call venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple torch==2.2.2+cu121 torchaudio==2.2.2+cu121
python -m pip install -r requirements.txt

echo Iniciando servidor...
venv\Scripts\python.exe -m streamlit run server.py --server.maxUploadSize=4096

pause
