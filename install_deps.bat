@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    py -3 -m venv venv 2>nul
    if errorlevel 1 (
        python -m venv venv
        if errorlevel 1 (
            echo Falha ao criar o ambiente virtual.
            pause
            exit /b 1
        )
    )
)

call venv\Scripts\activate
if errorlevel 1 (
    echo Falha ao ativar o ambiente virtual.
    pause
    exit /b 1
)

python -m pip install --upgrade pip
if errorlevel 1 (
    echo Falha ao atualizar o pip.
    pause
    exit /b 1
)

python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple torch==2.2.2+cu121 torchaudio==2.2.2+cu121
if errorlevel 1 (
    echo Falha ao instalar PyTorch com CUDA. Verifique o driver NVIDIA e o Python (64-bit).
    pause
    exit /b 1
)

python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Falha ao instalar as dependencias definidas em requirements.txt.
    pause
    exit /b 1
)

echo Dependencias instaladas com sucesso.
pause
exit /b 0
