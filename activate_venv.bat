@echo off
REM Batch script to activate virtual environment
REM Usage: activate_venv.bat

set "VENV_PATH=%~dp0venv"

if not exist "%VENV_PATH%" (
    echo Error: Virtual environment not found at '%VENV_PATH%'
    echo Please run setup_venv.ps1 or python -m venv venv first to create the virtual environment.
    exit /b 1
)

set "ACTIVATE_SCRIPT=%VENV_PATH%Scripts\activate.bat"

if exist "%ACTIVATE_SCRIPT%" (
    call "%ACTIVATE_SCRIPT%"
    echo Virtual environment activated!
) else (
    echo Error: Activation script not found at '%ACTIVATE_SCRIPT%'
    exit /b 1
)

