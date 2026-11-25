@echo off
rem Local launcher for the Kestrel scoring service on Windows.
rem Double-click this file, or run it from a terminal. It creates the virtual
rem environment on first use, then serves on http://127.0.0.1:%PORT%
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  py -3.10 -m venv .venv 2>nul || py -3 -m venv .venv 2>nul || python -m venv .venv
  if not exist ".venv\Scripts\python.exe" (
    echo Could not create a virtual environment. Install Python 3.10 or newer and retry.
    pause
    exit /b 1
  )
  echo Installing dependencies...
  ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
  ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
  if errorlevel 1 (
    echo Dependency install failed.
    pause
    exit /b 1
  )
)

set "PYTHONPATH=%~dp0"
if "%PORT%"=="" set "PORT=8020"
rem Single-threaded math libraries: multi-threaded OpenBLAS can abort at import on Windows.
set "OPENBLAS_NUM_THREADS=1"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"

echo.
echo   Kestrel scoring service: http://127.0.0.1:%PORT%
echo   Health check:            http://127.0.0.1:%PORT%/health
echo   Press Ctrl+C to stop.
echo.
".venv\Scripts\python.exe" -m src.serve
pause
