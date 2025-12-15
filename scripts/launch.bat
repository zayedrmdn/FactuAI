@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

REM ─────────────────────────────────────────────────────────────
REM   launch.bat - FactuAI Local Development Launcher
REM
REM   Usage:
REM     launch.bat       - Starts backend + frontend (default)
REM
REM   Windows:
REM     - Backend:  uvicorn app.main:app --reload
REM     - Frontend: pnpm dev
REM ─────────────────────────────────────────────────────────────

REM Setup paths
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."
set "BACKEND_DIR=%PROJECT_ROOT%\backend"
set "FRONTEND_DIR=%PROJECT_ROOT%\frontend"
set "VENV_DIR=%BACKEND_DIR%\venv"

echo.
echo ═══════════════════════════════════════════════════════════
echo   FactuAI Launcher
echo ═══════════════════════════════════════════════════════════
echo.

REM Check if backend venv exists, create if not
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [INFO] Creating backend virtual environment...
    cd /d "%BACKEND_DIR%"
    python -m venv venv
    call "%VENV_DIR%\Scripts\activate.bat"
    echo [INFO] Installing dependencies...
    pip install -r requirements-core.txt
    echo [INFO] Virtual environment ready.
) else (
    echo [INFO] Using existing backend venv
)

echo.
echo Starting FactuAI services...
echo.

REM Backend
start "FactuAI Backend" cmd /k ^
  "cd /d %BACKEND_DIR% && call %VENV_DIR%\Scripts\activate.bat && uvicorn app.main:app --reload"

REM Frontend
start "FactuAI Frontend" cmd /k ^
  "cd /d %FRONTEND_DIR% && pnpm dev"

echo [✓] Backend and Frontend started in new windows
echo.
pause