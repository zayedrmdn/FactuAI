@echo off
REM ─────────────────────────────────────────────────────────────
REM   launch.bat
REM   Location: scripts/launch.bat
REM   
REM   Usage:
REM     launch.bat cloud   - Start in Cloud Mode (lightweight, uses APIs)
REM     launch.bat local   - Start in Local Mode (full ML stack)
REM     launch.bat         - Defaults to Cloud Mode
REM
REM   Virtual Environments:
REM     .venv-cloud/  - For Cloud Mode (requirements-core.txt)
REM     .venv-local/  - For Local Mode (requirements-local.txt)
REM ─────────────────────────────────────────────────────────────

REM Get the project root (parent of scripts folder)
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

REM Determine run mode (default: cloud)
set "RUN_MODE=%1"
if "%RUN_MODE%"=="" set "RUN_MODE=cloud"

echo ═══════════════════════════════════════════════════════════
echo   FactuAI Launcher
echo   Mode: %RUN_MODE%
echo   Project Root: %PROJECT_ROOT%
echo ═══════════════════════════════════════════════════════════

REM Set venv path based on mode
if "%RUN_MODE%"=="local" (
    set "VENV_PATH=%PROJECT_ROOT%\.venv-local"
    set "REQUIREMENTS=%PROJECT_ROOT%\requirements-local.txt"
) else (
    set "VENV_PATH=%PROJECT_ROOT%\.venv-cloud"
    set "REQUIREMENTS=%PROJECT_ROOT%\requirements-core.txt"
)

REM Check if venv exists, create if not
if not exist "%VENV_PATH%\Scripts\activate.bat" (
    echo [INFO] Creating virtual environment at %VENV_PATH%...
    python -m venv "%VENV_PATH%"
    echo [INFO] Installing dependencies from %REQUIREMENTS%...
    call "%VENV_PATH%\Scripts\activate.bat"
    pip install -r "%REQUIREMENTS%"
) else (
    echo [INFO] Using existing venv: %VENV_PATH%
)

echo.
echo Starting FactuAI stack...

REM 1 ▸ Backend
start "FactuAI Backend" cmd /k ^
  "cd /d %PROJECT_ROOT%\backend && call %VENV_PATH%\Scripts\activate.bat && set APP_RUN_MODE=%RUN_MODE% && python app.py"

REM 2 ▸ Frontend
start "FactuAI Frontend" cmd /k ^
  "cd /d %PROJECT_ROOT%\frontend && npm run dev"

REM 3 ▸ Misc scripts shell
start "FactuAI Scripts" cmd /k ^
  "cd /d %PROJECT_ROOT%\scripts && call %VENV_PATH%\Scripts\activate.bat && set APP_RUN_MODE=%RUN_MODE%"

echo.
echo All windows launched. You can close this launcher now.
pause >nul
