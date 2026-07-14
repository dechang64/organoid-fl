@echo off
chcp 65001 >nul
cd /d %~dp0\..\..

echo ============================================================
echo  Phase B2: VLM Two-Step Reasoning TP/FP Evaluation
echo  Uses z-ai vision API via bun + Node.js SDK
echo ============================================================
echo.

REM Check bun available
where bun >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] bun not found. Installing...
    powershell -Command "irm bun.sh/install.ps1 | iex"
    REM Refresh PATH after install
    set "PATH=%USERPROFILE%\.bun\bin;%PATH%"
    where bun >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] bun install failed. Please install manually from https://bun.sh
        pause
        exit /b 1
    )
)

REM Install z-ai-web-dev-sdk if not present
bun add z-ai-web-dev-sdk 2>nul

REM Run full eval on all 4 datasets
python scripts\multiorg\vlm_reasoning_eval.py
if %errorlevel% neq 0 (
    echo [ERROR] VLM eval failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Done! Results: results\vlm_reasoning\vlm_reasoning_results.json
echo ============================================================
pause
