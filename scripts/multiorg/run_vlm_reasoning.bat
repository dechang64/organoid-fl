@echo off
chcp 65001 >nul
cd /d %~dp0\..\..

echo ============================================================
echo  Phase B2: VLM Two-Step Reasoning TP/FP Evaluation
echo  Uses z-ai vision API (no GPU needed)
echo ============================================================
echo.

REM Check bun available (for VLM SDK)
where bun >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] bun not found. Install: https://bun.sh
    pause
    exit /b 1
)

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
