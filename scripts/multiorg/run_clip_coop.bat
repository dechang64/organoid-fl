@echo off
chcp 65001 >nul
cd /d %~dp0\..\..

echo ============================================================
echo  Phase A3: CLIP CoOp Prompt Tuning + LOO
echo  Learn prompts, freeze CLIP backbone (preserve zero-shot)
echo ============================================================
echo.

pip install open_clip_torch 2>nul
python scripts\multiorg\clip_coop_loo.py --device cuda:0 --epochs 50
if %errorlevel% neq 0 (
    echo [ERROR] CoOp experiment failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Done! Results: results\clip_coop_loo\clip_coop_loo_results.json
echo ============================================================
pause
