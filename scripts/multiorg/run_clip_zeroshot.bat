@echo off
chcp 65001 >nul
cd /d %~dp0\..\..

echo ============================================================
echo  Phase A4: CLIP Zero-Shot TP/FP Evaluation
echo  5 prompt pairs × 4 datasets (MultiOrg + Mouse B1/B2/B3 + Intestinal)
echo ============================================================
echo.

python scripts\multiorg\clip_zeroshot_eval.py --device cuda:0
if %errorlevel% neq 0 (
    echo [ERROR] CLIP eval failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Done! Results: results\clip_zeroshot\clip_zeroshot_results.json
echo ============================================================
pause
