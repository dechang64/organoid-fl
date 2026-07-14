@echo off
chcp 65001 >nul
cd /d %~dp0\..\..

echo ============================================================
echo  Phase A1: CLIP + SupCon LOO Cross-Domain
echo  CLIP ViT-B/16 features → SupCon training → LOO eval
echo ============================================================
echo.

pip install open_clip_torch 2>nul
python scripts\multiorg\clip_supcon_loo.py --device cuda:0 --epochs 50
if %errorlevel% neq 0 (
    echo [ERROR] CLIP SupCon experiment failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Done! Results: results\clip_supcon_loo\clip_supcon_loo_results.json
echo ============================================================
pause
