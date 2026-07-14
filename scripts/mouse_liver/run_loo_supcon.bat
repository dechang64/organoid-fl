@echo off
chcp 65001 >nul

cd /d %~dp0\..\..

echo ============================================================
echo  Leave-One-Out Cross-Domain SupCon (True Cross-Domain)
echo ============================================================
echo.

python scripts\mouse_liver\run_loo_supcon.py --device cuda:0 --epochs 50
if %errorlevel% neq 0 (
    echo [ERROR] LOO experiment failed!
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Done! Check results/loo_supcon/loo_summary.json
echo ============================================================
pause
