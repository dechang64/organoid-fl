@echo off
chcp 65001 >nul
echo ============================================================
echo  Cross-Domain Evaluation: MultiOrg Slot Model -> Mouse Liver
echo ============================================================
echo.

set CKPT=results\supcon_8s_d128_p256_t0.07_b0.1_20260713_003826\best.pt

echo [1/6] Generating B1 crops...
python scripts\mouse_liver\generate_mouse_crops.py --batch b1 --weights runs\mouse_liver_v2\b1\full\checkpoint_best_regular.pth --src mouse_liver_data_correct\batch1\images --annotations mouse_liver_data_correct\batch1\annotations.json --dst data\mouse_crops\b1 --resolution 544
if %errorlevel% neq 0 (
    echo [ERROR] B1 crop generation failed!
    pause
    exit /b 1
)
echo.

echo [2/6] Cross-domain eval B1...
python scripts\mouse_liver\cross_domain_eval.py --checkpoint %CKPT% --metadata data\mouse_crops\b1\crop_metadata.json --crops-dir data\mouse_crops\b1\crops --device cuda:0 --tag b1
echo.

echo [3/6] Generating B2 crops...
python scripts\mouse_liver\generate_mouse_crops.py --batch b2 --weights runs\mouse_liver_v2\b2\full\checkpoint_best_regular.pth --src mouse_liver_data_correct\batch2\images --annotations mouse_liver_data_correct\batch2\annotations.json --dst data\mouse_crops\b2 --resolution 768
if %errorlevel% neq 0 (
    echo [ERROR] B2 crop generation failed!
    pause
    exit /b 1
)
echo.

echo [4/6] Cross-domain eval B2...
python scripts\mouse_liver\cross_domain_eval.py --checkpoint %CKPT% --metadata data\mouse_crops\b2\crop_metadata.json --crops-dir data\mouse_crops\b2\crops --device cuda:0 --tag b2
echo.

echo [5/6] Generating B3 crops...
python scripts\mouse_liver\generate_mouse_crops.py --batch b3 --weights runs\mouse_liver_v2\b3\full\checkpoint_best_regular.pth --src mouse_liver_data_correct\batch3\images --annotations mouse_liver_data_correct\batch3\annotations.json --dst data\mouse_crops\b3 --resolution 768
if %errorlevel% neq 0 (
    echo [ERROR] B3 crop generation failed!
    pause
    exit /b 1
)
echo.

echo [6/6] Cross-domain eval B3...
python scripts\mouse_liver\cross_domain_eval.py --checkpoint %CKPT% --metadata data\mouse_crops\b3\crop_metadata.json --crops-dir data\mouse_crops\b3\crops --device cuda:0 --tag b3
echo.

echo ============================================================
echo  Done! Results in results\cross_domain_b1\, b2\, b3\
echo  Send me: cross_domain_results.json from each folder
echo ============================================================
pause
