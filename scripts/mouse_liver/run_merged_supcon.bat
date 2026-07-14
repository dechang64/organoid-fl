@echo off
chcp 65001 >nul

REM Switch to project root (2 levels up from this bat)
cd /d %~dp0\..\..

echo ============================================================
echo  Multi-Dataset Joint SupCon Training + Cross-Domain Eval
echo  MultiOrg + Mouse Liver + Intestinal
echo ============================================================
echo.
echo  Working dir: %CD%
echo.

REM ============================================================
REM Step 1: Merge datasets
REM ============================================================
echo [1/6] Merging datasets...
python scripts\mouse_liver\merge_datasets.py
if %errorlevel% neq 0 (
    echo [ERROR] Merge failed!
    pause
    exit /b 1
)
echo.

REM ============================================================
REM Step 2: Train SupCon on merged data
REM ============================================================
set CKPT=results\supcon_merged\best.pt

if exist %CKPT% (
    echo [2/6] SupCon checkpoint already exists: %CKPT%
    echo      Skipping training. Delete the checkpoint to retrain.
) else (
    echo [2/6] Training SupCon on merged dataset...
    python scripts\multiorg\slot_supcon.py --metadata data\merged_crops\crop_metadata.json --crops-dir data\merged_crops\crops --output-dir results\supcon_merged --num-slots 8 --dim-slots 128 --proj-dim 256 --temperature 0.07 --supcon-weight 0.1 --epochs 50 --batch-size 32 --device cuda:0
    if %errorlevel% neq 0 (
        echo [ERROR] SupCon training failed!
        pause
        exit /b 1
    )
)
echo.

REM ============================================================
REM Step 3-6: Cross-domain eval on each dataset
REM ============================================================
echo [3/6] Eval: merged model -> Mouse B1
python scripts\mouse_liver\cross_domain_eval.py --checkpoint %CKPT% --metadata data\mouse_crops\b1\crop_metadata.json --crops-dir data\mouse_crops\b1\crops --device cuda:0 --tag merged_b1
echo.

echo [4/6] Eval: merged model -> Mouse B2
python scripts\mouse_liver\cross_domain_eval.py --checkpoint %CKPT% --metadata data\mouse_crops\b2\crop_metadata.json --crops-dir data\mouse_crops\b2\crops --device cuda:0 --tag merged_b2
echo.

echo [5/6] Eval: merged model -> Mouse B3
python scripts\mouse_liver\cross_domain_eval.py --checkpoint %CKPT% --metadata data\mouse_crops\b3\crop_metadata.json --crops-dir data\mouse_crops\b3\crops --device cuda:0 --tag merged_b3
echo.

echo [6/6] Eval: merged model -> Intestinal
python scripts\mouse_liver\cross_domain_eval.py --checkpoint %CKPT% --metadata data\intestinal_crops\val\crop_metadata.json --crops-dir data\intestinal_crops\val\crops --device cuda:0 --tag merged_intestinal
echo.

REM ============================================================
REM Pack results
REM ============================================================
echo Packing results...
powershell -Command "Compress-Archive -Path 'results\cross_domain_merged_b1','results\cross_domain_merged_b2','results\cross_domain_merged_b3','results\cross_domain_merged_intestinal' -DestinationPath 'results\cross_domain_merged.zip' -Force"

if exist results\cross_domain_merged.zip (
    echo.
    echo ============================================================
    echo  Done! Results packed into: results\cross_domain_merged.zip
    echo  Send me this zip file.
    echo ============================================================
) else (
    echo [WARN] Zip creation failed. Results are in:
    echo   results\cross_domain_merged_b1\cross_domain_results.json
    echo   results\cross_domain_merged_b2\cross_domain_results.json
    echo   results\cross_domain_merged_b3\cross_domain_results.json
    echo   results\cross_domain_merged_intestinal\cross_domain_results.json
)

pause
