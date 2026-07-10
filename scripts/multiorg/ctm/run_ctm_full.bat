@echo off
chcp 65001 >nul
cd /d C:\Users\decha\organoid-fl

echo ============================================================
echo  Step 1: Generate crops (16198 detections from 55 images)
echo  Source: D:\datasets\mutliorg\MultiOrg_v2
echo  Output: data\ctm_crops\
echo ============================================================
echo.

python scripts\multiorg\ctm\ctm_generate_crops.py --sam2-results results\multiorg_sam2_zeroshot\multiorg_sam2_results.json --images-root "D:\datasets\mutliorg\MultiOrg_v2" --output-dir data\ctm_crops --pad-ratio 0.2

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Crop generation failed. Check errors above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  Step 2: Train CTM (full dataset, 50 epochs, ~8h)
echo  Output auto-generated: results\ctm_20ticks_d256_TIMESTAMP\
echo ============================================================
echo.

python scripts\multiorg\ctm\ctm_train.py --metadata data\ctm_crops\ctm_metadata.json --crops-dir data\ctm_crops --device cuda:0 --epochs 50 --n-ticks 20 --d-internal 256 --batch-size 32 --num-workers 4

echo.
echo ============================================================
echo  Done. Check results\ctm_20ticks_d256_*
echo  Run: for /d %%d in (results\ctm_*) do python scripts\multiorg\ctm\ctm_evaluate.py --checkpoint "%%d\best.pt" --metadata data\ctm_crops\ctm_metadata.json --crops-dir data\ctm_crops --device cuda:0
echo ============================================================
pause
