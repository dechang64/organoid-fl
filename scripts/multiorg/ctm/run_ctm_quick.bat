@echo off
chcp 65001 >nul
cd /d C:\Users\decha\organoid-fl

echo ============================================================
echo  CTM Quick Test (100 crops, 50 epochs)
echo  Output auto-generated, no overwrite
echo ============================================================
echo.

python scripts\multiorg\ctm\ctm_train.py --metadata results\phase2_vlm_100\vlm_results.json --crops-dir results\phase2_vlm_100\crops --device cuda:0 --epochs 50 --n-ticks 20 --d-internal 256 --batch-size 32

echo.
echo ============================================================
echo  Training done. Check results\ctm_20ticks_d256_*
echo  Run evaluate.bat next for tick-wise analysis
echo ============================================================
pause
