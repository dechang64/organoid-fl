@echo off
REM FP clustering + DINOv2 DPMM verification
REM Usage: cd C:\Users\decha\organoid-fl && scripts\multiorg\run_fp_analysis.bat

cd /d C:\Users\decha\organoid-fl
call .\.venv\Scripts\activate

pip install timm tifffile scikit-learn matplotlib

python scripts\multiorg\fp_clustering_analysis.py --results-json results\multiorg_sam2_zeroshot\multiorg_sam2_results.json --data-root D:\datasets\mutliorg\MultiOrg_v2 --annotator t1_b --output-dir results\fp_analysis_zero_shot --device cuda

echo.
echo === Done! Output: results\fp_analysis_zero_shot\ ===
dir results\fp_analysis_zero_shot\
pause
