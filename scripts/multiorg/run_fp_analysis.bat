@echo off
REM FP clustering + DINOv2 DPMM verification
REM Usage: run_fp_analysis.bat
REM Run from organoid-fl root: C:\Users\decha\organoid-fl

echo === Installing dependencies ===
pip install timm tifffile scikit-learn matplotlib

echo.
echo === Running FP Clustering Analysis ===
python scripts\multiorg\fp_clustering_analysis.py --results-json D:\datasets\MultiOrg_v4_640\runs\multiorg_sam2_zero_shot\multiorg_sam2_results.json --data-root D:\datasets\mutliorg\MultiOrg_v2 --annotator t1_b --output-dir results\fp_analysis_zero_shot --device cuda

echo.
echo === Done! Output: results\fp_analysis_zero_shot\ ===
dir results\fp_analysis_zero_shot\
pause
