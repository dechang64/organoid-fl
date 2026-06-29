@echo off
REM FP missed annotation analysis
REM Usage: cd C:\Users\decha\organoid-fl && scripts\multiorg\run_fp_missed_analysis.bat

cd /d C:\Users\decha\organoid-fl
call .\.venv\Scripts\activate

python scripts\multiorg\fp_missed_annotation_analysis.py --results-json results\multiorg_sam2_zeroshot\multiorg_sam2_results.json --data-root D:\datasets\mutliorg\MultiOrg_v2 --annotator t1_b --output-dir results\fp_missed_analysis --iou-threshold 0.5

echo.
echo === Done! Output: results\fp_missed_analysis\ ===
dir results\fp_missed_analysis\
echo.
echo === napari_missed subfolder ===
dir results\fp_missed_analysis\napari_missed\
pause
