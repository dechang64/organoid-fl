@echo off
REM Ensemble Inference: RF-DETR + YOLOv12
REM Usage: cd C:\Users\decha\organoid-fl && scripts\multiorg\run_ensemble.bat

cd /d C:\Users\decha\organoid-fl
call .\.venv\Scripts\activate

REM 先跑 intersection 策略（严格，砍 FP）
python scripts\multiorg\ensemble_inference.py --rfdetr-weights output\checkpoint_best_regular.pth --rfdetr-variant small --yolo-weights D:\datasets\MultiOrg_v4_640\runs\multiorg_v5_12s_freebies-2\best.pt --src D:\datasets\mutliorg\MultiOrg_v2\test --dst results\ensemble_intersection --strategy intersection --match-iou 0.5 --annotator t1_b

echo.
echo ==========================================
echo Intersection done. Now running union...
echo ==========================================
echo.

REM 再跑 union 策略（宽松，保留 recall）
python scripts\multiorg\ensemble_inference.py --rfdetr-weights output\checkpoint_best_regular.pth --rfdetr-variant small --yolo-weights D:\datasets\MultiOrg_v4_640\runs\multiorg_v5_12s_freebies-2\best.pt --src D:\datasets\mutliorg\MultiOrg_v2\test --dst results\ensemble_union --strategy union --match-iou 0.5 --unmatched-penalty 0.7 --annotator t1_b

echo.
echo === Done! Compare results: ===
type results\ensemble_intersection\ensemble_results.json | findstr "mean_ap50"
type results\ensemble_union\ensemble_results.json | findstr "mean_ap50"
pause
