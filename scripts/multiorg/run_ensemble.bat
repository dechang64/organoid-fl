@echo off
REM Ensemble Inference: RF-DETR + YOLOv12
REM Usage: cd C:\Users\decha\organoid-fl && scripts\multiorg\run_ensemble.bat

cd /d C:\Users\decha\organoid-fl
call .\.venv\Scripts\activate

REM 先安装 ensemble-boxes（WBF 实现）
pip install ensemble-boxes >nul 2>&1

REM === 策略 1: WBF (SOTA, 推荐) ===
REM 论文: Solovyev et al., IVC 2021 (arXiv 1910.13302)
REM 所有 box 按 score 聚类，fused score = sum(scores)*T/N
REM 比 NMS/Soft-NMS/NMW 全面最优，cluster 机制天然去重
python scripts\multiorg\ensemble_inference.py --rfdetr-weights output\checkpoint_best_regular.pth --rfdetr-variant small --yolo-weights D:\datasets\MultiOrg_v4_640\runs\multiorg_v5_12s_freebies-2\best.pt --src D:\datasets\mutliorg\MultiOrg_v2\test --dst results\ensemble_wbf --strategy wbf --match-iou 0.55 --annotator t1_b

echo.
echo ==========================================
echo WBF done. Now running intersection...
echo ==========================================
echo.

REM === 策略 2: intersection (严格 baseline) ===
python scripts\multiorg\ensemble_inference.py --rfdetr-weights output\checkpoint_best_regular.pth --rfdetr-variant small --yolo-weights D:\datasets\MultiOrg_v4_640\runs\multiorg_v5_12s_freebies-2\best.pt --src D:\datasets\mutliorg\MultiOrg_v2\test --dst results\ensemble_intersection --strategy intersection --match-iou 0.3 --annotator t1_b

echo.
echo ==========================================
echo Intersection done. Now running union...
echo ==========================================
echo.

REM === 策略 3: union (宽松 baseline) ===
python scripts\multiorg\ensemble_inference.py --rfdetr-weights output\checkpoint_best_regular.pth --rfdetr-variant small --yolo-weights D:\datasets\MultiOrg_v4_640\runs\multiorg_v5_12s_freebies-2\best.pt --src D:\datasets\mutliorg\MultiOrg_v2\test --dst results\ensemble_union --strategy union --match-iou 0.3 --unmatched-penalty 0.7 --annotator t1_b

echo.
echo === Done! Compare results: ===
echo.
echo --- WBF ---
type results\ensemble_wbf\ensemble_results.json | findstr "mean_ap50"
echo.
echo --- Intersection ---
type results\ensemble_intersection\ensemble_results.json | findstr "mean_ap50"
echo.
echo --- Union ---
type results\ensemble_union\ensemble_results.json | findstr "mean_ap50"
pause
