@echo off
REM Ensemble Inference: RF-DETR + YOLOv12
REM Usage: scripts\multiorg\run_ensemble.bat <yolo_weights_path>
REM Example: scripts\multiorg\run_ensemble.bat "D:\datasets\MultiOrg_v4_640\runs\multiorg_v5_12s_freebies-2\weights\best.pt"

cd /d C:\Users\decha\organoid-fl
call .\.venv\Scripts\activate

REM === 接收 YOLO checkpoint 路径作为参数 ===
set YOLO_CKPT=%1
if "%YOLO_CKPT%"=="" (
    echo ERROR: Please provide YOLO checkpoint path
    echo Usage: scripts\multiorg\run_ensemble.bat ^<yolo_weights_path^>
    echo Example: scripts\multiorg\run_ensemble.bat "D:\path\to\best.pt"
    pause
    exit /b 1
)

REM === 验证文件存在 ===
if not exist "%YOLO_CKPT%" (
    echo ERROR: YOLO checkpoint not found: %YOLO_CKPT%
    echo Please verify the path.
    pause
    exit /b 1
)

REM === 验证 RF-DETR checkpoint 存在 ===
if not exist "output\checkpoint_best_regular.pth" (
    echo ERROR: RF-DETR checkpoint not found: output\checkpoint_best_regular.pth
    echo Please verify the path.
    pause
    exit /b 1
)

REM 先安装 ensemble-boxes（WBF 实现）
pip install ensemble-boxes >nul 2>&1

echo.
echo === YOLO checkpoint: %YOLO_CKPT% ===
echo === RF-DETR checkpoint: output\checkpoint_best_regular.pth ===
echo.

REM === 策略 1: WBF (SOTA, 推荐) ===
python scripts\multiorg\ensemble_inference.py --rfdetr-weights output\checkpoint_best_regular.pth --rfdetr-variant small --yolo-weights "%YOLO_CKPT%" --src D:\datasets\mutliorg\MultiOrg_v2\test --dst results\ensemble_wbf --strategy wbf --match-iou 0.55 --annotator t1_b

echo.
echo ==========================================
echo WBF done. Now running intersection...
echo ==========================================
echo.

REM === 策略 2: intersection (严格 baseline) ===
python scripts\multiorg\ensemble_inference.py --rfdetr-weights output\checkpoint_best_regular.pth --rfdetr-variant small --yolo-weights "%YOLO_CKPT%" --src D:\datasets\mutliorg\MultiOrg_v2\test --dst results\ensemble_intersection --strategy intersection --match-iou 0.3 --annotator t1_b

echo.
echo ==========================================
echo Intersection done. Now running union...
echo ==========================================
echo.

REM === 策略 3: union (宽松 baseline) ===
python scripts\multiorg\ensemble_inference.py --rfdetr-weights output\checkpoint_best_regular.pth --rfdetr-variant small --yolo-weights "%YOLO_CKPT%" --src D:\datasets\mutliorg\MultiOrg_v2\test --dst results\ensemble_union --strategy union --match-iou 0.3 --unmatched-penalty 0.7 --annotator t1_b

echo.
echo === Done! Compare results: ===
echo.
echo --- WBF ---
if exist results\ensemble_wbf\ensemble_results.json (type results\ensemble_wbf\ensemble_results.json | findstr "mean_ap50") else (echo FAILED)
echo.
echo --- Intersection ---
if exist results\ensemble_intersection\ensemble_results.json (type results\ensemble_intersection\ensemble_results.json | findstr "mean_ap50") else (echo FAILED)
echo.
echo --- Union ---
if exist results\ensemble_union\ensemble_results.json (type results\ensemble_union\ensemble_results.json | findstr "mean_ap50") else (echo FAILED)
pause
