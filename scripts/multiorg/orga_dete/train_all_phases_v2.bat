@echo off
REM Orga-Dete v2: yolo12s instead of yolo11n
REM yolo11n 2.6M only got 44.2% (ep24), yolo12s 9.4M baseline 62.3%
REM Expected: 65-70% with MPCA + EMASlideLoss
REM
REM Usage: scripts\multiorg\orga_dete\train_all_phases_v2.bat
REM Prerequisite: data.yaml at D:\datasets\MultiOrg_v4_640\data.yaml

cd /d C:\Users\decha\organoid-fl
call .\.venv\Scripts\activate

REM 验证数据集
if not exist "D:\datasets\MultiOrg_v4_640\data.yaml" (
    echo ERROR: data.yaml not found at D:\datasets\MultiOrg_v4_640\data.yaml
    pause
    exit /b 1
)

echo ================================================================
echo Phase 1: yolo12s + MPCA (baseline + MPCA)
echo ================================================================
python -c "import sys; sys.path.insert(0, 'scripts/multiorg/orga_dete'); from orga_dete_modules import MPCA; import ultralytics.nn.tasks as tasks; import ultralytics.nn.modules as modules; tasks.MPCA = MPCA; modules.MPCA = MPCA; from ultralytics import YOLO; model = YOLO('scripts/multiorg/orga_dete/orga_dete_yolo12s.yaml'); model.load('yolo12s.pt'); model.train(data='D:\\datasets\\MultiOrg_v4_640\\data.yaml', epochs=300, imgsz=640, batch=8, device=0, project='runs/orga_dete', name='phase1_12s_mpca', cos_lr=True, close_mosaic=15, patience=50, label_smoothing=0.1, copy_paste=0.1, mixup=0.1, workers=8)"

echo.
echo ================================================================
echo Phase 3: yolo12s + MPCA + EMASlideLoss
echo ================================================================
python scripts\multiorg\orga_dete\train_with_ema_slide.py --model scripts\multiorg\orga_dete\orga_dete_yolo12s.yaml --data D:\datasets\MultiOrg_v4_640\data.yaml --epochs 300 --imgsz 640 --batch 8 --device 0 --project runs\orga_dete --name phase3_12s_mpca_ema --pretrained yolo12s.pt --workers 8

echo.
echo === All phases complete! ===
echo Results in: runs\orga_dete\
echo   phase1_12s_mpca\       — yolo12s + MPCA
echo   phase3_12s_mpca_ema\   — yolo12s + MPCA + EMASlideLoss
pause
