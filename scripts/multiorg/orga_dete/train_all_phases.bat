@echo off
REM Orga-Dete 三阶段训练对比
REM Phase 1: MPCA only
REM Phase 2: MPCA + BiFPN (代码构建模型)
REM Phase 3: MPCA + EMASlideLoss
REM
REM Usage: scripts\multiorg\orga_dete\train_all_phases.bat
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
echo Phase 1: MPCA only (baseline + MPCA)
echo ================================================================
python -c "import sys; sys.path.insert(0, 'scripts/multiorg/orga_dete'); from orga_dete_modules import MPCA; import ultralytics.nn.tasks as tasks; import ultralytics.nn.modules as modules; tasks.MPCA = MPCA; modules.MPCA = MPCA; from ultralytics import YOLO; model = YOLO('scripts/multiorg/orga_dete/orga_dete_yolo11n.yaml'); model.load('yolo11n.pt'); model.train(data='D:\\datasets\\MultiOrg_v4_640\\data.yaml', epochs=300, imgsz=640, batch=8, device=0, project='runs/orga_dete', name='phase1_mpca', cos_lr=True, close_mosaic=15, patience=50, label_smoothing=0.1, copy_paste=0.1, mixup=0.1)"

echo.
echo ================================================================
echo Phase 3: MPCA + EMASlideLoss
echo ================================================================
python scripts\multiorg\orga_dete\train_with_ema_slide.py --model scripts\multiorg\orga_dete\orga_dete_yolo11n.yaml --data D:\datasets\MultiOrg_v4_640\data.yaml --epochs 300 --imgsz 640 --batch 8 --device 0 --project runs\orga_dete --name phase3_mpca_ema --pretrained yolo11n.pt

echo.
echo === All phases complete! ===
echo Results in: runs\orga_dete\
echo   phase1_mpca\       — MPCA only
echo   phase3_mpca_ema\   — MPCA + EMASlideLoss
pause
