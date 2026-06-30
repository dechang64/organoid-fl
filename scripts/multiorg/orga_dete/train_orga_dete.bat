@echo off
REM Orga-Dete Training: YOLOv11n + MPCA + EMASlideLoss
REM Phase 1: MPCA only (BiFPN + EMASlideLoss in later phases)
REM
REM Usage: scripts\multiorg\orga_dete\train_orga_dete.bat
REM Prerequisite: data.yaml at D:\datasets\MultiOrg_v4_640\data.yaml

cd /d C:\Users\decha\organoid-fl
call .\.venv\Scripts\activate

REM 验证数据集存在
if not exist "D:\datasets\MultiOrg_v4_640\data.yaml" (
    echo ERROR: data.yaml not found at D:\datasets\MultiOrg_v4_640\data.yaml
    pause
    exit /b 1
)

REM 训练 Orga-Dete (Phase 1: MPCA only)
python -c "import sys; sys.path.insert(0, 'scripts/multiorg/orga_dete'); from orga_dete_modules import MPCA; import ultralytics.nn.tasks as tasks; import ultralytics.nn.modules as modules; tasks.MPCA = MPCA; modules.MPCA = MPCA; from ultralytics import YOLO; model = YOLO('scripts/multiorg/orga_dete/orga_dete_yolo11n.yaml'); model.load('yolo11n.pt'); model.train(data='D:\\datasets\\MultiOrg_v4_640\\data.yaml', epochs=300, imgsz=640, batch=8, device=0, project='runs/orga_dete', name='phase1_mpca', cos_lr=True, close_mosaic=15, patience=50, label_smoothing=0.1, copy_paste=0.1, mixup=0.1)"

echo.
echo === Training complete! ===
echo Results: runs\orga_dete\phase1_mpca\
pause
