@echo off
chcp 65001 >nul
echo ============================================================
echo  Organoid-FL 本地环境搭建 (Windows + RTX 3060)
echo ============================================================
echo.

REM ── 1. 创建项目目录 ──
echo [1/5] 创建项目目录...
if not exist "organoid-fl" mkdir organoid-fl
cd organoid-fl
if not exist "data" mkdir data
if not exist "results" mkdir results
if not exist "results\fl_training" mkdir results\fl_training
if not exist "results\phase2_yolo" mkdir results\phase2_yolo
echo   ✓ 目录结构就绪

REM ── 2. 创建虚拟环境 ──
echo.
echo [2/5] 创建 Python 虚拟环境...
if not exist ".venv" (
    python -m venv .venv
    echo   ✓ 虚拟环境已创建
) else (
    echo   ✓ 虚拟环境已存在
)

REM ── 3. 激活并安装依赖 ──
echo.
echo [3/5] 安装 Python 依赖 (需要几分钟)...
call .venv\Scripts\activate.bat

pip install --upgrade pip -q
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 -q
pip install ultralytics -q
pip install pillow scikit-learn matplotlib -q
pip install streamlit plotly -q

echo   ✓ 依赖安装完成

REM ── 4. 验证环境 ──
echo.
echo [4/5] 验证环境...
python -c "import torch; print(f'  PyTorch {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '  ⚠️ GPU未检测到')" 2>nul
python -c "import ultralytics; print(f'  Ultralytics {ultralytics.__version__}')" 2>nul
python -c "from torchvision import models; print('  torchvision ✓')" 2>nul

REM ── 5. 检查数据集 ──
echo.
echo [5/5] 检查数据集...
if exist "data\organoid_patches" (
    echo   ✓ organoid_patches 已就绪
) else (
    echo   ⏳ organoid_patches 缺失 — 请从微信下载后解压到 data\organoid_patches\
)
if exist "data\intestinal_organoid" (
    echo   ✓ intestinal_organoid 已就绪
) else (
    echo   ⏳ intestinal_organoid 缺失 — 请从微信下载后解压到 data\intestinal_organoid\
)

echo.
echo ============================================================
echo  环境搭建完成！
echo ============================================================
echo.
echo  下一步:
echo    1. 把微信收到的 fl_classify_sim.py 和 fl_detect_sim.py 放到 organoid-fl\
echo    2. 把数据集文件夹放到 data\
echo    3. 运行快速测试:
echo       .venv\Scripts\activate
echo       python fl_classify_sim.py --data ./data/organoid_patches --quick --device 0
echo       python fl_detect_sim.py --data ./data/intestinal_organoid/OrganoidDataset/data.yaml --quick --device 0
echo.
pause
