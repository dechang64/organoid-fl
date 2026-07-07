@echo off
REM 鼠肝 organoid v2 全量实验一键运行
REM 
REM 前提:
REM   1. 已运行 prepare_data.py 创建数据分配
REM   2. 已有 RF-DETR MultiOrg pretrained checkpoint (output\checkpoint_best_regular.pth)
REM   3. 已有 SAM2 checkpoint (sam2_hiera_small.pt)
REM   4. 冬生本地 Win11 + RTX 3060 12GB

set DATA_ROOT=D:\datasets\mouse_liver_split
set OUTPUT=runs\mouse_liver_v2
set PRETRAINED=output\checkpoint_best_regular.pth
set SAM2_CKPT=sam2_hiera_small.pt

echo ========================================
echo 鼠肝 organoid v2 全量实验
echo ========================================
echo Data root: %DATA_ROOT%
echo Output: %OUTPUT%
echo Pretrained: %PRETRAINED%
echo.

REM Step 1: 数据分配
echo [1/6] 数据分配...
python scripts\mouse_liver\v2\prepare_data.py --output %DATA_ROOT%
if errorlevel 1 goto error
echo.

REM Step 2: 全量训练 (RF-DETR)
echo [2/6] 全量训练 (RF-DETR)...
python scripts\mouse_liver\v2\train_full.py --batch all --data-root %DATA_ROOT% --pretrained %PRETRAINED% --output %OUTPUT%
if errorlevel 1 goto error
echo.

REM Step 3: few-shot 训练 (RF-DETR 3-shot)
echo [3/6] few-shot 训练 (3-shot)...
python scripts\mouse_liver\v2\train_fewshot.py --batch all --data-root %DATA_ROOT% --pretrained %PRETRAINED% --output %OUTPUT%
if errorlevel 1 goto error
echo.

REM Step 4: 传统CV基线
echo [4/6] 传统CV基线...
for %%b in (b1 b2 b3) do (
    python scripts\mouse_liver\traditional_cv.py --src %DATA_ROOT%\%%b\test\images --gt %DATA_ROOT%\%%b\test\labels --dst %OUTPUT%\%%b\traditional
)
echo.

REM Step 5: SAM2 分割 (zero-shot, 带 resolution 参数)
echo [5/6] SAM2 分割 (zero-shot)...
REM B1 用 544, B2/B3 用 768
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b1\full\checkpoint_best_regular.pth --src %DATA_ROOT%\b1\test\images --gt %DATA_ROOT%\b1\test\labels --dst %OUTPUT%\b1\sam2_full --sam2-checkpoint %SAM2_CKPT% --resolution 544
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b1\fewshot\checkpoint_best_regular.pth --src %DATA_ROOT%\b1\test\images --gt %DATA_ROOT%\b1\test\labels --dst %OUTPUT%\b1\sam2_fewshot --sam2-checkpoint %SAM2_CKPT% --resolution 544
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b2\full\checkpoint_best_regular.pth --src %DATA_ROOT%\b2\test\images --gt %DATA_ROOT%\b2\test\labels --dst %OUTPUT%\b2\sam2_full --sam2-checkpoint %SAM2_CKPT% --resolution 768
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b2\fewshot\checkpoint_best_regular.pth --src %DATA_ROOT%\b2\test\images --gt %DATA_ROOT%\b2\test\labels --dst %OUTPUT%\b2\sam2_fewshot --sam2-checkpoint %SAM2_CKPT% --resolution 768
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b3\full\checkpoint_best_regular.pth --src %DATA_ROOT%\b3\test\images --gt %DATA_ROOT%\b3\test\labels --dst %OUTPUT%\b3\sam2_full --sam2-checkpoint %SAM2_CKPT% --resolution 768
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b3\fewshot\checkpoint_best_regular.pth --src %DATA_ROOT%\b3\test\images --gt %DATA_ROOT%\b3\test\labels --dst %OUTPUT%\b3\sam2_fewshot --sam2-checkpoint %SAM2_CKPT% --resolution 768
echo.

REM Step 6: FL 实验 (4组)
echo [6/6] FL 实验...
python scripts\mouse_liver\v2\run_fl.py --gate none --order b1_b2_b3 --tag F1 --data-root %DATA_ROOT% --output %OUTPUT%\fl
python scripts\mouse_liver\v2\run_fl.py --gate soft --order b1_b2_b3 --tag F2 --data-root %DATA_ROOT% --output %OUTPUT%\fl
python scripts\mouse_liver\v2\run_fl.py --gate hard --order b1_b2_b3 --tag F3 --data-root %DATA_ROOT% --output %OUTPUT%\fl
python scripts\mouse_liver\v2\run_fl.py --gate hard --order b3_b2_b1 --tag F4 --data-root %DATA_ROOT% --output %OUTPUT%\fl
echo.

REM 评估
echo [评估]...
for %%b in (b1 b2 b3) do (
    python scripts\mouse_liver\v2\evaluate.py --batch %%b --weights %OUTPUT%\%%b\full\checkpoint_best_regular.pth --data-root %DATA_ROOT% --output %OUTPUT%
    python scripts\mouse_liver\v2\evaluate.py --batch %%b --weights %OUTPUT%\%%b\fewshot\checkpoint_best_regular.pth --data-root %DATA_ROOT% --output %OUTPUT%
)
echo.

echo ========================================
echo 全部实验完成!
echo ========================================
goto end

:error
echo.
echo [ERROR] 实验失败, 请检查日志

:end
pause
