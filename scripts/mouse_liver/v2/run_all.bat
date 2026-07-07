@echo off
REM 鼠肝 organoid v2 全量实验一键运行
REM 
REM 前提:
REM   1. batch1/2/3 数据在 D:\datasets\mouse_liver_correct\ (含 images/ 和 labels/)
REM   2. SAM2 checkpoint 在 sam2_checkpoints\sam2_hiera_small.pt
REM   3. 冬生本地 Win11 + RTX 3060 12GB
REM
REM RF-DETR 用 COCO 预训练从头训练, 不用 MultiOrg checkpoint
REM
REM 输出路径:
REM   DATA_ROOT (D:\datasets\mouse_liver_split) — 数据分配
REM   OUTPUT (runs\mouse_liver_v2) — 所有训练结果
REM     {b1,b2,b3}\full\ — RF-DETR 全量训练 checkpoint
REM     {b1,b2,b3}\fewshot\ — RF-DETR 3-shot 训练 checkpoint
REM     {b1,b2,b3}\traditional\ — 传统CV结果
REM     {b1,b2,b3}\sam2_full\ — SAM2 分割结果 (full)
REM     {b1,b2,b3}\sam2_fewshot\ — SAM2 分割结果 (fewshot)
REM     {b1,b2,b3}\{full,fewshot}\eval_test.json — bbox 评估
REM     fl\{F1,F2,F3,F4}\ — FL 实验结果

set SRC_ROOT=D:\datasets\mouse_liver_correct
set DATA_ROOT=D:\datasets\mouse_liver_split
set OUTPUT=runs\mouse_liver_v2
set SAM2_CKPT=sam2_checkpoints\sam2_hiera_small.pt

echo ========================================
echo 鼠肝 organoid v2 全量实验
echo ========================================
echo Source: %SRC_ROOT%
echo Data root: %DATA_ROOT%
echo Output: %OUTPUT%
echo SAM2: %SAM2_CKPT%
echo.

REM Step 1: 数据分配
echo [1/7] 数据分配...
python scripts\mouse_liver\v2\prepare_data.py --data-root %SRC_ROOT% --output %DATA_ROOT%
if errorlevel 1 goto error
echo.

REM Step 2: 全量训练 (RF-DETR, COCO 预训练)
echo [2/7] 全量训练 (RF-DETR, COCO pretrained)...
python scripts\mouse_liver\v2\train_full.py --batch all --data-root %DATA_ROOT% --output %OUTPUT%
if errorlevel 1 goto error
echo.

REM Step 3: few-shot 训练 (RF-DETR 3-shot, COCO 预训练)
echo [3/7] few-shot 训练 (3-shot, COCO pretrained)...
python scripts\mouse_liver\v2\train_fewshot.py --batch all --data-root %DATA_ROOT% --output %OUTPUT%
if errorlevel 1 goto error
echo.

REM Step 4: 传统CV基线
echo [4/7] 传统CV基线...
for %%b in (b1 b2 b3) do (
    python scripts\mouse_liver\traditional_cv.py --src %DATA_ROOT%\%%b\test\images --gt %DATA_ROOT%\%%b\test\labels --dst %OUTPUT%\%%b\traditional
)
echo.

REM Step 5: SAM2 分割 (zero-shot, 带 resolution 参数)
echo [5/7] SAM2 分割 (zero-shot)...
REM B1 用 544, B2/B3 用 768
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b1\full\checkpoint_best_regular.pth --src %DATA_ROOT%\b1\test\images --gt %DATA_ROOT%\b1\test\labels --dst %OUTPUT%\b1\sam2_full --sam2-checkpoint %SAM2_CKPT% --resolution 544
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b1\fewshot\checkpoint_best_regular.pth --src %DATA_ROOT%\b1\test\images --gt %DATA_ROOT%\b1\test\labels --dst %OUTPUT%\b1\sam2_fewshot --sam2-checkpoint %SAM2_CKPT% --resolution 544
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b2\full\checkpoint_best_regular.pth --src %DATA_ROOT%\b2\test\images --gt %DATA_ROOT%\b2\test\labels --dst %OUTPUT%\b2\sam2_full --sam2-checkpoint %SAM2_CKPT% --resolution 768
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b2\fewshot\checkpoint_best_regular.pth --src %DATA_ROOT%\b2\test\images --gt %DATA_ROOT%\b2\test\labels --dst %OUTPUT%\b2\sam2_fewshot --sam2-checkpoint %SAM2_CKPT% --resolution 768
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b3\full\checkpoint_best_regular.pth --src %DATA_ROOT%\b3\test\images --gt %DATA_ROOT%\b3\test\labels --dst %OUTPUT%\b3\sam2_full --sam2-checkpoint %SAM2_CKPT% --resolution 768
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b3\fewshot\checkpoint_best_regular.pth --src %DATA_ROOT%\b3\test\images --gt %DATA_ROOT%\b3\test\labels --dst %OUTPUT%\b3\sam2_fewshot --sam2-checkpoint %SAM2_CKPT% --resolution 768
echo.

REM Step 6: FL 实验 (4组)
echo [6/7] FL 实验...
python scripts\mouse_liver\v2\run_fl.py --gate none --order b1_b2_b3 --tag F1 --data-root %DATA_ROOT% --output %OUTPUT%\fl
python scripts\mouse_liver\v2\run_fl.py --gate soft --order b1_b2_b3 --tag F2 --data-root %DATA_ROOT% --output %OUTPUT%\fl
python scripts\mouse_liver\v2\run_fl.py --gate hard --order b1_b2_b3 --tag F3 --data-root %DATA_ROOT% --output %OUTPUT%\fl
python scripts\mouse_liver\v2\run_fl.py --gate hard --order b3_b2_b1 --tag F4 --data-root %DATA_ROOT% --output %OUTPUT%\fl
echo.

REM Step 7: 评估
echo [7/7] 评估...
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
