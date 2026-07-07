@echo off
REM 鼠肝 organoid v2 全量实验 (14个实验, 三步走)
REM
REM 实验设计 (2026-07-06 确认):
REM   Step 1: 底线和天花板 (4个)
REM     1. B1 底线: 6 train, COCO pretrained, RF-DETR
REM     2. B2 底线: 6 train, COCO pretrained, RF-DETR
REM     3. B3 底线: 12 train, COCO pretrained, RF-DETR
REM     4. 天花板: 24 train (集中式), COCO pretrained, RF-DETR
REM   Step 2: 跨域迁移 (4个)
REM     5. B1→B2 zeroshot: B1 模型直接推理 B2
REM     6. B1→B2 few-shot: B1 checkpoint + B2 3张微调
REM     7. B1→B3 zeroshot: B1 模型直接推理 B3
REM     8. B1→B3 few-shot: B1 checkpoint + B3 3张微调
REM   Step 3: 联邦学习 (6个)
REM     9-14. FL 策略对比 + 顺序对比
REM
REM 前提:
REM   1. batch1/2/3 数据在 D:\datasets\mouse_liver_correct\
REM   2. SAM2 checkpoint 在 sam2_checkpoints\sam2_hiera_small.pt
REM   3. 冬生本地 Win11 + RTX 3060 12GB

set SRC_ROOT=D:\datasets\mouse_liver_correct
set DATA_ROOT=D:\datasets\mouse_liver_split
set OUTPUT=runs\mouse_liver_v2
set SAM2_CKPT=sam2_checkpoints\sam2_hiera_small.pt
set B1_CKPT=%OUTPUT%\b1\full\checkpoint_best_regular.pth

echo ========================================
echo 鼠肝 organoid v2 全量实验 (14个)
echo ========================================
echo Source: %SRC_ROOT%
echo Data: %DATA_ROOT%
echo Output: %OUTPUT%
echo.

REM === Step 1: 底线和天花板 ===
echo [1/8] 数据分配...
python scripts\mouse_liver\v2\prepare_data.py --data-root %SRC_ROOT% --output %DATA_ROOT%
if errorlevel 1 goto error
echo.

echo [2/8] 底线训练 (B1+B2+B3, COCO pretrained)...
python scripts\mouse_liver\v2\train_full.py --batch all --data-root %DATA_ROOT% --output %OUTPUT%
if errorlevel 1 goto error
echo.

echo [3/8] 天花板训练 (集中式, B1+B2+B3 合并)...
python scripts\mouse_liver\v2\train_central.py --data-root %DATA_ROOT% --output %OUTPUT%
if errorlevel 1 goto error
echo.

REM === Step 2: 跨域迁移 ===
echo [4/8] B1→B2/B3 zeroshot 评估 (B1 模型直接推理)...
python scripts\mouse_liver\v2\evaluate.py --batch b2 --weights %B1_CKPT% --data-root %DATA_ROOT% --output %OUTPUT% --tag b1_to_b2_zeroshot
python scripts\mouse_liver\v2\evaluate.py --batch b3 --weights %B1_CKPT% --data-root %DATA_ROOT% --output %OUTPUT% --tag b1_to_b3_zeroshot
if errorlevel 1 goto error
echo.

echo [5/8] B1→B2/B3 few-shot 微调 (B1 checkpoint + 3张)...
python scripts\mouse_liver\v2\train_fewshot.py --target all --data-root %DATA_ROOT% --b1-ckpt %B1_CKPT% --output %OUTPUT%
if errorlevel 1 goto error
echo.

REM === 传统CV + SAM2 ===
echo [6/8] 传统CV 基线...
for %%b in (b1 b2 b3) do (
    python scripts\mouse_liver\traditional_cv.py --src %DATA_ROOT%\%%b\test\images --gt %DATA_ROOT%\%%b\test\labels --dst %OUTPUT%\%%b\traditional
)
if errorlevel 1 goto error
echo.

echo [7/8] SAM2 分割 (zero-shot)...
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b1\full\checkpoint_best_regular.pth --src %DATA_ROOT%\b1\test\images --gt %DATA_ROOT%\b1\test\labels --dst %OUTPUT%\b1\sam2_full --sam2-checkpoint %SAM2_CKPT% --resolution 544
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b2\full\checkpoint_best_regular.pth --src %DATA_ROOT%\b2\test\images --gt %DATA_ROOT%\b2\test\labels --dst %OUTPUT%\b2\sam2_full --sam2-checkpoint %SAM2_CKPT% --resolution 768
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b3\full\checkpoint_best_regular.pth --src %DATA_ROOT%\b3\test\images --gt %DATA_ROOT%\b3\test\labels --dst %OUTPUT%\b3\sam2_full --sam2-checkpoint %SAM2_CKPT% --resolution 768
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b2\fewshot\checkpoint_best_regular.pth --src %DATA_ROOT%\b2\test\images --gt %DATA_ROOT%\b2\test\labels --dst %OUTPUT%\b2\sam2_fewshot --sam2-checkpoint %SAM2_CKPT% --resolution 768
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b3\fewshot\checkpoint_best_regular.pth --src %DATA_ROOT%\b3\test\images --gt %DATA_ROOT%\b3\test\labels --dst %OUTPUT%\b3\sam2_fewshot --sam2-checkpoint %SAM2_CKPT% --resolution 768
if errorlevel 1 goto error
echo.

REM === Step 3: 联邦学习 ===
echo [8/8] FL 实验 (6组)...
python scripts\mouse_liver\v2\run_fl.py --gate none --order b1_b2_b3 --tag F1 --data-root %DATA_ROOT% --output %OUTPUT%\fl
python scripts\mouse_liver\v2\run_fl.py --gate soft --order b1_b2_b3 --tag F2 --data-root %DATA_ROOT% --output %OUTPUT%\fl
python scripts\mouse_liver\v2\run_fl.py --gate hard --order b1_b2_b3 --tag F3 --data-root %DATA_ROOT% --output %OUTPUT%\fl
python scripts\mouse_liver\v2\run_fl.py --gate hard --order b3_b2_b1 --tag F4 --data-root %DATA_ROOT% --output %OUTPUT%\fl
if errorlevel 1 goto error
echo.

REM === 最终评估 ===
echo [评估] 底线 + 天花板 + few-shot...
REM full 评估: 所有 batch
for %%b in (b1 b2 b3) do (
    python scripts\mouse_liver\v2\evaluate.py --batch %%b --weights %OUTPUT%\%%b\full\checkpoint_best_regular.pth --data-root %DATA_ROOT% --output %OUTPUT%
)
REM fewshot 评估: 只有 b2/b3 (B1→B2/B3 跨域迁移, 没有 b1 fewshot)
python scripts\mouse_liver\v2\evaluate.py --batch b2 --weights %OUTPUT%\b2\fewshot\checkpoint_best_regular.pth --data-root %DATA_ROOT% --output %OUTPUT%
python scripts\mouse_liver\v2\evaluate.py --batch b3 --weights %OUTPUT%\b3\fewshot\checkpoint_best_regular.pth --data-root %DATA_ROOT% --output %OUTPUT%
REM 集中式评估: 用 resolution=640 (训练时用的)
python scripts\mouse_liver\v2\evaluate.py --batch all --weights %OUTPUT%\central\checkpoint_best_regular.pth --data-root %DATA_ROOT% --output %OUTPUT% --tag central --resolution 640
echo.

echo ========================================
echo 全部 14 个实验完成!
echo ========================================
goto end

:error
echo.
echo [ERROR] 实验失败, 请检查日志

:end
pause
