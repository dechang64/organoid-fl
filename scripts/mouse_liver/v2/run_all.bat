@echo off
chcp 65001 >nul
REM Mouse Liver Organoid v2 - Full Experiment Suite (14 experiments)
REM
REM Design (confirmed 2026-07-06):
REM   Step 1: Baseline + Ceiling (4)
REM     1. B1 baseline: 6 train, COCO pretrained, RF-DETR
REM     2. B2 baseline: 6 train, COCO pretrained, RF-DETR
REM     3. B3 baseline: 12 train, COCO pretrained, RF-DETR
REM     4. Ceiling: 24 train (centralized), COCO pretrained, RF-DETR
REM   Step 2: Cross-domain Transfer (4)
REM     5. B1->B2 zeroshot: B1 model directly infer on B2
REM     6. B1->B2 few-shot: B1 checkpoint + B2 3-shot fine-tune
REM     7. B1->B3 zeroshot: B1 model directly infer on B3
REM     8. B1->B3 few-shot: B1 checkpoint + B3 3-shot fine-tune
REM   Step 3: Federated Learning (6)
REM     9-14. FL strategy comparison + order comparison
REM
REM Prerequisites:
REM   1. batch1/2/3 data in D:\datasets\mouse_liver_correct\
REM   2. SAM2 checkpoint in sam2_checkpoints\sam2_hiera_small.pt
REM   3. Win11 + RTX 3060 12GB

set SRC_ROOT=D:\datasets\mouse_liver_correct
set DATA_ROOT=D:\datasets\mouse_liver_split
set OUTPUT=runs\mouse_liver_v2
set SAM2_CKPT=sam2_checkpoints\sam2_hiera_small.pt
set B1_CKPT=%OUTPUT%\b1\full\checkpoint_best_regular.pth

echo ========================================
echo Mouse Liver Organoid v2 (14 experiments)
echo ========================================
echo Source: %SRC_ROOT%
echo Data:   %DATA_ROOT%
echo Output: %OUTPUT%
echo.

REM === Step 1: Baseline + Ceiling ===
echo [1/8] Data split...
python scripts\mouse_liver\v2\prepare_data.py --data-root %SRC_ROOT% --output %DATA_ROOT%
if errorlevel 1 goto error
echo.

echo [2/8] Baseline training (B1+B2+B3, COCO pretrained)...
python scripts\mouse_liver\v2\train_full.py --batch all --data-root %DATA_ROOT% --output %OUTPUT%
if errorlevel 1 goto error
echo.

echo [3/8] Ceiling training (centralized, B1+B2+B3 merged)...
python scripts\mouse_liver\v2\train_central.py --data-root %DATA_ROOT% --output %OUTPUT%
if errorlevel 1 goto error
echo.

REM === Step 2: Cross-domain Transfer ===
echo [4/8] B1->B2/B3 zeroshot evaluation...
python scripts\mouse_liver\v2\evaluate.py --batch b2 --weights %B1_CKPT% --data-root %DATA_ROOT% --output %OUTPUT% --tag b1_to_b2_zeroshot
python scripts\mouse_liver\v2\evaluate.py --batch b3 --weights %B1_CKPT% --data-root %DATA_ROOT% --output %OUTPUT% --tag b1_to_b3_zeroshot
if errorlevel 1 goto error
echo.

echo [5/8] B1->B2/B3 few-shot fine-tune (B1 checkpoint + 3 images)...
python scripts\mouse_liver\v2\train_fewshot.py --target all --data-root %DATA_ROOT% --b1-ckpt %B1_CKPT% --output %OUTPUT%
if errorlevel 1 goto error
echo.

REM === Traditional CV + SAM2 ===
echo [6/8] Traditional CV baseline...
for %%b in (b1 b2 b3) do (
    python scripts\mouse_liver\traditional_cv.py --src %DATA_ROOT%\%%b\test\images --gt %DATA_ROOT%\%%b\test\labels --dst %OUTPUT%\%%b\traditional
)
if errorlevel 1 goto error
echo.

echo [7/8] SAM2 segmentation (zero-shot)...
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b1\full\checkpoint_best_regular.pth --src %DATA_ROOT%\b1\test\images --gt %DATA_ROOT%\b1\test\labels --dst %OUTPUT%\b1\sam2_full --sam2-checkpoint %SAM2_CKPT% --resolution 544
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b2\full\checkpoint_best_regular.pth --src %DATA_ROOT%\b2\test\images --gt %DATA_ROOT%\b2\test\labels --dst %OUTPUT%\b2\sam2_full --sam2-checkpoint %SAM2_CKPT% --resolution 768
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b3\full\checkpoint_best_regular.pth --src %DATA_ROOT%\b3\test\images --gt %DATA_ROOT%\b3\test\labels --dst %OUTPUT%\b3\sam2_full --sam2-checkpoint %SAM2_CKPT% --resolution 768
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b2\fewshot\checkpoint_best_regular.pth --src %DATA_ROOT%\b2\test\images --gt %DATA_ROOT%\b2\test\labels --dst %OUTPUT%\b2\sam2_fewshot --sam2-checkpoint %SAM2_CKPT% --resolution 768
python scripts\mouse_liver\sam2_segment.py --weights %OUTPUT%\b3\fewshot\checkpoint_best_regular.pth --src %DATA_ROOT%\b3\test\images --gt %DATA_ROOT%\b3\test\labels --dst %OUTPUT%\b3\sam2_fewshot --sam2-checkpoint %SAM2_CKPT% --resolution 768
if errorlevel 1 goto error
echo.

REM === Step 3: Federated Learning ===
echo [8/8] FL experiments (4 groups)...
python scripts\mouse_liver\v2\run_fl.py --gate none --order b1_b2_b3 --tag F1 --data-root %DATA_ROOT% --output %OUTPUT%\fl
python scripts\mouse_liver\v2\run_fl.py --gate soft --order b1_b2_b3 --tag F2 --data-root %DATA_ROOT% --output %OUTPUT%\fl
python scripts\mouse_liver\v2\run_fl.py --gate hard --order b1_b2_b3 --tag F3 --data-root %DATA_ROOT% --output %OUTPUT%\fl
python scripts\mouse_liver\v2\run_fl.py --gate hard --order b3_b2_b1 --tag F4 --data-root %DATA_ROOT% --output %OUTPUT%\fl
if errorlevel 1 goto error
echo.

REM === Final Evaluation ===
echo [Eval] Baseline + Ceiling + few-shot...
REM full eval: all batches
for %%b in (b1 b2 b3) do (
    python scripts\mouse_liver\v2\evaluate.py --batch %%b --weights %OUTPUT%\%%b\full\checkpoint_best_regular.pth --data-root %DATA_ROOT% --output %OUTPUT%
)
REM fewshot eval: only b2/b3 (B1->B2/B3 transfer, no b1 fewshot)
python scripts\mouse_liver\v2\evaluate.py --batch b2 --weights %OUTPUT%\b2\fewshot\checkpoint_best_regular.pth --data-root %DATA_ROOT% --output %OUTPUT%
python scripts\mouse_liver\v2\evaluate.py --batch b3 --weights %OUTPUT%\b3\fewshot\checkpoint_best_regular.pth --data-root %DATA_ROOT% --output %OUTPUT%
REM central eval: resolution=640 (training resolution)
python scripts\mouse_liver\v2\evaluate.py --batch all --weights %OUTPUT%\central\checkpoint_best_regular.pth --data-root %DATA_ROOT% --output %OUTPUT% --tag central --resolution 640
echo.

echo ========================================
echo All 14 experiments done!
echo ========================================
goto end

:error
echo.
echo [ERROR] Experiment failed, check log

:end
pause
