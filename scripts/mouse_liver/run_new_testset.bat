@echo off
REM Mouse Liver New Test Set Evaluation
REM Usage: double-click or run in PowerShell

cd /d C:\Users\decha\organoid-fl
call .\.venv\Scripts\activate

REM === Configuration ===
set WEIGHTS=runs\mouse_liver_fewshot\checkpoint_best_regular.pth
set SAM2_CKPT=sam2_checkpoints\sam2_hiera_small.pt
set SRC=D:\datasets\mouse_liver_new\orig
set ANNOT=D:\datasets\mouse_liver_new\annot
set DST=results\mouse_liver_new_testset

REM === Create data dirs if needed ===
if not exist "%SRC%" mkdir "%SRC%"
if not exist "%ANNOT%" mkdir "%ANNOT%"

echo ============================================================
echo Mouse Liver New Test Set Evaluation
echo ============================================================
echo Weights:  %WEIGHTS%
echo SAM2:     %SAM2_CKPT%
echo Orig:     %SRC%
echo Annot:    %ANNOT%
echo Output:   %DST%
echo ============================================================
echo.
echo [IMPORTANT] Put 10 original images in %SRC%
echo [IMPORTANT] Put 10 annotated images in %ANNOT%
echo Images are paired by sorted filename order.
echo.
pause

python scripts\mouse_liver\eval_new_testset.py ^
    --weights "%WEIGHTS%" ^
    --src "%SRC%" ^
    --annot "%ANNOT%" ^
    --sam2-checkpoint "%SAM2_CKPT%" ^
    --dst "%DST%" ^
    --threshold 0.25

echo.
echo Done! Results in %DST%
pause
