@echo off
REM SAM2 Self-Distillation: Zero-shot pseudo label → finetune
REM Step 1: Generate pseudo masks with zero-shot SAM2
REM Step 2: Finetune SAM2 with pseudo masks

cd /d C:\Users\decha\organoid-fl
call .\.venv\Scripts\activate

echo ============================================================
echo Step 1: Generate pseudo-label data (zero-shot SAM2)
echo ============================================================
python scripts\multiorg\prepare_sam2_pseudo.py ^
    --src D:\datasets\mutliorg\MultiOrg_v2\train ^
    --dst data\multiorg_sam2_pseudo ^
    --sam2-checkpoint sam2_checkpoints\sam2_hiera_small.pt ^
    --annotator Annotator_A

echo.
echo ============================================================
echo Step 2: Finetune SAM2 with pseudo masks
echo ============================================================
python scripts\multiorg\finetune_sam2.py ^
    --data data\multiorg_sam2_pseudo ^
    --checkpoint sam2_checkpoints\sam2_hiera_small.pt ^
    --dst runs\sam2_finetune_pseudo ^
    --epochs 5 --lr 1e-5 --batch-size 4

echo.
echo Done! Finetuned model: runs\sam2_finetune_pseudo\sam2_finetuned.pt
pause
