@echo off
REM Step 2: Finetune SAM2 mask decoder on MultiOrg
REM Usage: cd C:\Users\decha\organoid-fl && scripts\multiorg\run_sam2_finetune.bat

cd /d C:\Users\decha\organoid-fl
call .\.venv\Scripts\activate

python scripts\multiorg\finetune_sam2.py --data data\multiorg_sam2 --checkpoint sam2_checkpoints\sam2_hiera_small.pt --dst runs\sam2_finetune --epochs 5 --lr 1e-5 --batch-size 2

pause
