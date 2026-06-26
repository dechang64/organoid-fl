@echo off
REM Step 3: Run SAM2 morphology filter with FINETUNED SAM2
REM Usage: cd C:\Users\decha\organoid-fl && scripts\multiorg\run_sam2_test_finetuned.bat

cd /d C:\Users\decha\organoid-fl
call .\.venv\Scripts\activate

python scripts\multiorg\multiorg_sam2.py --weights output\checkpoint_best_regular.pth --model-variant small --src D:\datasets\mutliorg\MultiOrg_v2\test --dst results\multiorg_sam2_finetuned_v2 --windows 512 --overlap 0.3 --conf 0.25 --score-filter 0.3 --sam2-checkpoint runs\sam2_finetune_v2\sam2_finetuned.pt --max-images 5 --save-vis

pause
