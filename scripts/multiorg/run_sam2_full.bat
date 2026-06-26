@echo off
REM MultiOrg SAM2 - full 55 images
REM Usage: cd C:\Users\decha\organoid-fl && multiorg_sam2_full.bat

cd /d C:\Users\decha\organoid-fl
call .\.venv\Scripts\activate

python scripts\multiorg\multiorg_sam2.py --weights output\checkpoint_best_regular.pth --model-variant small --src D:\datasets\mutliorg\MultiOrg_v2\test --dst results\multiorg_sam2_full --windows 512 --overlap 0.3 --conf 0.25 --score-filter 0.3 --sam2-checkpoint sam2_hiera_small.pt

pause
