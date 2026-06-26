@echo off
REM MultiOrg SAM2 - 5 image quick test
REM Usage: cd C:\Users\decha\organoid-fl && multiorg_sam2_test.bat

cd /d C:\Users\decha\organoid-fl
call .\.venv\Scripts\activate

python scripts\multiorg\multiorg_sam2.py --weights output\small_512\checkpoint_best_regular.pth --model-variant small --src D:\datasets\mutliorg\MultiOrg_v2\test --dst results\multiorg_sam2 --windows 512 --overlap 0.3 --conf 0.25 --score-filter 0.3 --sam2-checkpoint sam2_hiera_small.pt --max-images 5 --save-vis

pause
