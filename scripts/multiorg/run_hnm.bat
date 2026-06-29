@echo off
REM Hard Negative Mining - RF-DETR retraining with FP negatives
REM Usage: cd C:\Users\decha\organoid-fl && scripts\multiorg\run_hnm.bat

cd /d C:\Users\decha\organoid-fl
call .\.venv\Scripts\activate

python scripts\multiorg\hard_negative_mining.py --checkpoint output\checkpoint_best_regular.pth --data-yaml D:\datasets\MultiOrg_v4_640\data.yaml --model-variant small --output-dir runs\rfdetr_hnm --hnm-epochs 50 --imgsz 512 --iou-threshold 0.3 --conf 0.25

echo.
echo === Done! Output: runs\rfdetr_hnm\ ===
dir runs\rfdetr_hnm\
pause
