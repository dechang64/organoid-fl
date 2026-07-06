@echo off
REM B3 imgsz 对比实验 — 验证"B3 目标尺度不匹配是 FL 更低的根因"
REM
REM E9:  B3 独立 imgsz=1280 (baseline 脚本里)
REM E10: FL soft gate, B3@1280, B1/B2@640
REM E11: 集中式 imgsz=1280 (baseline 脚本里)
REM
REM 用法:
REM   1. 先跑 baseline (含 E9+E11):
REM      python scripts\mouse_liver\train_baseline.py
REM   2. 再跑 E10 (FL B3@1280):
REM      scripts\mouse_liver\run_b3_imgsz.bat
REM
REM 或者直接跑这个 bat 会依次执行两步

cd /d C:\Users\decha\organoid-fl
call .venv\Scripts\activate

echo ============================================================
echo Step 1/2: Baseline + E9 (B3@1280) + E11 (centralized@1280)
echo ============================================================
python scripts\mouse_liver\train_baseline.py

echo ============================================================
echo Step 2/2: E10 — FL soft gate, B3@1280
echo ============================================================
python scripts\mouse_liver\fl_sequential.py --gate soft --order b1_b2_b3 --signal mAP --tag E10_soft_b3_1280 --b3-imgsz 1280

echo.
echo ============================================================
echo 全部完成! 结果文件:
echo   Baseline: runs\mouse_liver_baseline\baseline_results.json
echo   E10:      runs\mouse_liver_fl_seq\E10_soft_b3_1280_results.json
echo ============================================================
pause
