@echo off
REM B3 imgsz 对比实验 — 验证"B3 目标尺度不匹配是 FL 更低的根因"
REM
REM 审计修复版: 删除旧 baseline_results.json 强制重跑
REM (旧结果用 train=val=images + auto optimizer, 和新代码不可比)
REM
REM E9:  B3 独立 imgsz=1280
REM E10: FL soft gate, B3@1280, B1/B2@640
REM E11: 集中式 imgsz=1280

cd /d C:\Users\decha\organoid-fl
call .venv\Scripts\activate

echo ============================================================
echo 删除旧 baseline_results.json (代码已改, 旧结果不可比)
echo ============================================================
if exist "runs\mouse_liver_baseline\baseline_results.json" (
    del "runs\mouse_liver_baseline\baseline_results.json"
    echo 已删除
) else (
    echo 不存在, 跳过
)

echo ============================================================
echo Step 1/2: Baseline (B1/B2/B3/centralized @640) + E9 + E11
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
