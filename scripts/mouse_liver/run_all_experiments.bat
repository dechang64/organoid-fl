@echo off
REM 完整实验重跑 — 审计修复后全部重跑
REM
REM 旧代码问题: train=val=images + auto optimizer → 旧 E4-E8 和 baseline 不可比
REM 新代码: val 指向统一 val_set + 显式 AdamW lr0=0.002
REM
REM 实验矩阵:
REM   Baseline: B1/B2/B3/centralized @640 (2min+2min+7min+8min ≈ 19min)
REM   E9:       B3 @1280 (≈13min)
REM   E11:      centralized @1280 (≈32min)
REM   E4-E8:    顺序链式 FL 5组 (每组 ≈20min, 共 ≈100min)
REM   E10:      FL soft gate B3@1280 (≈25min)
REM
REM 总计: ≈ 3h
REM
REM 用法:
REM   scripts\mouse_liver\run_all_experiments.bat

cd /d C:\Users\decha\organoid-fl
call .venv\Scripts\activate

echo ============================================================
echo 删除旧结果 (代码已改, 旧结果不可比)
echo ============================================================
if exist "runs\mouse_liver_baseline\baseline_results.json" del "runs\mouse_liver_baseline\baseline_results.json"
if exist "runs\mouse_liver_fl_seq\E4_seq_none_results.json" del "runs\mouse_liver_fl_seq\E4_seq_none_results.json"
if exist "runs\mouse_liver_fl_seq\E5_seq_hard_results.json" del "runs\mouse_liver_fl_seq\E5_seq_hard_results.json"
if exist "runs\mouse_liver_fl_seq\E6_seq_soft_results.json" del "runs\mouse_liver_fl_seq\E6_seq_soft_results.json"
if exist "runs\mouse_liver_fl_seq\E7_seq_hard_rev_results.json" del "runs\mouse_liver_fl_seq\E7_seq_hard_rev_results.json"
if exist "runs\mouse_liver_fl_seq\E8_seq_local_results.json" del "runs\mouse_liver_fl_seq\E8_seq_local_results.json"
echo 已清理旧结果

echo ============================================================
echo Step 1/3: Baseline + E9 + E11
echo ============================================================
python scripts\mouse_liver\train_baseline.py

echo ============================================================
echo Step 2/3: E4-E8 顺序链式 FL 矩阵
echo ============================================================
echo.
echo [E4] gate=none, b1→b2→b3
python scripts\mouse_liver\fl_sequential.py --gate none --order b1_b2_b3 --tag E4_seq_none

echo.
echo [E5] gate=hard, b1→b2→b3, signal=mAP
python scripts\mouse_liver\fl_sequential.py --gate hard --order b1_b2_b3 --signal mAP --margin 0.0 --tag E5_seq_hard

echo.
echo [E6] gate=soft, b1→b2→b3, signal=mAP
python scripts\mouse_liver\fl_sequential.py --gate soft --order b1_b2_b3 --signal mAP --tag E6_seq_soft

echo.
echo [E7] gate=hard, b3→b2→b1, signal=mAP
python scripts\mouse_liver\fl_sequential.py --gate hard --order b3_b2_b1 --signal mAP --margin 0.0 --tag E7_seq_hard_rev

echo.
echo [E8] gate=local, b1→b2→b3, signal=mAP
python scripts\mouse_liver\fl_sequential.py --gate local --order b1_b2_b3 --signal mAP --margin 0.0 --tag E8_seq_local

echo ============================================================
echo Step 3/3: E10 FL soft gate B3@1280
echo ============================================================
python scripts\mouse_liver\fl_sequential.py --gate soft --order b1_b2_b3 --signal mAP --tag E10_soft_b3_1280 --b3-imgsz 1280

echo.
echo ============================================================
echo 全部完成! 结果文件:
echo   Baseline: runs\mouse_liver_baseline\baseline_results.json
echo   E4-E8:    runs\mouse_liver_fl_seq\E*_results.json
echo   E10:      runs\mouse_liver_fl_seq\E10_soft_b3_1280_results.json
echo ============================================================
pause
