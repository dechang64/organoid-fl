@echo off
REM 完整实验重跑 — 数据修正后全部重跑
REM
REM 数据修正: B2 从 2592×1944(旧错误) → 4000×3000(正确)
REM 旧 B2 用了 B1 标注图(有红色折线), 训练数据被污染
REM 所有旧实验结果无效, 全部重跑
REM
REM 实验矩阵:
REM   Step 1: Baseline (B1/B2/B3/centralized @640) + E9 (B3@1280) + E11 (centralized@1280)
REM   Step 2: E4-E8 顺序链式 FL 矩阵 (5组)
REM   Step 3: E10 FL soft gate B3@1280
REM   Step 4: Phase 1 Perception (RF-DETR + SAM2 + 形态学特征)
REM
REM 总计: ≈ 3.5h
REM
REM 用法:
REM   scripts\mouse_liver\run_all_experiments.bat

cd /d C:\Users\decha\organoid-fl
call .venv\Scripts\activate

echo ============================================================
echo 清理所有旧结果 (数据已修正, 旧结果全部无效)
echo ============================================================

REM 删除整个 runs 目录 (checkpoint + fl_split + val_set + json 全部清除)
if exist "runs\mouse_liver_baseline" rmdir /s /q "runs\mouse_liver_baseline"
if exist "runs\mouse_liver_fl_seq" rmdir /s /q "runs\mouse_liver_fl_seq"
if exist "runs\mouse_liver_fl" rmdir /s /q "runs\mouse_liver_fl"
if exist "runs\mouse_liver_phase1" rmdir /s /q "runs\mouse_liver_phase1"

REM 删除各 batch 下的 fl_split (旧数据残留)
if exist "D:\datasets\mouse_liver_correct\batch1\fl_split" rmdir /s /q "D:\datasets\mouse_liver_correct\batch1\fl_split"
if exist "D:\datasets\mouse_liver_correct\batch2\fl_split" rmdir /s /q "D:\datasets\mouse_liver_correct\batch2\fl_split"
if exist "D:\datasets\mouse_liver_correct\batch3\fl_split" rmdir /s /q "D:\datasets\mouse_liver_correct\batch3\fl_split"

echo 已清理所有旧结果

echo ============================================================
echo Step 1/4: Baseline + E9 (B3@1280) + E11 (centralized@1280)
echo ============================================================
python scripts\mouse_liver\train_baseline.py

echo ============================================================
echo Step 2/4: E4-E8 顺序链式 FL 矩阵
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
echo Step 3/4: E10 FL soft gate B3@1280
echo ============================================================
python scripts\mouse_liver\fl_sequential.py --gate soft --order b1_b2_b3 --signal mAP --tag E10_soft_b3_1280 --b3-imgsz 1280

echo ============================================================
echo Step 4/4: Phase 1 Perception 层验证
echo ============================================================
python scripts\mouse_liver\phase1_perception.py

echo.
echo ============================================================
echo 全部完成! 结果文件:
echo   Baseline:  runs\mouse_liver_baseline\baseline_results.json
echo   E4-E8:     runs\mouse_liver_fl_seq\E*_results.json
echo   E10:       runs\mouse_liver_fl_seq\E10_soft_b3_1280_results.json
echo   Phase 1:   runs\mouse_liver_phase1\phase1_results.json
echo ============================================================
pause
