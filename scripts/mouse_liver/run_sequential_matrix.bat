@echo off
REM 顺序链式 FL 实验矩阵 — 5组核心实验
REM 在冬生本地运行: cd C:\Users\decha\organoid-fl && .\.venv\Scripts\activate

echo ============================================================
echo  顺序链式 FL 实验矩阵
echo ============================================================

REM E4: 顺序链式 + 无门控 (vs E1 并行 FedAvg)
echo.
echo [E4] sequential, gate=none, order=b1_b2_b3
python scripts\mouse_liver\fl_sequential.py --gate none --order b1_b2_b3 --tag E4_seq_none

REM E5: 顺序链式 + 硬门控 (冬生方案)
echo.
echo [E5] sequential, gate=hard, order=b1_b2_b3, signal=mAP, margin=0
python scripts\mouse_liver\fl_sequential.py --gate hard --order b1_b2_b3 --signal mAP --margin 0.0 --tag E5_seq_hard

REM E6: 顺序链式 + 软门控 (EWA 加权)
echo.
echo [E6] sequential, gate=soft, order=b1_b2_b3, signal=mAP
python scripts\mouse_liver\fl_sequential.py --gate soft --order b1_b2_b3 --signal mAP --tag E6_seq_soft

REM E7: 顺序链式 + 硬门控 + 反序 (B3先跑)
echo.
echo [E7] sequential, gate=hard, order=b3_b2_b1, signal=mAP, margin=0
python scripts\mouse_liver\fl_sequential.py --gate hard --order b3_b2_b1 --signal mAP --margin 0.0 --tag E7_seq_hard_rev

REM E8: 顺序链式 + 本地门控
echo.
echo [E8] sequential, gate=local, order=b1_b2_b3, signal=mAP, margin=0
python scripts\mouse_liver\fl_sequential.py --gate local --order b1_b2_b3 --signal mAP --margin 0.0 --tag E8_seq_local

echo.
echo ============================================================
echo  全部实验完成!
echo  结果: runs\mouse_liver_fl_seq\E*_results.json
echo  可视化: runs\mouse_liver_fl_seq\E*_convergence.png
echo ============================================================
pause
