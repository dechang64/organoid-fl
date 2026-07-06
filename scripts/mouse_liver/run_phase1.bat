@echo off
REM Phase 1: Perception 层验证 — bbox→SAM2→形态学特征
REM
REM 前置:
REM   - RF-DETR checkpoint: output\checkpoint_best_regular.pth (B1 8张训练)
REM   - SAM2 checkpoint: sam2_checkpoints\sam2_hiera_small.pt
REM   - B3 标注图: D:\datasets\mouse_liver_annotated_20260702\ (20张红色折线)
REM
REM 用法:
REM   cd C:\Users\decha\organoid-fl
REM   .\.venv\Scripts\activate
REM   scripts\mouse_liver\run_phase1.bat

cd /d C:\Users\decha\organoid-fl
call .venv\Scripts\activate

echo ============================================================
echo Phase 1: Perception 层验证
echo ============================================================

python scripts\mouse_liver\phase1_perception.py

echo.
echo ============================================================
echo 完成! 结果: runs\mouse_liver_phase1\phase1_results.json
echo ============================================================
pause
