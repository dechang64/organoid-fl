@echo off
REM Organoid-FL Simulation Runner (Windows + RTX 3060)
REM ===================================================
REM
REM Phase 1: Classification (ResNet18, 23K patches)
REM   python fl_classify_sim.py --data ./organoid_patches --rounds 10 --epochs 5 --device 0
REM
REM Phase 2: Detection (YOLOv12n, 840 images)
REM   python fl_detect_sim.py --data ./intestinal_organoid/OrganoidDataset/data.yaml --device 0
REM
REM Quick test (2 rounds, 2 epochs):
REM   python fl_classify_sim.py --data ./organoid_patches --quick --device 0
REM   python fl_detect_sim.py --data ./intestinal_organoid/OrganoidDataset/data.yaml --quick --device 0
REM
REM Full experiment matrix (4 Non-IID × 3 strategies):
REM   python fl_classify_sim.py --data ./organoid_patches --matrix --device 0

echo ========================================
echo  Organoid-FL Quick Test
echo ========================================
echo.

echo [Phase 1] Classification FL (ResNet18, 2 rounds)...
python fl_classify_sim.py --data ./organoid_patches --quick --device 0
echo.

echo [Phase 2] Detection FL (YOLOv12n, 2 rounds)...
python fl_detect_sim.py --data ./intestinal_organoid/OrganoidDataset/data.yaml --quick --device 0
echo.

echo Done! Check fl_classify_results/ and fl_detect_results/
pause
