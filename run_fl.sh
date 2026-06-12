#!/bin/bash
# Organoid-FL Simulation Runner (Linux/Mac)
# ==========================================

echo "========================================"
echo " Organoid-FL Quick Test"
echo "========================================"
echo

echo "[Phase 1] Classification FL (ResNet18, 2 rounds)..."
python3 fl_classify_sim.py --data ./organoid_patches --quick --device 0
echo

echo "[Phase 2] Detection FL (YOLOv12n, 2 rounds)..."
python3 fl_detect_sim.py --data ./intestinal_organoid/OrganoidDataset/data.yaml --quick --device 0
echo

echo "Done! Check fl_classify_results/ and fl_detect_results/"
