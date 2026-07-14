@echo off
chcp 65001 >nul

REM Switch to project root (2 levels up from this bat)
cd /d %~dp0\..\..

echo ============================================================
echo  Cross-Domain Eval: MultiOrg Slot Model -> Intestinal Organoid
echo ============================================================
echo.
echo  Working dir: %CD%
echo.

REM ============================================================
REM Config - modify paths here if needed
REM ============================================================
set CKPT=results\supcon_8s_d128_p256_t0.07_b0.1_20260713_003826\best.pt
set YOLO_CKPT=runs\detect\train\weights\best.pt
set DATA_ROOT=data\intestinal_organoid\OrganoidDataset
set OUT_ZIP=results\cross_domain_intestinal.zip

REM Verify checkpoint exists
if not exist %CKPT% (
    echo [ERROR] SupCon checkpoint not found: %CKPT%
    echo Please check the path.
    pause
    exit /b 1
)

REM Verify YOLO checkpoint exists
if not exist %YOLO_CKPT% (
    echo [ERROR] YOLO checkpoint not found: %YOLO_CKPT%
    echo This should be the yolo12s checkpoint trained on intestinal_organoid.
    echo Typical locations:
    echo   runs\detect\train\weights\best.pt
    echo   runs\detect\yolo12s_freebies\weights\best.pt
    echo Update YOLO_CKPT in this bat if needed.
    pause
    exit /b 1
)

REM Verify data root exists
if not exist %DATA_ROOT% (
    echo [ERROR] Data root not found: %DATA_ROOT%
    echo Current dir: %CD%
    pause
    exit /b 1
)

echo  SupCon checkpoint: %CKPT%
echo  YOLO checkpoint:   %YOLO_CKPT%
echo  Data root:         %DATA_ROOT%
echo.

REM ============================================================
REM Step 1: Generate crops from val set
REM ============================================================
echo [1/2] Generating intestinal val crops...
python scripts\mouse_liver\generate_intestinal_crops.py --weights %YOLO_CKPT% --data-root %DATA_ROOT% --split val --dst data\intestinal_crops\val --imgsz 1088
if %errorlevel% neq 0 (
    echo [ERROR] Crop generation failed!
    pause
    exit /b 1
)
echo.

REM ============================================================
REM Step 2: Cross-domain eval
REM ============================================================
echo [2/2] Cross-domain eval...
python scripts\mouse_liver\cross_domain_eval.py --checkpoint %CKPT% --metadata data\intestinal_crops\val\crop_metadata.json --crops-dir data\intestinal_crops\val\crops --device cuda:0 --tag intestinal_val
if %errorlevel% neq 0 (
    echo [ERROR] Cross-domain eval failed!
    pause
    exit /b 1
)
echo.

REM ============================================================
REM Pack results
REM ============================================================
echo Packing results into zip...
powershell -Command "Compress-Archive -Path 'results\cross_domain_intestinal_val' -DestinationPath '%OUT_ZIP%' -Force"

if exist %OUT_ZIP% (
    echo.
    echo ============================================================
    echo  Done! Results packed into: %OUT_ZIP%
    echo  Send me this zip file.
    echo ============================================================
) else (
    echo [WARN] Zip creation failed. Results are in:
    echo   results\cross_domain_intestinal_val\cross_domain_results.json
)

pause
