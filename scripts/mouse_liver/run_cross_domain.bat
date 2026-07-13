@echo off
chcp 65001 >nul

REM Switch to project root (2 levels up from this bat)
cd /d %~dp0\..\..

echo ============================================================
echo  Cross-Domain Evaluation: MultiOrg Slot Model -> Mouse Liver
echo ============================================================
echo.
echo  Working dir: %CD%
echo.

REM ============================================================
REM Config - modify paths here if needed
REM ============================================================
set CKPT=results\supcon_8s_d128_p256_t0.07_b0.1_20260713_003826\best.pt
set DATA_ROOT=mouse_liver_data_correct
set RUNS_ROOT=runs\mouse_liver_v2
set OUT_ZIP=results\cross_domain_results.zip

REM Verify checkpoint exists
if not exist %CKPT% (
    echo [ERROR] Checkpoint not found: %CKPT%
    echo Please check the path. You may need to update CKPT in this bat.
    pause
    exit /b 1
)

REM Verify data root exists
if not exist %DATA_ROOT% (
    echo [ERROR] Data root not found: %DATA_ROOT%
    echo Current dir: %CD%
    echo Please check the path. You may need to update DATA_ROOT in this bat.
    pause
    exit /b 1
)

echo  Checkpoint: %CKPT%
echo  Data root:  %DATA_ROOT%
echo  Runs root:  %RUNS_ROOT%
echo.

REM ============================================================
REM B1
REM ============================================================
echo [1/6] Generating B1 crops...
set B1_WEIGHTS=%RUNS_ROOT%\b1\full\checkpoint_best_regular.pth
if not exist %B1_WEIGHTS% (
    echo [ERROR] B1 weights not found: %B1_WEIGHTS%
    echo Looking for alternatives...
    dir /s /b %RUNS_ROOT%\b1\*checkpoint*.pth 2>nul
    pause
    exit /b 1
)
python scripts\mouse_liver\generate_mouse_crops.py --batch b1 --weights %B1_WEIGHTS% --src %DATA_ROOT%\batch1\images --annotations %DATA_ROOT%\batch1\annotations.json --dst data\mouse_crops\b1 --resolution 544
if %errorlevel% neq 0 (
    echo [ERROR] B1 crop generation failed!
    pause
    exit /b 1
)
echo.

echo [2/6] Cross-domain eval B1...
python scripts\mouse_liver\cross_domain_eval.py --checkpoint %CKPT% --metadata data\mouse_crops\b1\crop_metadata.json --crops-dir data\mouse_crops\b1\crops --device cuda:0 --tag b1
echo.

REM ============================================================
REM B2
REM ============================================================
echo [3/6] Generating B2 crops...
set B2_WEIGHTS=%RUNS_ROOT%\b2\full\checkpoint_best_regular.pth
if not exist %B2_WEIGHTS% (
    echo [ERROR] B2 weights not found: %B2_WEIGHTS%
    dir /s /b %RUNS_ROOT%\b2\*checkpoint*.pth 2>nul
    pause
    exit /b 1
)
python scripts\mouse_liver\generate_mouse_crops.py --batch b2 --weights %B2_WEIGHTS% --src %DATA_ROOT%\batch2\images --annotations %DATA_ROOT%\batch2\annotations.json --dst data\mouse_crops\b2 --resolution 768
if %errorlevel% neq 0 (
    echo [ERROR] B2 crop generation failed!
    pause
    exit /b 1
)
echo.

echo [4/6] Cross-domain eval B2...
python scripts\mouse_liver\cross_domain_eval.py --checkpoint %CKPT% --metadata data\mouse_crops\b2\crop_metadata.json --crops-dir data\mouse_crops\b2\crops --device cuda:0 --tag b2
echo.

REM ============================================================
REM B3
REM ============================================================
echo [5/6] Generating B3 crops...
set B3_WEIGHTS=%RUNS_ROOT%\b3\full\checkpoint_best_regular.pth
if not exist %B3_WEIGHTS% (
    echo [ERROR] B3 weights not found: %B3_WEIGHTS%
    dir /s /b %RUNS_ROOT%\b3\*checkpoint*.pth 2>nul
    pause
    exit /b 1
)
python scripts\mouse_liver\generate_mouse_crops.py --batch b3 --weights %B3_WEIGHTS% --src %DATA_ROOT%\batch3\images --annotations %DATA_ROOT%\batch3\annotations.json --dst data\mouse_crops\b3 --resolution 768
if %errorlevel% neq 0 (
    echo [ERROR] B3 crop generation failed!
    pause
    exit /b 1
)
echo.

echo [6/6] Cross-domain eval B3...
python scripts\mouse_liver\cross_domain_eval.py --checkpoint %CKPT% --metadata data\mouse_crops\b3\crop_metadata.json --crops-dir data\mouse_crops\b3\crops --device cuda:0 --tag b3
echo.

REM ============================================================
REM Pack results into zip
REM ============================================================
echo Packing results into zip...
powershell -Command "Compress-Archive -Path 'results\cross_domain_b1\cross_domain_results.json','results\cross_domain_b1\embeddings.npy','results\cross_domain_b1\labels.npy','results\cross_domain_b1\confs.npy','results\cross_domain_b2\cross_domain_results.json','results\cross_domain_b2\embeddings.npy','results\cross_domain_b2\labels.npy','results\cross_domain_b2\confs.npy','results\cross_domain_b3\cross_domain_results.json','results\cross_domain_b3\embeddings.npy','results\cross_domain_b3\labels.npy','results\cross_domain_b3\confs.npy' -DestinationPath '%OUT_ZIP%' -Force"

if exist %OUT_ZIP% (
    echo.
    echo ============================================================
    echo  Done! All results packed into: %OUT_ZIP%
    echo  Send me this zip file.
    echo ============================================================
) else (
    echo [WARN] Zip creation failed. Results are in:
    echo   results\cross_domain_b1\cross_domain_results.json
    echo   results\cross_domain_b2\cross_domain_results.json
    echo   results\cross_domain_b3\cross_domain_results.json
)

pause
