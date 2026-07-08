@echo off
chcp 65001 >nul
REM Pack mouse_liver_v2 results (exclude .pth checkpoints)
REM Run this in C:\Users\decha\organoid-fl\

set SRC=runs\mouse_liver_v2
set DST=mouse_liver_v2_results.zip

if not exist %SRC% (
    echo [ERROR] %SRC% not found
    pause
    exit /b 1
)

echo Packing %SRC% -> %DST% (excluding .pth files)...
powershell -Command "Get-ChildItem -Path '%SRC%' -Recurse -File | Where-Object { $_.Extension -ne '.pth' } | Compress-Archive -DestinationPath '%DST%' -Force"

if exist %DST% (
    echo.
    echo Done! %DST%
    echo Send this file to the assistant.
    dir %DST%
) else (
    echo [ERROR] Pack failed
)
pause
