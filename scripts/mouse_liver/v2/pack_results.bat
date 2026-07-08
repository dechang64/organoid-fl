@echo off
chcp 65001 >nul
REM Pack mouse_liver_v2 results — only JSON + TXT + CSV + YAML
REM Run this in C:\Users\decha\organoid-fl\

set SRC=runs\mouse_liver_v2
set DST=mouse_liver_v2_results.zip

if not exist %SRC% (
    echo [ERROR] %SRC% not found
    pause
    exit /b 1
)

echo Packing %SRC% -> %DST%
echo Only .json .txt .csv .yaml .yml files, everything else excluded
echo.
powershell -Command "Get-ChildItem -Path '%SRC%' -Recurse -File | Where-Object { $_.Extension -in '.json','.txt','.csv','.yaml','.yml' } | Compress-Archive -DestinationPath '%DST%' -Force"

if exist %DST% (
    echo.
    echo Done! %DST%
    dir %DST%
) else (
    echo [ERROR] Pack failed
)
pause
