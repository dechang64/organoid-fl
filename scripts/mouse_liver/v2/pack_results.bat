@echo off
chcp 65001 >nul
REM Pack mouse_liver_v2 results — preserve directory structure
REM Only .json .txt .csv .yaml .yml files, skip all binary
REM Run this in C:\Users\decha\organoid-fl\

set SRC=runs\mouse_liver_v2
set DST=mouse_liver_v2_results.zip

if not exist %SRC% (
    echo [ERROR] %SRC% not found
    pause
    exit /b 1
)

echo Packing %SRC% -> %DST%
echo Only .json .txt .csv .yaml .yml files, preserving directory structure
echo.

python -c ^
"import zipfile, os, sys^
src = r'%SRC%'^
dst = r'%DST%'^
n = 0^
with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zf:^
    for root, dirs, files in os.walk(src):^
        for f in files:^
            ext = os.path.splitext(f)[1].lower()^
            if ext in ('.json', '.txt', '.csv', '.yaml', '.yml'):^
                full = os.path.join(root, f)^
                rel = os.path.relpath(full, src)^
                zf.write(full, rel)^
                n += 1^
print(f'Done! {n} files -> {dst}')^
size_mb = os.path.getsize(dst) / 1024 / 1024^
print(f'Size: {size_mb:.1f} MB')"

pause
