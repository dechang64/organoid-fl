@echo off
chcp 65001 >nul
cd /d %~dp0\..\..\..
python scripts\mouse_liver\v2\pack_results.py
pause
