@echo off
REM Step 1: Prepare SAM2 training data (polygons → masks)
REM Usage: cd C:\Users\decha\organoid-fl && scripts\multiorg\run_sam2_prepare.bat

cd /d C:\Users\decha\organoid-fl
call .\.venv\Scripts\activate

REM 先跑 20 张测试，确认没问题再全量跑（去掉 --max-images 20）
python scripts\multiorg\prepare_sam2_data.py --src D:\datasets\mutliorg\MultiOrg_v2\train --dst data\multiorg_sam2 --annotator Annotator_A --max-images 20

pause
