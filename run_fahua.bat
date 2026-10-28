@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONUNBUFFERED=1
call "%USERPROFILE%\anaconda3\Scripts\activate.bat" || (echo Anaconda not found & pause & exit /b 1)
call conda activate fahua || (echo 先运行 setup.bat & pause & exit /b 1)
powershell -NoProfile -ExecutionPolicy Bypass -Command "$OutputEncoding=[Console]::OutputEncoding=[Text.Encoding]::UTF8; python run_fahua.py %* 2>&1 | Tee-Object -FilePath run_log.txt"
pause
