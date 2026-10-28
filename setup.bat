@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在安装，过程会同时写入 setup_log.txt（下载模型约 9GB，需要一段时间）……
powershell -NoProfile -ExecutionPolicy Bypass -Command "$OutputEncoding=[Console]::OutputEncoding=[Text.Encoding]::UTF8; cmd /c _setup_steps.bat 2>&1 | Tee-Object -FilePath setup_log.txt"
echo.
echo 安装结束。最后一行是 SETUP_OK 就说明成功了。
pause
