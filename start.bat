@echo off
chcp 65001 >nul
echo.
echo  ╔══════════════════════════════════════╗
echo  ║   🎯 Person Detection Web App         ║
echo  ║   เปิด: http://localhost:5000         ║
echo  ╚══════════════════════════════════════╝
echo.
start "" "http://localhost:5000"
python app.py
pause
