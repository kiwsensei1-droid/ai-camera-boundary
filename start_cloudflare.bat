@echo off
chcp 65001 >nul
title Cloudflare Online Host
echo.
echo  ==============================================================
echo     🌐 กำลังสร้างลิงก์ออนไลน์ Cloudflare Tunnel (ฟรี ปลอดภัย)
echo  ==============================================================
echo.
echo  [1/2] กำลังเปิดเซิร์ฟเวอร์ AI Detection...
start "AI Camera Server" cmd /k "python app.py"

echo  [2/2] กำลังเชื่อมต่อ Cloudflare...
echo.
echo  ==============================================================
echo   👉 สังเกตหาลิงก์ด้านล่างที่ลงท้ายด้วย: .trycloudflare.com
echo   👉 นำลิงก์นั้นไปเปิดบนมือถือหรือส่งให้คนอื่นได้เลย (เข้าได้ทุกที่!)
echo  ==============================================================
echo.

cloudflared.exe tunnel --url http://localhost:5000
pause
