@echo off
title QF Terminal
color 0A
cls
echo.
echo  ==========================================
echo    Q F   T E R M I N A L
echo    Quantitative Finance Dashboard
echo  ==========================================
echo.
echo  Starting server at http://127.0.0.1:8050
echo  Press Ctrl+C to stop.
echo.

"C:\Program Files\Python313\python.exe" "%~dp0dashboard\app.py"

echo.
echo  Server stopped.
pause
