@echo off
echo ===================================================
echo     Smart Road Safety System - 1-Click Startup
echo ===================================================
echo.

echo Starting Flask Backend Server on Port 5000...
start cmd /k "cd backend && python app.py"

echo Starting React Frontend Server on Port 5173...
start cmd /k "cd frontend && npm run dev"

echo.
echo Both servers are starting up! 
echo A new terminal window has opened for each server.
echo.
echo You can view your dashboard at: http://localhost:5173/
echo.
pause
