@echo off
echo ==========================================
echo   StreamClipMaker AI Cockpit Launcher
echo ==========================================

:: Start Backend API in a new window
echo Starting Backend API...
start "Cockpit Backend" cmd /k "uvicorn api:app --reload"

:: Wait a moment for backend to initialize
timeout /t 2 >nul

:: Start Frontend UI in a new window
echo Starting Frontend UI...
start "Cockpit Frontend" cmd /k "cd frontend && npm run dev"

:: Wait for Vite to spin up and open browser
timeout /t 5 >nul
echo Opening Cockpit...
start http://localhost:5173

echo.
echo Dashboard is now launching!
echo Keep the other two windows open while working.
echo.
pause
