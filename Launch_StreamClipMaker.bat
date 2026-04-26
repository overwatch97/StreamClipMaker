@echo off
title StreamClipMaker Auto-Shorts Launcher
color 0a

:: Change to the directory of this batch script
cd /d "%~dp0"

:: Check if the user dragged a file onto the script
if "%~1" == "" goto PROMPT_PATH
set VIDEO_PATH=%~1
goto RUN_SCRIPT

:PROMPT_PATH
echo ========================================================
echo Welcome to StreamClipMaker!
echo ========================================================
echo Hint: You can simply drag and drop a .mp4 file directly onto this .bat file next time!
echo.
set /p VIDEO_PATH="Please paste the full path to your video file and press Enter: "

if "%VIDEO_PATH%" == "" (
    echo Error: No video path provided!
    pause
    exit /b
)

:RUN_SCRIPT
:: Strip any surrounding quotes the user might have pasted
set VIDEO_PATH=%VIDEO_PATH:"=%

echo.
echo ========================================================
echo Launching StreamClipMaker Engine
echo Video: "%VIDEO_PATH%"
echo ========================================================
echo.

:: Ensure the output doesn't get messed up by emojis or buffering
set PYTHONIOENCODING=utf-8

:: Run the script using the local virtual environment!
call .\venv\Scripts\python.exe -u main.py "%VIDEO_PATH%"

echo.
echo ========================================================
echo Process Fully Completed! You may now close this window.
echo ========================================================
pause
