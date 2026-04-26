@echo off
title StreamClipMaker GUI
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
start "" /B .\venv\Scripts\pythonw.exe gui.py %*
