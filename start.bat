@echo off
chcp 65001 >nul
title VizML Platform Launcher
cd /d "%~dp0"

echo ============================================
echo    VizML Machine Learning Visual Lab
echo ============================================
echo.

if exist ".venv\python.exe" (
    echo [1/2] Starting with project virtualenv ...
    ".venv\python.exe" start.py
) else if exist ".venv\Scripts\python.exe" (
    echo [1/2] Starting with project virtualenv ...
    ".venv\Scripts\python.exe" start.py
) else (
    echo [1/2] virtualenv not found, using system Python ...
    python start.py
)

echo.
echo Server stopped.
pause
