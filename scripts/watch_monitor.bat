@echo off
title STRIDE - Training Monitor
cd /d "%~dp0.."
".venv\Scripts\python.exe" "code\watch_training.py"
pause
