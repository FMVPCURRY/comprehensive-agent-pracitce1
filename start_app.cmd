@echo off
setlocal
cd /d "%~dp0"

set PYTHON_EXE=D:\anaconda\envs\nlp\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

echo Starting Smart Shield at http://127.0.0.1:7871
"%PYTHON_EXE%" -B web_app.py --host 127.0.0.1 --port 7871
