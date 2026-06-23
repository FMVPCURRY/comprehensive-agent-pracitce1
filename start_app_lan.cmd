@echo off
setlocal
cd /d "%~dp0"

set PYTHON_EXE=D:\anaconda\envs\nlp\python.exe
if not exist "%PYTHON_EXE%" set PYTHON_EXE=python

echo Starting Smart Shield for LAN access at http://0.0.0.0:7871
echo Use ipconfig to find this computer's IPv4 address, then open http://IPv4:7871 on another device.
"%PYTHON_EXE%" -B web_app.py --host 0.0.0.0 --port 7871
