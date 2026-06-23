@echo off
cd /d D:\third2\comprehensive\Baseline\ChiFraud-main
echo Starting Smart Shield on all network interfaces...
echo Open http://YOUR-IP:7871 from another device in the same LAN.
D:\anaconda\envs\nlp\python.exe -B web_app.py --host 0.0.0.0 --port 7871
