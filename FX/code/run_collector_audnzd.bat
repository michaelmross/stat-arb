@echo off
REM Phase 1b collector, arm: AUD/NZD (AUD_USD + NZD_USD -> AUD_NZD), account -002
REM From Mon 2026-09-14. Same 03:00-17:00 ET window as the EUR/GBP census so
REM the pre-registered comparison (k below 13.7) is made over the same sessions.
REM Data: config_audnzd.toml [storage].root. Its own lock, so the arms cannot
REM block each other. Launched hidden by launch_arm.vbs; see run_collector.bat
REM for why python is called directly and never wrapped in `start`.

setlocal
REM Interpreter: honour a preset PY, else this machine's per-user install,
REM else whatever python is on PATH.
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if not defined PY set "PY=python"
cd /d "%~dp0"

echo. >> collector_audnzd.log
echo ================================================== >> collector_audnzd.log
echo START %DATE% %TIME% >> collector_audnzd.log

"%PY%" collect_ticks.py --config config_audnzd.toml --source oanda --until 17:00 --quiet >> collector_audnzd.log 2>&1
set RC=%ERRORLEVEL%

echo END %DATE% %TIME% exit=%RC% >> collector_audnzd.log
exit /b %RC%
