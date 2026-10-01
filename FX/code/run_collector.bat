@echo off
REM Phase 1 daily collector, 2026-08-31 .. 2026-09-11.
REM
REM Starts at 03:00 ET to cover the London open through the NY close. The spec
REM emphasises the 08:00-17:00 ET overlap, and that window is what the analysis
REM filters to -- but EUR/GBP is a European cross whose most active hours are
REM the London morning, and that data is additive: it is tagged session="other"
REM so it can never contaminate an overlap-filtered view.
REM
REM Collection deliberately runs THROUGH the 16:55 ET rollover. The spread
REM blowout is itself a measurement (see collect_ticks.py).
REM
REM Appends to collector.log. Every gap, reconnect and kill switch also lands in
REM data/events/<date>.jsonl, which is the authoritative record.

setlocal
REM Interpreter: honour a preset PY, else this machine's per-user install,
REM else whatever python is on PATH.
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if not defined PY set "PY=python"
cd /d "%~dp0"

echo. >> collector.log
echo ================================================== >> collector.log
echo START %DATE% %TIME% >> collector.log

REM Call python DIRECTLY. Do NOT wrap this in `start` -- `start` redirects only
REM itself, so the collector's output would bypass collector.log, and the child
REM would get a visible console window that kills the run if anyone closes it.
REM Priority is raised in-process instead (collect_ticks.raise_priority).
"%PY%" collect_ticks.py --source oanda --until 17:00 --quiet >> collector.log 2>&1
set RC=%ERRORLEVEL%

echo END %DATE% %TIME% exit=%RC% >> collector.log
exit /b %RC%
