@echo off
REM 6-Hour Prediction & Backfill Scheduler
REM Runs continuously - No admin access required
REM Schedule: 08:30, 14:30, 20:30, 02:30 daily
REM Keep command window open - Press Ctrl+C to stop

setlocal enabledelayedexpansion

echo ============================================================
echo 🚀 6-Hour Prediction ^& Backfill Scheduler
echo ============================================================
echo 📅 Schedule: 08:30, 14:30, 20:30, 02:30 daily
echo ⚡ No admin access required
echo 🖥️  Keep this window open
echo 🛑 Press Ctrl+C to stop
echo ============================================================
echo.

cd /d "c:\Users\nayabb.fatima\AI-Agents\predictive_modeling"

:main_loop
REM Get current time
for /f "tokens=1,2 delims=:" %%a in ("%time%") do (
    set hour=%%a
    set minute=%%b
)
set /a hour=%hour: =%
set /a minute=%minute: =%
set /a current_minutes=%hour%*60+%minute%

REM Calculate next run time (8:30=510, 14:30=870, 20:30=1230, 2:30=150 next day)
set "next_run_time="
set "next_description="

if %current_minutes% LSS 510 (
    set "next_run_time=08:30"
    set "next_description=8:30 AM today"
) else if %current_minutes% LSS 870 (
    set "next_run_time=14:30"
    set "next_description=2:30 PM today"
) else if %current_minutes% LSS 1230 (
    set "next_run_time=20:30"
    set "next_description=8:30 PM today"
) else (
    set "next_run_time=02:30"
    set "next_description=2:30 AM tomorrow"
)

echo ⏰ Current time: %date% %time%
echo 🎯 Next run: !next_description!
echo.

REM Calculate sleep time in seconds
set /a target_minutes=0
if "!next_run_time!"=="08:30" set /a target_minutes=510
if "!next_run_time!"=="14:30" set /a target_minutes=870
if "!next_run_time!"=="20:30" set /a target_minutes=1230
if "!next_run_time!"=="02:30" set /a target_minutes=150+1440

set /a sleep_minutes=!target_minutes!-!current_minutes!
if !sleep_minutes! LEQ 0 set /a sleep_minutes=!sleep_minutes!+1440

echo 😴 Waiting !sleep_minutes! minutes until next run...
echo    (Keep this window open)
echo.

REM Wait until run time (convert minutes to seconds, max 3600 per timeout)
set /a sleep_seconds=!sleep_minutes!*60
:wait_loop
if !sleep_seconds! GTR 3600 (
    timeout /t 3600 /nobreak >nul
    set /a sleep_seconds=!sleep_seconds!-3600
    goto wait_loop
) else (
    timeout /t !sleep_seconds! /nobreak >nul
)

REM Run the pipeline
echo.
echo 🚀 PIPELINE START    %date% %time%
echo --------------------------------------------------

echo 📊 Running prediction pipeline...
python -m app.cli predict
if !errorlevel! neq 0 (
    echo ❌ Prediction failed
    echo ⚠️  Skipping backfill due to prediction failure
    goto next_cycle
) else (
    echo ✅ Prediction completed successfully
)

timeout /t 5 /nobreak >nul

echo 🔄 Running backfill pipeline...
python -m app.cli backfill
if !errorlevel! neq 0 (
    echo ❌ Backfill failed
    echo ⚠️  PIPELINE COMPLETED WITH ERRORS
) else (
    echo ✅ Backfill completed successfully
    echo 🎉 PIPELINE COMPLETED SUCCESSFULLY!
)

:next_cycle
echo --------------------------------------------------
echo 🏁 PIPELINE END      %date% %time%
echo.
echo 🔄 Next cycle in 6 hours
echo ============================================================
echo.

REM Sleep for 6 hours (21600 seconds) in chunks
set /a remaining=21600
:sleep_6_hours
if !remaining! GTR 3600 (
    timeout /t 3600 /nobreak >nul
    set /a remaining=!remaining!-3600
    goto sleep_6_hours
) else (
    timeout /t !remaining! /nobreak >nul
)

goto main_loop
