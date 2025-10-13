@echo off
REM TEST VERSION of 6-Hour Prediction & Backfill Scheduler
REM This version runs in the NEXT MINUTE for testing purposes
REM Schedule: Every minute for testing
REM Keep command window open - Press Ctrl+C to stop

setlocal enabledelayedexpansion

echo ============================================================
echo 🧪 TEST VERSION - 6-Hour Prediction ^& Backfill Scheduler
echo ============================================================
echo 📅 TEST Schedule: Runs in the NEXT MINUTE
echo ⚡ No admin access required
echo 🖥️  Keep this window open
echo 🛑 Press Ctrl+C to stop
echo ============================================================
echo.

cd /d "c:\Users\nayabb.fatima\AI-Agents\predictive_modeling"

:main_loop
REM Get current time
for /f "tokens=1,2,3 delims=:." %%a in ("%time%") do (
    set hour=%%a
    set minute=%%b
    set second=%%c
)
set /a hour=%hour: =%
set /a minute=%minute: =%
set /a second=%second: =%

REM Calculate next minute
set /a next_minute=%minute%+1
set /a next_hour=%hour%
if %next_minute% GEQ 60 (
    set /a next_minute=%next_minute%-60
    set /a next_hour=%next_hour%+1
    if !next_hour! GEQ 24 set /a next_hour=0
)

echo ⏰ Current time: %date% %time%
echo 🎯 Next test run: !next_hour!:!next_minute!:00
echo.

REM Calculate seconds to wait (wait until next minute)
set /a seconds_to_wait=60-%second%
if %seconds_to_wait% EQU 60 set /a seconds_to_wait=0

echo 😴 Waiting !seconds_to_wait! seconds until next test run...
echo    (This is a TEST - normally waits much longer)
echo.

REM Wait until next minute
if %seconds_to_wait% GTR 0 (
    timeout /t %seconds_to_wait% /nobreak >nul
)

REM Run the pipeline
echo.
echo 🚀 TEST PIPELINE START    %date% %time%
echo --------------------------------------------------

echo 📊 Running prediction pipeline (TEST MODE)...
python -m app.cli predict
if !errorlevel! neq 0 (
    echo ❌ Prediction failed
    echo ⚠️  Skipping backfill due to prediction failure
    goto next_cycle
) else (
    echo ✅ Prediction completed successfully
)

timeout /t 5 /nobreak >nul

echo 🔄 Running backfill pipeline (TEST MODE)...
python -m app.cli backfill
if !errorlevel! neq 0 (
    echo ❌ Backfill failed
    echo ⚠️  TEST PIPELINE COMPLETED WITH ERRORS
) else (
    echo ✅ Backfill completed successfully
    echo 🎉 TEST PIPELINE COMPLETED SUCCESSFULLY!
)

:next_cycle
echo --------------------------------------------------
echo 🏁 TEST PIPELINE END      %date% %time%
echo.
echo 🔄 TEST: Next cycle in 1 minute (instead of 6 hours)
echo ============================================================
echo.

REM For testing: Just wait 60 seconds instead of 6 hours
echo 😴 TEST MODE: Waiting 60 seconds for next cycle...
timeout /t 60 /nobreak >nul

goto main_loop
