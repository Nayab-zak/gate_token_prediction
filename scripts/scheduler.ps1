# 6-Hour Prediction & Backfill Scheduler
# Runs continuously - No admin access required
# Schedule: 08:30, 14:30, 20:30, 02:30 daily
# Keep PowerShell window open - Press Ctrl+C to stop

function Write-Banner {
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "🚀 6-Hour Prediction & Backfill Scheduler" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "📅 Schedule: 08:30, 14:30, 20:30, 02:30 daily" -ForegroundColor Green
    Write-Host "⚡ No admin access required" -ForegroundColor Green
    Write-Host "🖥️  Keep this PowerShell window open" -ForegroundColor Green
    Write-Host "🛑 Press Ctrl+C to stop" -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Get-NextRunTime {
    $now = Get-Date
    
    # Define schedule times for today
    $scheduleToday = @(
        $now.Date.AddHours(8).AddMinutes(30)   # 8:30 AM
        $now.Date.AddHours(14).AddMinutes(30)  # 2:30 PM
        $now.Date.AddHours(20).AddMinutes(30)  # 8:30 PM
    )
    
    # Find next time today
    foreach ($scheduleTime in $scheduleToday) {
        if ($now -lt $scheduleTime) {
            return $scheduleTime
        }
    }
    
    # If all times today have passed, use 2:30 AM tomorrow
    $tomorrow = $now.Date.AddDays(1).AddHours(2).AddMinutes(30)
    return $tomorrow
}

function Run-PredictionPipeline {
    Write-Host "📊 Running prediction pipeline..." -ForegroundColor Blue
    try {
        $result = & python -m app.cli predict 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Prediction completed successfully" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ Prediction failed: $result" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ Prediction error: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Run-BackfillPipeline {
    Write-Host "🔄 Running backfill pipeline..." -ForegroundColor Blue
    try {
        $result = & python -m app.cli backfill 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Backfill completed successfully" -ForegroundColor Green
            return $true
        } else {
            Write-Host "❌ Backfill failed: $result" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "❌ Backfill error: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

function Run-FullPipeline {
    $startTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host ""
    Write-Host "🚀 PIPELINE START    $startTime" -ForegroundColor Yellow
    Write-Host "--------------------------------------------------" -ForegroundColor Gray
    
    # Step 1: Prediction
    $predictionSuccess = Run-PredictionPipeline
    
    if (-not $predictionSuccess) {
        Write-Host "⚠️  Skipping backfill due to prediction failure" -ForegroundColor Yellow
        return $false
    }
    
    # Wait 5 seconds between operations
    Start-Sleep -Seconds 5
    
    # Step 2: Backfill
    $backfillSuccess = Run-BackfillPipeline
    
    Write-Host "--------------------------------------------------" -ForegroundColor Gray
    $endTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    
    if ($predictionSuccess -and $backfillSuccess) {
        Write-Host "🎉 PIPELINE COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    } else {
        Write-Host "⚠️  PIPELINE COMPLETED WITH ERRORS" -ForegroundColor Yellow
    }
    Write-Host "🏁 PIPELINE END      $endTime" -ForegroundColor Yellow
    
    return ($predictionSuccess -and $backfillSuccess)
}

function Format-Duration {
    param([int]$Seconds)
    
    $hours = [math]::Floor($Seconds / 3600)
    $minutes = [math]::Floor(($Seconds % 3600) / 60)
    
    if ($hours -gt 0) {
        return "${hours}h ${minutes}m"
    } else {
        return "${minutes}m"
    }
}

function Start-Scheduler {
    Write-Banner
    
    try {
        while ($true) {
            $now = Get-Date
            $nextRun = Get-NextRunTime
            
            Write-Host "⏰ Current time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor White
            Write-Host "🎯 Next run: $($nextRun.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor White
            
            # Calculate sleep time
            $sleepSeconds = [math]::Max(0, ($nextRun - $now).TotalSeconds)
            
            if ($sleepSeconds -gt 0) {
                $duration = Format-Duration -Seconds $sleepSeconds
                Write-Host "😴 Waiting $duration until next run..." -ForegroundColor Cyan
                Write-Host "   (Keep this window open)" -ForegroundColor Gray
                Write-Host ""
                
                # Sleep until next run time
                Start-Sleep -Seconds $sleepSeconds
            }
            
            # Run the full pipeline
            Run-FullPipeline
            
            $nextCycle = $nextRun.AddHours(6)
            Write-Host ""
            Write-Host "🔄 Next cycle in 6 hours at $($nextCycle.ToString('HH:mm'))" -ForegroundColor Magenta
            Write-Host "============================================================" -ForegroundColor Cyan
            Write-Host ""
            
            # Sleep for 6 hours minus a few seconds to avoid timing issues
            Start-Sleep -Seconds (6 * 3600 - 10)
        }
    }
    catch [System.Management.Automation.PipelineStoppedException] {
        Write-Host ""
        Write-Host ""
        Write-Host "🛑 Scheduler stopped by user" -ForegroundColor Red
        Write-Host "👋 Goodbye!" -ForegroundColor Yellow
        exit 0
    }
    catch {
        Write-Host ""
        Write-Host "❌ Scheduler error: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "🔄 Restarting in 60 seconds..." -ForegroundColor Yellow
        Start-Sleep -Seconds 60
        Start-Scheduler  # Restart on error
    }
}

# Main execution
# Change to project directory
$projectDir = Split-Path -Parent $PSScriptRoot
Set-Location $projectDir

# Start the scheduler
Start-Scheduler
