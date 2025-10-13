# Prediction and Actuals Management

## Overview
This document explains how to handle predictions when actual values aren't immediately available - a common challenge in real-time ML systems.

## The Problem
When making predictions for future time periods (e.g., predicting token counts for the next hour), the actual values won't be available until that time period has passed. However, you need to store predictions immediately for:
- Real-time decision making
- Model performance monitoring  
- Business reporting

## Our Solution: Backfill Strategy

### 1. **Initial Prediction Storage**
- Store predictions immediately with `TokenCount_actual = NULL`
- Include all metadata (model version, prediction timestamp, etc.)
- Use unique identifiers (`SUR_GKEY`, `BI_BATCH_ID`) for tracking

### 2. **Automated Backfill Process** 
- Run periodically (every 6 hours) to update predictions with actual values
- Only update predictions where enough time has passed for actuals to be available
- Track when each record was last updated (`BI_UPDATED`)

### 3. **Performance Monitoring**
- Compute accuracy metrics (MAE, RMSE) for predictions with actuals
- Compare against expected model performance from cross-validation
- Alert on model degradation

## Usage

### Manual Backfill
```bash
# Dry run to see what would be updated
python -m app.cli backfill --days-back 7 --dry-run

# Actually update the predictions  
python -m app.cli backfill --days-back 7
```

### Check Model Accuracy
```bash
# Get accuracy metrics for last 30 days
python -m app.cli accuracy --days-back 30
```

### Automated Setup
1. Update `config/backfill.yaml` with your source table details
2. Set up Windows Task Scheduler to run `scripts/automated_backfill.bat` every 6 hours
3. Configure email notifications if desired

## Database Schema
```sql
-- Predictions table structure
CREATE TABLE predictions (
    MoveDate DATE,
    MoveHour INT, 
    MoveType VARCHAR(64),
    TerminalID VARCHAR(64),
    Desig VARCHAR(64),
    TokenCount_actual FLOAT,      -- NULL until backfilled
    TokenCount_pred FLOAT,        -- Available immediately
    prediction_ts_utc TIMESTAMP,  -- When prediction was made
    model_name VARCHAR(64),
    model_version VARCHAR(64),
    SUR_GKEY INT,                 -- Unique identifier
    BI_CREATED TIMESTAMP,         -- Record creation time
    BI_UPDATED TIMESTAMP,         -- Last update time (backfill)
    BI_BATCH_ID VARCHAR(64),      -- Batch identifier
    -- ... other columns
);
```

## Best Practices

### ✅ Do
- Store predictions immediately, even without actuals
- Use proper timestamps to track prediction and actual times
- Implement automated backfill processes
- Monitor model performance continuously
- Use unique identifiers for tracking individual predictions

### ❌ Don't  
- Wait for actuals before storing predictions
- Manually update actuals (automate it)
- Ignore model performance monitoring
- Use the same timestamp for predictions and actuals
- Forget to handle cases where actuals never arrive

## Monitoring Queries

### Find predictions needing backfill
```sql
SELECT COUNT(*) as predictions_needing_actuals
FROM predictions  
WHERE TokenCount_actual IS NULL
  AND prediction_ts_utc < CURRENT_TIMESTAMP - INTERVAL '2 HOURS';
```

### Model performance over time
```sql  
SELECT 
    DATE(prediction_ts_utc) as prediction_date,
    AVG(ABS(TokenCount_actual - TokenCount_pred)) as daily_mae
FROM predictions
WHERE TokenCount_actual IS NOT NULL
  AND prediction_ts_utc >= CURRENT_DATE - INTERVAL '30 DAYS'
GROUP BY DATE(prediction_ts_utc)
ORDER BY prediction_date;
```

## Troubleshooting

### "No actuals found for backfill"
- Check if your source table name is correct in `config/backfill.yaml`
- Verify the time window - actuals might not be available yet
- Check if there are any data pipeline delays

### "Model accuracy degrading"
- Compare with recent model retraining results
- Check for data drift in features
- Verify prediction timestamps are correct
- Consider model retraining if degradation persists

## Configuration Files
- `config/backfill.yaml` - Backfill settings and thresholds
- `scripts/automated_backfill.bat` - Windows automation script
- `app/pipelines/backfill_actuals.py` - Core backfill logic
