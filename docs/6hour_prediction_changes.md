# 6-Hour Prediction Horizon Changes

## Overview
This document outlines the changes made to adjust the training pipeline from 1-hour to 6-hour prediction horizon.

## Key Changes Made

### 1. Feature Engineering (`app/data/feature_gen.py`)
- **Modified `add_lags()`**: Added `prediction_horizon_hours` parameter that adjusts lag features
  - For 6-hour prediction: `lag_1h` becomes effective `lag_7h` (1 + 6)
  - For 6-hour prediction: `lag_24h` becomes effective `lag_30h` (24 + 6)

- **Modified `add_rollings()`**: Added `prediction_horizon_hours` parameter that adjusts rolling window base
  - Rolling windows now start from `shift(1 + prediction_horizon_hours)` instead of `shift(1)`

- **Added `create_target_for_horizon()`**: Creates future target variable
  - For 6-hour prediction: Creates `TokenCount_future_6h` by shifting target backwards by 6 hours

### 2. Training Pipeline (`app/pipelines/train.py`)
- **Added horizon awareness**: Gets `prediction_horizon_hours` from config (defaults to 6)
- **Modified target creation**: Uses future target (`TokenCount_future_6h`) instead of current target
- **Updated feature generation**: Passes horizon parameter to lag and rolling functions
- **Enhanced metadata**: Records prediction horizon in model metadata for validation

### 3. Prediction Pipeline (`app/pipelines/predict.py`)
- **Consistent feature generation**: Uses same horizon-adjusted features as training
- **Aligned with training**: Ensures prediction uses same feature engineering logic

### 4. Feature Configuration (`config/features.yaml`)
- **Enhanced lag features**: Added `lag_6h` for better 6-hour prediction
- **Extended rolling windows**: Added 48-hour rolling mean for longer-term patterns

### 5. Main Configuration (`config/config.yaml`)
- **Documentation**: Added clear comments explaining dual role of `prediction_frequency_hours`

## Technical Details

### How 6-Hour Prediction Works
1. **Training Data**: 
   - Features use data from 7+ hours ago (lag_1h + 6h horizon)
   - Target is the actual value 6 hours in the future
   - Model learns: `f(data_t-7, data_t-12, data_t-30) = actual_t+6`

2. **Prediction**:
   - Uses same feature engineering (data from 7+ hours ago relative to prediction time)
   - Predicts 6 hours ahead from the base time
   - Maintains temporal consistency with training

### Validation
- **Leakage Protection**: Updated to check against future target
- **Null Handling**: Accounts for additional nulls from future target creation
- **Metadata Tracking**: Records prediction horizon for model validation

## Benefits
1. **Temporal Consistency**: Training and prediction now use the same time relationships
2. **Better Features**: Longer lag features capture more relevant historical patterns
3. **Validation**: Model metadata includes horizon information for verification
4. **Flexibility**: Easy to change horizon by updating config value

## Next Steps
1. **Retrain Model**: Run training pipeline to create 6-hour horizon model
2. **Validate**: Check that prediction accuracy improves with proper temporal alignment
3. **Monitor**: Ensure predictions are generated for correct 6-hour ahead timeframe
