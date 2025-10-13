# LightGBM Feature Names Warning Fix

## Problem
LightGBM was showing this warning:
```
UserWarning: X does not have valid feature names, but LGBMRegressor was fitted with feature names
```

## Root Cause
The issue occurred because:
1. **Training**: LightGBM models were trained with pandas DataFrames (which have feature names)
2. **Prediction**: The preprocessor was converting DataFrames to numpy arrays (which lose feature names)
3. **Mismatch**: LightGBM expected feature names during prediction but received numpy arrays

## Solution Applied

### 1. Enhanced Preprocessor (`app/models/preprocessors.py`)
- Created `FeatureNamePreservingTransformer` class
- Wraps sklearn's `ColumnTransformer` but returns DataFrames instead of numpy arrays
- Preserves feature names through the transformation pipeline
- Maintains backward compatibility with existing interface

### 2. Updated LightGBM Model (`app/models/lightgbm_.py`)
- Removed conversions from DataFrame to numpy array
- Now accepts and works with DataFrames directly
- Preserves feature names throughout training and prediction

### 3. Updated Quantile Models (`app/models/intervals.py`)
- Modified `QuantileLGBM` class to work with DataFrames
- Removed numpy array conversions
- Maintains feature name consistency

### 4. Updated Prediction Pipeline (`app/pipelines/predict.py`)
- Changed variable name from `X_np` to `X_processed` to reflect DataFrame usage
- Updated `predict_intervals` function to handle DataFrames
- Maintains feature name consistency throughout prediction pipeline

## Technical Details

### Before (Problematic Flow):
```
DataFrame → Preprocessor → numpy array → LightGBM.predict() → WARNING
```

### After (Fixed Flow):
```
DataFrame → Preprocessor → DataFrame (with feature names) → LightGBM.predict() → ✓
```

### Feature Name Generation
The preprocessor automatically generates proper feature names:
- **Numeric features**: Keep original names (e.g., `lag_1h`, `roll_mean_24h`)  
- **Categorical features**: Use one-hot encoded names (e.g., `TerminalID_T1`, `MoveType_In`)

## Benefits
1. **No More Warnings**: Eliminates the sklearn validation warning
2. **Better Debugging**: Feature names are preserved for model interpretability
3. **Consistency**: Same data types used in training and prediction
4. **Backward Compatible**: No changes needed to existing model training code

## Testing
- All modified files have no syntax errors
- Preprocessor maintains same interface as original
- Feature name generation works for both numeric and categorical features
- Compatible with all existing model types (not just LightGBM)

## Files Modified
- `app/models/preprocessors.py` - Enhanced with feature name preservation
- `app/models/lightgbm_.py` - Updated to work with DataFrames
- `app/models/intervals.py` - Updated quantile models for DataFrames  
- `app/pipelines/predict.py` - Updated to handle DataFrames throughout pipeline
