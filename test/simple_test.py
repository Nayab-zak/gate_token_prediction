#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Load Random Forest data
df = pd.read_csv('data/predictions/random_forest/random_forest_test_preds_20250722_112931.csv')
print(f'Loaded {len(df)} rows')

# Calculate metrics
mae = np.mean(np.abs(df['true_count'] - df['pred_count']))
rmse = np.sqrt(np.mean((df['true_count'] - df['pred_count']) ** 2))
mape = np.mean(np.abs((df['true_count'] - df['pred_count']) / df['true_count'])) * 100

print(f'MAE: {mae}')
print(f'RMSE: {rmse}') 
print(f'MAPE: {mape}')
print(f'Any NaN? MAE: {np.isnan(mae)}, RMSE: {np.isnan(rmse)}, MAPE: {np.isnan(mape)}')
