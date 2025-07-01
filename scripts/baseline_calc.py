import pandas as pd
import numpy as np
import json
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
from agents.config import PREPROCESSED_FEATURES_PATH, BASELINE_METRICS_PATH

# 1. Load cleaned data (parquet or csv)
input_path = PREPROCESSED_FEATURES_PATH
if os.path.exists(input_path):
    df = pd.read_csv(input_path, parse_dates=['MoveDate'])
else:
    input_path = 'data/preprocessed/preprocessed_features.parquet'
    df = pd.read_parquet(input_path)

# 2. Compute hourly mean and same-hour-last-week predictions
assert 'MoveDate' in df.columns and 'TokenCount' in df.columns, 'Required columns missing.'
df = df.sort_values('MoveDate')
df['hour'] = df['MoveDate'].dt.hour

# Hourly mean prediction
hourly_mean = df.groupby('hour')['TokenCount'].mean()
df['hourly_mean_pred'] = df['hour'].map(hourly_mean)

# Same-hour-last-week prediction (shift by 168 hours for weekly data)
df['last_week_pred'] = df['TokenCount'].shift(168)

# 3. Print MAE/RMSE for both baselines (drop NA rows for fair comparison)
results = {}
for col in ['hourly_mean_pred', 'last_week_pred']:
    mask = ~df[col].isna() & ~df['TokenCount'].isna()
    mae = mean_absolute_error(df.loc[mask, 'TokenCount'], df.loc[mask, col])
    # Remove 'squared' argument for compatibility
    mse = mean_squared_error(df.loc[mask, 'TokenCount'], df.loc[mask, col])
    rmse = mse ** 0.5
    print(f"{col}: MAE={mae:.2f}, RMSE={rmse:.2f}")
    results[col] = {'MAE': mae, 'RMSE': rmse}

# 4. Save outputs to outputs/baseline_metrics.json
os.makedirs('outputs', exist_ok=True)
with open(BASELINE_METRICS_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Metrics saved to {BASELINE_METRICS_PATH}')
