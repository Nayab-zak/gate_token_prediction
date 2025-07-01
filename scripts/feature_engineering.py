import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from agents.feature_engineer_agent import FeatureEngineerAgent
from agents.config import PREPROCESSED_FEATURES_PATH, EDA_PATH

PREP_PATH = PREPROCESSED_FEATURES_PATH
OUT_PATH = PREP_PATH  # Overwrite or change if you want a new file

if not os.path.exists(PREP_PATH):
    raise FileNotFoundError(f"Preprocessed input file not found: {PREP_PATH}")

# 1. Load preprocessed data
df = pd.read_csv(PREP_PATH, parse_dates=['MoveDate'])

# 2. Feature engineering
agent = FeatureEngineerAgent()
df_feat = agent.add_holiday_flags(df)
df_feat = agent.add_lagged_features(df_feat, cols=['TokenCount'], lags=[1, 24, 168])
df_feat = agent.add_rolling_features(df_feat, cols=['TokenCount'], windows=[24, 168], aggs=['mean', 'max'])
# Add year, month, MoveHour, and is_weekend columns for EDA filters
if 'MoveDate' in df_feat.columns:
    df_feat['year'] = pd.to_datetime(df_feat['MoveDate'], errors='coerce').dt.year
    df_feat['month'] = pd.to_datetime(df_feat['MoveDate'], errors='coerce').dt.month
    # Use MoveHour from the data, not from MoveDate
    if 'MoveHour' in df_feat.columns:
        df_feat['MoveHour'] = pd.to_numeric(df_feat['MoveHour'], errors='coerce')
    else:
        df_feat['MoveHour'] = pd.to_datetime(df_feat['MoveDate'], errors='coerce').dt.hour
    # Drop rows with missing MoveDate or MoveHour
    df_feat = df_feat[df_feat['MoveDate'].notna() & df_feat['MoveHour'].notna()]
    df_feat['MoveHour'] = df_feat['MoveHour'].astype(int)
    # Add is_weekend column (Saturday=5, Sunday=6)
    df_feat['is_weekend'] = pd.to_datetime(df_feat['MoveDate'], errors='coerce').dt.dayofweek.isin([5, 6])
# 3. Save engineered features
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df_feat.to_csv(OUT_PATH, index=False)
# Save a separate Excel for EDA
eda_path = EDA_PATH
df_feat.to_excel(eda_path, index=False)
print(f"[feature_engineering.py] Feature engineered data saved to {OUT_PATH} and {eda_path}")
