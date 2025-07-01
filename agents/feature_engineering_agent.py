import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INPUT_FILE_PATH

import pandas as pd
import numpy as np

def compute_features(df):
    col_map = {
        'MOVEDATE':'MoveDate', 'MOVEHOUR':'MoveHour',
        'TERMINAL_ID':'TerminalID', 'Tokencount':'TokenCount', 'contr_cnt':'ContainerCount'
    }
    for k, v in col_map.items():
        if k in df.columns:
            df.rename(columns={k: v}, inplace=True)
    if 'MoveDate' in df.columns:
        df['MoveDate'] = pd.to_datetime(df['MoveDate'], dayfirst=True, errors='coerce')
    if 'MoveHour' in df.columns:
        if np.issubdtype(df['MoveHour'].dtype, np.datetime64):
            df['MoveHour'] = df['MoveHour'].dt.hour
        else:
            df['MoveHour'] = pd.to_numeric(df['MoveHour'], errors='coerce')
        df = df[df['MoveHour'].notna()]
        df['MoveHour'] = df['MoveHour'].astype(int)
    if 'MoveDate' in df.columns:
        df['year'] = df.MoveDate.dt.year
        df['month'] = df.MoveDate.dt.month
        df['dayofweek'] = df.MoveDate.dt.dayofweek
    if 'MoveHour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df.MoveHour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.MoveHour / 24)
    # --- Robust date_hour creation for dashboard time series ---
    if 'MoveDate' in df.columns and 'MoveHour' in df.columns:
        df['date_hour'] = df['MoveDate'].dt.strftime('%Y-%m-%d') + ' ' + df['MoveHour'].astype(str).str.zfill(2) + ':00'
        df['date_hour'] = pd.to_datetime(df['date_hour'], format='%Y-%m-%d %H:%M', errors='coerce')
        df = df[df['date_hour'].notna()]
        # Convert to Python datetime for downstream compatibility
        df['date_hour'] = df['date_hour'].dt.to_pydatetime()
    return df

def main():
    PREPROCESSED_PATH = os.path.join('data', 'preprocessed', 'preprocessed_features.csv')
    os.makedirs(os.path.dirname(PREPROCESSED_PATH), exist_ok=True)
    ext = os.path.splitext(INPUT_FILE_PATH)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(INPUT_FILE_PATH)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(INPUT_FILE_PATH)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    df = compute_features(df)
    df.to_csv(PREPROCESSED_PATH, index=False)
    print(f"Preprocessed data saved to {PREPROCESSED_PATH}")

if __name__ == "__main__":
    main()
